"""Inference: window blending, TTA, devices, progress/cancel, area stats, exports.

Most tests run a tiny deterministic stand-in for the U-Net, so they are fast
and need no data. Tests marked ``model`` compare against the trained
checkpoint on a VAL crop; the ``slow`` one runs the full VAL areas and only
when slow tests are asked for (``-m slow`` or URBANSEG_RUN_SLOW=1).
"""
import hashlib
import io
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from rasterio import features as rfeatures
from rasterio.transform import Affine
from shapely.geometry import Polygon, shape
from shapely.ops import transform as shp_transform

import helpers
from urbanseg import config, dataset, imagery, infer, labels


# --- frozen copies of the functions as they were before blending/TTA/area changes ---

@torch.no_grad()
def orig_predict_tta(model, img, **kw):
    accum = None
    for k, flip in ((0, False), (2, False), (0, True), (2, True)):
        v = np.rot90(img, k)
        if flip:
            v = v[:, ::-1]
        _, p = orig_predict(model, np.ascontiguousarray(v), **kw)
        if flip:
            p = p[:, :, ::-1]
        p = np.rot90(p, -k, axes=(1, 2))
        accum = p if accum is None else accum + p
    accum /= 4.0
    return accum.argmax(0).astype(np.uint8), accum


@torch.no_grad()
def orig_predict(model, img, window=1024, stride=768, batch=1):
    H, W = img.shape[:2]
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    im = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    Hp, Wp = im.shape[:2]
    win = min(window, Hp, Wp)
    probs = np.zeros((config.N_CLASSES, Hp, Wp), dtype=np.float32)
    count = np.zeros((Hp, Wp), dtype=np.float32)
    ys = sorted(set(list(range(0, max(Hp - win, 0) + 1, stride)) + [max(Hp - win, 0)]))
    xs = sorted(set(list(range(0, max(Wp - win, 0) + 1, stride)) + [max(Wp - win, 0)]))
    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        xb = torch.stack([dataset.normalize(im[y:y + win, x:x + win]) for y, x in chunk])
        out = torch.softmax(model(xb), 1).numpy()
        for (y, x), p in zip(chunk, out):
            probs[:, y:y + win, x:x + win] += p
            count[y:y + win, x:x + win] += 1
    probs /= np.maximum(count, 1)
    probs = probs[:, :H, :W]
    return probs.argmax(0).astype(np.uint8), probs


def orig_render_overlay(img, classmap, alpha=0.45):
    color = np.zeros_like(img)
    for k, c in config.CLASS_COLORS.items():
        color[classmap == k] = c
    return (img * (1 - alpha) + color * alpha).astype(np.uint8)


def orig_area_stats(classmap, meta):
    lat = meta["center"][0]
    mpp = imagery.meters_per_map_px(lat, meta["zoom"]) / meta["scale"]
    px_area = mpp * mpp
    out = {}
    for k, name in enumerate(config.CLASS_NAMES):
        n = int((classmap == k).sum())
        out[name] = {"pixels": n, "area_m2": round(n * px_area, 1),
                     "share": round(n / classmap.size, 4)}
    return out


# --- helpers ---

class TinyNet(torch.nn.Module):
    """Deterministic stand-in for the U-Net: 4 logits per pixel from one conv.

    k=3 (random, asymmetric kernel, zero padding) is not equivariant under
    flips/turns and depends on the window position, so TTA and blending
    bookkeeping errors show; k=1 sees one pixel, so every window agrees and
    any correctly normalised blend must reproduce it exactly.
    """

    def __init__(self, k=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = torch.nn.Conv2d(3, config.N_CLASSES, k, padding=k // 2)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g))
            self.conv.bias.copy_(torch.randn(config.N_CLASSES, generator=g))
        self.calls = 0
        self.eval()

    def forward(self, x):
        self.calls += x.shape[0]
        return self.conv(x)


def rand_img(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def dihedral(a, k, flip, axes=(0, 1)):
    """The TTA transform (k quarter turns, then a horizontal flip) of an image or CxHxW map."""
    v = np.rot90(a, k, axes=axes)
    return v[:, :, ::-1] if flip and axes == (1, 2) else v[:, ::-1] if flip else v


def grid_meta(lat, lng, nx, ny):
    """The meta imagery.fetch_area returns, without fetching."""
    ux, uy = config.USABLE_X, config.USABLE_Y
    cx, cy = imagery.latlng_to_world(lat, lng)
    return {"zoom": config.ZOOM, "scale": config.SCALE, "x0": cx - nx * ux / 2.0,
            "y0": cy - ny * uy / 2.0, "width_map_px": nx * ux, "height_map_px": ny * uy,
            "center": [lat, lng], "nx": nx, "ny": ny}


def rasterize_back(fc, meta, shape_):
    """Class ids of the exported polygons on the image grid (255 = not covered)."""
    s, z = meta["scale"], meta["zoom"]
    tr = Affine(1.0 / s, 0, meta["x0"], 0, 1.0 / s, meta["y0"])
    ids = {name: k for k, name in enumerate(config.CLASS_NAMES)}
    shapes = [(shp_transform(lambda x, y: labels._world_arrays(x, y, z), shape(f["geometry"])),
               ids[f["properties"]["class"]]) for f in fc["features"]]
    if not shapes:
        return np.full(shape_, 255, np.uint8)
    return rfeatures.rasterize(shapes, out_shape=shape_, transform=tr, fill=255, dtype="uint8")


def seam_ratio(cm, window=1024, stride=768):
    """Class-transition rate on internal window-edge rows/cols over the median row/col rate."""
    H, W = cm.shape
    win, coords = infer._windows(H + (32 - H % 32) % 32, W + (32 - W % 32) % 32, window, stride)
    ex = {x for _, x in coords if 0 < x < W} | {x + win for _, x in coords if x + win < W}
    ey = {y for y, _ in coords if 0 < y < H} | {y + win for y, _ in coords if y + win < H}
    tx = np.r_[np.nan, (cm[:, 1:] != cm[:, :-1]).mean(0)]  # tx[e]: change between cols e-1, e
    ty = np.r_[np.nan, (cm[1:] != cm[:-1]).mean(1)]
    return float(np.mean([tx[e] / np.nanmedian(tx) for e in ex]
                         + [ty[e] / np.nanmedian(ty) for e in ey]))


@pytest.fixture
def run_slow(request):
    expr = request.config.getoption("markexpr") or ""
    if not (os.environ.get("URBANSEG_RUN_SLOW") == "1"
            or ("slow" in expr and "not slow" not in expr)):
        pytest.skip("full-area run: select with -m slow or URBANSEG_RUN_SLOW=1")


# --- sliding window ---

@pytest.mark.parametrize("blend", ["gaussian", "uniform"])
def test_blends_are_normalised(blend):
    net = TinyNet(k=1)
    img = rand_img(150, 230)
    ref = torch.softmax(net(dataset.normalize(img)[None]), 1)[0].detach().numpy()
    cm, p = infer.predict(net, img, window=64, stride=48, blend=blend)
    assert p.dtype == np.float32 and p.shape == (config.N_CLASSES, 150, 230)
    assert cm.dtype == np.uint8 and np.array_equal(cm, p.argmax(0))
    np.testing.assert_allclose(p, ref, atol=1e-6)


@pytest.mark.parametrize("kw", [dict(window=64, stride=48), dict(window=64, stride=48, batch=3),
                                dict(window=96, stride=32), dict()])
def test_uniform_blend_is_the_original_predict(kw):
    net = TinyNet()
    img = rand_img(150, 230)
    c0, p0 = orig_predict(net, img, **kw)
    c1, p1 = infer.predict(net, img, blend="uniform", **kw)
    assert np.array_equal(p0, p1) and np.array_equal(c0, c1)


def test_gaussian_blend_reweights_overlaps_only():
    net = TinyNet()
    img = rand_img(150, 230)
    _, pu = infer.predict(net, img, window=64, stride=48, blend="uniform")
    _, pg = infer.predict(net, img, window=64, stride=48)
    assert np.abs(pu - pg).max() > 1e-3  # the conv's padding makes windows disagree at edges
    np.testing.assert_allclose(pg.sum(0), 1, atol=1e-5)
    # a single window has nothing to blend
    _, pu1 = infer.predict(net, img[:64, :64], window=64, blend="uniform")
    _, pg1 = infer.predict(net, img[:64, :64], window=64)
    np.testing.assert_allclose(pu1, pg1, atol=1e-6)


def test_narrow_image_is_fully_covered():
    net = TinyNet(k=1)
    img = rand_img(70, 300)  # 96 px windows: the default 768 stride skipped a band
    _, p_old = orig_predict(net, img)
    assert (p_old.sum(0) == 0).any()
    _, p = infer.predict(net, img)
    np.testing.assert_allclose(p.sum(0), 1, atol=1e-5)


def test_blend_and_quality_are_validated():
    net, img = TinyNet(), rand_img(64, 64)
    with pytest.raises(ValueError):
        infer.predict(net, img, blend="hann")
    with pytest.raises(ValueError):
        infer.predict_tta(net, img, n=2)
    with pytest.raises(ValueError):
        infer.predict_quality(net, img, "ultra")


# --- test-time augmentation ---

def test_tta4_uniform_is_the_original_predict_tta():
    net = TinyNet()
    img = rand_img(150, 230)
    c0, p0 = orig_predict_tta(net, img, window=64, stride=48)
    c1, p1 = infer.predict_tta(net, img, n=4, window=64, stride=48, blend="uniform")
    assert np.array_equal(p0, p1) and np.array_equal(c0, c1)
    assert infer.TTA_TRANSFORMS[:4] == ((0, False), (2, False), (0, True), (2, True))


def test_tta1_is_predict():
    net = TinyNet()
    img = rand_img(100, 140)
    c0, p0 = infer.predict(net, img, window=64, stride=48)
    c1, p1 = infer.predict_tta(net, img, n=1, window=64, stride=48)
    assert np.array_equal(p0, p1) and np.array_equal(c0, c1)


@pytest.mark.parametrize("k, flip", [(k, f) for k in range(4) for f in (False, True)])
def test_tta8_averages_over_the_whole_dihedral_group(k, flip):
    # averaging over the full group makes the result transform with the input,
    # whatever the model does; a missing or mis-inverted transform breaks this
    net = TinyNet()
    img = rand_img(96, 160)
    kw = dict(window=64, stride=48)
    _, p = infer.predict_tta(net, img, n=8, **kw)
    _, pg = infer.predict_tta(net, np.ascontiguousarray(dihedral(img, k, flip)), n=8, **kw)
    np.testing.assert_allclose(pg, dihedral(p, k, flip, axes=(1, 2)), atol=1e-6)
    if k % 2:  # the 4-way set lacks the quarter turns
        _, p4 = infer.predict_tta(net, img, n=4, **kw)
        _, pg4 = infer.predict_tta(net, np.ascontiguousarray(dihedral(img, k, flip)), n=4, **kw)
        assert np.abs(pg4 - dihedral(p4, k, flip, axes=(1, 2))).max() > 1e-3


def test_quality_levels():
    net = TinyNet()
    img = rand_img(100, 140)
    kw = dict(window=64, stride=48)
    for got, want in ((infer.predict_quality(net, img, "standard", **kw), infer.predict(net, img, **kw)),
                      (infer.predict_quality(net, img, "high", **kw),
                       infer.predict_tta(net, img, n=8, **kw))):
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    assert infer.QUALITIES == ("standard", "high")


# --- progress / cancel ---

def test_progress_counts_windows_and_cancel_stops_between_them():
    net = TinyNet()
    img = rand_img(150, 230)  # 64/48 windows: 3 rows x 5 cols
    seen = []
    infer.predict(net, img, window=64, stride=48, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(i, 15) for i in range(1, 16)] and net.calls == 15
    seen.clear()
    infer.predict_tta(net, img, n=8, window=64, stride=48, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(i, 120) for i in range(1, 121)]
    seen.clear()
    infer.predict(net, img, window=64, stride=48, batch=4, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (15, 15) and len(seen) == 15

    net.calls = 0
    with pytest.raises(infer.Cancelled):
        infer.predict(net, img, window=64, stride=48, cancel=lambda: net.calls >= 2)
    assert net.calls == 2
    with pytest.raises(infer.Cancelled):  # TTA passes cancel through
        infer.predict_quality(net, img, "high", window=64, stride=48, cancel=lambda: net.calls >= 20)
    assert net.calls == 20
    assert issubclass(infer.Cancelled, RuntimeError)


# --- devices ---

def test_device_selection(monkeypatch):
    monkeypatch.delenv("URBANSEG_DEVICE", raising=False)
    assert infer.resolve_device().type == "cpu"
    monkeypatch.setenv("URBANSEG_DEVICE", "auto")
    want = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
    assert infer.resolve_device().type == want
    assert infer.resolve_device("cpu").type == "cpu"  # an explicit device wins over the env
    monkeypatch.setenv("URBANSEG_DEVICE", "CPU ")
    assert infer.resolve_device().type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        infer.resolve_device("mps")
    monkeypatch.setenv("URBANSEG_DEVICE", "auto")
    if not torch.cuda.is_available():
        assert infer.resolve_device().type == "cpu"


def test_load_model_defaults_to_cpu_eval(tmp_path, monkeypatch):
    from urbanseg.train import build_model
    p = tmp_path / "weights.pt"
    torch.save(build_model(pretrained=False).state_dict(), p)
    monkeypatch.delenv("URBANSEG_DEVICE", raising=False)
    m = infer.load_model(p)
    assert next(m.parameters()).device.type == "cpu" and not m.training


def test_models_resolve_by_id_setting_or_path(tmp_path, monkeypatch):
    monkeypatch.delenv("URBANSEG_MODEL", raising=False)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    default = config.DEFAULT_MODEL
    assert infer.resolve_model() == (default, tmp_path / config.MODELS[default]["file"])
    assert infer.resolve_model("v2") == ("v2", tmp_path / "unet_r34_v2.pt")
    monkeypatch.setenv("URBANSEG_MODEL", "v2")
    assert infer.resolve_model()[0] == "v2" and infer.resolve_model("v1")[0] == "v1"
    other = tmp_path / "elsewhere" / "w.pt"
    assert infer.resolve_model(other) == (None, other)
    assert infer.resolve_model(str(tmp_path / "unet_r34.pt"))[0] == "v1"  # keeps its id
    with pytest.raises(ValueError, match="unknown model"):
        infer.resolve_model("v9")


def test_load_model_records_what_it_loaded(tmp_path, monkeypatch):
    from urbanseg.train import build_model
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    p = tmp_path / "unet_r34_v2.pt"
    torch.save(build_model(pretrained=False).state_dict(), p)
    assert infer.load_model("v2").info == {
        "id": "v2", "file": "unet_r34_v2.pt", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


@pytest.mark.data
@pytest.mark.model
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_matches_cpu(model):
    img, _, _ = helpers.load_area("lbnagar")
    crop = img[1000:1512, 1200:1712]
    m = infer.load_model("v1", device="mps")  # the fixture's checkpoint
    assert next(m.parameters()).device.type == "mps"
    c, p = infer.predict(m, crop)
    assert isinstance(p, np.ndarray) and p.dtype == np.float32
    c0, p0 = infer.predict(model, crop)
    assert np.abs(p - p0).max() < 1e-3 and (c == c0).mean() > 0.999


# --- area statistics ---

def test_area_stats_is_wgs84_correct():
    meta = grid_meta(17.4482, 78.3915, 3, 3)  # madhapur
    cm = np.ones((meta["height_map_px"] * 2, meta["width_map_px"] * 2), np.uint8)
    w, s, e, n = imagery.bbox_latlng(meta)
    k = 200  # densify: a geodesic between two corners bows off the parallel
    ring = ([(w + (e - w) * i / k, s) for i in range(k)] + [(e, s + (n - s) * i / k) for i in range(k)]
            + [(e - (e - w) * i / k, n) for i in range(k)] + [(w, n - (n - s) * i / k) for i in range(k)])
    geodesic = abs(infer.GEOD.geometry_area_perimeter(Polygon(ring))[0])
    got = infer.area_stats(cm, meta)["residential"]["area_m2"]
    assert got == pytest.approx(geodesic, rel=1e-5)
    old = orig_area_stats(cm, meta)["residential"]["area_m2"]
    assert old / geodesic - 1 == pytest.approx(0.0055, abs=0.0003)  # the spherical over-statement


def test_area_stats_mask_and_nodata():
    meta = grid_meta(17.4, 78.48, 1, 1)
    cm = np.zeros((10, 10), np.uint8)
    cm[:, 5:] = 1
    cm[0, :] = infer.NODATA
    cm[9, 9] = 3
    st = infer.area_stats(cm, meta)
    assert list(st) == config.CLASS_NAMES
    assert [st[c]["pixels"] for c in config.CLASS_NAMES] == [45, 44, 0, 1]
    assert sum(v["share"] for v in st.values()) == pytest.approx(1.0, abs=1e-3)
    assert st["other"]["share"] == round(45 / 90, 4)
    px = infer.pixel_area_m2(17.4)
    assert st["residential"]["area_m2"] == round(44 * px, 1)

    mask = np.zeros_like(cm, bool)
    mask[:, :5] = True
    st = infer.area_stats(cm, meta, mask=mask)
    assert [st[c]["pixels"] for c in config.CLASS_NAMES] == [45, 0, 0, 0]
    assert st["other"]["share"] == 1.0
    st = infer.area_stats(cm, meta, mask=np.zeros_like(cm, bool))
    assert all(v == {"pixels": 0, "area_m2": 0.0, "share": 0.0} for v in st.values())
    with pytest.raises(ValueError):
        infer.area_stats(cm, meta, mask=mask[:5])
    st = infer.area_stats(cm, dict(meta, m_per_px=0.5))  # a metric-scale image
    assert st["residential"]["area_m2"] == 11.0
    json.dumps(st)


def test_area_stats_keeps_counts_and_shares():
    meta = grid_meta(17.4482, 78.3915, 1, 1)
    cm = np.random.default_rng(1).integers(0, 4, (300, 400)).astype(np.uint8)
    new, old = infer.area_stats(cm, meta), orig_area_stats(cm, meta)
    for c in config.CLASS_NAMES:
        assert new[c]["pixels"] == old[c]["pixels"] and new[c]["share"] == old[c]["share"]
        assert new[c]["area_m2"] == pytest.approx(old[c]["area_m2"] / 1.0055, rel=3e-4)


# --- GeoJSON ---

def blocks_classmap():
    """A 200 x 300 classmap: free land with two touching buildings, a road, a NODATA block."""
    cm = np.full((200, 300), 3, np.uint8)
    cm[20:80, 30:110] = 1
    cm[20:80, 110:150] = 2
    cm[120:140, :] = 0
    cm[150:200, 250:300] = infer.NODATA
    cm[170:175, 20:25] = 1  # 25 px: below min_area_px
    return cm


def test_geojson_features_and_properties():
    meta = grid_meta(17.4482, 78.3915, 1, 1)
    cm = blocks_classmap()
    fc = infer.to_geojson(cm, meta)
    assert fc["type"] == "FeatureCollection"
    classes = [f["properties"]["class"] for f in fc["features"]]
    assert sorted(set(classes)) == sorted(config.CLASS_NAMES)  # all 4 by default
    assert classes.count("residential") == 1  # the 25-px blob is dropped
    px = infer.pixel_area_m2(meta["center"][0])
    by = {f["properties"]["class"]: f for f in fc["features"] if f["properties"]["class"] != "free_land"}
    for name, n_px in (("residential", 60 * 80), ("commercial", 60 * 40), ("other", 20 * 300)):
        assert by[name]["properties"]["area_m2"] == pytest.approx(n_px * px, rel=1e-3)
        ring = by[name]["geometry"]["coordinates"][0]
        assert Polygon(ring).exterior.is_ccw  # RFC 7946 right-hand rule
    coords = json.dumps(fc).replace("]", ",").replace("[", ",").split(",")
    decimals = [len(c.split(".")[1]) for c in coords if c.strip().startswith(("78.", "17."))]
    assert decimals and max(decimals) <= 7

    fc1 = infer.to_geojson(cm, meta, classes=(1,), min_area_px=0)
    assert [f["properties"]["class"] for f in fc1["features"]] == ["residential"] * 2
    fc_full = infer.to_geojson(cm, meta, classes=(1,), precision=None)
    full = json.dumps(fc_full).replace("]", ",").replace("[", ",").split(",")
    assert max(len(c.split(".")[1]) for c in full if "." in c and c.strip()[:1].isdigit()) > 7
    assert infer.to_geojson(np.full((50, 50), infer.NODATA, np.uint8), meta)["features"] == []


def test_geojson_regions_share_edges():
    meta = grid_meta(17.4482, 78.3915, 1, 1)
    cm = blocks_classmap()
    back = rasterize_back(infer.to_geojson(cm, meta, min_area_px=0), meta, cm.shape)
    valid = cm != infer.NODATA
    assert (back[~valid] == 255).all()  # nothing exported on NODATA
    assert (back[valid] == cm[valid]).mean() > 0.999


def test_geojson_mask_clips_at_pixels():
    meta = grid_meta(17.4482, 78.3915, 1, 1)
    cm = blocks_classmap()
    mask = np.zeros(cm.shape, bool)
    mask[10:100, 60:200] = True
    fc = infer.to_geojson(cm, meta, simplify_m=0, min_area_px=0, mask=mask)
    back = rasterize_back(fc, meta, cm.shape)
    assert (back[~mask] == 255).all()
    assert np.array_equal(back[mask], cm[mask])
    with pytest.raises(ValueError):
        infer.to_geojson(cm, meta, mask=mask[:10])


# --- rendering ---

def test_render_overlay_mask_and_nodata():
    img = rand_img(60, 80)
    cm = np.random.default_rng(2).integers(0, 4, (60, 80)).astype(np.uint8)
    assert np.array_equal(infer.render_overlay(img, cm), orig_render_overlay(img, cm))
    mask = np.zeros(cm.shape, bool)
    mask[10:50, 10:70] = True
    cm[12, 12] = infer.NODATA
    out = infer.render_overlay(img, cm, mask=mask)
    off = ~mask | (cm == infer.NODATA)
    dim = (img * infer.OUTSIDE_DIM).astype(np.uint8)
    assert np.array_equal(out[off], dim[off])
    assert np.array_equal(out[~off], orig_render_overlay(img, cm)[~off])


def test_render_class_png_is_transparent_outside():
    cm = np.random.default_rng(3).integers(0, 4, (60, 80)).astype(np.uint8)
    cm[0, :] = infer.NODATA
    mask = np.ones(cm.shape, bool)
    mask[:, :10] = False
    png = infer.render_class_png(cm, mask=mask)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    shown = mask & (cm != infer.NODATA)
    assert (rgba[..., 3][shown] == 255).all() and (rgba[..., 3][~shown] == 0).all()
    for k, c in config.CLASS_COLORS.items():
        assert (rgba[..., :3][shown & (cm == k)] == c).all()
    rgba = np.asarray(Image.open(io.BytesIO(infer.render_class_png(cm))).convert("RGBA"))
    assert (rgba[1:, :, 3] == 255).all() and (rgba[0, :, 3] == 0).all()


@pytest.mark.data
def test_render_class_png_is_small():
    _, mask, _ = helpers.load_area("lbnagar")
    png = infer.render_class_png(mask[:2400, :2560])  # a 2x2-tile area
    print(f"\nclass PNG for a 2x2-tile area: {len(png)} bytes")
    assert len(png) < 200_000


# --- pipelines ---

def test_segment_image_quality_and_pseudo_meta():
    net = TinyNet()
    img = rand_img(130, 170)
    cm, probs, meta = infer.segment_image(net, img)
    assert meta == {"zoom": config.ZOOM, "scale": config.SCALE, "x0": 0, "y0": 0,
                    "center": [17.4, 78.48], "width_map_px": 170 // config.SCALE,
                    "height_map_px": 130 // config.SCALE}
    assert np.array_equal(probs, infer.predict(net, img)[1])
    cm8, probs8, _ = infer.segment_image(net, img, quality="high")
    assert np.array_equal(probs8, infer.predict_tta(net, img, n=8)[1])


def test_segment_latlng_quality(monkeypatch):
    net = TinyNet()
    img = rand_img(140, 160)
    meta0 = grid_meta(17.4482, 78.3915, 1, 1)
    monkeypatch.setattr(infer.imagery, "fetch_area",
                        lambda lat, lng, nx, ny: (Image.fromarray(img), dict(meta0)))
    calls = []
    monkeypatch.setattr(infer, "fuse_footprints",
                        lambda cm, probs, meta: calls.append(probs.shape) or cm)
    arr, cm, meta = infer.segment_latlng(net, 17.4482, 78.3915, grid=1, fuse=False, quality="high")
    assert np.array_equal(arr, img) and meta == meta0 and not calls
    assert np.array_equal(cm, infer.predict_tta(net, img, n=8)[0])
    infer.segment_latlng(net, 17.4482, 78.3915, grid=1)
    assert calls == [(config.N_CLASSES, 140, 160)]


# --- the trained model on a VAL crop ---

_VAL = {}  # computed once; a module-scoped fixture would run before conftest sets the data paths


@pytest.fixture
def val_crop():
    if "crop" not in _VAL:
        img, mask, meta = helpers.load_area("lbnagar")
        _VAL["crop"] = helpers.crop(img, mask, meta, 1000, 2216, 1200, 2704)  # 1216 x 1504: 4 windows
    return _VAL["crop"]


@pytest.fixture
def val_preds(model, val_crop):
    if "preds" not in _VAL:
        img = val_crop[0]
        _VAL["preds"] = {"orig": orig_predict(model, img),
                         "uniform": infer.predict(model, img, blend="uniform"),
                         "gaussian": infer.predict(model, img)}
    return _VAL["preds"]


@pytest.mark.data
@pytest.mark.model
def test_uniform_blend_reproduces_the_original_on_the_model(val_preds):
    (c0, p0), (c1, p1) = val_preds["orig"], val_preds["uniform"]
    assert np.array_equal(p0, p1) and np.array_equal(c0, c1)


@pytest.mark.data
@pytest.mark.model
def test_gaussian_blend_removes_window_seams(val_crop, val_preds):
    mask = val_crop[1]
    (cu, _), (cg, pg) = val_preds["uniform"], val_preds["gaussian"]
    np.testing.assert_allclose(pg.sum(0), 1, atol=1e-5)
    su, sg, s_gt = seam_ratio(cu), seam_ratio(cg), seam_ratio(mask)
    print(f"\nseam ratio on the VAL crop: uniform {su:.2f}, gaussian {sg:.2f} (labels {s_gt:.2f})")
    assert su > 3 and sg < 0.5 * su
    assert (cg == mask).mean() >= (cu == mask).mean() - 0.002  # no accuracy cost


@pytest.mark.data
@pytest.mark.model
def test_geojson_round_trip_on_a_prediction(val_crop, val_preds):
    meta = val_crop[2]
    cm = val_preds["gaussian"][0]
    fc = infer.to_geojson(cm, meta)
    back = rasterize_back(fc, meta, cm.shape)
    covered = back != 255
    agree = (back[covered] == cm[covered]).mean()
    print(f"\nGeoJSON round trip: {agree:.4f} agreement on {covered.mean():.4f} of the pixels")
    assert agree >= 0.985 and covered.mean() > 0.98
    st = infer.area_stats(cm, meta)
    total = sum(f["properties"]["area_m2"] for f in fc["features"])
    assert total == pytest.approx(sum(v["area_m2"] for v in st.values()), rel=0.02)


# --- full VAL areas (slow) ---

@pytest.mark.data
@pytest.mark.model
@pytest.mark.slow
def test_high_quality_gain_on_val(model, run_slow):
    hits = {"standard_uniform": 0, "high": 0}
    total = 0
    for name in ("lbnagar", "kompally"):
        img, mask, _ = helpers.load_area(name)
        hits["standard_uniform"] += int((infer.predict(model, img, blend="uniform")[0] == mask).sum())
        hits["high"] += int((infer.predict_quality(model, img, "high")[0] == mask).sum())
        total += mask.size
    gain = (hits["high"] - hits["standard_uniform"]) / total * 100
    print(f"\nVAL pixel accuracy: {hits['standard_uniform'] / total:.4f} -> "
          f"{hits['high'] / total:.4f} ({gain:+.2f} pt)")
    assert gain > 1.0
