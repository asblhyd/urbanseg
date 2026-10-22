"""Inference: sliding-window segmentation + footprint/POI fusion + outputs."""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import shapely
import torch
from PIL import Image
from pyproj import Geod
from rasterio import features as rfeatures
from rasterio.transform import Affine
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

from . import config, dataset, fusion, imagery

# fusion rules used by fuse_footprints, and so by the API, CLI and evaluate
FUSION_MODE = os.environ.get("URBANSEG_FUSION_MODE", "roof")
NODATA = 255  # classmap value of pixels without valid imagery (upload nodata)
QUALITIES = ("standard", "high")
# Gaussian blend sigma as a fraction of the window: selected on val, where
# W/4 left the fewest window seams (accuracy is the same for any blend)
GAUSS_SIGMA = 0.25
# dihedral TTA transforms: (quarter turns, then horizontal flip). The first
# four are the original 4-way set in its original order, so their sums stay
# bit-identical; the last four add the 90-degree turns (the transposes).
TTA_TRANSFORMS = ((0, False), (2, False), (0, True), (2, True),
                  (1, False), (3, False), (1, True), (3, True))
OUTSIDE_DIM = 0.4  # brightness kept where render_overlay shows no class
GEOD = Geod(ellps="WGS84")


class Cancelled(RuntimeError):
    """predict() stopped between windows because cancel() returned True."""


def resolve_device(device=None):
    """torch.device for "cpu", "mps", "auto" or any torch device string.

    None reads URBANSEG_DEVICE (default "cpu"). "auto" picks CUDA, then MPS,
    then CPU; an explicit "mps" that is unavailable raises instead of
    silently running on the CPU.
    """
    name = str(device or os.environ.get("URBANSEG_DEVICE") or "cpu").strip().lower()
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available here; "
                           "use 'cpu' or 'auto'")
    return dev


def resolve_model(model=None):
    """(id, checkpoint path) of a model: an id of config.MODELS, or a .pt path
    (id None). None means $URBANSEG_MODEL, else config.DEFAULT_MODEL."""
    m = model if model is not None else (os.environ.get("URBANSEG_MODEL")
                                         or config.DEFAULT_MODEL)
    if isinstance(m, str) and m in config.MODELS:
        return m, config.MODELS_DIR / config.MODELS[m]["file"]
    p = Path(m)
    if p.suffix != ".pt" and not p.is_file():
        raise ValueError(f"unknown model {str(m)!r}: one of {', '.join(config.MODELS)}, "
                         "or a .pt checkpoint")
    for k, entry in config.MODELS.items():  # a path to a registered checkpoint keeps its id
        if p.resolve() == (config.MODELS_DIR / entry["file"]).resolve():
            return k, p
    return None, p


def load_model(path=None, device=None):
    """The trained U-Net in eval mode on resolve_device(device).

    path: a model id of config.MODELS or a checkpoint path (resolve_model;
    None = the default model). The model's .info is {"id", "file", "sha256"},
    which results record.

    Weights stay in the default memory layout: channels-last is ~1.55x
    faster per window on an M5 CPU but moves probabilities by up to 1.9e-5
    (val areas), so outputs would no longer match the reference runs.
    """
    from .train import build_model
    model_id, p = resolve_model(path)
    raw = Path(p).read_bytes()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu"))
    model.to(resolve_device(device))
    model.eval()
    model.info = {"id": model_id, "file": Path(p).name,
                  "sha256": hashlib.sha256(raw).hexdigest()}
    return model


def _device_of(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def predict_tta(model, img, n=8, **kw):
    """Average predict() over n dihedral transforms of img (n = 1, 4 or 8).

    n=4 is the original flip/rotate-180 set (with blend="uniform" it
    reproduces the original predict_tta exactly); n=8 adds the 90-degree
    turns. kw go to predict(); progress counts windows over all n passes.
    """
    if n not in (1, 4, 8):
        raise ValueError(f"n must be 1, 4 or 8, not {n!r}")
    progress = kw.pop("progress", None)
    accum = None
    for j, (k, flip) in enumerate(TTA_TRANSFORMS[:n]):
        v = np.rot90(img, k)
        if flip:
            v = v[:, ::-1]
        if progress is not None:
            # every pass has the same window count: a quarter turn swaps H, W
            kw["progress"] = lambda done, total, j=j: progress(j * total + done, n * total)
        _, p = predict(model, np.ascontiguousarray(v), **kw)
        if flip:
            p = p[:, :, ::-1]
        p = np.rot90(p, -k, axes=(1, 2))
        if accum is None:
            accum = p
        else:
            accum += p
        del p  # else this pass's buffer stays alive through the next one
    accum /= float(n)
    return accum.argmax(0).astype(np.uint8), accum


def predict_quality(model, img, quality="standard", **kw):
    """predict() for quality "standard"; 8-way TTA for "high".

    On the val areas "high" scores +1.5 pt pixel accuracy (0.7179 ->
    0.7333) and +2.6 pt mIoU for 8x the time: a 3x3-tile area takes ~15 s
    vs ~125 s on an M5 CPU with 2 threads (~2.5 s vs ~19 s on MPS), and on
    the CPU it peaks ~0.5 GB above "standard". kw go to predict().
    """
    if quality == "standard":
        return predict(model, img, **kw)
    if quality == "high":
        return predict_tta(model, img, n=8, **kw)
    raise ValueError(f"quality must be one of {QUALITIES}, not {quality!r}")


def _windows(Hp, Wp, window, stride):
    """(win, top-left corners) of the windows covering a padded Hp x Wp image."""
    win = min(window, Hp, Wp)
    # a window shrunk to fit a narrow image must not step past its own
    # width: that left uncovered bands, which came out as class 0
    stride = min(stride, win)
    ys = sorted(set(list(range(0, max(Hp - win, 0) + 1, stride)) + [max(Hp - win, 0)]))
    xs = sorted(set(list(range(0, max(Wp - win, 0) + 1, stride)) + [max(Wp - win, 0)]))
    return win, [(y, x) for y in ys for x in xs]


def _blend_weights(win):
    """win x win separable Gaussian centred on the window (sigma GAUSS_SIGMA * win)."""
    i = np.arange(win, dtype=np.float64) + 0.5
    g = np.exp(-0.5 * ((i - win / 2) / (GAUSS_SIGMA * win)) ** 2)
    return np.maximum(np.outer(g, g), 1e-4).astype(np.float32)


@torch.no_grad()
def predict(model, img, window=1024, stride=768, batch=1, *, blend="gaussian",
            progress=None, cancel=None):
    """img HxWx3 uint8 -> (classmap HxW uint8, probs CxHxW float32).

    The image is reflect-padded to a multiple of 32 and covered by
    window x window tiles `stride` apart. blend="gaussian" weights each
    tile's softmax by a separable Gaussian (sigma = window/4) and divides by
    the summed weights, so that tiles hand over smoothly where the model
    sees least context; plain averaging ("uniform", the original behaviour,
    reproduced exactly) leaves straight class seams along tile edges.
    Tiles run on the model's device; results come back as numpy.

    progress(done, total) is called after every tile; cancel() is polled
    before every tile, and True raises Cancelled.

    batch=1: on CPU larger batches are no faster, only heavier. The 1024-px
    forward pass sets the peak: ~4 GB RSS on macOS for a 3x3-tile area at
    batch 1, ~7.4 GB at batch 2.
    """
    if blend not in ("gaussian", "uniform"):
        raise ValueError(f"blend must be 'gaussian' or 'uniform', not {blend!r}")
    H, W = img.shape[:2]
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    im = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    Hp, Wp = im.shape[:2]
    win, coords = _windows(Hp, Wp, window, stride)
    wm = _blend_weights(win) if blend == "gaussian" else None
    device = _device_of(model)
    probs = np.zeros((config.N_CLASSES, Hp, Wp), dtype=np.float32)
    weight = np.zeros((Hp, Wp), dtype=np.float32)
    for i in range(0, len(coords), batch):
        if cancel is not None and cancel():
            raise Cancelled("prediction cancelled")
        chunk = coords[i:i + batch]
        xb = torch.stack([dataset.normalize(im[y:y + win, x:x + win]) for y, x in chunk])
        out = torch.softmax(model(xb.to(device)), 1).float().cpu().numpy()
        for j, ((y, x), p) in enumerate(zip(chunk, out)):
            if wm is None:
                probs[:, y:y + win, x:x + win] += p
                weight[y:y + win, x:x + win] += 1
            else:
                probs[:, y:y + win, x:x + win] += p * wm
                weight[y:y + win, x:x + win] += wm
            if progress is not None:
                progress(i + j + 1, len(coords))
    # every pixel has weight > 0; for "uniform" it is the window count, so
    # this is the original division
    probs /= np.maximum(weight, 1e-12)
    probs = probs[:, :H, :W]
    return probs.argmax(0).astype(np.uint8), probs


def fuse_footprints(classmap, probs, meta, use_osm=True, *, mode=None, layers=None,
                    osm_info=None):
    """Refine a georeferenced prediction with footprints and OSM (fusion.fuse).

    mode None -> FUSION_MODE: env URBANSEG_FUSION_MODE, default "roof", which
    keeps colours on the roofs as imaged; "legacy" is the original snapping
    rules. layers / osm_info: OSM fetched by the caller (None fetches it).
    Records the sources in meta["fusion"] and the details in
    meta["fusion_info"]; never raises.
    """
    return fusion.fuse(classmap, probs, meta, use_osm=use_osm, mode=mode or FUSION_MODE,
                       layers=layers, osm_info=osm_info)


def _as_mask(mask, classmap):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != classmap.shape:
        raise ValueError(f"mask shape {mask.shape} != classmap shape {classmap.shape}")
    return mask


def render_overlay(img, classmap, alpha=0.45, mask=None):
    """Class colours blended over the image (HxWx3 uint8).

    Pixels outside `mask` (bool HxW) and NODATA pixels show the imagery
    dimmed instead of coloured.
    """
    color = np.zeros_like(img)
    for k, c in config.CLASS_COLORS.items():
        color[classmap == k] = c
    out = (img * (1 - alpha) + color * alpha).astype(np.uint8)
    off = classmap == NODATA
    if mask is not None:
        off |= ~_as_mask(mask, classmap)
    if off.any():
        out[off] = (img[off] * OUTSIDE_DIM).astype(np.uint8)
    return out


def render_class_png(classmap, mask=None):
    """PNG bytes of the classmap for a map layer.

    Class colours are opaque; pixels outside `mask` (bool HxW) and NODATA
    pixels are transparent. A 5-entry palette PNG with a tRNS chunk (4 bits
    per pixel) decodes as RGBA at about half the size of an RGBA PNG:
    80-110 KB for a 2x2-tile area of model output.
    """
    n = config.N_CLASSES
    idx = np.where(classmap < n, classmap, n).astype(np.uint8)
    if mask is not None:
        idx[~_as_mask(mask, classmap)] = n
    im = Image.fromarray(idx)
    im.putpalette([v for k in range(n) for v in config.CLASS_COLORS[k]] + [0, 0, 0])
    buf = io.BytesIO()
    im.save(buf, format="PNG", transparency=n, optimize=True)
    return buf.getvalue()


LEGEND_ITEMS = [
    (1, "Residential building"),
    (2, "Commercial building"),
    (3, "Free land"),
    (0, "Roads / rail / water"),
]


def add_legend(overlay_arr, stats=None):
    """Attach a legend panel to the right of the overlay (never covers pixels)."""
    from PIL import ImageDraw, ImageFont

    h, w = overlay_arr.shape[:2]
    panel_w = 400
    canvas = Image.new("RGB", (w + panel_w, h), (24, 26, 30))
    canvas.paste(Image.fromarray(overlay_arr), (0, 0))
    d = ImageDraw.Draw(canvas)

    try:
        f_title = ImageFont.load_default(size=30)
        f_label = ImageFont.load_default(size=24)
        f_small = ImageFont.load_default(size=18)
    except TypeError:  # very old Pillow: bitmap font only
        f_title = f_label = f_small = ImageFont.load_default()

    x = w + 28
    y = 36
    d.text((x, y), "Legend", fill=(255, 255, 255), font=f_title)
    y += 58
    sw = 34  # swatch size
    for k, label in LEGEND_ITEMS:
        d.rectangle([x, y, x + sw, y + sw], fill=config.CLASS_COLORS[k],
                    outline=(255, 255, 255), width=1)
        d.text((x + sw + 14, y + 3), label, fill=(235, 235, 235), font=f_label)
        if stats is not None:
            share = stats[config.CLASS_NAMES[k]]["share"] * 100
            area = stats[config.CLASS_NAMES[k]]["area_m2"]
            d.text((x + sw + 14, y + 33),
                   f"{share:.1f}%  ({area:,.0f} sq m)",
                   fill=(160, 165, 175), font=f_small)
            y += 26
        y += sw + 26
    # tiles are cropped above the provider watermark, so credit it here
    d.text((x, h - 40), "Imagery © Google", fill=(160, 165, 175), font=f_small)
    return np.asarray(canvas)


def pixel_area_m2(lat, zoom=config.ZOOM, scale=config.SCALE):
    """Ground area (m²) of one image pixel of the Web-Mercator grid at `lat`.

    WGS84-correct: a pixel spans k = 2*pi / (256 * 2**zoom * scale) radians
    of longitude and k*cos(lat) of latitude, i.e. N*k*cos(lat) by
    M*k*cos(lat) metres, with N and M the ellipsoid's prime-vertical and
    meridional radii of curvature. imagery.meters_per_map_px squared uses
    the equatorial radius for both and over-states the area by 0.55% at
    Hyderabad (east-west +0.03%, north-south +0.58%).
    """
    phi = np.radians(lat)
    k = 2 * np.pi / (256 * 2 ** zoom * scale)
    w = 1 - GEOD.es * np.sin(phi) ** 2
    return (k * np.cos(phi) * GEOD.a) ** 2 * (1 - GEOD.es) / w ** 2


def area_stats(classmap, meta, mask=None):
    """{class name: {"pixels", "area_m2", "share"}} for the 4 classes.

    Counts only pixels inside `mask` (bool HxW) that are not NODATA; shares
    are of the counted pixels. The pixel area is pixel_area_m2 at the image
    centre, meta["center"]; across a 3x3-tile area it changes by only 0.01%
    from top to bottom, evenly about the centre. A meta with "m_per_px" (an
    image resampled to that many ground metres per pixel) uses that square
    pixel size instead.
    """
    vals = classmap if mask is None else classmap[_as_mask(mask, classmap)]
    counts = np.bincount(vals.ravel(), minlength=256)[:config.N_CLASSES]
    total = int(counts.sum())
    mpp = meta.get("m_per_px")
    px_area = (mpp * mpp if mpp else
               pixel_area_m2(meta["center"][0], meta["zoom"], meta["scale"]))
    return {name: {"pixels": int(n), "area_m2": round(float(n * px_area), 1),
                   "share": round(int(n) / total, 4) if total else 0.0}
            for name, n in zip(config.CLASS_NAMES, counts)}


def to_geojson(classmap, meta, min_area_px=400, simplify_m=1.0, *,
               classes=(0, 1, 2, 3), mask=None, precision=7):
    """Class regions as a GeoJSON FeatureCollection in WGS84 lon/lat.

    One feature per 4-connected region of each class in `classes`, with
    properties {"class": name, "area_m2": geodesic area of the exported
    geometry}. Regions under min_area_px image pixels are dropped. Outlines
    are simplified to a simplify_m tolerance (metres; 0 keeps the pixel
    outlines) as one coverage, so neighbouring regions keep sharing their
    edges: simplifying each region on its own left 0.3-0.6% of an area
    covered twice. Rings follow the RFC 7946 right-hand rule; coordinates
    are snapped to `precision` decimals (7 is ~1 cm; None keeps them all).
    `mask` (bool HxW) clips the regions to its True pixels before
    simplification; NODATA is never exported.
    """
    s, zoom = meta["scale"], meta["zoom"]
    inside = None if mask is None else _as_mask(mask, classmap)
    polys, ks = [], []
    for k in classes:
        m = classmap == k
        if inside is not None:
            m &= inside
        for geom, _ in rfeatures.shapes(m.astype(np.uint8), mask=m):
            poly = shape(geom)  # image px, so pixel corners are exact integers
            if poly.area >= min_area_px:
                polys.append(poly)
                ks.append(k)
    geoms = np.array(polys, dtype=object)
    if simplify_m and len(geoms):
        tol = simplify_m * s / imagery.meters_per_map_px(meta["center"][0], zoom)
        # a vertex at every pixel corner gives regions meeting at a
        # T-junction the shared vertex that coverage simplification needs
        geoms = shapely.coverage_simplify(shapely.segmentize(geoms, 1.0), tol)
    x0, y0 = meta["x0"], meta["y0"]
    geoms = shapely.transform(geoms, lambda c: np.column_stack(
        _ll_arrays(x0 + c[:, 0] / s, y0 + c[:, 1] / s, zoom)))
    if precision is not None:
        geoms = shapely.set_precision(geoms, 10.0 ** -precision)
    geoms = shapely.orient_polygons(geoms)
    feats = [{"type": "Feature",
              "properties": {"class": config.CLASS_NAMES[k],
                             "area_m2": round(float(GEOD.geometry_area_perimeter(g)[0]), 1)},
              "geometry": mapping(g)}
             for g, k in zip(geoms, ks) if not g.is_empty]
    return {"type": "FeatureCollection", "features": feats}


def _ll_arrays(x, y, zoom):
    w = 256 * (2 ** zoom)
    lng = np.asarray(x) / w * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * np.asarray(y) / w))))
    return lng, lat


def segment_latlng(model, lat, lng, grid=2, fuse=True, *, quality="standard"):
    """Full pipeline for a coordinate query -> (img, classmap, meta)."""
    img, meta = imagery.fetch_area(lat, lng, grid, grid)
    arr = np.asarray(img)
    classmap, probs = predict_quality(model, arr, quality)
    if fuse:
        classmap = fuse_footprints(classmap, probs, meta)
    return arr, classmap, meta


def segment_image(model, img_arr, *, quality="standard"):
    """Pipeline for an uploaded image (no geo metadata -> no fusion)."""
    classmap, probs = predict_quality(model, img_arr, quality)
    meta = {"zoom": config.ZOOM, "scale": config.SCALE, "x0": 0, "y0": 0,
            "center": [17.4, 78.48],
            "width_map_px": img_arr.shape[1] // config.SCALE,
            "height_map_px": img_arr.shape[0] // config.SCALE}
    return classmap, probs, meta
