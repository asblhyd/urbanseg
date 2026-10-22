"""Uploaded images: scale, georeference, limits and safety (the uploads module)."""
import functools
import io
import math
import pickle
import struct
import warnings
import zlib

import numpy as np
import pytest
import rasterio
import rasterio.shutil
from PIL import Image, ImageFile
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import calculate_default_transform, reproject

import helpers
from urbanseg import imagery, infer, uploads
from urbanseg.uploads import (NODATA, UploadError, grid_transform, load_upload,
                            to_source_grid)

N_MPP = uploads.native_mpp()  # 0.2848 m/px at the nominal latitude
CROP = ("kompally", 1200, 1968, 1300, 2068)  # 768 x 768 px (~220 m) of a val area
RGBA_CI = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
pytestmark = [
    # rasterio's own helpers still use Affine '*'; writing plain test TIFFs warns
    pytest.mark.filterwarnings("ignore:Use `@` matmul:PendingDeprecationWarning:rasterio"),
    pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning"),
]


# ---- helpers -----------------------------------------------------------------

def png(arr, **kw):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", **kw)
    return buf.getvalue()


def _chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))


def png_header(w, h):
    """A PNG whose header claims w x h px (pixel data absent): a bomb stand-in."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(b"\0" * 64)) + _chunk(b"IEND", b""))


def png16(arr, colortype, chunks=()):
    """A 16-bit PNG written by hand (Pillow cannot write 16-bit colour);
    `chunks` are (type, data) pairs placed before the pixels."""
    h, w = arr.shape[:2]
    a = np.ascontiguousarray(arr.astype(">u2"))
    rows = b"".join(b"\0" + a[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 16, colortype, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + b"".join(_chunk(*c) for c in chunks)
            + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b""))


def to_jp2(tmp_path, arr, name="img", **kw):
    """arr CxHxW uint8 -> JPEG 2000 bytes; kw: GeoTIFF profile (crs, transform)
    and JP2OpenJPEG creation options (upper case)."""
    co = {k: v for k, v in kw.items() if k.isupper()}
    gtiff = tmp_path / f"{name}.tif"
    write_tif(gtiff, arr, photometric="RGB", **{k: v for k, v in kw.items() if not k.isupper()})
    with rasterio.open(gtiff) as src:  # JP2OpenJPEG writes by copy only
        rasterio.shutil.copy(src, tmp_path / f"{name}.jp2", driver="JP2OpenJPEG", **co)
    return (tmp_path / f"{name}.jp2").read_bytes()


def jp2_claiming(raw, w, h, tile, levels=None):
    """A JPEG 2000 file whose SIZ marker claims w x h px in tile x tile px
    tiles (and its COD marker `levels` decomposition levels): a header bomb
    whose pixel data is absent."""
    b = bytearray(raw)
    s = b.index(b"\xff\x4f\xff\x51")  # SOC + SIZ
    struct.pack_into(">II", b, s + 8, w, h)        # Xsiz, Ysiz
    struct.pack_into(">II", b, s + 24, tile, tile)  # XTsiz, YTsiz
    if levels is not None:
        j = s + 4 + struct.unpack_from(">H", b, s + 4)[0]  # the marker after SIZ
        assert b[j:j + 2] == b"\xff\x52"  # COD
        b[j + 9] = levels
    return bytes(b)


def texture(h=300, w=400, seed=0):
    """Smooth random RGB texture (survives resampling and JPEG well)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC))


def total_m2(classmap, meta):
    return sum(v["area_m2"] for v in infer.area_stats(classmap, meta).values())


def psnr(a, b, ok=None):
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2
    mse = d[ok].mean() if ok is not None else d.mean()
    return 10 * math.log10(255 ** 2 / max(mse, 1e-12))


@functools.lru_cache(maxsize=None)
def area_crop(name, y0, y1, x0, x1):
    img, mask, meta = helpers.load_area(name)
    return helpers.crop(img, mask, meta, y0, y1, x0, x1)


def sample_at(ref, meta_ref, meta, shape):
    """ref (on meta_ref's grid) sampled bilinearly at the px centres of a grid
    `meta` of `shape` - what a perfectly georeferenced upload should hold."""
    s = meta["scale"]
    ox, oy = (meta["x0"] - meta_ref["x0"]) * s, (meta["y0"] - meta_ref["y0"]) * s
    ix, iy = math.floor(ox), math.floor(oy)
    fx, fy = ox - ix, oy - iy
    h, w = shape
    pad = h + w + 4
    r = np.full((ref.shape[0] + 2 * pad, ref.shape[1] + 2 * pad, 3), np.nan)
    r[pad:-pad, pad:-pad] = ref
    y, x = iy + pad, ix + pad
    q = lambda dy, dx: r[y + dy:y + dy + h, x + dx:x + dx + w]  # noqa: E731
    return ((1 - fy) * ((1 - fx) * q(0, 0) + fx * q(0, 1))
            + fy * ((1 - fx) * q(1, 0) + fx * q(1, 1)))


def residual_shift(a, b):
    """Sub-px (dy, dx) of b relative to a by phase correlation of grey patches."""
    g = lambda x: np.nan_to_num(x).astype(np.float64) @ [0.299, 0.587, 0.114]  # noqa: E731
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, B = (np.fft.fft2((g(x) - g(x).mean()) * win) for x in (a, b))
    cc = np.real(np.fft.ifft2(B * A.conj()))
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    out = []
    for ax in (0, 1):
        n = cc.shape[ax]
        v = [cc[tuple((p + d) % n if i == ax else p for i, p in enumerate(peak))]
             for d in (-1, 0, 1)]
        sub = 0.5 * (v[0] - v[2]) / (v[0] - 2 * v[1] + v[2])
        p = peak[ax] if peak[ax] <= n // 2 else peak[ax] - n
        out.append(p + sub)
    return out


def write_tif(path, arr, **kw):
    """arr CxHxW -> GeoTIFF (tiled, deflate)."""
    prof = dict(driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                dtype=arr.dtype, tiled=True, blockxsize=256, blockysize=256,
                compress="deflate")
    prof.update(kw)
    colorinterp = prof.pop("colorinterp", None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)
        if colorinterp:
            dst.colorinterp = colorinterp
    return path.read_bytes()


def warped(img, meta, dst_crs, dst_tr, shape):
    """The crop reprojected onto (dst_crs, dst_tr, shape) -> 4xHxW RGBA with
    alpha 0, and RGB 0 as orthos usually have, outside its footprint."""
    src = np.concatenate([img.transpose(2, 0, 1), np.full((1,) + img.shape[:2], 255, np.uint8)])
    dst = np.zeros((4,) + shape, np.uint8)
    reproject(src, dst, src_transform=grid_transform(meta), src_crs="EPSG:3857",
              dst_transform=dst_tr, dst_crs=dst_crs, resampling=Resampling.bilinear,
              src_alpha=4, dst_alpha=4)
    dst[:, dst[3] < 255] = 0
    return dst


def to_crs(img, meta, crs, res, hole=False):
    """The crop at `res` (CRS units) in `crs` -> (RGBA, transform); with an
    elliptical hole and a ragged collar of alpha 0 when `hole`."""
    h, w = img.shape[:2]
    dtr, dw, dh = calculate_default_transform("EPSG:3857", crs, w, h,
                                              *array_bounds(h, w, grid_transform(meta)),
                                              resolution=res)
    dst = warped(img, meta, crs, dtr, (dh, dw))
    if hole:
        rr, cc = np.mgrid[0:dh, 0:dw]
        bad = (((cc - 0.3 * dw) / (0.08 * dw)) ** 2 + ((rr - 0.35 * dh) / (0.06 * dh)) ** 2 <= 1)
        bad |= cc > dw * (0.94 + 0.02 * np.sin(rr / (dh / 9.0)))
        dst[:, bad] = 0
    return dst, dtr


# ---- plain images: scale ------------------------------------------------------

def test_unknown_scale_assumes_native_and_says_so():
    arr = texture()
    up = load_upload(png(arr), "shot.png")
    assert up["img"].shape == arr.shape and np.array_equal(up["img"], arr)
    assert up["img"].flags.writeable and up["img"].flags.c_contiguous
    assert up["valid"] is None and up["georeferenced"] is False and up["crs"] is None
    assert up["scale_source"] == "assumed" and up["source_mpp"] is None
    assert uploads.SCALE_UNKNOWN in up["warnings"]
    assert up["original_size"] == up["native_size"] == [400, 300]
    assert up["format"] == "PNG"
    m = up["meta"]
    assert m["georeferenced"] is False and (m["zoom"], m["scale"]) == (18, 2)
    # the pseudo grid is self-consistent: its centre is the nominal point
    lat, lng = imagery.img_px_to_latlng(200, 150, m)
    assert abs(lat - uploads.NOMINAL_LAT) < 1e-9 and abs(lng - uploads.NOMINAL_LNG) < 1e-9
    assert total_m2(np.zeros((300, 400), np.uint8), m) == pytest.approx(
        400 * 300 * N_MPP ** 2, rel=0.01)


def test_declared_mpp_sets_area_4x_at_half_resolution():
    """A 1000x1000 upload declared at 0.57 m/px reports 4x the area of the same
    image declared native (the model sees it enlarged to 0.285 m/px)."""
    raw = png(texture(1000, 1000))
    nat = load_upload(raw, mpp=N_MPP)
    half = load_upload(raw, mpp=0.57)
    assert nat["native_size"] == [1000, 1000] and nat["scale_source"] == "mpp"
    wn = round(1000 * 0.57 / N_MPP)
    assert half["native_size"] == [wn, wn] and half["img"].shape == (wn, wn, 3)
    a_nat = total_m2(np.zeros((1000, 1000), np.uint8), nat["meta"])
    a_half = total_m2(np.zeros((wn, wn), np.uint8), half["meta"])
    assert a_half / a_nat == pytest.approx((0.57 / N_MPP) ** 2, rel=2e-3)
    assert a_half / a_nat == pytest.approx(4.0, rel=0.01)
    assert a_half == pytest.approx(1e6 * 0.57 ** 2, rel=0.01)  # true ground area
    assert "resampled x2.00" in half["scale_note"]
    # 2x coarser than native: still degraded after resampling, so say so
    assert any("2.0x coarser" in w for w in half["warnings"])
    assert not any("coarser" in w for w in nat["warnings"])


@pytest.mark.parametrize("zoom,dpr,factor", [
    (18, 2, 1.0),      # Static Maps zoom 18 scale 2 / a Retina screenshot at zoom 18
    (18, None, 2.0),   # zoom 18 on a 1x screen: 0.57 m/px
    (18, 0, 2.0),      # the contract's `dpr or 1`: 0 means not given
    (17, 2, 2.0),
    (19, 2, 0.5),      # zoomed in: shrunk to native
    (18, 1.5, 4 / 3),  # 150 % OS display scaling
])
def test_zoom_and_dpr(zoom, dpr, factor):
    up = load_upload(png(texture(240, 360)), zoom=zoom, dpr=dpr)
    assert up["scale_source"] == "zoom"
    assert up["source_mpp"] == pytest.approx(
        imagery.meters_per_map_px(17.4, zoom) / (dpr or 1))
    assert up["native_size"] == [round(360 * factor), round(240 * factor)]
    assert not any("ignored" in w for w in up["warnings"])


def test_mpp_wins_over_zoom_and_dpr_needs_zoom():
    up = load_upload(png(texture()), mpp=0.3, zoom=17, dpr=2)
    assert up["source_mpp"] == 0.3 and any("zoom / dpr ignored" in w for w in up["warnings"])
    up = load_upload(png(texture()), dpr=2)
    assert any("dpr ignored" in w for w in up["warnings"])
    assert up["scale_source"] == "assumed"


@pytest.mark.parametrize("kw,param", [
    (dict(mpp=0), "mpp"), (dict(mpp=-1), "mpp"), (dict(mpp="abc"), "mpp"),
    (dict(mpp=float("nan")), "mpp"), (dict(zoom=40), "zoom"), (dict(zoom=18, dpr=-1), "dpr"),
    (dict(zoom=18, dpr=50), "dpr")])
def test_bad_scale_values(kw, param):
    with pytest.raises(UploadError) as e:
        load_upload(png(texture()), **kw)
    # .param names the bad request parameter, so an API can answer 422, not 415
    assert e.value.code == "unsupported" and e.value.param == param


def test_pseudo_meta_carries_the_ground_pixel_size():
    """infer.area_stats reads meta["m_per_px"] when present: the exact ground
    size of a native px of a plain upload, so its m² do not depend on how the
    grid formula models the Earth (spherical vs WGS84: 0.55 % apart)."""
    raw = png(texture(333, 517))
    up = load_upload(raw, mpp=0.57)
    wn, hn = up["native_size"]
    m = up["meta"]["m_per_px"]
    assert m ** 2 * wn * hn == pytest.approx(517 * 333 * 0.57 ** 2, rel=1e-12)
    assert m == pytest.approx(N_MPP, rel=2e-3)
    assert load_upload(raw)["meta"]["m_per_px"] == N_MPP  # assumed native
    # a zoom-declared screenshot holds true Web-Mercator px: the grid formula applies
    assert "m_per_px" not in load_upload(raw, zoom=17, dpr=2)["meta"]


def test_coarse_scales_warn_then_refuse():
    up = load_upload(png(texture(200, 200)), mpp=0.8)
    assert any("coarser" in w for w in up["warnings"])
    with pytest.raises(UploadError) as e:
        load_upload(png(texture(200, 200)), mpp=5)
    assert e.value.code == "unsupported" and "too coarse" in str(e.value)


def test_fine_scale_shrinks_to_native():
    up = load_upload(png(texture(800, 1000)), mpp=0.0712)  # 4x finer: a drone photo
    assert up["native_size"] == [round(1000 * 0.0712 / N_MPP), round(800 * 0.0712 / N_MPP)]


# ---- plain images: decoding ----------------------------------------------------

def test_16bit_png_is_stretched_not_white():
    rng = np.random.default_rng(1)
    dn = (966 + (texture(200, 300)[..., 0].astype(np.float64) / 255) * (9353 - 966))
    dn = (dn + rng.normal(0, 20, dn.shape)).clip(0, 65535).astype(np.uint16)
    raw = png(dn)
    assert Image.open(io.BytesIO(raw)).mode == "I;16"
    up = load_upload(raw)
    img = up["img"]
    assert (img == 255).mean() < 0.05 and (img == 0).mean() < 0.05
    assert np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2])
    assert np.corrcoef(img[..., 0].ravel(), dn.ravel())[0, 1] > 0.99
    assert any("stretched to 8 bit" in w for w in up["warnings"])


@pytest.mark.parametrize("colortype", [2, 6, 4])  # RGB, RGBA, grey + alpha
def test_16bit_colour_png_is_stretched_not_cut_to_the_high_byte(colortype):
    rgb = texture(200, 300)
    dn = rgb.astype(np.uint16) * 30 + 900  # 12-bit-like DN 900..8550, as QGIS / GEE export
    alpha = np.full((200, 300), 65535, np.uint16)
    alpha[:, :40] = 0
    arr = {2: dn, 6: np.dstack([dn, alpha]), 4: np.dstack([dn[..., 0], alpha])}[colortype]
    raw = png16(arr, colortype)
    # Pillow decodes these to 8 bit by keeping the high byte: DN 3..33 (near black)
    assert Image.open(io.BytesIO(raw)).mode in ("RGB", "RGBA", "LA")
    up = load_upload(raw, mpp=N_MPP)
    img = up["img"]
    assert up["format"] == "PNG" and not up["georeferenced"]
    assert any("uint16 imagery stretched" in w for w in up["warnings"])
    ok = alpha > 0 if colortype != 2 else np.ones(alpha.shape, bool)
    if colortype != 2:
        assert np.array_equal(up["valid"], ok)
    assert np.percentile(img[ok], 99) - np.percentile(img[ok], 1) > 150  # the full range
    ref = rgb if colortype != 4 else np.repeat(rgb[..., :1], 3, -1)
    assert np.corrcoef(img[ok].ravel(), ref[ok].ravel())[0, 1] > 0.99


def test_large_interlaced_16bit_png_rejected_from_header(no_decode):
    ihdr = struct.pack(">IIBBBBB", 6000, 6000, 16, 2, 0, 0, 1)  # Adam7, 216 MB decoded
    raw = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IEND", b"")
    with pytest.raises(UploadError) as e:
        load_upload(raw, mpp=0.2)
    assert e.value.code == "unsupported" and "interlac" in str(e.value)


def test_16bit_colour_png_exif_orientation():
    rgb = texture(120, 200)
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated 90 deg counter-clockwise
    stored = np.rot90(rgb.astype(np.uint16) * 257, 1)
    raw = png16(stored, 2, [(b"eXIf", exif.tobytes()[6:])])  # PNG eXIf: no "Exif\0\0"
    up = load_upload(raw, mpp=N_MPP)
    assert up["original_size"] == [200, 120] and up["img"].shape == (120, 200, 3)
    assert np.corrcoef(up["img"].ravel(), rgb.ravel())[0, 1] > 0.99


def test_exif_rotated_jpeg_is_upright():
    upright = texture(200, 320)
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 deg clockwise to display"
    buf = io.BytesIO()
    Image.fromarray(upright).transpose(Image.Transpose.ROTATE_90).save(
        buf, "JPEG", quality=95, exif=exif.tobytes())
    up = load_upload(buf.getvalue(), mpp=N_MPP)
    assert up["original_size"] == [320, 200] and up["img"].shape == (200, 320, 3)
    assert psnr(up["img"], upright) > 30
    assert up["format"] == "JPEG"


def test_orientation_table_matches_pillow():
    a = texture(30, 50)
    for o, f in uploads._ORIENT.items():
        exif = Image.Exif()
        exif[0x0112] = o
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "PNG", exif=exif.tobytes())
        ref = np.asarray(uploads.ImageOps.exif_transpose(Image.open(buf)))
        assert np.array_equal(f(a), ref), o


def test_transparency_becomes_invalid_and_is_filled():
    rgb = texture(300, 300)
    yy, xx = np.mgrid[0:300, 0:300]
    hole = (yy - 150) ** 2 + (xx - 150) ** 2 < 60 ** 2
    rgba = np.dstack([rgb, np.where(hole, 0, 255).astype(np.uint8)])
    rgba[hole, :3] = 0  # black under transparency
    up = load_upload(png(rgba))
    assert np.array_equal(up["valid"], ~hole)
    assert np.array_equal(up["img"][~hole], rgb[~hole])
    # filled with nearby colours, not left black (the model would see a road / pond)
    assert up["img"][hole].mean() > 0.5 * rgb.mean()
    cm = to_source_grid(np.full((300, 300), 1, np.uint8), up)
    assert np.array_equal(cm == NODATA, hole)
    # resampled: validity follows the hole at the new scale
    up2 = load_upload(png(rgba), mpp=0.57)
    wn = up2["native_size"][0]
    yy, xx = np.mgrid[0:wn, 0:wn]
    hole2 = (yy - wn / 2 + 0.5) ** 2 + (xx - wn / 2 + 0.5) ** 2 < (60 * wn / 300) ** 2
    assert (up2["valid"] == ~hole2).mean() > 0.995
    back = to_source_grid(np.ones((wn, wn), np.uint8), up2)
    assert back.shape == (300, 300) and ((back == NODATA) == hole).mean() > 0.995


@pytest.mark.parametrize("kind", ["png", "tiff"])
def test_fully_transparent_image_is_unreadable(tmp_path, kind):
    rgba = np.dstack([texture(), np.zeros((300, 400), np.uint8)])
    raw = (png(rgba) if kind == "png" else
           write_tif(tmp_path / "t.tif", rgba.transpose(2, 0, 1), photometric="RGB",
                     alpha="YES", colorinterp=RGBA_CI))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing averaged over zero pixels on the way
        with pytest.raises(UploadError) as e:
            load_upload(raw)
    assert e.value.code == "unreadable" and "no valid pixels" in str(e.value)


def test_fill_near_the_gap_and_its_cost_follows_the_outline(monkeypatch):
    """A 2000x2000 upload with only a 160 px square valid (e.g. a drone
    footprint): GDAL's full-resolution fill runs only on blocks near the
    square, not over the whole gap (that took 20-37 s at 73-79 Mpx with 50-86 %
    nodata, e.g. a rotated ortho or a diagonal corridor survey)."""
    rgb = texture(2000, 2000, seed=3)
    ok = np.zeros((2000, 2000), bool)
    ok[920:1080, 900:1060] = True
    sizes = []
    real = uploads.fillnodata

    def spy(image, mask=None, **kw):
        sizes.append(image.size)
        return real(image, mask=mask, **kw)

    monkeypatch.setattr(uploads, "fillnodata", spy)
    up = load_upload(png(np.dstack([rgb, np.where(ok, 255, 0).astype(np.uint8)])))
    assert np.array_equal(up["valid"], ok) and np.array_equal(up["img"][ok], rgb[ok])
    assert sum(sizes) < 0.15 * 3 * 2000 * 2000
    # next to the square: the colours just inside it
    img = up["img"].astype(int)
    assert np.abs(img[920:1080, 899] - rgb[920:1080, 900]).mean() < 6
    assert np.abs(img[919, 900:1060] - rgb[920, 900:1060]).mean() < 6
    # far away: a smooth fill around the square's mean colour, never black
    far = up["img"][:200, :200].reshape(-1, 3)
    assert np.abs(far.mean(0) - rgb[ok].mean(0)).max() < 3 and far.std(0).max() < 2


def test_palette_transparency_and_cmyk():
    rgb = texture(120, 160)
    q = Image.fromarray(rgb).quantize(31)
    idx = np.array(q)
    idx[:20, :20] = 31  # an extra palette entry, transparent (tRNS), used only here
    pal = Image.fromarray(idx, "P")
    pal.putpalette(q.getpalette()[:31 * 3] + [0, 0, 0])
    buf = io.BytesIO()
    pal.save(buf, "PNG", transparency=31)
    up = load_upload(buf.getvalue())
    assert up["valid"] is not None and not up["valid"][:20, :20].any()
    assert up["valid"][20:, 20:].all()
    buf = io.BytesIO()
    Image.fromarray(rgb).convert("CMYK").save(buf, "JPEG", quality=95)
    up = load_upload(buf.getvalue())
    assert up["img"].shape == rgb.shape and psnr(up["img"], rgb) > 25


def test_opaque_alpha_is_dropped():
    up = load_upload(png(np.dstack([texture(), np.full((300, 400), 255, np.uint8)])))
    assert up["valid"] is None


def test_to_source_grid_inverts_the_resampling():
    up = load_upload(png(texture(300, 400)), mpp=0.57)
    wn, hn = up["native_size"]
    cm = (np.arange(wn)[None, :] // 40 % 4 * np.ones((hn, 1))).astype(np.uint8)
    back = to_source_grid(cm, up)
    assert back.shape == (300, 400)
    expect = (np.arange(400) * wn / 400 // 40 % 4).astype(np.uint8)  # nearest px centre
    agree = (back == expect[None, :]).mean()
    assert agree > 0.98
    with pytest.raises(ValueError):
        to_source_grid(cm[:, :-1], up)


# ---- limits and safety -----------------------------------------------------------

@pytest.fixture
def no_decode(monkeypatch):
    """Fail the test if any pixel data gets decoded."""
    def boom(*a, **k):
        raise AssertionError("pixels were decoded")
    monkeypatch.setattr(ImageFile.ImageFile, "load", boom)
    monkeypatch.setattr(rasterio.io.DatasetReader, "read", boom)
    monkeypatch.setattr(rasterio.io.DatasetReader, "dataset_mask", boom)


def test_byte_cap_before_decoding(no_decode, monkeypatch):
    raw = png_header(1000, 1000) + b"\0" * 3000
    with pytest.raises(UploadError) as e:
        load_upload(raw, max_bytes=2000)
    assert e.value.code == "too_large" and "MB" in str(e.value)
    monkeypatch.setattr(uploads, "MAX_UPLOAD_MB", 0.001)  # the env default
    with pytest.raises(UploadError) as e:
        load_upload(raw)
    assert e.value.code == "too_large"


@pytest.mark.parametrize("side", [13000, 40000])  # ours (150 Mpx), Pillow's (179 Mpx)
def test_decompression_bomb_header_rejected_without_decoding(no_decode, side):
    with pytest.raises(UploadError) as e:
        load_upload(png_header(side, side), mpp=0.03)
    assert e.value.code == "too_large"


def test_native_limit_rejects_before_decoding_and_explains(no_decode):
    with pytest.raises(UploadError) as e:  # 81 Mpx assumed native > 80 Mpx
        load_upload(png_header(9000, 9000))
    msg = str(e.value)
    assert e.value.code == "too_large" and "6.6 km²" in msg and "6.5 km²" in msg
    assert "crop" in msg and "give mpp or zoom" in msg
    with pytest.raises(UploadError) as e:  # 1140 m square -> 16 Mpx native
        load_upload(png_header(2000, 2000), mpp=0.57, max_native_mpx=10)
    assert e.value.code == "too_large" and "1.3 km²" in str(e.value)
    assert "0.8 km² (10 Mpx)" in str(e.value)


def test_geotiff_bomb_rejected_from_header(tmp_path, no_decode):
    tr = Affine(0.3, 0, 8.7e6, 0, -0.3, 1.98e6)
    with rasterio.open(tmp_path / "bomb.tif", "w", driver="GTiff", width=60000, height=60000,
                       count=3, dtype="uint8", crs="EPSG:3857", transform=tr, tiled=True,
                       compress="deflate", SPARSE_OK=True):
        pass  # header only: no tile is ever written
    raw = (tmp_path / "bomb.tif").read_bytes()
    assert len(raw) < 2e6
    with pytest.raises(UploadError) as e:
        load_upload(raw)
    assert e.value.code == "too_large" and "3600 Mpx" in str(e.value)


def test_geotiff_native_limit_from_bounds(tmp_path, no_decode):
    tr = Affine(0.2986, 0, 8.7e6, 0, -0.2986, 1.98e6)  # native resolution, 144 Mpx
    with rasterio.open(tmp_path / "big.tif", "w", driver="GTiff", width=12000, height=12000,
                       count=3, dtype="uint8", crs="EPSG:3857", transform=tr, tiled=True,
                       compress="deflate", SPARSE_OK=True):
        pass
    with pytest.raises(UploadError) as e:
        load_upload((tmp_path / "big.tif").read_bytes())
    assert e.value.code == "too_large" and "bounding box" in str(e.value)


@pytest.mark.parametrize("wh,kw", [((40, 40), {}), ((63, 500), {}),
                                   ((200, 200), dict(mpp=0.05))])  # 35 px at native
def test_too_small(wh, kw):
    with pytest.raises(UploadError) as e:
        load_upload(png(texture(wh[1], wh[0])), **kw)
    assert e.value.code == "too_small"


@pytest.mark.parametrize("raw", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"x" * 50,
                                 b"II*\x00" + b"\x00" * 100, b"\xff\xd8\xff" + b"\x00" * 30])
def test_corrupt_bytes_are_unreadable(raw):
    with pytest.raises(UploadError) as e:
        load_upload(raw)
    assert e.value.code == "unreadable"


def test_truncated_png_is_unreadable():
    raw = png(texture(300, 400))
    with pytest.raises(UploadError) as e:
        load_upload(raw[: len(raw) // 2])
    assert e.value.code == "unreadable"


def test_heic_is_unsupported():
    raw = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64
    with pytest.raises(UploadError) as e:
        load_upload(raw, "IMG_0001.HEIC")
    assert e.value.code == "unsupported" and "JPEG" in str(e.value)
    with pytest.raises(UploadError) as e:  # an MP4 video is not a HEIC photo
        load_upload(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64)
    assert e.value.code == "unreadable"


def test_error_codes_are_checked():
    with pytest.raises(ValueError):
        UploadError("nope", "x")
    assert isinstance(UploadError("too_small", "x"), ValueError)
    # picklable, e.g. raised in a process-pool worker
    for err in (UploadError("too_large", "x"), UploadError("unsupported", "bad", param="mpp")):
        back = pickle.loads(pickle.dumps(err))
        assert (type(back), back.code, str(back), back.param) == (
            UploadError, err.code, str(err), err.param)


# ---- JPEG 2000 structure ---------------------------------------------------------

@pytest.fixture
def small_jp2(tmp_path):
    """A valid 256 px RGB JPEG 2000 file (one tile, 2 resolution levels)."""
    return to_jp2(tmp_path, texture(256, 256).transpose(2, 0, 1), "small", QUALITY=50,
                  BLOCKXSIZE=256, BLOCKYSIZE=256, RESOLUTIONS=2)


@pytest.fixture
def opened(monkeypatch):
    """Records every rasterio MemoryFile.open (GDAL parses the file there)."""
    calls = []
    real = rasterio.io.MemoryFile.open

    def spy(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(rasterio.io.MemoryFile, "open", spy)
    return calls


@pytest.mark.parametrize("w,tile,code,msg", [
    # 62,500 tiles in a 45 KB file: OpenJPEG's per-tile state took 0.8 GB at open
    (4000, 16, "unsupported", "62500 tiles"),
    # one 169 Mpx tile: decoded window by window, re-reading the tile each time
    (13000, 13000, "unsupported", "13000x13000 px tiles"),
])
def test_jp2_tile_structure_checked_before_opening(small_jp2, opened, w, tile, code, msg):
    with pytest.raises(UploadError) as e:
        load_upload(jp2_claiming(small_jp2, w, w, tile), mpp=0.1)
    assert e.value.code == code and msg in str(e.value)
    assert not opened


def test_jp2_decode_cost_counts_resolution_levels(small_jp2, opened):
    # 400 Mpx at 5 cm without resolution levels: every px would be decoded
    with pytest.raises(UploadError) as e:
        load_upload(jp2_claiming(small_jp2, 20000, 20000, 1024, levels=0), mpp=0.05)
    assert e.value.code == "too_large" and "resolution level" in str(e.value)
    assert not opened
    # with 6 levels (the GDAL / opj_compress default) GDAL reads a reduced
    # level: the header passes (and the absent pixel data is then unreadable)
    with pytest.raises(UploadError) as e:
        load_upload(jp2_claiming(small_jp2, 20000, 20000, 1024, levels=5), mpp=0.05)
    assert e.value.code == "unreadable" and opened


def test_jp2_single_tile_limit_on_a_real_file(tmp_path, monkeypatch):
    rgb = texture(600, 700)
    raw = to_jp2(tmp_path, rgb.transpose(2, 0, 1), "one", QUALITY=100, REVERSIBLE=True,
                 BLOCKXSIZE=700, BLOCKYSIZE=600)
    assert np.array_equal(load_upload(raw, mpp=N_MPP)["img"], rgb)
    monkeypatch.setattr(uploads, "MAX_JP2_TILE_MPX", 0.4)
    with pytest.raises(UploadError) as e:
        load_upload(raw, mpp=N_MPP)
    assert e.value.code == "unsupported" and "700x600 px tiles" in str(e.value)
    assert "BLOCKXSIZE=1024" in str(e.value)


@pytest.fixture
def gdal_envs(monkeypatch):
    """Records the GDAL_CACHEMAX of every rasterio.Env that uploads opens."""
    seen, real = [], rasterio.Env

    def spy(**kw):
        seen.append(kw.get("GDAL_CACHEMAX"))
        return real(**kw)

    monkeypatch.setattr(uploads.rasterio, "Env", spy)
    return seen


@pytest.mark.parametrize("georef", [False, True])
@pytest.mark.parametrize("resolutions,rejected", [(1, True), (3, False)])
def test_jp2_read_cache_counts_the_level_the_read_uses(tmp_path, monkeypatch, gdal_envs,
                                                       georef, resolutions, rejected):
    # One 700x600 px tile read ~5.7x decimated (0.05 m px). Without reduced
    # levels the read decodes the full tile, 1.26 MB of RGB; with 3
    # resolutions it decodes the 4x level, 175x150 px (79 KB).
    geo = dict(crs="EPSG:32644", transform=Affine(0.05, 0, 217000, 0, -0.05, 1931000))
    raw = to_jp2(tmp_path, texture(600, 700).transpose(2, 0, 1), f"r{resolutions}", QUALITY=50,
                 BLOCKXSIZE=700, BLOCKYSIZE=600, RESOLUTIONS=resolutions,
                 **(geo if georef else {}))
    monkeypatch.setattr(uploads, "MAX_JP2_CACHE_MB", 0.5)
    kw = {} if georef else {"mpp": 0.05}
    if not rejected:
        up = load_upload(raw, **kw)
        assert up["georeferenced"] == georef and min(up["native_size"]) >= 100
        assert gdal_envs[-1] == uploads.GDAL_CACHE_MB << 20  # fits the default cache
        return
    with pytest.raises(UploadError) as e:
        load_upload(raw, **kw)
    assert e.value.code == "unsupported"
    assert "700x600 px tiles with 1 resolution level" in str(e.value)
    assert "BLOCKXSIZE=1024" in str(e.value)


def test_jp2_read_cache_holds_a_row_of_decoded_tiles(gdal_envs):
    # 12 tiles of 1024x12000 px (what GDAL writes for a 12000 px JP2 asked
    # for one tile). With 1 resolution level the 128 MB cache thrashed: 143 s,
    # vs 4.4 s with 512 MB. The row of decoded tiles is 12 x 1024 x 12000 x 3 B.
    one_level = (12000, 12000, 1024, 12000, 12, 3, 0)
    with uploads._jp2_read_env(one_level, 5, 1):
        pass
    assert gdal_envs[-1] == int(12 * 1024 * 12000 * 3 * 1.25)
    # with 6 resolutions a 5x decimated read decodes the 4x level: 27.6 MB
    with uploads._jp2_read_env(one_level[:-1] + (5,), 5, 1):
        pass
    assert gdal_envs[-1] == uploads.GDAL_CACHE_MB << 20
    with pytest.raises(UploadError) as e:  # 16-bit, a 40000 px single tile
        uploads._jp2_read_env((40000, 40000, 40000, 40000, 1, 3, 0), 1, 2)
    assert e.value.code == "unsupported" and "MB of decoded tiles" in str(e.value)
    assert uploads._jp2_read_env(None, 5, 1).__class__.__name__ == "nullcontext"


@pytest.mark.parametrize("raw", [b"\x00\x00\x00\x0cjP  \r\n\x87\n" + b"\x00" * 40,
                                 b"\xffO\xffQ\x00\x29" + b"\x00" * 20])
def test_jp2_without_a_readable_header_is_unreadable(raw):
    with pytest.raises(UploadError) as e:
        load_upload(raw)
    assert e.value.code == "unreadable"


# ---- plain (non-georeferenced) TIFF ------------------------------------------------

def test_plain_16bit_rgb_tiff_and_orientation(tmp_path):
    rgb = texture(160, 240)
    dn = (rgb.astype(np.uint16) * 30 + 900).transpose(2, 0, 1)  # 12-bit-like DN
    raw = write_tif(tmp_path / "plain.tif", dn, photometric="RGB")
    up = load_upload(raw, mpp=N_MPP)
    assert not up["georeferenced"] and up["format"] == "GTiff"
    assert up["img"].shape == (160, 240, 3)
    assert np.corrcoef(up["img"].ravel(), rgb.ravel())[0, 1] > 0.99  # Pillow: 8 % of range
    assert any("uint16 imagery stretched" in w for w in up["warnings"])
    # GDAL ignores the TIFF Orientation tag; we apply it
    buf = io.BytesIO()
    Image.fromarray(rgb).transpose(Image.Transpose.ROTATE_90).save(
        buf, "TIFF", tiffinfo={0x0112: 6})
    up = load_upload(buf.getvalue(), mpp=N_MPP)
    assert up["original_size"] == [240, 160] and np.array_equal(up["img"], rgb)


def test_plain_tiff_shrinks_with_area_average(tmp_path):
    rgb = texture(400, 400)
    up = load_upload(write_tif(tmp_path / "p.tif", rgb.transpose(2, 0, 1), photometric="RGB"),
                     mpp=N_MPP / 2)
    assert up["native_size"] == [200, 200]
    ref = rgb.reshape(200, 2, 200, 2, 3).mean((1, 3))
    assert np.abs(up["img"].astype(float) - ref).max() <= 1


# ---- georeferenced rasters ---------------------------------------------------------

@pytest.mark.data
def test_geotiff_3857_on_the_grid_is_copied_exactly(tmp_path):
    img, _, meta = area_crop(*CROP)
    raw = write_tif(tmp_path / "g.tif", img.transpose(2, 0, 1), crs="EPSG:3857",
                    transform=grid_transform(meta), photometric="RGB")
    up = load_upload(raw, "g.tif", mpp=0.5)
    m = up["meta"]
    assert up["georeferenced"] and m["georeferenced"] and m["crs"] == "EPSG:3857"
    assert up["crs"] == "EPSG:3857" and up["scale_source"] == "georeference"
    assert abs(m["x0"] - meta["x0"]) * 2 < 1e-6 and abs(m["y0"] - meta["y0"]) * 2 < 1e-6
    assert np.array_equal(up["img"], img) and up["valid"] is None
    assert up["source_mpp"] == pytest.approx(uploads.native_mpp(meta["center"][0]), rel=1e-3)
    assert any("ignored" in w for w in up["warnings"])  # mpp=0.5 was not used
    assert imagery.bbox_latlng(m) == pytest.approx(imagery.bbox_latlng(meta), abs=1e-9)
    assert total_m2(np.zeros(img.shape[:2], np.uint8), m) == pytest.approx(
        total_m2(np.zeros(img.shape[:2], np.uint8), meta), rel=1e-9)


def _check_roundtrip(up, img, meta, min_psnr=30):
    m = up["meta"]
    h, w = up["img"].shape[:2]
    exp = sample_at(img.astype(np.float64), meta, m, (h, w))
    ok = (up["valid"] if up["valid"] is not None else np.ones((h, w), bool))
    ok = ok & np.isfinite(exp).all(-1)
    assert ok.mean() > 0.5  # enough px compared (a rotated footprint fills ~80 %)
    assert psnr(up["img"], np.nan_to_num(exp), ok) >= min_psnr
    c = (slice(h // 2 - 128, h // 2 + 128), slice(w // 2 - 128, w // 2 + 128))
    dy, dx = residual_shift(exp[c], up["img"][c])
    assert math.hypot(dx, dy) < 0.1  # georeferenced to a tenth of a pixel
    return exp


@pytest.mark.data
def test_geotiff_utm_10cm_with_alpha_roundtrip(tmp_path, monkeypatch):
    img, _, meta = area_crop(*CROP)
    rgba, dtr = to_crs(img, meta, "EPSG:32644", 0.10, hole=True)
    raw = write_tif(tmp_path / "utm.tif", rgba, crs="EPSG:32644", transform=dtr,
                    photometric="RGB", alpha="YES", colorinterp=RGBA_CI)
    up = load_upload(raw, "ortho.tif")
    m, s = up["meta"], up["meta"]["scale"]
    assert up["crs"] == "EPSG:32644" and up["source_mpp"] == pytest.approx(0.1, rel=0.01)
    assert up["original_size"] == [rgba.shape[2], rgba.shape[1]]
    # lands on the original grid's extent (to the strict edge px)
    assert abs(m["x0"] - meta["x0"]) * s < 1.5 and abs(m["y0"] - meta["y0"]) * s < 1.5
    _check_roundtrip(up, img, meta)
    # the hole and the collar are invalid; colours under them are filled
    assert 0.03 < 1 - up["valid"].mean() < 0.15
    assert up["img"][~up["valid"]].mean() > 0.5 * img.mean()
    # m² of the valid area agree with the UTM raster's own (within edge px)
    cm = np.where(up["valid"], 3, NODATA).astype(np.uint8)
    utm_m2 = (rgba[3] == 255).sum() * 0.10 ** 2
    assert total_m2(cm, m) == pytest.approx(utm_m2, rel=0.02)
    # back onto the ortho's own grid: classes where valid, NODATA outside
    back = to_source_grid(cm, up)
    assert back.shape == rgba.shape[1:]
    inside = rgba[3] == 255
    assert (back[inside] == 3).mean() > 0.98 and (back[~inside] == NODATA).mean() > 0.98
    monkeypatch.setattr(uploads, "MAX_DECODE_MPX", 1)  # e.g. a 3 cm ortho's own grid
    with pytest.raises(UploadError) as e:
        to_source_grid(cm, up)
    assert e.value.code == "too_large" and "native grid" in str(e.value)


@pytest.mark.data
def test_geotiff_alpha_without_alpha_flag(tmp_path):
    img, _, meta = area_crop(*CROP)
    rgba, dtr = to_crs(img, meta, "EPSG:32644", 0.20, hole=True)
    undeclared = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined]
    raw = write_tif(tmp_path / "u4.tif", rgba, crs="EPSG:32644", transform=dtr,
                    photometric="RGB", colorinterp=undeclared)
    up = load_upload(raw)
    assert any("no alpha flag" in w for w in up["warnings"])
    ref = load_upload(write_tif(tmp_path / "a4.tif", rgba, crs="EPSG:32644", transform=dtr,
                                photometric="RGB", alpha="YES", colorinterp=RGBA_CI))
    assert np.array_equal(up["valid"], ref["valid"]) and np.array_equal(up["img"], ref["img"])
    # a 4th band with data (e.g. near infrared) is not a mask
    rgbn = rgba.copy()
    rgbn[3] = rgba[1] // 2 + 7
    up = load_upload(write_tif(tmp_path / "n4.tif", rgbn, crs="EPSG:32644", transform=dtr,
                               photometric="RGB", colorinterp=undeclared))
    assert not any("no alpha flag" in w for w in up["warnings"])
    assert any("only bands [1, 2, 3]" in w for w in up["warnings"])


@pytest.mark.data
def test_geotiff_utm_5cm_nodata0_roundtrip(tmp_path):
    img, _, meta = area_crop(*CROP)
    sub, _, smeta = helpers.crop(img, None, meta, 200, 560, 240, 600)  # 360 px ~ 100 m
    rgba, dtr = to_crs(sub, smeta, "EPSG:32644", 0.05, hole=True)
    rgb = rgba[:3].copy()
    black = (rgba[3] == 255) & (rgb.max(0) == 0)
    rgb[:, black] = 1  # a true-black px would read as nodata
    raw = write_tif(tmp_path / "nd0.tif", rgb, crs="EPSG:32644", transform=dtr, nodata=0,
                    photometric="RGB")
    up = load_upload(raw)
    assert up["source_mpp"] == pytest.approx(0.05, rel=0.01)
    _check_roundtrip(up, sub, smeta)
    assert up["valid"] is not None and not up["valid"].all()


@pytest.mark.data
def test_geotiff_rotated_geotransform(tmp_path):
    img, _, meta = area_crop(*CROP)
    tr = grid_transform(meta)
    rot = Affine.translation(tr.c, tr.f) @ Affine.rotation(12) @ Affine.scale(0.2, -0.2)
    rgba = warped(img, meta, "EPSG:3857", rot, (1000, 1000))
    raw = write_tif(tmp_path / "rot.tif", rgba, crs="EPSG:3857", transform=rot,
                    photometric="RGB", alpha="YES", colorinterp=RGBA_CI)
    up = load_upload(raw)
    assert up["source_mpp"] == pytest.approx(0.2 * math.cos(math.radians(17.55)), rel=0.01)
    _check_roundtrip(up, img, meta)


@pytest.mark.data
def test_rotated_raster_is_warped_in_tiles_with_bounded_reads(tmp_path, monkeypatch):
    """45 deg: a full-width output strip maps onto a diagonal band whose
    bounding window is most of the raster (5 GB for an 81 Mpx ortho); tiles
    keep every read near STRIP_PX and join without seams."""
    img, _, meta = area_crop(*CROP)
    tr = grid_transform(meta)
    rot = Affine.translation(tr.c, tr.f) @ Affine.rotation(45) @ Affine.scale(0.25, -0.25)
    rgba = warped(img, meta, "EPSG:3857", rot, (900, 900))
    raw = write_tif(tmp_path / "rot45.tif", rgba, crs="EPSG:3857", transform=rot,
                    photometric="RGB", alpha="YES", colorinterp=RGBA_CI)
    whole = load_upload(raw)  # one strip or tile at the default STRIP_PX
    reads, grid = [], []
    read_rgba, pieces = uploads._read_rgba, uploads._warp_pieces

    def spy_read(*a):
        reads.append(a[-2][0] * a[-2][1])  # px read (decimated window)
        return read_rgba(*a)

    def spy_pieces(hd, wd, jac):
        grid.append(hd * wd)  # the warped grid: the raster's bounding box
        return pieces(hd, wd, jac)

    monkeypatch.setattr(uploads, "_read_rgba", spy_read)
    monkeypatch.setattr(uploads, "_warp_pieces", spy_pieces)
    monkeypatch.setattr(uploads, "STRIP_PX", 4e4)  # many pieces on a small raster
    up = load_upload(raw)
    assert len(reads) > 20 and max(reads) <= 2 * uploads.STRIP_PX
    assert sum(reads) < 2.5 * grid[0]  # full-width strips read 13.6x here
    _check_roundtrip(up, img, meta)
    assert np.array_equal(up["valid"], whole["valid"]) and up["meta"] == whole["meta"]
    d = np.abs(up["img"].astype(int) - whole["img"])[up["valid"]]
    assert (d > 2).mean() < 0.002 and d.mean() < 0.05


@pytest.mark.parametrize("tr", [
    Affine(0.3, 0.3, 8.7e6, 0.3, 0.3, 1.98e6),     # singular: both axes the same
    Affine(0.3, 0.29, 8.7e6, 0.0, -0.05, 1.98e6),  # pixel axes 10 deg apart
])
def test_degenerate_geotransform_is_unsupported(tmp_path, tr):
    raw = write_tif(tmp_path / "d.tif", texture(256, 256).transpose(2, 0, 1),
                    crs="EPSG:3857", transform=tr, photometric="RGB")
    with pytest.raises(UploadError) as e:
        load_upload(raw)
    assert e.value.code == "unsupported" and "degenerate" in str(e.value)


def test_uint16_palette_geotiff(tmp_path):
    idx = (np.arange(256 * 300).reshape(1, 256, 300) % 1000).astype(np.uint16)
    cmap = {i: (i % 256, (i * 7) % 256, (i * 13) % 256, 255) for i in range(1000)}
    res = uploads.MERC / imagery.WORLD(18) / 2  # on the grid: copied exactly
    path = tmp_path / "pal16.tif"
    with rasterio.open(path, "w", driver="GTiff", width=300, height=256, count=1,
                       dtype="uint16", crs="EPSG:3857", photometric="PALETTE",
                       transform=Affine(res, 0, 8.7e6, 0, -res, 1.98e6)) as dst:
        dst.write(idx)
        dst.write_colormap(1, cmap)
    up = load_upload(path.read_bytes())
    lut = np.array([cmap[i][:3] for i in range(1000)], np.uint8)
    assert np.array_equal(up["img"], lut[idx[0]])
    assert "m_per_px" not in up["meta"]  # a real grid: the grid formula gives its m²


@pytest.mark.data
def test_geotiff_epsg4326(tmp_path):
    img, _, meta = area_crop(*CROP)
    rgba, dtr = to_crs(img, meta, "EPSG:4326", 1e-6)
    raw = write_tif(tmp_path / "ll.tif", rgba, crs="EPSG:4326", transform=dtr,
                    photometric="RGB", alpha="YES", colorinterp=RGBA_CI)
    up = load_upload(raw)
    assert up["crs"] == "EPSG:4326" and up["source_mpp"] == pytest.approx(0.108, rel=0.03)
    _check_roundtrip(up, img, meta)


@pytest.mark.data
def test_geotiff_uint16_4band_and_single_band(tmp_path):
    img, _, meta = area_crop(*CROP)
    nir = img[..., :1].astype(np.uint16) * 37
    rgbn = np.concatenate([img.astype(np.uint16) * 37 + 400, nir], -1).transpose(2, 0, 1)
    up = load_upload(write_tif(tmp_path / "rgbn.tif", rgbn, crs="EPSG:3857",
                               transform=grid_transform(meta)))
    assert any("uint16 imagery stretched" in w for w in up["warnings"])
    assert any("only bands [1, 2, 3]" in w for w in up["warnings"])
    for b in range(3):
        assert np.corrcoef(up["img"][..., b].ravel(), img[..., b].ravel())[0, 1] > 0.99
    grey = img.mean(-1).astype(np.uint8)[None]
    up = load_upload(write_tif(tmp_path / "grey.tif", grey, crs="EPSG:3857",
                               transform=grid_transform(meta)))
    assert np.array_equal(up["img"][..., 0], grey[0]) and np.array_equal(
        up["img"][..., 0], up["img"][..., 2])
    assert any("single-band" in w for w in up["warnings"])


@pytest.mark.data
def test_geotiff_palette(tmp_path):
    img, _, meta = area_crop(*CROP)
    q = Image.fromarray(img).quantize(64)
    idx = np.asarray(q)[None]
    pal = np.asarray(q.getpalette()[:64 * 3], np.uint8).reshape(-1, 3)
    path = tmp_path / "pal.tif"
    with rasterio.open(path, "w", driver="GTiff", width=idx.shape[2], height=idx.shape[1],
                       count=1, dtype="uint8", crs="EPSG:3857", transform=grid_transform(meta),
                       photometric="PALETTE") as dst:
        dst.write(idx)
        dst.write_colormap(1, {i: (*pal[i], 255) for i in range(64)})
    up = load_upload(path.read_bytes())
    assert np.array_equal(up["img"], pal[idx[0]])


@pytest.mark.data
def test_jpeg2000_georeferenced_and_plain(tmp_path):
    img, _, meta = area_crop(*CROP)
    for crs in ("EPSG:3857", None):
        geo = dict(crs=crs, transform=grid_transform(meta)) if crs else {}
        gtiff = tmp_path / f"src_{bool(crs)}.tif"
        write_tif(gtiff, img.transpose(2, 0, 1), photometric="RGB", **geo)
        jp2 = tmp_path / f"img_{bool(crs)}.jp2"
        with rasterio.open(gtiff) as src:  # JP2OpenJPEG writes by copy only
            rasterio.shutil.copy(src, jp2, driver="JP2OpenJPEG", QUALITY=100, REVERSIBLE=True)
        up = load_upload(jp2.read_bytes())
        assert up["format"] == "JP2OpenJPEG" and up["georeferenced"] == bool(crs)
        assert np.array_equal(up["img"], img)  # lossless, already on the grid / native
        if crs:
            assert abs(up["meta"]["x0"] - meta["x0"]) * 2 < 1e-6


@pytest.mark.data
def test_truncated_geotiff_is_unreadable(tmp_path):
    img, _, meta = area_crop(*CROP)
    raw = write_tif(tmp_path / "t.tif", img.transpose(2, 0, 1), crs="EPSG:3857",
                    transform=grid_transform(meta), photometric="RGB")
    with pytest.raises(UploadError) as e:
        load_upload(raw[: len(raw) * 6 // 10])
    assert e.value.code == "unreadable"


@pytest.mark.data
def test_geotiff_outside_hyderabad_warns(tmp_path):
    img, _, meta = area_crop(*CROP)
    tr = grid_transform(meta)
    far = Affine(tr.a, 0, tr.c - 3e5, 0, tr.e, tr.f)  # ~300 km west
    up = load_upload(write_tif(tmp_path / "far.tif", img.transpose(2, 0, 1),
                               crs="EPSG:3857", transform=far, photometric="RGB"))
    assert any("outside greater Hyderabad" in w for w in up["warnings"])


# ---- the model at the wrong scale (measurement) ------------------------------------

def _metrics(gt, pred):
    ok = pred != NODATA
    gt, pred = gt[ok], pred[ok]
    gb, pb = np.isin(gt, (1, 2)), np.isin(pred, (1, 2))
    return {"acc": float((gt == pred).mean()),
            "b_iou": float((gb & pb).sum() / max((gb | pb).sum(), 1)),
            "b_ratio": float(pb.sum() / max(gb.sum(), 1))}


@pytest.mark.data
@pytest.mark.model
@pytest.mark.slow
def test_native_rescale_restores_quality_on_val_crop(model, capsys):
    """kompally (val) 1500x1500 crop shown at s x the native resolution
    (s < 1 coarser, s > 1 finer), segmented as is (naive) and through
    load_upload with the true mpp (rescaled), scored on the upload's grid
    against the auto-label resampled the same way. 512 px windows keep the
    process under 4 GB (1024 px ones peak at 3.8 GB in infer.predict alone).

    Measured (2026-10, unet_r34.pt, 512 / 384 px windows, peak RSS 1.8 GB):
    native acc 0.697 / building IoU 0.568; s=2.0 naive 0.594 / 0.193 ->
    rescaled 0.697 / 0.557; s=0.5 naive 0.581 / 0.583 (building area 1.23x)
    -> rescaled 0.653 / 0.430 (0.56x); with 1024 px windows every figure is
    within 0.010 of these. Finer uploads get native quality back; coarser
    ones gain accuracy but lose small buildings either way (hence the
    'coarser' warning)."""
    img, gt, _ = area_crop("kompally", 1050, 2550, 1170, 2670)
    rows = {}
    win = dict(window=512, stride=384)
    for s in (0.5, 2.0):
        size = (round(1500 * s), round(1500 * s))
        up_img = np.asarray(Image.fromarray(img).resize(size, Image.BICUBIC))
        up_gt = np.asarray(Image.fromarray(gt).resize(size, Image.NEAREST))
        naive, _ = infer.predict(model, up_img, **win)
        up = load_upload(png(up_img), mpp=N_MPP / s)
        cm, _ = infer.predict(model, up["img"], **win)
        rows[s] = {"naive": _metrics(up_gt, naive),
                   "rescaled": _metrics(up_gt, to_source_grid(cm, up))}
    base, _ = infer.predict(model, img, **win)
    rows[1.0] = {"native": _metrics(gt, base)}
    with capsys.disabled():
        for s, r in sorted(rows.items()):
            print(f"\n  s={s}: " + "; ".join(
                f"{k} acc {v['acc']:.3f} bIoU {v['b_iou']:.3f} b_ratio {v['b_ratio']:.2f}"
                for k, v in r.items()))
    for s in (0.5, 2.0):
        assert rows[s]["rescaled"]["acc"] > rows[s]["naive"]["acc"] + 0.03
    fine, native = rows[2.0]["rescaled"], rows[1.0]["native"]
    assert rows[2.0]["naive"]["b_iou"] < native["b_iou"] - 0.2  # the bug being fixed
    assert fine["acc"] == pytest.approx(native["acc"], abs=0.01)
    assert fine["b_iou"] == pytest.approx(native["b_iou"], abs=0.03)
