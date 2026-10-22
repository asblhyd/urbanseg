"""Uploaded images -> model-ready imagery at the native scale, with correct m².

The model was trained on Google zoom-18 / scale-2 imagery (~0.285 m/px in
Hyderabad) and only works near that scale. On val crops (kompally, lbnagar)
a 0.14 m/px upload (zoomed in, drone) drops building IoU from 0.58 / 0.78 to
0.20 / 0.39, and a 0.57 m/px one (a zoomed-out screenshot) over-predicts
buildings 1.1-1.25x. Resampling to the native scale first gives finer uploads
the native result back and raises accuracy on coarser ones (their small
buildings stay unreliable). Area per pixel follows from the same scale. The
scale is never guessed from the pixels (automatic estimation was within 25 %
only 11-17 % of the time); it comes from

- the georeference of a GeoTIFF / COG / JPEG 2000 with a CRS: the raster is
  reprojected onto the zoom-18 / scale-2 Web-Mercator grid that coordinate
  mode uses, so its meta supports fusion, GeoJSON and m² exactly like a
  fetched area;
- for plain images (PNG, JPEG, WebP, plain TIFF ...): ``mpp`` (ground metres
  per pixel), or the Google Maps ``zoom`` and device-pixel ratio ``dpr`` of a
  screenshot (see _plain_scale). Without either, the native scale is assumed
  and a warning says so.

Nothing is downscaled silently: an upload that would exceed the native pixel
budget is rejected with its ground area and how to fix it. Invalid pixels
(alpha, nodata, outside a reprojected raster) are returned in ``valid``.
"""
import io
import math
import os
import struct
import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
import rasterio
from PIL import Image, ImageOps, TiffImagePlugin
from rasterio.enums import ColorInterp, Interleaving, Resampling
from rasterio.errors import CRSError, NotGeoreferencedWarning
from rasterio.fill import fillnodata
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import reproject, transform as warp_transform, transform_bounds
from rasterio.windows import Window

from . import config, imagery

NODATA = 255  # classmap value for invalid pixels (same as infer.NODATA)
MIN_SIDE = 64  # px per side, before and after resampling
# plain images carry no location: their scale is evaluated, and their pseudo
# grid placed, at this nominal Hyderabad point (the grid scale varies < 0.4 %
# across config.HYD_BBOX)
NOMINAL_LAT, NOMINAL_LNG = 17.4, 78.48
MAX_UPLOAD_MB = float(os.environ.get("URBANSEG_MAX_UPLOAD_MB", "200"))
# native px after resampling; 80 Mpx at 0.285 m/px is about 6.5 km²
MAX_NATIVE_MPX = float(os.environ.get("URBANSEG_MAX_NATIVE_MPX", "80"))
# PNG/JPEG/WebP are decoded whole: this bounds their memory (~4 B/px); it sits
# just under Pillow's own 179 Mpx bomb error so our message is the one shown
MAX_DECODE_MPX = float(os.environ.get("URBANSEG_MAX_DECODE_MPX", "150"))
# TIFF/JP2 are read decimated and in strips, so memory follows the output;
# this bounds the decode time of e.g. a compressed bomb
MAX_SOURCE_MPX = float(os.environ.get("URBANSEG_MAX_SOURCE_MPX", "2000"))
MAX_BLOCK_MB = 256  # a TIFF block is always decoded whole
# JPEG 2000 structure, checked from its SIZ / COD markers before GDAL parses
# the file (see _check_jp2): OpenJPEG keeps ~13 KB of state per tile from the
# header on (a 45 KB file claiming 62,500 tiles took 0.8 GB to open), and it
# decodes a big tile window by window, re-reading the whole tile each time
# (one 64 Mpx tile: 5.0 s vs 2.5 s in 1024 px tiles; one 144 Mpx tile: 12 s;
# memory stays bounded)
MAX_JP2_TILE_BANDS = 32768  # tiles x bands, e.g. 8192 RGBA tiles
MAX_JP2_TILE_MPX = 150
# A JPEG 2000 read decodes whole tiles into GDAL's block cache. When one row
# of decoded tiles (at the resolution level read) does not fit, GDAL evicts
# and re-decodes them for every chunk of rows: 12 tiles of 1024x12000 px with
# no reduced levels took 143 s with the 128 MB cache, 4.4 s with 512 MB. So
# JP2 reads get a cache sized to that row (see _jp2_read_env), up to this cap
MAX_JP2_CACHE_MB = float(os.environ.get("URBANSEG_MAX_JP2_CACHE_MB", "768"))
GDAL_CACHE_MB = 128  # everything else streams through a small cache
# s = native m/px / upload m/px. From 2x coarser (0.57 m/px, s < COARSE_WARN)
# buildings degrade even after resampling (areas 0.8-1.2x on test crops);
# below COARSE_LIMIT (2.3 m/px) a building is a few px and nothing is left
COARSE_WARN, COARSE_LIMIT = 0.55, 0.125
# nodata fill (see _fill_invalid): reach of the far field, computed on a
# 1/FILL_COARSE grid; full-resolution fill within NEAR_PX of valid px, run on
# FILL_BLOCK px blocks near the gap only
FILL_PX, FILL_COARSE, NEAR_PX, FILL_BLOCK = 128, 8, 16, 128
GDAL_THREADS = int(os.environ.get("URBANSEG_GDAL_THREADS", "2"))
STRIP_PX = 16e6  # decimated source px read per warp strip or tile
MERC = 2 * math.pi * 6378137.0  # EPSG:3857 world width (m)
# the only Pillow decoders tried (no EPS / Ghostscript etc.); JPEG also opens
# multi-picture (MPO) files; AVIF only where this Pillow build has it
PIL_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP", "AVIF")
HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1",
               b"avif", b"avis"}  # ISO-BMFF ftyp brands of phone photos
# the only GDAL drivers allowed to parse each detected format; PNG goes
# through GDAL only for 16-bit colour, which Pillow cuts to 8 bit
RASTER_DRIVERS = {"tiff": ["GTiff"], "jp2": ["JP2OpenJPEG"], "png": ["PNG"]}
RASTER_NAMES = {"tiff": "TIFF", "jp2": "JPEG 2000", "png": "PNG"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SCALE_UNKNOWN = "scale unknown: areas assume 0.285 m/px; give mpp or zoom for correct results"
_U16_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I", "F")
# Image.Transpose of each EXIF orientation (as ImageOps.exif_transpose), on arrays
_ORIENT = {2: lambda a: a[:, ::-1], 3: lambda a: a[::-1, ::-1], 4: lambda a: a[::-1],
           5: lambda a: a.swapaxes(0, 1), 6: lambda a: np.rot90(a, -1),
           7: lambda a: a[::-1, ::-1].swapaxes(0, 1), 8: lambda a: np.rot90(a, 1)}


class UploadError(ValueError):
    """An upload that cannot be used. ``code`` is one of CODES (an API maps them
    to e.g. 400 / 413 / 422 / 415); the message says why and what to do, and
    never echoes the file name or contents. ``param`` names the bad request
    parameter ("mpp", "zoom" or "dpr") when that is the problem, else None:
    an API should then answer 422, not 415."""
    CODES = ("unreadable", "too_large", "too_small", "unsupported")

    def __init__(self, code, message, param=None):
        if code not in self.CODES:
            raise ValueError(f"unknown UploadError code {code!r}")
        super().__init__(message)
        self.code = code
        self.param = param

    def __reduce__(self):  # picklable, e.g. raised in a process-pool worker
        return type(self), (self.code, str(self), self.param)


def native_mpp(lat=NOMINAL_LAT):
    """Ground m per px of the model's native imagery (zoom 18, scale 2) at lat."""
    return imagery.meters_per_map_px(lat, config.ZOOM) / config.SCALE


def grid_transform(meta):
    """EPSG:3857 Affine (image px -> Web-Mercator m) of a zoom/scale grid meta,
    e.g. to write the classmap of a georeferenced upload as a GeoTIFF."""
    res = MERC / imagery.WORLD(meta["zoom"]) / meta["scale"]
    X0, Y0 = _world_to_merc(meta["x0"], meta["y0"], meta["zoom"])
    return Affine(res, 0.0, X0, 0.0, -res, Y0)


def load_upload(raw, filename="", *, mpp=None, zoom=None, dpr=None,
                max_bytes=None, max_native_mpx=None):
    """Decode an uploaded file into imagery at the model's native scale.

    raw: the file's bytes. filename: only a hint (never trusted or echoed).
    mpp / zoom / dpr: the scale of a plain image (ignored, with a warning,
    for a georeferenced raster). max_bytes defaults to URBANSEG_MAX_UPLOAD_MB
    (200 MB); max_native_mpx to URBANSEG_MAX_NATIVE_MPX (80 Mpx ~ 6.5 km²).

    Returns a dict:
      img            HxWx3 uint8 at native_mpp(lat) m/px, invalid px filled with
                     nearby colours (the model input)
      meta           georeferenced: the zoom-18 / scale-2 Web-Mercator grid meta
                     (x0, y0, width_map_px, height_map_px, zoom, scale, center,
                     georeferenced=True, crs="EPSG:3857"); otherwise a pseudo
                     grid of the same size centred on NOMINAL_LAT/LNG
                     (georeferenced=False): infer.area_stats gives correct m²
                     on it, but its position means nothing (no fusion, no
                     georeferenced export). For a scale given as mpp, or
                     assumed, it also holds m_per_px, the exact ground m per
                     px (a zoom-declared screenshot has true grid px)
      valid          HxW bool (False = no data, exclude from stats) or None
      georeferenced  bool
      source_mpp     ground m per upload px; None when the scale is unknown
      scale_source   "georeference" | "mpp" | "zoom" | "assumed"
      scale_note     one line on the scale and the resampling
      warnings       list of user-facing strings
      original_size  [w, h] of the upload's own grid (after EXIF orientation)
      native_size    [w, h] of img
      crs            the source CRS (georeferenced) or None
      source_transform  the source Affine as 6 numbers (georeferenced only)
      format         detected format, e.g. "PNG" or "GTiff"
    Raises UploadError. Size limits are checked from the header, before any
    pixel is decoded.
    """
    max_bytes = int(MAX_UPLOAD_MB * 2**20) if max_bytes is None else int(max_bytes)
    limit = (MAX_NATIVE_MPX if max_native_mpx is None else float(max_native_mpx)) * 1e6
    if not raw:
        raise UploadError("unreadable", "the upload is empty")
    if len(raw) > max_bytes:
        raise UploadError("too_large", f"the upload is {len(raw) / 2**20:.1f} MB; "
                          f"the limit is {max_bytes / 2**20:.0f} MB")
    raw = raw if isinstance(raw, bytes) else bytes(raw)
    scale = (mpp, zoom, dpr)
    head = raw[:32]
    # TIFF and JPEG 2000 go through GDAL: any band count / dtype, georeference
    if head[:4] in (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"):
        return _load_raster(raw, "tiff", scale, limit)
    if head[:12] == b"\0\0\0\x0cjP  \r\n\x87\n" or head[:4] == b"\xffO\xffQ":
        return _load_raster(raw, "jp2", scale, limit)
    # 16-bit RGB / RGBA / grey+alpha PNG (IHDR bit depth, colour type): Pillow
    # keeps only the high byte of each sample, which turns 12-bit data black
    if (head[:8] == PNG_SIGNATURE and head[12:16] == b"IHDR"
            and head[24:26] in (b"\x10\x02", b"\x10\x04", b"\x10\x06")):
        return _load_raster(raw, "png", scale, limit)
    return _load_picture(raw, filename, scale, limit)


def to_source_grid(classmap, upload):
    """Map a classmap on upload["img"]'s grid back onto the upload's own pixel
    grid (nearest), for display over, or export alongside, the original file.

    Invalid pixels (upload["valid"] False, or outside the reprojected area of a
    georeferenced raster) become NODATA. Returns HxW uint8 with
    (W, H) == upload["original_size"].
    """
    cm = np.asarray(classmap, dtype=np.uint8)
    if [cm.shape[1], cm.shape[0]] != list(upload["native_size"]):
        raise ValueError(f"classmap is {cm.shape[1]}x{cm.shape[0]} px, the upload's "
                         f"native image {upload['native_size']}")
    if upload.get("valid") is not None:
        cm = np.where(upload["valid"], cm, NODATA).astype(np.uint8)
    w, h = upload["original_size"]
    if not upload["georeferenced"]:
        if (w, h) == (cm.shape[1], cm.shape[0]):
            return cm.copy()
        return np.array(Image.fromarray(cm).resize((w, h), Image.NEAREST))
    if w * h > MAX_DECODE_MPX * 1e6:  # e.g. a 3 cm ortho: export the native grid
        raise UploadError("too_large", f"the upload's own grid has {w * h / 1e6:.0f} Mpx; "
                          "export the native grid (meta, grid_transform) instead")
    out = np.full((h, w), NODATA, np.uint8)
    reproject(cm, out, src_transform=grid_transform(upload["meta"]), src_crs="EPSG:3857",
              src_nodata=NODATA, dst_transform=Affine(*upload["source_transform"]),
              dst_crs=upload["crs"], dst_nodata=NODATA, resampling=Resampling.nearest,
              num_threads=GDAL_THREADS)
    return out


# ---- plain images (Pillow) -------------------------------------------------

def _load_picture(raw, filename, scale, limit):
    warns = []
    Image.init()
    try:
        with _quiet():  # Pillow's bomb warning: our own limit is checked below
            im = Image.open(io.BytesIO(raw), formats=[f for f in PIL_FORMATS if f in Image.OPEN])
    except Image.DecompressionBombError:
        raise UploadError("too_large", "the image has more than "
                          f"{2 * Image.MAX_IMAGE_PIXELS / 1e6:.0f} Mpx: crop or downscale "
                          "it (and give its new scale), or upload it as a tiled GeoTIFF"
                          ) from None
    except (OSError, ValueError) as ex:
        ext = os.path.splitext(filename or "")[1].lower()
        if (raw[4:8] == b"ftyp" and raw[8:12] in HEIF_BRANDS) or ext in (".heic", ".heif"):
            raise UploadError("unsupported", "HEIC/HEIF images are not supported: export "
                              "the photo as JPEG or PNG") from ex
        raise UploadError("unreadable", "not a readable image (PNG, JPEG, WebP, TIFF/"
                          "GeoTIFF or JPEG 2000)") from ex
    fmt = im.format
    _check_source(*im.size, MAX_DECODE_MPX)
    src_mpp, how, note = _plain_scale(*scale, warns)
    _native_dims(*im.size, src_mpp, how, limit, [])  # area checks: orientation-free
    if getattr(im, "n_frames", 1) > 1:
        warns.append("the file has several frames: only the first is used")
    try:
        with _quiet():
            im.load()
            # after load(): a PNG may carry its eXIf chunk behind the pixels
            ImageOps.exif_transpose(im, in_place=True)
    except Exception as ex:  # noqa: BLE001 - any decoder failure means a bad file
        raise UploadError("unreadable", "the image data is corrupt or truncated "
                          f"({type(ex).__name__})") from ex
    try:
        im, stretched = _to_8bit(im)
    except ValueError as ex:
        raise UploadError("unsupported", f"colour mode {im.mode} is not supported") from ex
    if stretched is not None:
        warns.append(_stretch_note(stretched))
    w, h = im.size
    wn, hn = _native_dims(w, h, src_mpp, how, limit, warns)
    if (wn, hn) != (w, h):
        shrink = wn < w
        # Pillow resizes RGBA premultiplied: transparent colours do not bleed in
        im = im.resize((wn, hn), Image.LANCZOS if shrink else Image.BICUBIC,
                       reducing_gap=3.0 if shrink else None)
    arr = np.asarray(im)
    valid = arr[..., 3] >= 128 if arr.shape[2] == 4 else None
    return _finish_plain(arr[..., :3], valid, w, h, src_mpp, how, note, warns, fmt)


def _to_8bit(im):
    """Pillow image in any mode -> ("RGB" or "RGBA" image, stretched dtype or None)."""
    stretched = None
    if im.mode in _U16_MODES:  # convert("RGB") would clip these to white
        a = np.asarray(im)
        ok = np.isfinite(a) if a.dtype.kind == "f" else None
        im = Image.fromarray(_stretch_band(a, *_percentiles(a, ok)))
        stretched = a.dtype
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        if im.mode != "RGBA":  # convert() to the same mode would copy
            im = im.convert("RGBA")
        if im.getchannel("A").getextrema()[0] == 255:  # opaque after all
            im = im.convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    return im, stretched


# ---- TIFF / JPEG 2000 / 16-bit PNG (rasterio) -------------------------------

def _load_raster(raw, kind, scale, limit):
    jp2 = None
    if kind == "jp2":
        jp2 = _check_jp2(raw, limit)  # GDAL's own header parse is the costly part
    elif kind == "png":
        _check_png16(raw)
    # GDAL's block cache is process-wide (default 5 % of RAM): keep it small
    # while a large raster streams through it
    with _quiet(), rasterio.Env(GDAL_NUM_THREADS=GDAL_THREADS, GDAL_CACHEMAX=GDAL_CACHE_MB << 20), \
            MemoryFile(raw, ext="." + {"tiff": "tif"}.get(kind, kind)) as mf:
        try:
            src = mf.open(driver=RASTER_DRIVERS[kind])
        except Exception as ex:  # noqa: BLE001
            raise UploadError("unreadable", f"not a readable {RASTER_NAMES[kind]} file") from ex
        with src:
            # header only; a PNG is decoded row by row from the top (whole
            # when interlaced), like the formats Pillow decodes
            _check_source(src.width, src.height,
                          MAX_DECODE_MPX if kind == "png" else MAX_SOURCE_MPX)
            _check_raster(src)
            try:
                if src.crs is not None and not src.transform.is_identity:
                    return _read_geo(src, scale, limit, jp2)
                # GDAL ignores the TIFF orientation tag and the PNG eXIf chunk
                o = {"tiff": _tiff_orientation, "png": _png_orientation}.get(
                    kind, lambda _: 1)(raw)
                return _read_plain_raster(src, o, scale, limit, jp2)
            except UploadError:
                raise
            except CRSError as ex:
                raise UploadError("unsupported", "the file's CRS cannot be placed on the "
                                  "map: reproject it, e.g. gdalwarp -t_srs EPSG:3857") from ex
            except Exception as ex:
                if type(ex).__module__.split(".")[0] != "rasterio":
                    raise  # a bug, not a bad file
                raise UploadError("unreadable", "the raster data is corrupt or truncated "
                                  f"({type(ex).__name__})") from ex


def _check_jp2(raw, limit):
    """Header-only limits of a JPEG 2000 file (see MAX_JP2_*); limit: native px."""
    hdr = _jp2_header(raw)
    if hdr is None:
        raise UploadError("unreadable", "not a readable JPEG 2000 file")
    w, h, tw, th, tiles, bands, levels = hdr
    _check_source(w, h, MAX_SOURCE_MPX)
    resave = ("re-save it in 1024 px tiles, e.g. gdal_translate -of JP2OpenJPEG -co "
              "BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 in.jp2 out.jp2, or as a tiled GeoTIFF")
    if tiles * bands > MAX_JP2_TILE_BANDS:
        raise UploadError("unsupported", f"the JPEG 2000 file holds {tiles} tiles of {bands} "
                          "band(s), too many to decode safely: " + resave)
    if tw * th > MAX_JP2_TILE_MPX * 1e6:
        raise UploadError("unsupported", f"the JPEG 2000 file stores its pixels in {tw}x{th} "
                          "px tiles, too large to decode efficiently: " + resave)
    # a decimated read decodes the finest resolution level that is not
    # coarser than needed: with enough levels that is < 4x the native px
    # (~25 Mpx/s), without them every px of the file
    if w * h / 4 ** levels > 4 * limit:
        raise UploadError("too_large", f"the JPEG 2000 file has {w}x{h} px but only "
                          f"{levels + 1} resolution level(s), so reading it would decode "
                          f"{w * h / 4 ** levels / 1e6:.0f} Mpx: re-save it with more levels "
                          "(e.g. gdal_translate -of JP2OpenJPEG -co RESOLUTIONS=6 in.jp2 "
                          "out.jp2) or as a tiled GeoTIFF with overviews")
    return hdr


def _jp2_read_env(jp2, d, itemsize):
    """GDAL settings for reading a JPEG 2000 at d source px per px read (a
    no-op context for other formats): a block cache that holds one full row
    of decoded tiles at the resolution level the read uses, i.e. the finest
    reduced level not coarser than d. Raises UploadError when that row is
    over MAX_JP2_CACHE_MB. jp2: the _jp2_header tuple or None."""
    if jp2 is None:
        return nullcontext()
    w, _, tw, th, _, bands, levels = jp2
    f = 2 ** min(levels, max(0, math.floor(math.log2(max(d, 1.0)) + 1e-9)))
    row = -(-w // tw) * -(-tw // f) * -(-th // f) * bands * itemsize
    if row > MAX_JP2_CACHE_MB * 2**20:
        raise UploadError("unsupported", f"the JPEG 2000 file stores its pixels in {tw}x{th} "
                          f"px tiles with {levels + 1} resolution level(s): reading it needs "
                          f"{row / 2**20:.0f} MB of decoded tiles at once. Re-save it in 1024 "
                          "px tiles, e.g. gdal_translate -of JP2OpenJPEG -co BLOCKXSIZE=1024 "
                          "-co BLOCKYSIZE=1024 in.jp2 out.jp2, or as a tiled GeoTIFF")
    return rasterio.Env(GDAL_CACHEMAX=max(GDAL_CACHE_MB << 20, int(row * 1.25)))


def _check_png16(raw):
    """GDAL decodes an interlaced PNG in 100 MB chunks, each one a full decode
    of the file (6000x6000 RGB16: 11 s vs 3.7 s plain): one chunk at most."""
    w, h = struct.unpack_from(">II", raw, 16)
    if raw[28:29] == b"\x01" and w * h * 2 * {2: 3, 4: 2, 6: 4}[raw[25]] > 1e8:
        raise UploadError("unsupported", f"the PNG is interlaced with {w}x{h} 16-bit px, too "
                          "slow to decode: save it without interlacing, or as a GeoTIFF")


def _jp2_header(raw):
    """(w, h, largest tile w, h, tiles, bands, decomposition levels) from the
    SIZ and COD markers of a JPEG 2000 file (JP2 boxes or a bare code-stream),
    without decoding anything; None when they are missing or inconsistent."""
    i = 0
    if raw[:4] != b"\xffO\xffQ":  # JP2: walk the top-level boxes to the code-stream
        p = 0
        while True:
            if p + 8 > len(raw):
                return None
            n, box = struct.unpack_from(">I4s", raw, p)
            head = 8
            if n == 1 and p + 16 <= len(raw):  # 64-bit box length
                n, head = struct.unpack_from(">Q", raw, p + 8)[0], 16
            if box == b"jp2c":
                i = p + head
                break
            if n < head:  # 0: the box runs to the end of the file
                return None
            p += n
    if raw[i:i + 4] != b"\xffO\xffQ" or len(raw) < i + 42:
        return None
    lsiz, _, X, Y, XO, YO, XT, YT, XTO, YTO, C = struct.unpack_from(">HHIIIIIIIIH", raw, i + 4)
    if not (X > XO and Y > YO and XT and YT and XTO <= XO and YTO <= YO):
        return None
    levels, j = 0, i + 4 + lsiz  # no COD found: assume none (the costliest)
    while j + 10 <= len(raw):  # the main header's markers, up to the first tile
        m, n = struct.unpack_from(">HH", raw, j)
        if m == 0xFF52:  # COD: Lcod, Scod, SGcod (4 B), then the level count
            levels = raw[j + 9]
            break
        if m == 0xFF90 or n < 2:  # SOT (first tile-part), or corrupt
            break
        j += 2 + n
    tiles = -(-(X - XTO) // XT) * -(-(Y - YTO) // YT)
    return X - XO, Y - YO, min(XT, X - XO), min(YT, Y - YO), tiles, C, levels


def _check_raster(src):
    if any(np.dtype(d).kind == "c" for d in src.dtypes):
        raise UploadError("unsupported", "complex-valued rasters are not imagery")
    bh, bw = src.block_shapes[0]
    per_px = src.count if src.interleaving == Interleaving.pixel else 1
    if bh * bw * per_px * np.dtype(src.dtypes[0]).itemsize > MAX_BLOCK_MB * 2**20:
        raise UploadError("unsupported", f"the file stores its pixels in {bw}x{bh} px "
                          "blocks, too large to decode safely: re-save it tiled, e.g. "
                          "gdal_translate -co TILED=YES -co COMPRESS=DEFLATE in.tif out.tif")


def _read_geo(src, scale, limit, jp2=None):
    warns = []
    if any(v is not None for v in scale):
        warns.append("mpp / zoom / dpr ignored: the file is georeferenced")
    rgb, mask_band, lut = _bands(src, warns)
    zoom, s = config.ZOOM, config.SCALE
    res = MERC / imagery.WORLD(zoom) / s  # output px size in Mercator m
    # output grid = the raster's Mercator bounding box from its NW corner, so a
    # 3857 raster already at this resolution is copied pixel for pixel
    W, H = src.width, src.height
    xs, ys = src.transform @ (np.array([0, W, 0, W], float), np.array([0, 0, H, H], float))
    try:
        L, B, R, T = transform_bounds(src.crs, "EPSG:3857", xs.min(), ys.min(),
                                      xs.max(), ys.max(), densify_pts=21)
        # Mercator length of one source px along each source axis, at the centre
        c, r = W / 2, H / 2
        px, py = src.transform @ (np.array([c, c + 1, c]), np.array([r, r, r + 1]))
        X, Y = (np.asarray(v) for v in warp_transform(src.crs, "EPSG:3857", px, py))
    except Exception as ex:  # noqa: BLE001 - PROJ/GDAL refuse this CRS
        raise CRSError(str(ex)) from ex
    if not np.isfinite([L, B, R, T, *X, *Y]).all() or max(abs(B), abs(T)) > MERC / 2:
        raise UploadError("unsupported", "the raster lies outside the Web-Mercator range")
    # Mercator m per source px along its column / row axes (the columns of M)
    M = np.array([[X[1] - X[0], X[2] - X[0]], [Y[1] - Y[0], Y[2] - Y[0]]])
    step_x, step_y = np.hypot(*M)
    # |det| / (step_x * step_y) is the sine of the angle between the pixel
    # axes: 1 for north-up and rotated rasters, 0 for a singular transform
    if not (min(step_x, step_y) > 0 and abs(np.linalg.det(M)) >= 0.5 * step_x * step_y):
        raise UploadError("unsupported", "the raster's geotransform is degenerate (its pixel "
                          "axes are parallel or nearly so)")
    rx, ry = res / step_x, res / step_y  # source px per output px, per source axis
    lat_c = imagery.world_to_latlng(*_merc_to_world(X[0], Y[0], zoom), zoom)[0]
    n_mpp = native_mpp(lat_c)
    src_mpp = n_mpp / math.sqrt(rx * ry)
    _check_coarse(math.sqrt(rx * ry), src_mpp, warns)
    x0, y0 = _merc_to_world(L, T, zoom)
    x1, y1 = _merc_to_world(R, B, zoom)
    wd, hd = math.ceil((x1 - x0) * s - 1e-6), math.ceil((y1 - y0) * s - 1e-6)
    _check_extent(wd, hd, n_mpp, limit, False, "raster's bounding box")

    # integer box-average decimation to just finer than the target bounds the
    # read (GDAL uses overviews when the file has them); the warp then does
    # the exact fractional step
    dec = (max(1, math.floor(rx + 1e-9)), max(1, math.floor(ry + 1e-9)))
    read_env = _jp2_read_env(jp2, min(dec), np.dtype(src.dtypes[0]).itemsize)
    k = math.sqrt(rx / dec[0] * ry / dec[1])
    warp_rs = (Resampling.average if k > 1.05 else
               Resampling.bilinear if k >= 0.95 else Resampling.cubic)
    uniq = sorted(set(rgb))
    params = None
    if lut is None and src.dtypes[uniq[0] - 1] != "uint8":
        params = _stretch_params(src, uniq)
        warns.append(_stretch_note(src.dtypes[uniq[0] - 1]))
    dst_tr = Affine(res, 0.0, L, 0.0, -res, T)
    # output px -> decimated source px (rotation, scale), to plan the pieces
    jac = np.diag([1 / dec[0], 1 / dec[1]]) @ np.linalg.inv(M) @ np.diag([res, -res])
    with read_env:
        img, cover = _warp(src, uniq, [uniq.index(b) for b in rgb], mask_band, lut, params,
                           dst_tr, hd, wd, dec, warp_rs, jac)
    valid = cover == 255  # strict: partly covered edge px count as nodata
    if not valid.any():
        raise UploadError("unreadable", "the raster has no valid pixels "
                          "(everything is nodata or transparent)")
    # crop to the valid footprint by whole px, which keeps the grid lattice
    rows, cols = np.flatnonzero(valid.any(1)), np.flatnonzero(valid.any(0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    img, valid = img[r0:r1, c0:c1], valid[r0:r1, c0:c1]
    x0, y0 = x0 + c0 / s, y0 + r0 / s
    hn, wn = valid.shape
    _check_extent(wn, hn, n_mpp, limit, False, "raster's valid area")
    lat, lng = imagery.world_to_latlng(x0 + wn / s / 2, y0 + hn / s / 2, zoom)
    meta = {"zoom": zoom, "scale": s, "x0": x0, "y0": y0,
            "width_map_px": wn / s, "height_map_px": hn / s, "center": [lat, lng],
            "nx": None, "ny": None, "georeferenced": True, "crs": "EPSG:3857"}
    w_, s_, e_, n_ = config.HYD_BBOX
    if not (s_ <= lat <= n_ and w_ <= lng <= e_):
        warns.append("the raster is outside greater Hyderabad, where the model was "
                     "trained; expect lower accuracy")
    img, valid = _finish_valid(img, valid, warns)
    crs = src.crs.to_string()
    label = crs if len(crs) <= 24 else "custom CRS"  # WKT is not for a one-liner
    return {"img": img, "meta": meta, "valid": valid, "georeferenced": True,
            "source_mpp": src_mpp, "scale_source": "georeference",
            "scale_note": f"georeferenced ({label}, {src_mpp:.3f} m/px): reprojected to "
                          f"the zoom-18 Web-Mercator grid at {n_mpp:.3f} m/px",
            "warnings": warns, "original_size": [W, H], "native_size": [wn, hn],
            "crs": crs, "source_transform": list(src.transform)[:6], "format": src.driver}


def _warp(src, uniq, pos, mask_band, lut, params, dst_tr, hd, wd, dec, warp_rs, jac):
    """Reproject the source onto the dst_tr grid piece by piece (_warp_pieces),
    so memory stays O(STRIP_PX) whatever the source's size or rotation.
    -> (img HxWx3 uint8, coverage HxW 0..255)."""
    img = np.zeros((hd, wd, 3), np.uint8)
    cover = np.zeros((hd, wd), np.uint8)
    inv = ~src.transform
    dx, dy = dec
    # palette indexes cannot be averaged
    read_rs = Resampling.nearest if lut is not None else Resampling.average
    t = np.linspace(0.0, 1.0, 33)
    todo = _warp_pieces(hd, wd, jac)[::-1]
    while todo:
        r0, c0, h, w = todo.pop()
        # the piece's outline, densified (rotation, curvature) -> source px
        X, Y = dst_tr @ (c0 + np.concatenate([t * w, t * w, np.zeros(33), np.full(33, w)]),
                         r0 + np.concatenate([np.zeros(33), np.full(33, h), t * h, t * h]))
        sx, sy = warp_transform("EPSG:3857", src.crs, X, Y)
        cc, rr = inv @ (np.asarray(sx), np.asarray(sy))
        ok = np.isfinite(cc) & np.isfinite(rr)
        if not ok.any():
            continue
        # window on the decimation lattice, padded for the warp kernels
        sc0 = max(0, math.floor((cc[ok].min() - 3 * dx - 2) / dx) * dx)
        sc1 = min(src.width, math.ceil((cc[ok].max() + 3 * dx + 2) / dx) * dx)
        sr0 = max(0, math.floor((rr[ok].min() - 3 * dy - 2) / dy) * dy)
        sr1 = min(src.height, math.ceil((rr[ok].max() + 3 * dy + 2) / dy) * dy)
        if sc1 <= sc0 or sr1 <= sr0:
            continue
        ow, oh = -(-(sc1 - sc0) // dx), -(-(sr1 - sr0) // dy)
        if ow * oh > 4 * STRIP_PX and max(h, w) > 64:  # the plan was off (curved CRS): halve
            todo += ([(r0, c0 + w // 2, h, w - w // 2), (r0, c0, h, w // 2)] if w >= h else
                     [(r0 + h // 2, c0, h - h // 2, w), (r0, c0, h // 2, w)])
            continue
        win = Window(sc0, sr0, sc1 - sc0, sr1 - sr0)
        srcarr = _read_rgba(src, uniq, pos, mask_band, lut, params, win, (oh, ow), read_rs)
        w_tr = src.transform @ Affine.translation(sc0, sr0) @ Affine.scale(
            (sc1 - sc0) / ow, (sr1 - sr0) / oh)
        out = np.zeros((4, h, w), np.uint8)
        # alpha-weighted: colours of partly valid px come from valid sources only
        reproject(srcarr, out, src_transform=w_tr, src_crs=src.crs,
                  dst_transform=dst_tr @ Affine.translation(c0, r0), dst_crs="EPSG:3857",
                  resampling=warp_rs, src_alpha=4, dst_alpha=4, num_threads=GDAL_THREADS)
        img[r0:r0 + h, c0:c0 + w] = out[:3].transpose(1, 2, 0)
        cover[r0:r0 + h, c0:c0 + w] = out[3]
    return img, cover


def _warp_pieces(hd, wd, jac):
    """Output pieces (row, col, h, w) whose decimated source windows hold about
    STRIP_PX px; jac maps output px to decimated source px. Full-width strips
    while the source is north-up, so each source block is decoded once;
    square tiles when it is rotated: a full-width strip then maps onto a
    diagonal band whose bounding window is most of the raster (at 45 deg an
    81 Mpx ortho read 81 Mpx per strip, 5.3 GB peak)."""
    a = np.abs(jac)
    step = max(64, int(STRIP_PX / (wd * 4)))  # decimated reads are < 2x finer per axis
    if (a[0, 0] * wd + a[0, 1] * step) * (a[1, 0] * wd + a[1, 1] * step) <= 2 * STRIP_PX:
        return [(r, 0, min(step, hd - r), wd) for r in range(0, hd, step)]
    t = max(64, int(math.sqrt(STRIP_PX / ((a[0, 0] + a[0, 1]) * (a[1, 0] + a[1, 1])))))
    return [(r, c, min(t, hd - r), min(t, wd - c))
            for r in range(0, hd, t) for c in range(0, wd, t)]


def _read_plain_raster(src, orientation, scale, limit, jp2=None):
    warns = []
    if not src.transform.is_identity or src.gcps[0]:
        warns.append("the file has a geotransform or control points but no usable CRS: "
                     "it is treated as a plain image")
    rgb, mask_band, lut = _bands(src, warns)
    src_mpp, how, note = _plain_scale(*scale, warns)
    w, h = src.width, src.height
    wn, hn = _native_dims(w, h, src_mpp, how, limit, warns)
    # shrink straight to the native size with GDAL's area average (memory is
    # bounded by the output); enlarge from the full-resolution read
    shape = (hn, wn) if wn < w else (h, w)
    read_env = _jp2_read_env(jp2, min(w / shape[1], h / shape[0]),
                             np.dtype(src.dtypes[0]).itemsize)
    rs = Resampling.average if wn < w and lut is None else Resampling.nearest
    uniq = sorted(set(rgb))
    params = None
    if lut is None and src.dtypes[uniq[0] - 1] != "uint8":
        params = _stretch_params(src, uniq)
        warns.append(_stretch_note(src.dtypes[uniq[0] - 1]))
    with read_env:
        rgba = _read_rgba(src, uniq, [uniq.index(b) for b in rgb], mask_band, lut, params,
                          None, shape, rs)
    if shape != (hn, wn):
        im = Image.fromarray(np.ascontiguousarray(rgba.transpose(1, 2, 0)))
        rgba = np.asarray(im.resize((wn, hn), Image.BICUBIC)).transpose(2, 0, 1)
    arr, valid = rgba[:3].transpose(1, 2, 0), rgba[3] >= 128
    if orientation in _ORIENT:
        arr, valid = _ORIENT[orientation](arr), _ORIENT[orientation](valid)
        if orientation >= 5:
            w, h = h, w
    return _finish_plain(arr, valid, w, h, src_mpp, how, note, warns, src.driver)


def _bands(src, warns):
    """-> (1-based source bands for R, G, B; 8-bit alpha band to read with them,
    or None to use GDAL's dataset mask (nodata, internal masks, other alphas);
    palette LUT Nx4 or None)."""
    ci = list(src.colorinterp)
    if (src.count == 1 and ci[0] == ColorInterp.palette
            and np.dtype(src.dtypes[0]).kind in "ui"):
        cmap = src.colormap(1)
        # a uint16 palette (GTiff) has up to 65536 entries
        lut = np.zeros((max(256, max(cmap, default=0) + 1), 4), np.uint8)
        lut[:, 3] = 255
        for i, rgba in cmap.items():
            if i >= 0:
                lut[i] = rgba
        return [1, 1, 1], None, lut
    rgb_ci = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    has_alpha = ColorInterp.alpha in ci
    if all(c in ci for c in rgb_ci):
        rgb = [ci.index(c) + 1 for c in rgb_ci]
    elif src.count - has_alpha >= 3:
        rgb = [1, 2, 3]
        warns.append("the bands have no colour interpretation: bands 1-3 are used as "
                     "red, green, blue")
    else:
        rgb = [1, 1, 1]
        warns.append("single-band image: shown to the model as grey (it was trained "
                     "on colour imagery)")
    # an 8-bit alpha averages like GDAL's mask of it; reading it with the
    # colour bands decodes each (pixel-interleaved) block once instead of twice
    mask_band = None
    if has_alpha and src.dtypes[ci.index(ColorInterp.alpha)] == "uint8":
        mask_band = ci.index(ColorInterp.alpha) + 1
    elif not has_alpha and src.count in (2, 4) and src.dtypes[-1] == "uint8":
        # RGBA / grey+alpha written without the alpha flag (common): a last
        # band holding only 0/255 is a mask, not data. A central window keeps
        # this cheap (a decimated read would decode every block)
        last = src.read(src.count, window=_central_window(src, 1024))
        if np.isin(last, (0, 255)).mean() > 0.999 and (last == 255).any():
            mask_band = src.count
            warns.append(f"band {src.count} has no alpha flag but holds only 0/255: "
                         "used as transparency")
    alpha_bands = has_alpha or mask_band is not None
    if src.count > len(set(rgb)) + alpha_bands and src.count >= 3:
        warns.append(f"{src.count} bands: only bands {sorted(set(rgb))} are used as colour")
    return rgb, mask_band, None


def _read_rgba(src, uniq, pos, mask_band, lut, params, window, shape, rs):
    """Read a window at out shape `shape` -> 4xHxW uint8 R, G, B, validity."""
    idx = uniq + ([mask_band] if mask_band is not None else [])
    data = src.read(idx, window=window, out_shape=(len(idx),) + shape, resampling=rs)
    if mask_band is not None:
        return _rgba(data[:-1], pos, params, lut, data[-1])
    return _rgba(data, pos, params, lut,
                 src.dataset_mask(window=window, out_shape=shape, resampling=rs))


def _rgba(data, pos, params, lut, mask):
    """Band data (source dtype) -> 4xHxW uint8: R, G, B, validity 0..255."""
    out = np.empty((4,) + mask.shape, np.uint8)
    if lut is not None:
        q = np.take(lut, data[0], axis=0, mode="clip")  # an index past the palette: its end
        out[:3] = q[..., :3].transpose(2, 0, 1)
        out[3] = np.minimum(mask, q[..., 3])
        return out
    for i, p in enumerate(pos):
        out[i] = data[p] if params is None else _stretch_band(data[p], *params[p])
    out[3] = mask
    if data.dtype.kind == "f":
        out[3][~np.isfinite(data).all(0)] = 0
    return out


def _stretch_params(src, uniq):
    """Per-band 2-98 % stretch limits over the valid px of 4 x 4 windows spread
    over the raster: ~1 M samples, while a decimated preview would decode
    every block of a file without overviews."""
    def valid_values(data, ok):
        return [b[ok & np.isfinite(b) if b.dtype.kind == "f" else ok] for b in data]

    vals = [[] for _ in uniq]
    for fy in (0.125, 0.375, 0.625, 0.875):
        for fx in (0.125, 0.375, 0.625, 0.875):
            win = _central_window(src, 256, fx, fy)
            for v, b in zip(vals, valid_values(src.read(uniq, window=win),
                                               src.dataset_mask(window=win) > 0)):
                v.append(b)
    if sum(map(len, vals[0])) < 1000:  # mostly nodata there: sample the whole raster
        f = min(1.0, math.sqrt((1 << 20) / (src.width * src.height)))
        shape = (max(1, round(src.height * f)), max(1, round(src.width * f)))
        data = src.read(uniq, out_shape=(len(uniq),) + shape, resampling=Resampling.nearest)
        ok = src.dataset_mask(out_shape=shape, resampling=Resampling.nearest) > 0
        vals = [[b] for b in valid_values(data, ok)]
    return [_percentiles(np.concatenate(v)) for v in vals]


def _central_window(src, n, fx=0.5, fy=0.5):
    """An n x n (or smaller) window centred at (fx, fy) of the raster's extent."""
    w, h = min(n, src.width), min(n, src.height)
    c = min(max(0, round(fx * src.width) - w // 2), src.width - w)
    r = min(max(0, round(fy * src.height) - h // 2), src.height - h)
    return Window(c, r, w, h)


def _percentiles(a, ok=None):
    v = a[ok] if ok is not None else a.ravel()
    if v.size > 1 << 20:  # a strided 1 M sample gives the same limits
        v = v[::v.size // (1 << 20)]
    lo, hi = (float(x) for x in np.percentile(v, (2, 98))) if v.size else (0.0, 1.0)
    return lo, hi if hi > lo else lo + 1.0


def _stretch_band(b, lo, hi):
    x = (b.astype(np.float32) - lo) * np.float32(255.0 / (hi - lo)) + np.float32(0.5)
    np.clip(x, 0, 255, out=x)
    return np.nan_to_num(x, copy=False, nan=0.0).astype(np.uint8)


def _stretch_note(dtype):
    return (f"{np.dtype(dtype).name} imagery stretched to 8 bit (2-98 % per band); "
            "colours will differ from Google imagery")


def _tiff_orientation(raw):
    """Orientation tag of a TIFF's first IFD (GDAL ignores it); 1 if absent."""
    try:
        fp = io.BytesIO(raw)
        ifd = TiffImagePlugin.ImageFileDirectory_v2(fp.read(16 if 43 in raw[2:4] else 8))
        fp.seek(ifd.next)
        ifd.load(fp)
        return int(ifd.get(0x0112, 1))
    except Exception:  # noqa: BLE001 - best effort
        return 1


def _png_orientation(raw):
    """EXIF orientation of a PNG (its eXIf chunk holds a bare TIFF IFD); 1 if absent."""
    p = 8
    while p + 8 <= len(raw):
        n, chunk = struct.unpack_from(">I4s", raw, p)
        if chunk == b"eXIf":
            return _tiff_orientation(raw[p + 8:p + 8 + n])
        if chunk == b"IEND":
            break
        p += 12 + n  # length, type, data, CRC
    return 1


# ---- shared ----------------------------------------------------------------

def _plain_scale(mpp, zoom, dpr, warns):
    """Ground m per px of a plain image -> (mpp or None, scale_source, note).

    Google Maps on the web draws the world 256 * 2**zoom CSS px wide, so a CSS
    px covers imagery.meters_per_map_px(lat, zoom) = 156543.03 * cos(lat) /
    2**zoom m, and a screenshot holds dpr device px per CSS px:

        mpp = meters_per_map_px(17.4, zoom) / dpr

    dpr is window.devicePixelRatio: 2 on Retina / HiDPI screens, multiplied by
    the OS display scaling (1.25 / 1.5 at 125 / 150 %) and by the browser
    zoom. Zoom 18 on a 2x screen is the native 0.285 m/px; zoom 18 on a 1x
    screen, or zoom 17 on a 2x one, is 0.57 m/px. As in `dpr or 1`, a dpr
    of 0 or None means 1.
    """
    if mpp is not None:
        mpp = _number(mpp, "mpp (m per pixel)", 1e-4, 1e3)
        if zoom is not None or dpr:
            warns.append("zoom / dpr ignored: mpp was given")
        return mpp, "mpp", f"{mpp:.3f} m/px as given"
    if zoom is not None:
        z = _number(zoom, "zoom", 1, 23)
        d = _number(dpr, "dpr (device pixel ratio)", 0.25, 8) if dpr else 1.0
        m = imagery.meters_per_map_px(NOMINAL_LAT, z) / d
        return m, "zoom", (f"Google Maps zoom {z:g} at device pixel ratio {d:g}: "
                           f"{m:.3f} m/px at latitude {NOMINAL_LAT:g}")
    if dpr:
        warns.append("dpr ignored: it only applies together with zoom")
    warns.append(SCALE_UNKNOWN)
    return None, "assumed", SCALE_UNKNOWN


def _number(v, name, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = math.nan
    if not lo <= x <= hi:  # also rejects nan
        raise UploadError("unsupported", f"{name} must be a number from {lo:g} to {hi:g}",
                          param=name.split()[0])
    return x


def _native_dims(w, h, src_mpp, how, limit, warns):
    """Size at the native scale of a w x h plain image; rejects bad scales/sizes."""
    n = native_mpp()
    wn, hn = w, h
    if src_mpp is not None:
        _check_coarse(n / src_mpp, src_mpp, warns)
        wn, hn = max(1, round(w * src_mpp / n)), max(1, round(h * src_mpp / n))
    _check_extent(wn, hn, n, limit, how == "assumed", "image")
    return wn, hn


def _check_source(w, h, cap_mpx):
    if min(w, h) < MIN_SIDE:
        raise UploadError("too_small", f"the image is {w}x{h} px; at least {MIN_SIDE} px "
                          "per side are needed")
    if w * h > cap_mpx * 1e6:
        raise UploadError("too_large", f"the image is {w}x{h} px ({w * h / 1e6:.0f} Mpx), "
                          f"more than the {cap_mpx:.0f} Mpx that can be decoded: crop or "
                          "downscale it (and give its new scale), or upload it as a "
                          "tiled GeoTIFF")


def _check_coarse(s, src_mpp, warns):
    if s < COARSE_LIMIT:
        raise UploadError("unsupported", f"{src_mpp:.2f} m/px is too coarse for building "
                          f"detection: the model needs <= {native_mpp() / COARSE_LIMIT:.1f} "
                          f"m/px, ideally {native_mpp():.3f}")
    if s < COARSE_WARN:
        warns.append(f"the image is {1 / s:.1f}x coarser than the {native_mpp():.3f} m/px "
                     "the model was trained on: buildings and their areas are unreliable")


def _check_extent(wn, hn, n_mpp, limit, assumed, what):
    """Native size limits; n_mpp m per native px."""
    if min(wn, hn) < MIN_SIDE:
        raise UploadError("too_small", f"at {n_mpp:.3f} m/px the {what} is only {wn}x{hn} "
                          f"px ({wn * n_mpp:.0f} x {hn * n_mpp:.0f} m); at least {MIN_SIDE} "
                          "px per side are needed" + ("" if assumed else
                                                     " - check the scale given"))
    if wn * hn > limit:
        km2, lim = wn * hn * n_mpp ** 2 / 1e6, limit * n_mpp ** 2 / 1e6
        hint = ("; its scale is unknown and assumed to be native - if its pixels are "
                "finer than 0.285 m, give mpp or zoom" if assumed else "")
        raise UploadError("too_large", f"the {what} covers {km2:.1f} km² ({wn}x{hn} px at "
                          f"{n_mpp:.3f} m/px), more than the {lim:.1f} km² ({limit / 1e6:.0f}"
                          " Mpx) one run can analyse: crop it to the area of interest or "
                          f"split it into parts{hint}")


def _finish_plain(arr, valid, w, h, src_mpp, how, note, warns, fmt):
    hn, wn = arr.shape[:2]
    if (wn, hn) != (w, h):
        note = f"{note}; resampled x{wn / w:.2f} to {native_mpp():.3f} m/px"
    img, valid = _finish_valid(arr, valid, warns)
    # ground m per native px: exact for a declared mpp (the native size was
    # rounded); a zoom-declared screenshot has true grid px instead
    m_px = (None if how == "zoom" else native_mpp() if src_mpp is None else
            src_mpp * math.sqrt(w * h / (wn * hn)))
    return {"img": img, "meta": _pseudo_meta(wn, hn, m_px), "valid": valid,
            "georeferenced": False, "source_mpp": src_mpp, "scale_source": how,
            "scale_note": note, "warnings": warns, "original_size": [w, h],
            "native_size": [wn, hn], "crs": None, "format": fmt}


def _finish_valid(arr, valid, warns):
    """-> (writable contiguous img with invalid px filled, valid or None)."""
    if valid is None or valid.all():
        return np.array(arr, dtype=np.uint8, order="C"), None
    if not valid.any():
        raise UploadError("unreadable", "the image has no valid pixels (everything is "
                          "transparent or nodata)")
    valid = np.ascontiguousarray(valid)
    frac = 1 - valid.mean()
    if frac > 0.01:
        warns.append(f"{frac:.0%} of the image has no data (transparent / nodata) and "
                     "is excluded from areas")
    return _fill_invalid(arr, valid), valid


def _fill_invalid(img, valid):
    """Model input under nodata, in two levels (valid is not empty). Far field:
    valid colours averaged on a 1/FILL_COARSE grid, GDAL's inverse-distance
    fill there (FILL_PX reach), their mean beyond, bilinear back up. Near
    field: the same fill at full resolution from valid px within NEAR_PX,
    run only on blocks next to valid px, so the cost follows the gap's
    outline, not its area.

    What the model sees in a gap changes its predictions on the valid px next
    to it. On kompally / lbnagar val crops with a ragged collar and a hole,
    predictions 0-16 px from the gap kept the unmasked image's class for
    79 / 80 % of px, vs 79 / 72 % with a single 128 px full-resolution fill,
    79 / 71 % with a 16 px one, 76 / 47 % with the mean colour and 58 / 63 %
    with black (768 px windows). 79 Mpx with 86 % nodata: 2.1 s, vs 37 s for
    the single 128 px fill."""
    out = np.array(img, dtype=np.uint8, order="C")
    H, W = valid.shape
    inv, v8 = ~valid, valid.view(np.uint8)
    f = FILL_COARSE
    rows, cols = np.arange(0, H, f), np.arange(0, W, f)
    cnt = _block_sums(v8, rows, cols)
    cv = cnt > 0
    for b in range(3):
        band = out[..., b]
        c = np.full(cnt.shape, np.uint8(band.mean(where=valid) + 0.5))
        s = _block_sums(np.where(valid, band, 0), rows, cols)
        c[cv] = (s[cv] + cnt[cv] // 2) // cnt[cv]
        c = fillnodata(c, mask=cv.view(np.uint8), max_search_distance=FILL_PX / f,
                       smoothing_iterations=0)
        # coarse px centres sit at f/2 - 0.5 + f*i, as PIL's resize assumes
        up = Image.fromarray(c).resize((c.shape[1] * f, c.shape[0] * f), Image.BILINEAR)
        np.copyto(band, np.asarray(up)[:H, :W], where=inv)
    B, R = FILL_BLOCK, NEAR_PX
    rows, cols = np.arange(0, H, B), np.arange(0, W, B)
    n_valid = _block_sums(v8, rows, cols)
    n_px = np.outer(np.diff(rows, append=H), np.diff(cols, append=W))
    # valid px within R <= B of a block lie in it or in one of its neighbours
    p = np.pad(n_valid > 0, 1)
    near = np.zeros_like(n_valid, bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            near |= p[dy:dy + near.shape[0], dx:dx + near.shape[1]]
    for by, bx in np.argwhere(near & (n_valid < n_px)):
        y0, x0 = by * B, bx * B
        y1, x1 = min(H, y0 + B), min(W, x0 + B)
        py0, px0, py1, px1 = max(0, y0 - R), max(0, x0 - R), min(H, y1 + R), min(W, x1 + R)
        m = np.ascontiguousarray(v8[py0:py1, px0:px1])
        for b in range(3):
            res = fillnodata(np.ascontiguousarray(out[py0:py1, px0:px1, b]), mask=m,
                             max_search_distance=R, smoothing_iterations=0)
            out[y0:y1, x0:x1, b] = res[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
    return out


def _block_sums(a, rows, cols):
    """Sums of a 2-D uint8 array over the blocks starting at rows x cols, for
    blocks up to 257 rows tall: uint16 holds their column sums (a wider dtype
    makes reduceat copy the whole input first: 0.7 GB at 79 Mpx for int64)."""
    return np.add.reduceat(np.add.reduceat(a, rows, axis=0, dtype=np.uint16), cols, axis=1,
                           dtype=np.uint32).astype(np.int64)


def _pseudo_meta(w, h, m_per_px=None):
    """A zoom-18 / scale-2 grid of w x h px centred on the nominal point: no
    real position, but infer.area_stats reads native_mpp() m per px from it,
    or m_per_px when given (the exact ground size of a resampled image's px)."""
    s, z = config.SCALE, config.ZOOM
    cx, cy = imagery.latlng_to_world(NOMINAL_LAT, NOMINAL_LNG, z)
    meta = {"zoom": z, "scale": s, "x0": cx - w / s / 2, "y0": cy - h / s / 2,
            "width_map_px": w / s, "height_map_px": h / s,
            "center": [NOMINAL_LAT, NOMINAL_LNG], "nx": None, "ny": None,
            "georeferenced": False, "crs": None}
    if m_per_px:
        meta["m_per_px"] = m_per_px
    return meta


def _merc_to_world(X, Y, zoom):
    w = imagery.WORLD(zoom)
    return (X / MERC + 0.5) * w, (0.5 - Y / MERC) * w


def _world_to_merc(x, y, zoom):
    w = imagery.WORLD(zoom)
    return (x / w - 0.5) * MERC, (0.5 - y / w) * MERC


@contextmanager
def _quiet():
    """Silence decoder warnings we handle ourselves (bomb size, no georeference).
    catch_warnings is not thread-safe in 3.12; at worst it briefly hides or
    shows an unrelated warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        yield
