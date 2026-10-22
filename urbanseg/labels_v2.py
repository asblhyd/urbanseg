"""Version-2 training labels: buildings on their roofs, no enlargement, honest edges.

v1 labels (labels.py, data/masks) stay frozen. Measured problems they have,
which v2 addresses (docs/METHODOLOGY.md, "Known limitations"):

- Footprints are ground positions; Google's mosaic shows every roof displaced
  by its building's lean (0.6-7.6 m per area, more for taller buildings, in a
  direction that changes between captures). align_footprints() moves each
  footprint onto its roof as imaged, from the image itself.
- Buildings were dilated by 1.2 m (22% of label building area), which taught
  oversized buildings. v2 uses the outlines as they are, closes only gaps
  narrower than GAP_CLOSE_M between buildings, and marks EDGE_IGNORE_PX on
  each side of a building edge as IGNORE (residual misplacement, roof edges).
- Roads have one width per OSM class and centrelines that can be metres off:
  ROAD_EDGE_IGNORE_PX on each side of a buffer's edge is IGNORE; paths and
  tracks get 2 m instead of 5 m; closed ways (roundabouts) are kept (v3 parser).
- OSM water includes dry tank beds: inside water polygons, only pixels that
  look like water (dark, and smooth or blue-green: rippled lakes are not
  smooth) are "other"; the rest is IGNORE.
- Between a footprint and its roof lies what the lean shows of the building's
  walls (or, when the footprint came from another leaning capture, ground of
  unknown class): the strip each outline sweeps on its way to the roof is
  IGNORE, not free land.
- Open Buildings misses roofs (6-22% of the labelled building area of three
  areas checked in October 2026, more where tall buildings lean beyond the
  search): teacher_ignore() marks free land that a trained model is sure is
  a building, and that is as bright as the area's roofs and not green, as
  IGNORE. That is label version "v2t"; "v2" is without it.

Buildings win over road and water: their uncertain bands never reach into a
building. IGNORE (255) pixels carry no loss in training (train.build_loss).
"""
import numpy as np
from rasterio import features as rfeatures
from rasterio.transform import Affine
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform

from . import config, imagery, labels

IGNORE = 255
SEARCH_PX = 32          # +-9 m at 0.285 m/px: covers the leans measured (<= ~10 m)
FIELD_RADIUS_M = 120    # lean is coherent within a capture: neighbours vote
REFINE_PX = 8           # final search around the local field (+-2.3 m)
# Scores are edge strength on the outline (normalised per image) plus half
# the inside/outside contrast. Calibrated on kondapur, hitech, oldcity,
# madhapur and adibatla (790 footprints vs 1,377 placements >= 13 m from any
# footprint): a peak >= 0.9 is reached by 86% of footprints and 20% of the
# placements (some of which are unmapped buildings), >= 0.8 by 95% and 30%.
# The z-score of the peak separated far worse (median 4.4 vs 3.7).
FIELD_MIN_SCORE = 0.9   # a building votes in the local field
OWN_MIN_SCORE = 0.8     # ... and keeps its own shift near the field
OWN_MARGIN = 0.03       # only if that beats the field position itself
MIN_AREA_PX = 120       # smaller outlines carry too little edge to align alone
EDGE_IGNORE_PX = 2      # +-0.57 m around building edges
ROAD_EDGE_IGNORE_PX = 3
GAP_CLOSE_M = 1.2       # closes party-wall gaps narrower than ~2 x this
PATH_WIDTH_M = 2.0
SMALL_WAYS = {"footway", "path", "steps", "track", "pedestrian", "cycleway", "bridleway"}
# teacher_ignore: how sure the model must be, and how big a blob must be
TEACHER_MIN_P = 0.85
TEACHER_MIN_M2 = 15.0
TEACHER_MAX_EXG = 20.0       # 2G-R-B (5x5 mean) above this is vegetation
TEACHER_LUMA_PCT = 20        # at least this percentile of the area's roof luma
TEACHER_LUMA_FALLBACK = 120  # ... when the area has too few roofs to measure
WET_MAX_LUMA, WET_MAX_STD = 95.0, 14.0
# ... or, when rippled (Hussain Sagar: 9x9 std ~26), blue-green: rippled water
# measured (44, 72, 72) RGB, where dry beds are brown and trees have B << G
WET_MIN_BLUE_OVER_RED, WET_MIN_BLUE_OF_GREEN = 5.0, 0.85


# ------------------------------------------------------------------ geometry

def _to_px(geom, meta):
    """lon/lat geometry -> image px coordinates of meta's grid."""
    s, z = meta["scale"], meta["zoom"]
    x0, y0 = meta["x0"], meta["y0"]

    def f(lng, lat):
        wx, wy = labels._world_arrays(lng, lat, z)
        return (np.asarray(wx) - x0) * s, (np.asarray(wy) - y0) * s
    return shp_transform(f, geom)


def _from_px(geom, meta):
    s, z = meta["scale"], meta["zoom"]
    x0, y0 = meta["x0"], meta["y0"]
    w = 256 * 2 ** z

    def f(px, py):
        x = np.asarray(px) / s + x0
        y = np.asarray(py) / s + y0
        lng = x / w * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / w))))
        return lng, lat
    return shp_transform(f, geom)


def _edges(img):
    """Gradient magnitude of the (smoothed) grey image, normalised to ~[0, 1]."""
    g = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    g = ndi.gaussian_filter(g, 1.0)
    e = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))
    return g, e / max(float(np.percentile(e, 95)), 1e-6)


def _corr(field, kernel):
    """valid cross-correlation of field with kernel (sum over the kernel's support)."""
    return fftconvolve(field, kernel[::-1, ::-1], mode="valid")


def _score_map(grey, edges, poly_px, R=SEARCH_PX):
    """Score of every shift (dy, dx) in [-R, R]^2 of one footprint, or None.

    edge term: mean edge strength on the shifted outline (roof edges);
    contrast term: |mean grey inside - mean grey in a ring 2-6 px outside|.
    """
    H, W = grey.shape
    x0, y0, x1, y1 = poly_px.bounds
    pad = R + 7
    c0, r0 = int(np.floor(x0)) - pad, int(np.floor(y0)) - pad
    c1, r1 = int(np.ceil(x1)) + pad, int(np.ceil(y1)) + pad
    if c0 < 0 or r0 < 0 or c1 > W or r1 > H:
        return None  # too close to the image border to search
    h, w = r1 - r0 - 2 * R, c1 - c0 - 2 * R  # kernel size
    local = affinity.translate(poly_px, -(c0 + R), -(r0 + R))
    inside = rfeatures.rasterize([(local, 1)], out_shape=(h, w), transform=Affine.identity(),
                                 fill=0, dtype="uint8").astype(bool)
    n_in = inside.sum()
    if n_in < MIN_AREA_PX:
        return None
    outline = inside & ~ndi.binary_erosion(inside, iterations=1)
    ring = ndi.binary_dilation(inside, iterations=6) & ~ndi.binary_dilation(inside, iterations=2)
    G = grey[r0:r1, c0:c1]
    E = edges[r0:r1, c0:c1]
    edge = _corr(E, outline.astype(np.float32)) / max(outline.sum(), 1)
    m_in = _corr(G, inside.astype(np.float32)) / n_in
    m_out = _corr(G, ring.astype(np.float32)) / max(ring.sum(), 1)
    contrast = np.abs(m_in - m_out) / 40.0
    return edge + 0.5 * contrast  # (2R+1, 2R+1); [R, R] is zero shift


def _peak(score, center=None, radius=None):
    """(dy, dx, score) of the best shift, optionally within radius of center."""
    R = score.shape[0] // 2
    s = score
    if center is not None:
        cy, cx = int(round(center[0])) + R, int(round(center[1])) + R
        y0, y1 = max(0, cy - radius), min(score.shape[0], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(score.shape[1], cx + radius + 1)
        if y0 >= y1 or x0 >= x1:
            return center[0], center[1], -np.inf
        s = np.full_like(score, -np.inf)
        s[y0:y1, x0:x1] = score[y0:y1, x0:x1]
    iy, ix = np.unravel_index(np.argmax(s), s.shape)
    return iy - R, ix - R, float(s[iy, ix])


def _score_at(score, dy, dx):
    R = score.shape[0] // 2
    iy, ix = int(round(dy)) + R, int(round(dx)) + R
    if 0 <= iy < score.shape[0] and 0 <= ix < score.shape[1]:
        return float(score[iy, ix])
    return -np.inf


def _weighted_median(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2.0)])


def align_footprints(img, meta, footprints, *, return_details=False):
    """Move each footprint onto its roof as imaged.

    img HxWx3 uint8 on meta's grid; footprints GeoDataFrame (EPSG:4326).
    Returns a copy with translated geometries and columns shift_dx_px,
    shift_dy_px (image px; +x east, +y south), shift_conf (the score of the
    building's own evidence, 0 when unused) and shift_source ("own", "field"
    or "none").

    Each footprint's shift score is mapped over +-SEARCH_PX (_score_map). The
    best shifts of strongly matching buildings form a local field
    (score-weighted median within FIELD_RADIUS_M); each building then takes
    its best shift within REFINE_PX of the field when that scores high enough
    and better than the field position, else the field itself. Buildings
    without a usable score take the field.
    """
    fp = footprints.reset_index(drop=True).copy()
    n = len(fp)
    out = {"shift_dx_px": np.zeros(n), "shift_dy_px": np.zeros(n),
           "shift_conf": np.zeros(n), "shift_source": np.array(["none"] * n, dtype=object)}
    if n == 0:
        for k, v in out.items():
            fp[k] = v
        return (fp, {}) if return_details else fp
    grey, edges = _edges(img)
    polys = [_to_px(g, meta) for g in fp.geometry]
    cent = np.array([[p.centroid.y, p.centroid.x] for p in polys])
    maps = [None if p.is_empty else _score_map(grey, edges, p) for p in polys]
    raw = np.full((n, 3), np.nan)  # dy, dx, score of the global best shift
    for i, sm in enumerate(maps):
        if sm is not None:
            raw[i] = _peak(sm)
    good = np.flatnonzero(np.isfinite(raw[:, 2]) & (raw[:, 2] >= FIELD_MIN_SCORE))
    mpp = imagery.meters_per_map_px(meta["center"][0], meta["zoom"]) / meta["scale"]
    field = np.zeros((n, 2))
    if len(good):
        tree = cKDTree(cent[good])
        for i in range(n):
            idx = tree.query_ball_point(cent[i], FIELD_RADIUS_M / mpp)
            if not idx:
                idx = [tree.query(cent[i])[1]]
            j = good[idx]
            w = raw[j, 2] - FIELD_MIN_SCORE + 0.05
            field[i] = (_weighted_median(raw[j, 0], w), _weighted_median(raw[j, 1], w))
    for i, sm in enumerate(maps):
        if sm is not None:
            dy, dx, sc = _peak(sm, field[i], REFINE_PX)
            if sc >= OWN_MIN_SCORE and sc >= _score_at(sm, *field[i]) + OWN_MARGIN:
                out["shift_dy_px"][i], out["shift_dx_px"][i] = dy, dx
                out["shift_conf"][i], out["shift_source"][i] = sc, "own"
                continue
        if len(good):
            out["shift_dy_px"][i], out["shift_dx_px"][i] = field[i]
            out["shift_source"][i] = "field"
    moved = []
    for p, dx, dy in zip(polys, out["shift_dx_px"], out["shift_dy_px"]):
        moved.append(_from_px(affinity.translate(p, dx, dy), meta) if not p.is_empty else p)
    fp = fp.set_geometry(moved)
    for k, v in out.items():
        fp[k] = v
    if return_details:
        return fp, {"raw": raw, "field": field, "n_good": int(len(good))}
    return fp


# ------------------------------------------------------------------- labels

def _road_buffers(layers, utm):
    """(geoms lon/lat, widths m) for highways, rails and waterways, with v2 widths."""
    out = []
    roads = layers["roads"]
    if len(roads):
        widths = roads["highway"].map(
            lambda h: PATH_WIDTH_M if h in SMALL_WAYS else config.ROAD_WIDTHS.get(h, 5))
        r_utm = roads.to_crs(utm)
        out += [(g, float(wd)) for g, wd in zip(r_utm.geometry, widths)]
    if len(layers["rails"]):
        out += [(g, float(config.RAIL_WIDTH)) for g in layers["rails"].to_crs(utm).geometry]
    ww = layers.get("waterways")
    if ww is not None and len(ww):
        widths = ww["waterway"].map({"river": 25, "canal": 12, "stream": 8, "drain": 6,
                                     "ditch": 4}).fillna(6)
        out += [(g, float(wd)) for g, wd in zip(ww.to_crs(utm).geometry, widths)]
    return out


def _sweep(aligned, meta, shape):
    """Pixels the aligned outlines passed over on the way from their footprints:
    each outline swept along its shift (footprint + one parallelogram per edge)."""
    polys = []
    for g, dx, dy in zip(aligned.geometry, aligned.shift_dx_px, aligned.shift_dy_px):
        if g is None or g.is_empty or (dx == 0 and dy == 0):
            continue
        p = _to_px(g, meta)
        for q in getattr(p, "geoms", [p]):
            xy = np.asarray(q.exterior.coords)
            orig = xy - (dx, dy)
            polys.append(Polygon(orig))
            for a, b, c, d in zip(orig[:-1], orig[1:], xy[1:], xy[:-1]):
                quad = Polygon([a, b, c, d])
                if quad.area > 1e-6:
                    polys.append(quad)
    if not polys:
        return np.zeros(shape, bool)
    return rfeatures.rasterize([(q, 1) for q in polys], out_shape=shape,
                               transform=Affine.identity(), fill=0,
                               dtype="uint8").astype(bool)


def build_mask_v2(img, meta, layers, footprints):
    """v2 label (HxW uint8: 0-3 classes, 255 IGNORE) and the aligned footprints.

    layers: OSM parsed with osm_data.parse_overpass(data, version="v3");
    footprints: Open Buildings rows for the area (open_buildings.load_for_bbox).
    """
    import geopandas as gpd

    H, W = img.shape[:2]
    fp = labels.merge_footprints(footprints, layers["buildings"])
    cls = labels.classify_footprints(fp, layers)  # residential / commercial rules of v1
    fp = fp.reset_index(drop=True)
    fp["cls"] = cls
    aligned = align_footprints(img, meta, fp)
    tr = Affine.identity()
    mask = np.full((H, W), 3, np.uint8)  # free land
    ignore = np.zeros((H, W), bool)

    # water: "other" only where it looks wet, else unknown (dry tank beds)
    if len(layers["water"]):
        wet_poly = rfeatures.rasterize([(_to_px(g, meta), 1) for g in layers["water"].geometry
                                        if g is not None and not g.is_empty],
                                       out_shape=(H, W), transform=tr, fill=0,
                                       dtype="uint8").astype(bool)
        f = img.astype(np.float32)
        grey = f @ np.array([0.299, 0.587, 0.114], np.float32)
        mu = ndi.uniform_filter(grey, 9)
        sd = np.sqrt(np.maximum(ndi.uniform_filter(grey ** 2, 9) - mu ** 2, 0))
        r, g, b = (ndi.uniform_filter(f[..., c], 9) for c in range(3))
        teal = (b >= r + WET_MIN_BLUE_OVER_RED) & (b >= WET_MIN_BLUE_OF_GREEN * g)
        wet = (mu < WET_MAX_LUMA) & ((sd < WET_MAX_STD) | teal)
        mask[wet_poly & wet] = 0
        ignore |= wet_poly & ~wet

    # roads, rails, waterways: their buffers, with an uncertain band at the edge
    utm = labels.UTM
    bufs = _road_buffers(layers, utm)
    if bufs:
        geoms = gpd.GeoSeries([g.buffer(wd / 2, cap_style=2) for g, wd in bufs], crs=utm)
        geoms = geoms.to_crs(4326)
        road = rfeatures.rasterize([(_to_px(g, meta), 1) for g in geoms
                                    if g is not None and not g.is_empty],
                                   out_shape=(H, W), transform=tr, fill=0,
                                   dtype="uint8").astype(bool)
        mask[road] = 0
        k = ROAD_EDGE_IGNORE_PX
        ignore |= ndi.binary_dilation(road, iterations=k) & ~ndi.binary_erosion(road, iterations=k)

    # buildings on their roofs, painted last; class 0 for transport buildings
    bld_cls = np.zeros((H, W), np.uint8)
    for klass in (0, 1, 2):
        sel = aligned[aligned["cls"] == klass]
        shapes = [(_to_px(g, meta), 1) for g in sel.geometry if g is not None and not g.is_empty]
        if shapes:
            b = rfeatures.rasterize(shapes, out_shape=(H, W), transform=tr, fill=0,
                                    dtype="uint8").astype(bool)
            bld_cls[b] = klass + 1  # 1..3 = class + 1, 0 = no building
    bld = bld_cls > 0
    # close party-wall gaps (not the outer edge): new pixels take the nearest class
    mpp = imagery.meters_per_map_px(meta["center"][0], meta["zoom"]) / meta["scale"]
    r = max(1, int(round(GAP_CLOSE_M / mpp)))
    closed = ndi.binary_closing(bld, iterations=r) | bld  # closing may erode at the border
    gap = closed & ~bld
    if gap.any():
        _, (iy, ix) = ndi.distance_transform_edt(~bld, return_indices=True)
        bld_cls[gap] = bld_cls[iy[gap], ix[gap]]
        bld = closed
    mask[bld] = bld_cls[bld] - 1
    ignore &= ~bld  # buildings win over the road and water bands
    ignore |= _sweep(aligned, meta, (H, W)) & ~bld  # walls the lean shows
    k = EDGE_IGNORE_PX
    ignore |= ndi.binary_dilation(bld, iterations=k) & ~ndi.binary_erosion(bld, iterations=k)
    mask[ignore] = IGNORE
    return mask, aligned


def teacher_ignore(img, mask, p_building, meta):
    """Free-land pixels of a v2 mask that are probably a roof Open Buildings missed.

    p_building HxW: a trained model's P(residential) + P(commercial). A pixel
    qualifies when the label says free land, the model is at least
    TEACHER_MIN_P sure of a building, the pixel is as bright as the area's
    labelled roofs (TEACHER_LUMA_PCT-th percentile of their luma: shadowed
    lanes and setbacks stay free land, so spill is still taught away) and not
    green; blobs are opened by 2 px and kept from TEACHER_MIN_M2 up. Only ever
    turns free land into IGNORE, never into a building. Returns HxW bool.
    """
    f = img.astype(np.float32)
    luma = ndi.uniform_filter(f @ np.array([0.299, 0.587, 0.114], np.float32), 5)
    exg = ndi.uniform_filter(2 * f[..., 1] - f[..., 0] - f[..., 2], 5)
    roofs = ndi.binary_erosion(np.isin(mask, (1, 2)), iterations=3)
    floor = (float(np.percentile(luma[roofs], TEACHER_LUMA_PCT)) if roofs.sum() >= 1000
             else TEACHER_LUMA_FALLBACK)
    cand = (mask == 3) & (p_building >= TEACHER_MIN_P) & (luma >= floor) & (exg <= TEACHER_MAX_EXG)
    cand = ndi.binary_opening(cand, iterations=2)
    lbl, n = ndi.label(cand)
    if n == 0:
        return cand
    mpp = imagery.meters_per_map_px(meta["center"][0], meta["zoom"]) / meta["scale"]
    sizes = ndi.sum(cand, lbl, np.arange(1, n + 1))
    return np.isin(lbl, 1 + np.flatnonzero(sizes * mpp ** 2 >= TEACHER_MIN_M2))
