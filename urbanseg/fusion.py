"""Fuse a model prediction with building footprints and OpenStreetMap.

Footprints (Open Buildings, OSM building polygons) and OSM centrelines are
ground-referenced, but Google's mosaic shows every roof displaced from its
base by the building's lean (1-8 m in Hyderabad, in a direction set by the
capture). Pasting vector geometry over the prediction therefore colours
facades and gaps and cuts real roofs.

mode="roof" (default) keeps the colours on the roofs as imaged: the model
decides WHERE buildings are, footprints decide WHAT each building is
(residential, commercial or transport), and OSM paints roads and water only
where the imagery does not contradict it.

mode="legacy" is the decision logic of infer.fuse_footprints with injected
inputs: whole footprints are snapped and, when the footprint index is
present, OSM buffers are painted over everything.
"""
import logging
import os
import time

import numpy as np
from rasterio import features as rfeatures
from rasterio.transform import Affine
from scipy import ndimage as ndi
from shapely.ops import transform as shp_transform

from . import config, imagery, labels, open_buildings, osm_data

log = logging.getLogger(__name__)

NODATA = 255  # classmap value for invalid pixels (as infer.NODATA); never changed
MODES = ("roof", "legacy")
# a live request should fail over fast when Overpass is slow or down
LIVE_OVERPASS = {"timeout": 30, "rounds": 1, "pause": 0}
DECISIVE_TAGS = config.RES_BUILDING_TAGS | config.COM_BUILDING_TAGS


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# roof mode; the gate and tie-break are today's, the rest chosen on val
MIN_BLD_FRAC = 0.20  # a footprint the model barely sees as building may be stale
TIE_BREAK = 0.15     # |P(res) - P(com)| under which the commercial rules decide
# footprint class reaches roof pixels up to this far from the footprint (lean
# reaches ~8 m on tall blocks; on val the gain stops at ~4 m and 4-16 m tie)
LEAN_M = _env_float("URBANSEG_FUSION_LEAN_M", 8.0)
# model buildings are cut only in this central share of a road's width ...
CORE_FRAC = _env_float("URBANSEG_FUSION_CORE_FRAC", 0.5)
# ... and only where the model is less sure than this that they are buildings
# (blind check on val: below 0.5 a cut hit road 7 of 8 times, above 0.7 a roof
# 5 of 6 times, mostly roofs missing from the footprint data)
CORE_MAX_P_BUILDING = _env_float("URBANSEG_FUSION_CORE_MAX_P_BUILDING", 0.5)
# OSM water is painted only where P(free land) is below this (dry tank beds)
WATER_MAX_P_FREE = _env_float("URBANSEG_FUSION_WATER_MAX_P_FREE", 0.5)

# waterway widths (m) as labels.other_geoms_world paints them; roads and rails
# get config.ROAD_WIDTHS (5 m for any other highway value) and config.RAIL_WIDTH
WATERWAY_WIDTHS = {"river": 25, "canal": 12, "stream": 8, "drain": 6, "ditch": 4}


def fuse(classmap, probs, meta, *, use_osm=True, mode="roof", layers=None,
         footprints=None, osm_info=None):
    """Refine a georeferenced prediction with footprints and OSM.

    classmap HxW uint8 (classes 0-3, NODATA kept), probs CxHxW float (softmax),
    meta the grid meta of the image. layers: parsed OSM layers (as
    osm_data.parse_overpass); None with use_osm fetches them live
    (osm_data.get_live_osm), and use_osm=False ignores OSM. footprints: Open
    Buildings rows (as open_buildings.load_for_bbox); None loads the local
    index. osm_info describes injected layers and is passed through.

    Returns the refined classmap and never raises: on any failure it returns
    the input classmap and records the reason. Sets meta["fusion"] (sources
    used, a subset of ["footprints", "osm"]) and meta["fusion_info"] =
    {"mode", "sources", "osm", "footprints_error", "changed_px", "error"}:
    "osm" is the fetch info (or {"source": None, "error", "reason",
    "elapsed_s"} when OSM was unavailable, None without OSM),
    "footprints_error" why the footprint index could not be read (None when
    it was read, is not installed or footprints were given), and changed_px
    counts the pixels each rule changed plus their "total" (roof:
    class_change, station, road_paint, water_paint, disjoint; legacy: paint,
    then snap of the painted map).
    """
    info = {"mode": mode, "sources": [], "osm": osm_info, "footprints_error": None,
            "changed_px": {}, "error": None}
    try:
        meta["fusion"], meta["fusion_info"] = [], info
        if mode not in MODES:
            raise ValueError(f"unknown fusion mode {mode!r}")
        if tuple(np.shape(probs)[-2:]) != tuple(classmap.shape):
            raise ValueError("probs and classmap sizes differ")
        fp, layers = _sources(meta, use_osm, layers, footprints, info)
        out, changed = (_legacy if mode == "legacy" else _roof)(
            classmap, probs, meta, fp, layers)
        info["sources"] = list(meta["fusion"])
        if out is not classmap:
            out[classmap == NODATA] = NODATA
        changed["total"] = 0 if out is classmap else int((out != classmap).sum())
        info["changed_px"] = changed
        return out
    except Exception as ex:  # noqa: BLE001 - fusion is optional; fall back to the model
        log.warning("fusion failed; returning the model output", exc_info=True)
        info.update(sources=[], changed_px={}, error=_reason(ex))
        if isinstance(meta, dict):
            meta["fusion"], meta["fusion_info"] = [], info
        return classmap


def utm_crs(bbox):
    """EPSG code of the UTM zone at the centre of (west, south, east, north).

    Metric buffers are drawn in it: 32644 (UTM 44N) for Hyderabad.
    """
    w, s, e, n = bbox
    lng, lat = (w + e) / 2.0, (s + n) / 2.0
    zone = int((lng + 180.0) // 6.0) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def _reason(ex):
    return f"{type(ex).__name__}: {ex}"[:200]


def _sources(meta, use_osm, layers, footprints, info):
    """(Open Buildings rows, OSM layers or None); sets info["osm"] and, when the
    footprint index cannot be read, info["footprints_error"]."""
    bbox = imagery.bbox_latlng(meta)
    if footprints is None:
        try:
            footprints = open_buildings.load_for_bbox(bbox)
        except Exception as ex:  # noqa: BLE001 - a broken index must not block OSM fusion
            # recorded so that a broken index is not mistaken for a missing one:
            # either way roof mode runs as the OSM-only install
            log.warning("footprint index unreadable", exc_info=True)
            info["footprints_error"] = _reason(ex)
            footprints = open_buildings.empty()
    if not use_osm:
        info["osm"] = None
        return footprints, None
    if layers is None:
        layers, info["osm"] = _live_osm(bbox)
    return footprints, layers


def _live_osm(bbox):
    """(layers, info), or (None, {"source": None, "error"}) when OSM is unavailable."""
    t0 = time.time()
    try:
        get_live = getattr(osm_data, "get_live_osm", None)
        if get_live is not None:
            return get_live(bbox)
        # osm_data without get_live_osm (until package fetch lands): today's
        # query, uncached - get_area_osm would key a cache file by a 24-bit
        # hash of the bbox (collisions serve another area's OSM) and write it
        # into config.OSM_DIR, which holds the read-only label snapshots
        data = osm_data.fetch_overpass(bbox, **LIVE_OVERPASS)
        osm3s = data.get("osm3s")
        return osm_data.parse_overpass(data), {
            "source": "overpass", "mirror": None,
            "osm_base": osm3s.get("timestamp_osm_base") if isinstance(osm3s, dict) else None,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 2), "cache_file": None,
            "query_version": "v2"}
    except Exception as ex:  # noqa: BLE001 - fuse without OSM
        return None, {"source": None, "error": _reason(ex),
                      "reason": getattr(ex, "reason", None),
                      "elapsed_s": round(time.time() - t0, 2)}


def _affine(meta):
    s = meta["scale"]
    return Affine(1.0 / s, 0, meta["x0"], 0, 1.0 / s, meta["y0"])


def _world(g, zoom):
    return shp_transform(lambda x, y: labels._world_arrays(x, y, zoom), g)


def _burn(geoms, shape, tr):
    if not geoms:
        return np.zeros(shape, bool)
    return rfeatures.rasterize([(g, 1) for g in geoms], out_shape=shape, transform=tr,
                               fill=0, dtype="uint8").astype(bool)


def _legacy(classmap, probs, meta, fp, layers):
    """infer.fuse_footprints once its inputs are loaded (keep in step with it)."""
    meta["fusion"] = ["footprints"] if len(fp) else []
    if layers is not None:
        try:
            fp = labels.merge_footprints(fp, layers["buildings"])
            meta["fusion"].append("osm")
        except Exception:  # noqa: BLE001 - as today: fuse without OSM
            layers = None
    if not len(fp) and layers is None:
        return classmap, {}
    H, W = classmap.shape
    tr = _affine(meta)
    zoom = meta["zoom"]

    fp = fp.reset_index(drop=True)
    fp_ids = rfeatures.rasterize(
        [(_world(g, zoom), i + 1) for i, g in enumerate(fp.geometry)],
        out_shape=(H, W), transform=tr, fill=0, dtype="int32")

    # without the Open Buildings index the only outlines are OSM's: leave model
    # buildings under road buffers and snap only outlines with a decisive tag
    osm_only = "footprints" not in meta["fusion"]
    refined = classmap.copy()
    if layers is not None:
        geoms = labels.other_geoms_world(layers, zoom)
        if geoms:
            other = rfeatures.rasterize(
                [(g, 1) for g in geoms], out_shape=(H, W),
                transform=tr, fill=0, dtype="uint8").astype(bool)
            if osm_only:
                other &= ~np.isin(classmap, (1, 2))
            refined[other] = 0
    painted = refined.copy()
    res_p, com_p = probs[1], probs[2]
    osm_cls = labels.classify_footprints(fp, layers) if layers is not None else None
    flat = fp_ids.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_ids = flat[order]
    starts = np.searchsorted(sorted_ids, np.arange(1, len(fp) + 2))
    cm_flat, ref_flat = classmap.ravel(), refined.ravel()
    rp_flat, cp_flat = res_p.ravel(), com_p.ravel()
    for i in range(len(fp)):
        sel = order[starts[i]:starts[i + 1]]
        n = sel.size
        if n == 0:
            continue
        if osm_only and str(fp.iloc[i].get("tag") or "") not in DECISIVE_TAGS:
            continue
        bld_frac = float(np.isin(cm_flat[sel], (1, 2)).mean())
        if bld_frac < 0.20:
            continue  # model sees no building; footprint may be stale
        r, c = float(rp_flat[sel].mean()), float(cp_flat[sel].mean())
        klass = 1 if r >= c else 2
        if osm_cls is not None:
            tag = str(fp.iloc[i].get("tag") or "")
            if tag in config.RES_BUILDING_TAGS:
                klass = 1
            elif tag in config.COM_BUILDING_TAGS:
                klass = 2
            elif osm_cls[i] == 2 and abs(r - c) < 0.15:
                klass = 2
        ref_flat[sel] = klass
    return refined, {"paint": int((painted != classmap).sum()),
                     "snap": int((refined != painted).sum())}


def _roof(classmap, probs, meta, fp, layers):
    """Roof-aligned fusion: never adds building pixels the model does not see."""
    meta["fusion"] = (["footprints"] if len(fp) else []) + (["osm"] if layers is not None else [])
    osm_only = not len(fp)
    if layers is not None:
        fp = labels.merge_footprints(fp, layers["buildings"])
    if not len(fp) and layers is None:
        return classmap, {}
    shape, tr, zoom = classmap.shape, _affine(meta), meta["zoom"]
    m_px = imagery.meters_per_map_px(meta["center"][0], zoom) / meta["scale"]
    fp = fp.reset_index(drop=True)
    ids = np.zeros(shape, np.int32)
    if len(fp):
        ids = rfeatures.rasterize(
            [(_world(g, zoom), i + 1) for i, g in enumerate(fp.geometry)],
            out_shape=shape, transform=tr, fill=0, dtype="int32")
    bld = np.isin(classmap, (1, 2))
    valid = classmap != NODATA

    # what each building is: a footprint's vote holds for the model-building
    # pixels it covers and, because roofs lean off their footprints, for the
    # rest of the same blob up to LEAN_M away
    out = classmap.copy()
    vote = _votes(fp, ids, bld, valid, probs, layers)[ids]
    seeds = bld & (vote >= 0)
    out[seeds] = vote[seeds]
    # OSM outlines alone cover a minority of buildings, so the nearest one is
    # often another building's (on val spreading them lowered agreement)
    if seeds.any() and LEAN_M > 0 and not osm_only:
        _spread(out, seeds, bld, LEAN_M / m_px)

    # roads and water: OSM decides where the model sees no building; a model
    # building is cut only in a road's core, when no footprint vouches for it
    # and the model is unsure (centrelines are often metres off the imagery)
    paint = np.zeros(shape, bool)
    water = np.zeros(shape, bool)
    if layers is not None:
        full, core = _line_masks(layers, utm_crs(imagery.bbox_latlng(meta)), shape, tr, zoom)
        unsure = (probs[1] + probs[2]) < CORE_MAX_P_BUILDING
        paint = valid & ((full & ~bld) | (core & bld & unsure & (ids == 0)))
        # tanks mapped as water are often dry beds the model sees as free land
        polys = [_world(g, zoom) for g in layers["water"].geometry
                 if g is not None and not g.is_empty]
        water = valid & ~bld & ~paint & _burn(polys, shape, tr)
        water &= probs[3] < WATER_MAX_P_FREE
    recl = bld & (out != classmap) & ~paint & ~water
    changed = {"class_change": int((recl & (out != 0)).sum()),
               "station": int((recl & (out == 0)).sum()),
               "road_paint": int((paint & (classmap != 0)).sum()),
               "water_paint": int((water & (classmap != 0)).sum())}
    out[paint | water] = 0
    return out, changed


def _votes(fp, ids, bld, valid, probs, layers):
    """Class per footprint id (index 0 unused): 1 res, 2 com, 0 transport, -1 none.

    As infer.fuse_footprints decides, except that the mean P(res) / P(com) is
    taken over the model-building pixels only (the rest of a footprint is the
    imaged facade or ground), transport tags vote 'other' as in the labels, and
    OSM outlines without a decisive tag vote too (they only even out the
    model's class within one building; today's OSM-only install skips them).
    Footprints the model sees as building on less than MIN_BLD_FRAC of their
    valid pixels do not vote: NODATA (outside an upload) is no evidence.
    """
    n = len(fp)
    if not n:
        return np.full(1, -1, np.int8)
    b_ids = ids[bld]
    n_all = (np.bincount(ids.ravel(), minlength=n + 1)
             - np.bincount(ids[~valid], minlength=n + 1))[1:]
    n_bld = np.bincount(b_ids, minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.bincount(b_ids, weights=probs[1][bld], minlength=n + 1)[1:] / n_bld
        c = np.bincount(b_ids, weights=probs[2][bld], minlength=n + 1)[1:] / n_bld
        votes = (n_bld > 0) & (n_bld / n_all >= MIN_BLD_FRAC)
    klass = np.where(r >= c, 1, 2)
    if layers is not None:
        tag = fp["tag"].fillna("").astype(str).values
        rule = labels.classify_footprints(fp, layers)
        undecided = ~np.isin(tag, list(DECISIVE_TAGS)) & (rule == 2) & (np.abs(r - c) < TIE_BREAK)
        klass[undecided] = 2
        klass[np.isin(tag, list(config.RES_BUILDING_TAGS))] = 1
        klass[np.isin(tag, list(config.COM_BUILDING_TAGS))] = 2
        klass[np.isin(tag, list(config.OTHER_BUILDING_TAGS))] = 0
    return np.concatenate([[-1], np.where(votes, klass, -1)]).astype(np.int8)


def _spread(out, seeds, blobs, radius_px):
    """Give each pixel of `blobs` outside `seeds` the class of the nearest
    seed pixel of its own 8-connected blob, if one lies within radius_px.

    Each blob is solved alone in its bounding box, which holds all its seeds:
    a neighbouring blob's seed never shadows its own, and memory follows the
    blob rather than the image.
    """
    blob, n = ndi.label(blobs, structure=np.ones((3, 3), bool))
    todo = ((np.bincount(blob[seeds], minlength=n + 1) > 0)
            & (np.bincount(blob[blobs & ~seeds], minlength=n + 1) > 0))
    r2 = radius_px * radius_px
    for k, sl in enumerate(ndi.find_objects(blob), 1):
        if not todo[k]:
            continue
        own = blob[sl] == k
        seed = own & seeds[sl]
        # nearest-seed indices only; squared distances follow from them, which
        # spares a float64 distance map
        iy, ix = ndi.distance_transform_edt(~seed, return_distances=False,
                                            return_indices=True)
        ys, xs = np.nonzero(own & ~seed)
        sy, sx = iy[ys, xs], ix[ys, xs]
        near = (sy - ys) ** 2 + (sx - xs) ** 2 <= r2
        sub = out[sl]
        sub[ys[near], xs[near]] = sub[sy[near], sx[near]]


def _line_masks(layers, crs, shape, tr, zoom):
    """(full, core) masks of OSM road, rail and waterway buffers; the core is
    the central CORE_FRAC of each way's width. Widths and caps are those of
    labels.other_geoms_world, i.e. what is painted today."""
    parts = []
    roads = layers["roads"]
    if len(roads):
        parts.append((roads.geometry,
                      roads["highway"].map(config.ROAD_WIDTHS).fillna(5).values, "flat"))
    rails = layers["rails"]
    if len(rails):
        parts.append((rails.geometry, np.full(len(rails), float(config.RAIL_WIDTH)), "round"))
    ww = layers.get("waterways")
    if ww is not None and len(ww):
        parts.append((ww.geometry, ww["waterway"].map(WATERWAY_WIDTHS).fillna(6).values, "flat"))
    full, core = [], []
    for geoms, widths, cap in parts:
        metric = geoms.to_crs(crs)
        for frac, dst in ((1.0, full), (CORE_FRAC, core)):
            buf = metric.buffer(widths * (frac / 2.0), cap_style=cap).to_crs(4326)
            dst += [_world(g, zoom) for g in buf if g is not None and not g.is_empty]
    return _burn(full, shape, tr), _burn(core, shape, tr)
