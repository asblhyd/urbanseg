"""Inference: sliding-window segmentation + footprint/POI fusion + outputs."""
import json

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from rasterio import features as rfeatures
from rasterio.transform import Affine
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

from . import config, dataset, imagery, labels, open_buildings, osm_data

# a live request should fail over fast when Overpass is slow or down
LIVE_OVERPASS = {"timeout": 30, "rounds": 1, "pause": 0}
DECISIVE_TAGS = config.RES_BUILDING_TAGS | config.COM_BUILDING_TAGS


def load_model(path=None):
    from .train import build_model
    model = build_model(pretrained=False)
    p = path or (config.MODELS_DIR / "unet_r34.pt")
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def predict_tta(model, img, **kw):
    """Average predictions over 4 flips/rotations."""
    accum = None
    for k, flip in ((0, False), (2, False), (0, True), (2, True)):
        v = np.rot90(img, k)
        if flip:
            v = v[:, ::-1]
        _, p = predict(model, np.ascontiguousarray(v), **kw)
        if flip:
            p = p[:, :, ::-1]
        p = np.rot90(p, -k, axes=(1, 2))
        accum = p if accum is None else accum + p
    accum /= 4.0
    return accum.argmax(0).astype(np.uint8), accum


@torch.no_grad()
def predict(model, img, window=1024, stride=768, batch=1):
    """img HxWx3 uint8 -> (classmap HxW uint8, probs HxWxC float16).

    batch=1: on CPU larger batches are no faster and batch=4 nearly doubles
    peak memory (2.9 vs 1.6 GB on a 3x3-tile area); the output is identical.
    """
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


def fuse_footprints(classmap, probs, meta, use_osm=True):
    """Snap building predictions to footprints; paint OSM roads/water; POI fusion.

    Footprints are Open Buildings (if the local index has been built) plus
    live OSM building polygons. Only possible for geolocated inputs. Returns
    the refined classmap and records the sources used in meta["fusion"].
    """
    bbox = imagery.bbox_latlng(meta)
    meta["fusion"] = []
    try:
        fp = open_buildings.load_for_bbox(bbox)
    except Exception:  # noqa: BLE001
        fp = open_buildings.empty()
    if len(fp):
        meta["fusion"].append("footprints")
    layers = None
    if use_osm:
        try:
            layers = osm_data.get_area_osm(f"live_{hash(bbox) & 0xffffff:x}", bbox,
                                           **LIVE_OVERPASS)
            fp = labels.merge_footprints(fp, layers["buildings"])
            meta["fusion"].append("osm")
        except Exception:  # noqa: BLE001
            layers = None
    if not len(fp) and layers is None:
        return classmap
    s = meta["scale"]
    H, W = classmap.shape
    tr = Affine(1.0 / s, 0, meta["x0"], 0, 1.0 / s, meta["y0"])
    zoom = meta["zoom"]
    world = lambda g: shp_transform(  # noqa: E731
        lambda x, y: labels._world_arrays(x, y, zoom), g)

    fp = fp.reset_index(drop=True)
    fp_ids = rfeatures.rasterize(
        [(world(g), i + 1) for i, g in enumerate(fp.geometry)],
        out_shape=(H, W), transform=tr, fill=0, dtype="int32")

    # Without the Open Buildings index the only outlines are OSM's, which miss
    # most buildings and are often coarse. On the test split it scores best to
    # leave the model's buildings under road buffers and snap only outlines
    # with a decisive tag (75.2% vs 71.9% painting/snapping everything).
    osm_only = "footprints" not in meta["fusion"]
    refined = classmap.copy()
    if layers is not None:
        # authoritative vectors: paint OSM roads/rail/water as "other"
        geoms = labels.other_geoms_world(layers, zoom)
        if geoms:
            other = rfeatures.rasterize(
                [(g, 1) for g in geoms], out_shape=(H, W),
                transform=tr, fill=0, dtype="uint8").astype(bool)
            if osm_only:
                other &= ~np.isin(classmap, (1, 2))
            refined[other] = 0
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
    return refined


def render_overlay(img, classmap, alpha=0.45):
    color = np.zeros_like(img)
    for k, c in config.CLASS_COLORS.items():
        color[classmap == k] = c
    return (img * (1 - alpha) + color * alpha).astype(np.uint8)


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


def area_stats(classmap, meta):
    lat = meta["center"][0]
    mpp = imagery.meters_per_map_px(lat, meta["zoom"]) / meta["scale"]
    px_area = mpp * mpp
    out = {}
    for k, name in enumerate(config.CLASS_NAMES):
        n = int((classmap == k).sum())
        out[name] = {"pixels": n, "area_m2": round(n * px_area, 1),
                     "share": round(n / classmap.size, 4)}
    return out


def to_geojson(classmap, meta, min_area_px=400, simplify_m=1.0):
    s = meta["scale"]
    tr = Affine(1.0 / s, 0, meta["x0"], 0, 1.0 / s, meta["y0"])
    zoom = meta["zoom"]
    feats = []
    for k in (1, 2, 3):
        m = (classmap == k).astype(np.uint8)
        for geom, val in rfeatures.shapes(m, mask=m.astype(bool), transform=tr):
            poly = shape(geom)  # world map px coords
            if poly.area * (s * s) < min_area_px:
                continue
            poly = poly.simplify(simplify_m / imagery.meters_per_map_px(
                meta["center"][0], zoom))
            ll = shp_transform(lambda x, y: _ll_arrays(x, y, zoom), poly)
            feats.append({"type": "Feature",
                          "properties": {"class": config.CLASS_NAMES[k]},
                          "geometry": mapping(ll)})
    return {"type": "FeatureCollection", "features": feats}


def _ll_arrays(x, y, zoom):
    w = 256 * (2 ** zoom)
    lng = np.asarray(x) / w * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * np.asarray(y) / w))))
    return lng, lat


def segment_latlng(model, lat, lng, grid=2, fuse=True):
    """Full pipeline for a coordinate query."""
    img, meta = imagery.fetch_area(lat, lng, grid, grid)
    arr = np.asarray(img)
    classmap, probs = predict(model, arr)
    if fuse:
        classmap = fuse_footprints(classmap, probs, meta)
    return arr, classmap, meta


def segment_image(model, img_arr):
    """Pipeline for an uploaded image (no geo metadata -> no fusion)."""
    classmap, probs = predict(model, img_arr)
    meta = {"zoom": config.ZOOM, "scale": config.SCALE, "x0": 0, "y0": 0,
            "center": [17.4, 78.48],
            "width_map_px": img_arr.shape[1] // config.SCALE,
            "height_map_px": img_arr.shape[0] // config.SCALE}
    return classmap, probs, meta
