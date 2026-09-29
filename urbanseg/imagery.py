"""Google Static Maps satellite imagery: tile math, fetching, stitching.

All geometry is done in Web-Mercator "map pixels" at a fixed zoom
(world = 256 * 2**zoom map px). Images are SCALE x map px.
"""
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from . import config

WORLD = lambda zoom: 256 * (2 ** zoom)


def latlng_to_world(lat, lng, zoom=config.ZOOM):
    w = WORLD(zoom)
    x = (lng + 180.0) / 360.0 * w
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * w
    return x, y


def world_to_latlng(x, y, zoom=config.ZOOM):
    w = WORLD(zoom)
    lng = x / w * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / w))))
    return lat, lng


def meters_per_map_px(lat, zoom=config.ZOOM):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def _fetch_raw_tile(center_lat, center_lng, session=None, max_retries=4):
    """One 640x640 (map px) satellite image, SCALE x resolution."""
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{center_lat:.7f},{center_lng:.7f}",
        "zoom": str(config.ZOOM),
        "size": f"{config.TILE_MAP_PX}x{config.TILE_MAP_PX}",
        "scale": str(config.SCALE),
        "maptype": "satellite",
        "key": config.GOOGLE_MAPS_KEY,
    }
    sess = session or requests
    for attempt in range(max_retries):
        r = sess.get(url, params=params, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            if img.size == (config.TILE_MAP_PX * config.SCALE,) * 2:
                return img
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"tile fetch failed at {center_lat},{center_lng}: HTTP {r.status_code}")


def fetch_area(center_lat, center_lng, nx, ny, cache_dir: Path | None = None,
               name: str | None = None, sleep=0.05):
    """Fetch an nx x ny grid of tiles around a center; return (image, meta).

    meta georeferences the stitched image: img px -> world map px -> lat/lng.
    """
    s = config.SCALE
    ux, uy = config.USABLE_X, config.USABLE_Y
    cx, cy = latlng_to_world(center_lat, center_lng)
    x0 = cx - nx * ux / 2.0
    y0 = cy - ny * uy / 2.0
    meta = {
        "zoom": config.ZOOM, "scale": s, "x0": x0, "y0": y0,
        "width_map_px": nx * ux, "height_map_px": ny * uy,
        "center": [center_lat, center_lng], "nx": nx, "ny": ny,
    }
    full = Image.new("RGB", (nx * ux * s, ny * uy * s))
    session = requests.Session()
    for j in range(ny):
        for i in range(nx):
            cached = None
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached = cache_dir / f"{i}_{j}.png"
            if cached is not None and cached.exists():
                tile = Image.open(cached).convert("RGB")
            else:
                tx = x0 + i * ux
                ty = y0 + j * uy
                c_lat, c_lng = world_to_latlng(tx + config.TILE_MAP_PX / 2.0,
                                               ty + config.TILE_MAP_PX / 2.0)
                raw = _fetch_raw_tile(c_lat, c_lng, session=session)
                tile = raw.crop((0, 0, ux * s, uy * s))
                if cached is not None:
                    tile.save(cached)
                time.sleep(sleep)
            full.paste(tile, (i * ux * s, j * uy * s))
    if cache_dir is not None and name is not None:
        (cache_dir / "meta.json").write_text(json.dumps(meta))
        full.save(cache_dir.parent / f"{name}_full.png")
    return full, meta


def bbox_latlng(meta):
    """(west, south, east, north) of a stitched area."""
    n_lat, w_lng = world_to_latlng(meta["x0"], meta["y0"], meta["zoom"])
    s_lat, e_lng = world_to_latlng(meta["x0"] + meta["width_map_px"],
                                   meta["y0"] + meta["height_map_px"], meta["zoom"])
    return (w_lng, s_lat, e_lng, n_lat)


def latlng_to_img_px(lat, lng, meta):
    x, y = latlng_to_world(lat, lng, meta["zoom"])
    return (x - meta["x0"]) * meta["scale"], (y - meta["y0"]) * meta["scale"]


def img_px_to_latlng(px, py, meta):
    return world_to_latlng(meta["x0"] + px / meta["scale"],
                           meta["y0"] + py / meta["scale"], meta["zoom"])


def load_area(name):
    d = config.TILES_DIR / name
    meta = json.loads((d / "meta.json").read_text())
    img = Image.open(config.TILES_DIR / f"{name}_full.png").convert("RGB")
    return np.asarray(img), meta
