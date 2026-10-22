"""Small helpers shared by tests (import as ``import helpers``)."""
import json

import numpy as np
from PIL import Image

from urbanseg import config, imagery, osm_data


def load_area(name):
    """(img HxWx3 uint8, auto-label mask HxW uint8, meta) from the read-only data."""
    img, meta = imagery.load_area(name)
    mask = np.asarray(Image.open(config.MASKS_DIR / f"{name}.png")).copy()
    return img, mask, meta


def crop(img, mask, meta, y0, y1, x0, x1):
    """Crop image px [y0:y1, x0:x1] and return a meta that still georeferences it."""
    s = meta["scale"]
    m = dict(meta)
    m["x0"] = meta["x0"] + x0 / s
    m["y0"] = meta["y0"] + y0 / s
    m["width_map_px"] = (x1 - x0) / s
    m["height_map_px"] = (y1 - y0) / s
    m["center"] = list(imagery.img_px_to_latlng((x1 - x0) / 2, (y1 - y0) / 2, m))
    out_mask = None if mask is None else mask[y0:y1, x0:x1].copy()
    return img[y0:y1, x0:x1].copy(), out_mask, m


def osm_snapshot(name):
    """Parsed OSM layers of an area from its label snapshot (read-only, no network)."""
    p = config.DATA_DIR / "osm" / f"{name}_v2.json"
    return osm_data.parse_overpass(json.loads(p.read_text()))
