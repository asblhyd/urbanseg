"""Acquire imagery + OSM + Open Buildings and build masks for all areas."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urbanseg import config, imagery, labels, open_buildings, osm_data

STEP = sys.argv[1] if len(sys.argv) > 1 else "all"
ONLY = set(sys.argv[2:])
if ONLY:
    config.AREAS = [a for a in config.AREAS if a[0] in ONLY]

if STEP in ("all", "ob"):
    open_buildings.download_and_filter()

if STEP in ("all", "tiles"):
    for name, lat, lng, nx, ny, split in config.AREAS:
        d = config.TILES_DIR / name
        try:
            img, meta = imagery.fetch_area(lat, lng, nx, ny, cache_dir=d, name=name)
            print(f"[tiles] {name}: {img.size} {split}", flush=True)
        except Exception:
            print(f"[tiles] FAILED {name}", flush=True)
            traceback.print_exc()

if STEP in ("all", "osm"):
    import json
    for name, lat, lng, nx, ny, split in config.AREAS:
        meta = json.loads((config.TILES_DIR / name / "meta.json").read_text())
        bbox = imagery.bbox_latlng(meta)
        try:
            layers = osm_data.get_area_osm(name, bbox)
            print(f"[osm] {name}: b={len(layers['buildings'])} "
                  f"lu={len(layers['landuse'])} roads={len(layers['roads'])} "
                  f"pois={len(layers['pois'])}", flush=True)
        except Exception:
            print(f"[osm] FAILED {name}", flush=True)
            traceback.print_exc()

if STEP in ("all", "masks"):
    import json
    for name, lat, lng, nx, ny, split in config.AREAS:
        meta = json.loads((config.TILES_DIR / name / "meta.json").read_text())
        try:
            mask, fp, cls = labels.build_mask(name, meta)
            frac = {config.CLASS_NAMES[k]: round(float((mask == k).mean()), 3)
                    for k in range(config.N_CLASSES)}
            n_res, n_com = int((cls == 1).sum()), int((cls == 2).sum())
            print(f"[mask] {name}: {frac} bld(res={n_res},com={n_com})", flush=True)
        except Exception:
            print(f"[mask] FAILED {name}", flush=True)
            traceback.print_exc()

print("done", flush=True)
