"""Fetch and label the version-2 areas (config.AREAS_V2) into data_v2/.

data/ (the frozen v1 dataset) is never written. Steps, each skipping what is
already there (so a run can be resumed):

  python scripts/build_dataset_v2.py tiles    [names...]  # 9 Static Maps calls per area
  python scripts/build_dataset_v2.py osm      [names...]  # one Overpass query per area
  python scripts/build_dataset_v2.py masks    [names...]  # v1 rules -> data_v2/masks_v1/
  python scripts/build_dataset_v2.py teacher  [names...]  # P(building) of the shipped model
  python scripts/build_dataset_v2.py masks_v2 [names...]  # v2 and v2t labels

teacher and masks_v2 cover the v1 areas too; those are read from data/
(tiles, OSM snapshots), never written there. teacher stores the shipped
model's (models/unet_r34.pt, sha256 in data_v2/teacher/teacher.json)
P(residential) + P(commercial) as data_v2/teacher/<name>.png (0-255).

masks_v2 writes, per area, data_v2/masks_v2/<name>.png (labels_v2: footprints
moved onto their roofs, 255 = ignore), <name>.json (class shares, shifts),
<name>_footprints.parquet (the aligned outlines), data_v2/masks_v2t/<name>.png
(v2 + labels_v2.teacher_ignore) and a QC overlay in data_v2/preview_v2/.
Water relations the snapshot names without geometry (osm_data: "out tags")
are fetched once into data_v2/osm/<name>_water.json.
"""
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urbanseg import config, imagery, labels, labels_v2, open_buildings, osm_data  # noqa: E402

V2 = config.DATA_V2_DIR
STEP = sys.argv[1] if len(sys.argv) > 1 else "all"
ONLY = set(sys.argv[2:])
AREAS = [a for a in config.AREAS_V2 if not ONLY or a[0] in ONLY]
# v1 areas take part in teacher and masks_v2 only
WITH_OLD = ("all", "teacher", "masks_v2")
if ONLY:
    OLD = [a for a in config.AREAS if a[0] in ONLY]
else:
    OLD = list(config.AREAS) if STEP in WITH_OLD else []
if ONLY - {a[0] for a in AREAS + OLD}:
    sys.exit(f"unknown areas: {sorted(ONLY - {a[0] for a in AREAS + OLD})}")
if OLD and STEP not in WITH_OLD:
    sys.exit(f"only {', '.join(WITH_OLD[1:])} take v1 areas: {sorted(a[0] for a in OLD)}")
# every write of the reused v1 code goes to data_v2/
config.OSM_DIR = V2 / "osm"
config.MASKS_DIR = V2 / "masks_v1"
config.PREVIEW_DIR = V2 / "preview"
for d in (config.OSM_DIR, config.MASKS_DIR, config.PREVIEW_DIR):
    assert V2 in d.parents, d


def meta_of(name):
    return json.loads((V2 / "tiles" / name / "meta.json").read_text())


def osm_snapshot(name):
    """The OSM answer an area's labels are built from (v1 areas: data/osm, read-only)."""
    old = any(a[0] == name for a in config.AREAS)
    d = config.DATA_DIR / "osm" if old else V2 / "osm"
    return json.loads((d / f"{name}_v2.json").read_text())


def water_relations(name, bbox, main):
    """Members of the water relations `main` names without geometry, fetched once."""
    if not any(e["type"] == "relation" and not e.get("members") for e in main["elements"]):
        return []
    data = osm_data.fetch_overpass(bbox, cache_path=V2 / "osm" / f"{name}_water.json",
                                   hedge_s=15, timeout=300, pause=20,
                                   query=osm_data.QUERY_WATER_RELATIONS)
    return data["elements"]


def teacher_path(name):
    return V2 / "teacher" / f"{name}.png"


def save_png(path, arr):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(arr).save(tmp, format="PNG")
    os.replace(tmp, path)


def preview_v2(img, mask, step=3, alpha=0.45):
    """Every step-th px, classes tinted, ignore darkened: for eyeballing labels."""
    im = img[::step, ::step].astype(np.float32)
    m = mask[::step, ::step]
    out = im.copy()
    for k, rgb in config.CLASS_COLORS.items():
        sel = m == k
        out[sel] = (1 - alpha) * im[sel] + alpha * np.array(rgb, np.float32)
    out[m == labels_v2.IGNORE] *= 0.35
    return out.clip(0, 255).astype(np.uint8)


def build_v2(name):
    img, meta = imagery.load_area(name)
    bbox = imagery.bbox_latlng(meta)
    main = osm_snapshot(name)
    rel = water_relations(name, bbox, main)
    layers = osm_data.parse_overpass({**main, "elements": main["elements"] + rel}, version="v3")
    fp = open_buildings.load_for_bbox(bbox)
    mask, aligned = labels_v2.build_mask_v2(img, meta, layers, fp)
    out = V2 / "masks_v2"
    for d in (out, V2 / "masks_v2t", V2 / "preview_v2"):
        d.mkdir(parents=True, exist_ok=True)
    mask_t = None
    if teacher_path(name).exists():
        p_bld = np.asarray(Image.open(teacher_path(name)), np.float32) / 255.0
        mask_t = mask.copy()
        mask_t[labels_v2.teacher_ignore(img, mask, p_bld, meta)] = labels_v2.IGNORE
    inside = aligned[aligned.shift_source != "none"]
    mpp = imagery.meters_per_map_px(meta["center"][0], meta["zoom"]) / meta["scale"]
    stats = {
        "area": name,
        "shape": list(mask.shape),
        "classes": {config.CLASS_NAMES[k]: round(float((mask == k).mean()), 4)
                    for k in range(config.N_CLASSES)},
        "ignore": round(float((mask == labels_v2.IGNORE).mean()), 4),
        "footprints": len(aligned),
        "shift_source": aligned.shift_source.value_counts().to_dict(),
        "median_shift_m": [round(float(np.median(inside.shift_dx_px)) * mpp, 2),
                           round(float(np.median(inside.shift_dy_px)) * mpp, 2)]
        if len(inside) else None,
        "water_relations": len(rel),
        "teacher_ignore": (round(float(((mask_t == labels_v2.IGNORE) & (mask == 3)).mean()), 4)
                           if mask_t is not None else None),
    }
    aligned.to_parquet(out / f"{name}_footprints.parquet")
    (out / f"{name}.json").write_text(json.dumps(stats, indent=1))
    shown = mask if mask_t is None else mask_t
    Image.fromarray(preview_v2(img, shown)).save(V2 / "preview_v2" / f"{name}.jpg", quality=85)
    if mask_t is not None:
        save_png(V2 / "masks_v2t" / f"{name}.png", mask_t)
    save_png(out / f"{name}.png", mask)  # last: its presence marks the area done
    return stats


if STEP in ("all", "tiles"):
    for name, lat, lng, nx, ny, split, tex in AREAS:
        d = V2 / "tiles" / name
        if (d / "meta.json").exists() and (V2 / "tiles" / f"{name}_full.png").exists():
            print(f"[tiles] {name}: cached", flush=True)
            continue
        try:
            img, meta = imagery.fetch_area(lat, lng, nx, ny, cache_dir=d, name=name)
            print(f"[tiles] {name}: {img.size} {split} {tex}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[tiles] FAILED {name}: {imagery.redact(ex)}", flush=True)

if STEP in ("all", "osm"):
    for name, *_ in AREAS:
        try:
            bbox = imagery.bbox_latlng(meta_of(name))
            # v2 query, as the v1 labels used; mirrors hedged (the public ones are
            # often overloaded: one sequential round could take 18 min)
            layers = osm_data.get_area_osm(name, bbox, hedge_s=15, timeout=300, pause=20)
            print(f"[osm] {name}: b={len(layers['buildings'])} roads={len(layers['roads'])} "
                  f"water={len(layers['water'])} pois={len(layers['pois'])}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[osm] FAILED {name}: {imagery.redact(ex)}", flush=True)

if STEP in ("all", "masks"):
    for name, *_ in AREAS:
        try:
            mask, fp, cls = labels.build_mask(name, meta_of(name))
            frac = {config.CLASS_NAMES[k]: round(float((mask == k).mean()), 3)
                    for k in range(config.N_CLASSES)}
            print(f"[mask v1-rules] {name}: {frac} buildings={len(fp)}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[mask] FAILED {name}", flush=True)
            traceback.print_exc()

if STEP in ("all", "teacher"):
    from urbanseg import infer
    ckpt = config.MODELS_DIR / "unet_r34.pt"
    (V2 / "teacher").mkdir(parents=True, exist_ok=True)
    info = {"checkpoint": ckpt.name, "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "predict": "infer.predict defaults (gaussian blend), no TTA",
            "value": "round(255 * (P(residential) + P(commercial)))"}
    known = V2 / "teacher" / "teacher.json"
    if known.exists() and json.loads(known.read_text())["sha256"] != info["sha256"]:
        sys.exit(f"{known} was made with another checkpoint; move data_v2/teacher away first")
    known.write_text(json.dumps(info, indent=1))
    model = None
    for name, *_ in OLD + AREAS:
        if teacher_path(name).exists():
            print(f"[teacher] {name}: cached", flush=True)
            continue
        if model is None:
            model = infer.load_model(ckpt, device=os.environ.get("URBANSEG_DEVICE", "auto"))
        img, _ = imagery.load_area(name)
        _, probs = infer.predict(model, img)
        p_bld = np.clip(np.rint(255 * (probs[1] + probs[2])), 0, 255).astype(np.uint8)
        save_png(teacher_path(name), p_bld)
        print(f"[teacher] {name}: P(building) >= 0.5 on {np.mean(p_bld >= 128):.3f}", flush=True)

if STEP in ("all", "masks_v2"):
    for name, *_ in OLD + AREAS:
        if ((V2 / "masks_v2" / f"{name}.png").exists()
                and (V2 / "masks_v2t" / f"{name}.png").exists()):
            print(f"[mask v2] {name}: cached", flush=True)
            continue
        try:
            s = build_v2(name)
            print(f"[mask v2] {name}: {s['classes']} ignore={s['ignore']} "
                  f"teacher={s['teacher_ignore']} shift_m={s['median_shift_m']} "
                  f"{s['shift_source']} water_rel={s['water_relations']}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[mask v2] FAILED {name}", flush=True)
            traceback.print_exc()

print("done", flush=True)
