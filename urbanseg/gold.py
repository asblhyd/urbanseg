"""Gold set: hand-reviewed labels for measuring true accuracy.

The auto-labels (labels.py, labels_v2.py) carry their own errors, so scores
against them only say how well a model reproduces them. A gold sample is
imagery plus labels an analyst has reviewed in the web app's correction
editor:

  <config.GOLD_DIR>/<id>/image.png|image.jpg   the pixels the model is run on
  <config.GOLD_DIR>/<id>/labels.png            class ids 0-3, 255 = not reviewed
  <config.GOLD_DIR>/<id>/sample.json           where it came from, grid, seed, status

Samples come from a finished analysis (api: POST /api/gold) or from a window
of a dataset area, seeded with its auto-labels so that no model's output is
the starting point:

  python -m urbanseg.gold seed-area kondapur --window 1200 2000 640 [--labels v2t]
  python -m urbanseg.gold list
  python -m urbanseg.gold evaluate [--model v1 v2] [--tta 1|4|8] [--all]

evaluate scores reviewed samples only (--all: drafts too) with
evaluate.Scorer; unreviewed pixels (255) are never scored. A sample seeded
from a model's output favours that model where the analyst left it as it
was: sample.json records the seed, and evaluate reports scores per seed.
"""
import argparse
import io
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, dataset, imagery

IGNORE = 255
ID_RE = re.compile(r"g\d{14}_[0-9a-f]{6}")
# labels travel to and from the browser as grey levels this far apart, so a
# canvas's colour handling cannot turn one class into another
SPREAD = 60
MAX_SAMPLES = int(os.environ.get("URBANSEG_GOLD_MAX_SAMPLES", "2000"))
STATUSES = ("draft", "reviewed")


class GoldError(ValueError):
    """A sample that cannot be created or changed; .code for the API."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def gold_dir():
    return Path(config.GOLD_DIR)


# ------------------------------------------------------------------ labels

def encode_labels(labels):
    """PNG (RGB grey) of class ids: id * SPREAD, 255 = not reviewed."""
    lab = np.asarray(labels, np.uint8)
    grey = np.where(lab == IGNORE, 255, np.minimum(lab, 3) * SPREAD).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.repeat(grey[..., None], 3, axis=2)).save(buf, format="PNG")
    return buf.getvalue()


def decode_labels(raw, shape=None):
    """Class ids (uint8 HxW, 255 = not reviewed) from an encode_labels PNG.

    Grey levels snap to the nearest of 0, 60, 120, 180, 255; anything else
    far from them (another kind of image) is refused, as is a size that
    differs from shape (H, W)."""
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as ex:  # noqa: BLE001 - any decoder failure means "not a label image"
        raise GoldError("unreadable", "the labels are not a readable PNG") from ex
    grey = np.asarray(im.convert("RGB"), np.int16)[..., 0]
    if shape is not None and grey.shape != tuple(shape):
        raise GoldError("size_mismatch", f"labels are {grey.shape[1]}x{grey.shape[0]} px, "
                        f"the image {shape[1]}x{shape[0]}")
    levels = np.array([0, SPREAD, 2 * SPREAD, 3 * SPREAD, 255], np.int16)
    idx = np.abs(grey[..., None] - levels).argmin(-1)
    if np.abs(grey - levels[idx]).max() > SPREAD // 3:
        raise GoldError("unreadable", "the labels use grey levels the editor never writes")
    return np.array([0, 1, 2, 3, IGNORE], np.uint8)[idx]


def label_summary(labels):
    lab = np.asarray(labels)
    n = lab.size
    labelled = int(np.count_nonzero(lab != IGNORE))
    return {"labelled_share": round(labelled / n, 4) if n else 0.0,
            "classes": {config.CLASS_NAMES[k]: round(float(np.count_nonzero(lab == k)) /
                                                     max(labelled, 1), 4)
                        for k in range(config.N_CLASSES)}}


# ----------------------------------------------------------------- storage

def _new_id():
    return f"g{time.strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"


def _path(sample_id):
    if not isinstance(sample_id, str) or not ID_RE.fullmatch(sample_id):
        raise GoldError("not_found", "no such gold sample")
    p = gold_dir() / sample_id
    if not (p / "sample.json").is_file():
        raise GoldError("not_found", "no such gold sample")
    return p


def _write_atomic(path, data):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def create(image, labels, info, *, image_bytes=None, image_name="image.png"):
    """Store a new sample -> its info. image HxWx3 uint8 (or the encoded
    image_bytes as served, e.g. a job's JPEG); labels HxW uint8."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise GoldError("unreadable", "the image must be RGB")
    if labels.shape != image.shape[:2]:
        raise GoldError("size_mismatch", "labels and image differ in size")
    if info.get("status", "draft") not in STATUSES:
        raise GoldError("invalid", f"status must be one of {STATUSES}")
    root = gold_dir()
    root.mkdir(parents=True, exist_ok=True)
    if sum(1 for p in root.iterdir() if ID_RE.fullmatch(p.name)) >= MAX_SAMPLES:
        raise GoldError("full", f"the gold set already holds {MAX_SAMPLES} samples")
    sid = _new_id()
    tmp = root / f".{sid}.tmp"
    tmp.mkdir()
    if image_bytes is None:
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        image_bytes = buf.getvalue()
    (tmp / image_name).write_bytes(image_bytes)
    (tmp / "labels.png").write_bytes(encode_labels(labels))
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    info = {"id": sid, "status": "draft", "note": "", **info, "image": image_name,
            "size": [int(image.shape[1]), int(image.shape[0])], "created": now,
            "updated": now, **label_summary(labels)}
    (tmp / "sample.json").write_text(json.dumps(info, indent=1))
    os.replace(tmp, root / sid)
    return info


def info(sample_id):
    return json.loads((_path(sample_id) / "sample.json").read_text())


def samples():
    """Every sample's info, newest first."""
    root = gold_dir()
    if not root.is_dir():
        return []
    out = []
    for p in root.iterdir():
        if ID_RE.fullmatch(p.name) and (p / "sample.json").is_file():
            try:
                out.append(json.loads((p / "sample.json").read_text()))
            except (OSError, ValueError):
                continue
    return sorted(out, key=lambda s: s["id"], reverse=True)


def image_file(sample_id):
    p = _path(sample_id)
    return p / info(sample_id)["image"]


def load(sample_id):
    """(image HxWx3 uint8, labels HxW uint8, info)."""
    p = _path(sample_id)
    meta = info(sample_id)
    img = np.asarray(Image.open(p / meta["image"]).convert("RGB"))
    labels = decode_labels((p / "labels.png").read_bytes(), img.shape[:2])
    return img, labels, meta


def update(sample_id, labels_png, *, status=None, note=None):
    """New labels (encode_labels PNG) and optionally status / note -> info."""
    p = _path(sample_id)
    meta = info(sample_id)
    w, h = meta["size"]
    labels = decode_labels(labels_png, (h, w))
    if status is not None and status not in STATUSES:
        raise GoldError("invalid", f"status must be one of {STATUSES}")
    _write_atomic(p / "labels.png", encode_labels(labels))
    meta.update(label_summary(labels))
    if status is not None:
        meta["status"] = status
    if note is not None:
        meta["note"] = str(note)[:500]
    meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["edits"] = int(meta.get("edits", 0)) + 1
    _write_atomic(p / "sample.json", json.dumps(meta, indent=1).encode())
    return meta


def meta_of(sample):
    """The image grid of a sample (for evaluate.Scorer's metre tolerance)."""
    g = sample.get("grid") or {}
    w, h = sample["size"]
    scale = g.get("scale", config.SCALE)
    return {"zoom": g.get("zoom", config.ZOOM), "scale": scale,
            "x0": g.get("x0", 0.0), "y0": g.get("y0", 0.0),
            "width_map_px": w / scale, "height_map_px": h / scale,
            "center": sample.get("center") or [17.4, 78.48]}


# ------------------------------------------------------------ dataset areas

def seed_area(name, y0, x0, size, *, labels_version="v2t", note=""):
    """A draft sample: a size x size window of a dataset area, seeded with its
    auto-labels (IGNORE stays not reviewed)."""
    img, meta = imagery.load_area(name)
    lab = np.asarray(Image.open(dataset.mask_path(name, labels_version)))
    H, W = lab.shape
    if not (0 <= y0 < H and 0 <= x0 < W) or size < 64:
        raise GoldError("invalid", f"window outside {name} ({W}x{H} px) or under 64 px")
    y1, x1 = min(H, y0 + size), min(W, x0 + size)
    window = img[y0:y1, x0:x1].copy()
    s = meta["scale"]
    grid = {"zoom": meta["zoom"], "scale": s, "x0": meta["x0"] + x0 / s, "y0": meta["y0"] + y0 / s}
    sub = {**meta, **grid, "width_map_px": (x1 - x0) / s, "height_map_px": (y1 - y0) / s}
    w, south, e, n = imagery.bbox_latlng(sub)
    split = dict(dataset.area_splits(True)).get(name)
    return create(window, lab[y0:y1, x0:x1].copy(), {
        "title": f"{name} [{y0}:{y1}, {x0}:{x1}]",
        "source": {"kind": "area", "area": name, "split": split, "window": [y0, x0, y1, x1]},
        "seed": f"labels:{labels_version}", "grid": grid, "georeferenced": True,
        "center": [(south + n) / 2, (w + e) / 2], "note": note})


# --------------------------------------------------------------- evaluation

def evaluate_models(models, *, tta=1, include_drafts=False, device=None, out=print):
    """{model: {"pooled", "by_seed", "samples"}} of evaluate.summarize metrics."""
    from . import evaluate, infer
    chosen = [s for s in samples() if include_drafts or s["status"] == "reviewed"]
    if not chosen:
        raise GoldError("empty", "no reviewed gold samples yet"
                        + ("" if include_drafts else " (--all includes drafts)"))
    loaded = [load(s["id"]) for s in chosen]
    results = {}
    for m in models:
        net = infer.load_model(m, device=device)
        pooled, by_seed, per = None, {}, {}
        for img, lab, s in loaded:
            cm = (infer.predict(net, img)[0] if tta == 1
                  else infer.predict_tta(net, img, n=tta)[0])
            c = evaluate.Scorer(lab, meta_of(s)).counts(cm)["all"]
            per[s["id"]] = {"title": s.get("title"), "seed": s.get("seed"),
                            **evaluate.summarize(c)}
            pooled = _add(pooled, c)
            by_seed[s.get("seed")] = _add(by_seed.get(s.get("seed")), c)
        results[str(m)] = {"model": net.info, "pooled": evaluate.summarize(pooled),
                           "by_seed": {k: evaluate.summarize(v) for k, v in by_seed.items()},
                           "samples": per}
        p = results[str(m)]["pooled"]
        out(f"{m}: {len(loaded)} samples  acc={p['pixel_acc']} mIoU={p['mIoU']} "
            f"building BF={p['building_bf']['f1']} area ratio={p['building_area_ratio']}")
    return results


def _add(acc, c):
    return {k: v.copy() for k, v in c.items()} if acc is None else {k: acc[k] + c[k] for k in c}


def main(argv=None):
    ap = argparse.ArgumentParser(prog=f"python -m {__package__}.gold",
                                 description=f"Gold set in {gold_dir()}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the samples, newest first")
    sa = sub.add_parser("seed-area", help="a draft sample from a window of a dataset area")
    sa.add_argument("area")
    sa.add_argument("--window", nargs=3, type=int, required=True, metavar=("Y0", "X0", "SIZE"))
    sa.add_argument("--labels", default="v2t", help="auto-label version to start from")
    sa.add_argument("--note", default="")
    ev = sub.add_parser("evaluate", help="score models on the reviewed samples")
    ev.add_argument("--model", nargs="+", default=[None],
                    help="model ids or checkpoints (default: the default model)")
    ev.add_argument("--tta", type=int, default=1, choices=(1, 4, 8))
    ev.add_argument("--all", action="store_true", help="drafts too")
    ev.add_argument("--device")
    ev.add_argument("--json", help="also write the results here")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "list":
            for s in samples():
                print(f"{s['id']}  {s['status']:8s}  labelled {s['labelled_share']:.0%}  "
                      f"seed {s.get('seed')}  {s.get('title')}")
        elif a.cmd == "seed-area":
            s = seed_area(a.area, *a.window, labels_version=a.labels, note=a.note)
            print(f"created draft {s['id']}: {s['title']}; review it in the web app (Gold set)")
        else:
            res = evaluate_models(a.model, tta=a.tta, include_drafts=a.all, device=a.device)
            if a.json:
                Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    except GoldError as ex:
        sys.exit(f"gold: {ex}")


if __name__ == "__main__":
    main()
