"""Evaluate on held-out areas: raw model and footprint-fused (deployed) paths.

Run: python -m urbanseg.evaluate [split] [model_path] [--no-fuse]
"""
import json
import sys

import numpy as np
from PIL import Image

from . import config, dataset, imagery, infer
from .train import metrics_from_confusion


def eval_split(split="test", model_path=None, fuse=True, save_previews=True,
               tta=False):
    model = infer.load_model(model_path)
    results = {}
    cm_raw = np.zeros((config.N_CLASSES, config.N_CLASSES), dtype=np.int64)
    cm_fused = np.zeros_like(cm_raw)
    for name, img, mask, meta in dataset.load_split(split):
        pred_fn = infer.predict_tta if tta else infer.predict
        classmap, probs = pred_fn(model, img)
        idx = mask.ravel().astype(np.int64) * config.N_CLASSES + classmap.ravel()
        cm = np.bincount(idx, minlength=config.N_CLASSES ** 2).reshape(
            config.N_CLASSES, config.N_CLASSES)
        cm_raw += cm
        entry = {"raw_acc": float((classmap == mask).mean())}
        if fuse:
            fused = infer.fuse_footprints(classmap, probs, meta)
            idx = mask.ravel().astype(np.int64) * config.N_CLASSES + fused.ravel()
            cm_fused += np.bincount(idx, minlength=config.N_CLASSES ** 2).reshape(
                config.N_CLASSES, config.N_CLASSES)
            entry["fused_acc"] = float((fused == mask).mean())
            out_map = fused
        else:
            out_map = classmap
        if save_previews:
            ov = infer.render_overlay(img, out_map)
            Image.fromarray(ov).save(config.PREVIEW_DIR / f"pred_{name}.png")
        results[name] = entry
        print(f"{name}: {entry}", flush=True)

    def pack(cm):
        acc, ious, f1s = metrics_from_confusion(cm)
        return {"pixel_acc": round(acc, 4),
                "mIoU": round(float(np.mean(list(ious.values()))), 4),
                "iou": {k: round(v, 3) for k, v in ious.items()},
                "f1": {k: round(v, 3) for k, v in f1s.items()},
                "confusion": cm.tolist()}

    summary = {"split": split, "per_area": results, "raw": pack(cm_raw)}
    if fuse:
        summary["fused"] = pack(cm_fused)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / f"eval_{split}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_area"},
                     indent=2), flush=True)
    return summary


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    mp = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    eval_split(split, mp, fuse="--no-fuse" not in sys.argv,
               tta="--tta" in sys.argv)
