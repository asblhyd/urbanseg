"""Burn approved correction boxes into test ground-truth masks.

Usage: apply_corrections.py corrections.json
JSON: [{"area": name, "cell": "B3", "bbox_frac": [x0,y0,x1,y1] or null,
        "from_class": "free_land", "to_class": "other"}, ...]
A box repaints ONLY pixels currently equal to from_class ("any" = all).
Writes data/masks/{area}_gt.png (evaluation-only ground truth).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from urbanseg import config

GRID = 6
NAME2ID = {n: i for i, n in enumerate(config.CLASS_NAMES)}


def apply(corrections):
    by_area = {}
    for c in corrections:
        by_area.setdefault(c["area"], []).append(c)
    for area, corrs in by_area.items():
        mask = np.asarray(Image.open(config.MASKS_DIR / f"{area}.png")).copy()
        H, W = mask.shape
        ch, cw = H // GRID, W // GRID
        n_px = 0
        for c in corrs:
            cell = c["cell"].strip().upper()
            r, col = ord(cell[0]) - 65, int(cell[1:]) - 1
            y0, x0 = r * ch, col * cw
            bb = c.get("bbox_frac") or [0, 0, 1, 1]
            xa, ya = x0 + int(bb[0] * cw), y0 + int(bb[1] * ch)
            xb, yb = x0 + int(bb[2] * cw), y0 + int(bb[3] * ch)
            region = mask[ya:yb, xa:xb]
            if c["from_class"] == "any":
                sel = np.ones_like(region, dtype=bool)
            else:
                sel = region == NAME2ID[c["from_class"]]
            region[sel] = NAME2ID[c["to_class"]]
            n_px += int(sel.sum())
        Image.fromarray(mask).save(config.MASKS_DIR / f"{area}_gt.png")
        print(f"{area}: {len(corrs)} corrections, {n_px:,} px repainted "
              f"-> {area}_gt.png", flush=True)


if __name__ == "__main__":
    apply(json.loads(open(sys.argv[1]).read()))
