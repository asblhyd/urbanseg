"""Render side-by-side (raw | label overlay) QC panels for each area."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from urbanseg import config, imagery, labels

W = 980  # per-pane width in the panel


def panel(name):
    img, _ = imagery.load_area(name)
    mask = np.asarray(Image.open(config.MASKS_DIR / f"{name}.png"))
    ov = labels.save_preview(name, mask)
    h = int(img.shape[0] * W / img.shape[1])
    left = Image.fromarray(img).resize((W, h))
    right = Image.fromarray(ov).resize((W, h))
    out = Image.new("RGB", (W * 2 + 10, h), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (W + 10, 0))
    p = config.PREVIEW_DIR / f"qc_{name}.png"
    out.save(p)
    return p


if __name__ == "__main__":
    names = sys.argv[1:] or [a[0] for a in config.AREAS]
    for n in names:
        try:
            print(panel(n), flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"skip {n}: {ex}", flush=True)
