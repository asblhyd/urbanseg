"""Render gridded panels + per-cell crops for manual review of test ground truth."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from urbanseg import config, imagery, labels

GRID = 6  # 6x6 cells
CELL_OUT = 780  # per-pane width of cell crops


def render(name):
    img, _ = imagery.load_area(name)
    mask = np.asarray(Image.open(config.MASKS_DIR / f"{name}.png"))
    ov = labels.save_preview(name, mask)
    H, W = img.shape[:2]
    ch, cw = H // GRID, W // GRID
    outdir = config.PREVIEW_DIR / f"cells_{name}"
    outdir.mkdir(parents=True, exist_ok=True)

    # gridded full panel
    def gridded(arr):
        im = Image.fromarray(arr).convert("RGB")
        d = ImageDraw.Draw(im)
        for i in range(1, GRID):
            d.line([(i * cw, 0), (i * cw, H)], fill=(255, 255, 0), width=4)
            d.line([(0, i * ch), (W, i * ch)], fill=(255, 255, 0), width=4)
        for r in range(GRID):
            for c in range(GRID):
                cell = f"{chr(65 + r)}{c + 1}"
                d.text((c * cw + 14, r * ch + 10), cell, fill=(255, 255, 0),
                       font_size=60, stroke_width=3, stroke_fill=(0, 0, 0))
        return im

    pw = 980
    ph = int(H * pw / W)
    panel = Image.new("RGB", (pw * 2 + 10, ph), (255, 255, 255))
    panel.paste(gridded(img).resize((pw, ph)), (0, 0))
    panel.paste(gridded(ov).resize((pw, ph)), (pw + 10, 0))
    panel.save(config.PREVIEW_DIR / f"grid_{name}.png")

    # per-cell side-by-side crops at high res
    for r in range(GRID):
        for c in range(GRID):
            cell = f"{chr(65 + r)}{c + 1}"
            y0, x0 = r * ch, c * cw
            a = Image.fromarray(img[y0:y0 + ch, x0:x0 + cw])
            b = Image.fromarray(ov[y0:y0 + ch, x0:x0 + cw])
            hh = int(ch * CELL_OUT / cw)
            out = Image.new("RGB", (CELL_OUT * 2 + 8, hh), (255, 255, 255))
            out.paste(a.resize((CELL_OUT, hh)), (0, 0))
            out.paste(b.resize((CELL_OUT, hh)), (CELL_OUT + 8, 0))
            out.save(outdir / f"{cell}.png")
    print(f"{name}: grid panel + {GRID*GRID} cell crops -> {outdir}", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["madhapur", "begumpet", "kondapur"]):
        render(n)
