#!/usr/bin/env python
"""Segment a Hyderabad satellite view into residential / commercial / free land.

Usage:
  # by coordinates (fetches imagery with GOOGLE_MAPS_KEY from .env, full fusion):
  python scripts/segment.py --lat 17.4435 --lng 78.3772 [--grid 2]

  # by image file (no geo metadata -> model only, no OSM/footprint fusion):
  python scripts/segment.py --image path/to/satellite.png

  # options:
  #   --out DIR    output directory (default: outputs/<timestamp>)
  #   --no-tta     disable test-time augmentation (4x faster, ~1pp less accurate)
  #   --no-fuse    disable footprint/vector fusion for coordinate queries
  #   --model P    checkpoint path (default models/unet_r34.pt)

Outputs in --out:
  overlay.png    color overlay  (orange=residential, red=commercial,
                                 green=free land, gray=roads/water/rail)
  mask.png       raw class ids (0 other, 1 residential, 2 commercial, 3 free land)
  result.geojson polygons per class in WGS84 (coordinate queries only)
  stats.json     per-class pixel share and area in m2
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from urbanseg import config, imagery, infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lng", type=float)
    ap.add_argument("--image")
    ap.add_argument("--grid", type=int, default=2,
                    help="tiles per side for coordinate queries (2 = ~0.7x0.7 km)")
    ap.add_argument("--out")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-fuse", action="store_true")
    ap.add_argument("--model")
    args = ap.parse_args()
    if not args.image and (args.lat is None or args.lng is None):
        ap.error("give either --image or both --lat and --lng")

    out = Path(args.out or f"outputs/{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    model = infer.load_model(args.model)
    pred = infer.predict if args.no_tta else infer.predict_tta

    if args.image:
        img = np.asarray(Image.open(args.image).convert("RGB"))
        classmap, probs = pred(model, img)
        meta = {"zoom": config.ZOOM, "scale": config.SCALE, "x0": 0, "y0": 0,
                "center": [17.4, 78.48],
                "width_map_px": img.shape[1] // config.SCALE,
                "height_map_px": img.shape[0] // config.SCALE}
        geojson = None
        print("note: image input has no geolocation; OSM/footprint fusion off, "
              "area m2 assumes zoom-18 scale-2 (~0.28 m/px) imagery")
    else:
        pil, meta = imagery.fetch_area(args.lat, args.lng, args.grid, args.grid)
        img = np.asarray(pil)
        classmap, probs = pred(model, img)
        if not args.no_fuse:
            classmap = infer.fuse_footprints(classmap, probs, meta)
        geojson = infer.to_geojson(classmap, meta)

    stats = infer.area_stats(classmap, meta)
    overlay = infer.render_overlay(img, classmap)
    Image.fromarray(infer.add_legend(overlay, stats)).save(out / "overlay.png")
    Image.fromarray(classmap).save(out / "mask.png")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    if geojson is not None:
        (out / "result.geojson").write_text(json.dumps(geojson))

    print(f"\nwrote {out}/")
    for name, s in stats.items():
        print(f"  {name:<12} {s['share']*100:5.1f}%   {s['area_m2']:>12,.0f} m2")


if __name__ == "__main__":
    main()
