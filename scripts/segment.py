#!/usr/bin/env python
"""Segment a Hyderabad satellite view into residential / commercial / free land.

Usage:
  # a zone: GeoJSON polygon file, or a bounding box west,south,east,north
  python scripts/segment.py --aoi plot.geojson
  python scripts/segment.py --bbox 78.3770,17.4430,78.3785,17.4442

  # a point and a square of grid x grid tiles (1 = ~0.4 km, 2 = ~0.7 km, 3 = ~1.1 km)
  python scripts/segment.py --lat 17.4435 --lng 78.3772 [--grid 2]
  python scripts/segment.py --location "17,4435; 78,3772"   # also Google Maps links

  # an image you already have (GeoTIFF with coordinates gets map-data fusion)
  python scripts/segment.py --image ortho.tif
  python scripts/segment.py --image shot.png --zoom 18 --dpr 2   # Google Maps screenshot
  python scripts/segment.py --image drone.jpg --mpp 0.03         # known m per pixel

  # options:
  #   --out DIR         output directory (default: outputs/<timestamp>)
  #   --quality Q       high (default: 8-way test-time augmentation) or standard (8x faster)
  #   --no-fuse         model only, no OpenStreetMap / footprint fusion
  #   --fusion-mode M   roof (default: colours stay on the roofs) or legacy
  #   --allow-outside   analyse a location outside greater Hyderabad
  #   --model M         model id (v1, v2: see config.MODELS) or a checkpoint path
  #                     (default: $URBANSEG_MODEL, else config.DEFAULT_MODEL)

Outputs in --out: overlay.png (with legend), mask.png (class ids 0 other,
1 residential, 2 commercial, 3 free land, 255 no data / outside the zone),
classes.tif (GeoTIFF, EPSG:3857), result.geojson (polygons per class, clipped
to the zone, with area_m2), stats.csv / result.json (shares and areas in m2,
ha, acres, sq yd and guntas), plus the map layers the web app uses.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from urbanseg import aoi, config, infer, pipeline, uploads  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = ap.add_argument_group("what to analyse (one of)")
    where.add_argument("--aoi", help="GeoJSON file with the zone (polygon)")
    where.add_argument("--bbox", help="zone as west,south,east,north in degrees")
    where.add_argument("--location", help="'lat, lng' text or a Google Maps link")
    where.add_argument("--lat", type=float)
    where.add_argument("--lng", type=float)
    where.add_argument("--image", help="an image file (GeoTIFF, PNG, JPEG, JPEG 2000)")
    ap.add_argument("--grid", type=int, default=2,
                    help="tiles per side around a point (1 ~0.4 km, 2 ~0.7 km, 3 ~1.1 km)")
    ap.add_argument("--margin-m", type=float, default=aoi.DEFAULT_MARGIN_M,
                    help="context fetched around a zone (m)")
    ap.add_argument("--mpp", type=float, help="image: ground metres per pixel")
    ap.add_argument("--zoom", type=float, help="image: Google Maps zoom of a screenshot")
    ap.add_argument("--dpr", type=float, help="image: device pixel ratio of a screenshot")
    ap.add_argument("--out")
    ap.add_argument("--quality", choices=infer.QUALITIES, default="high")
    ap.add_argument("--no-tta", action="store_true", help="same as --quality standard")
    ap.add_argument("--no-fuse", action="store_true")
    ap.add_argument("--fusion-mode", choices=("roof", "legacy"))
    ap.add_argument("--allow-outside", action="store_true")
    ap.add_argument("--model", help="model id (v1, v2) or checkpoint path")
    args = ap.parse_args()

    if args.fusion_mode:
        infer.FUSION_MODE = args.fusion_mode
    quality = "standard" if args.no_tta else args.quality
    fuse = not args.no_fuse
    out = Path(args.out or f"outputs/{time.strftime('%Y%m%d_%H%M%S')}")

    def report(stage=None, label=None, fraction=None, **_):
        print(f"\r  {label or stage}... {fraction * 100:5.1f}%", end="", flush=True)

    try:
        if args.image:
            raw = Path(args.image).read_bytes()
            up = uploads.load_upload(raw, args.image, mpp=args.mpp, zoom=args.zoom, dpr=args.dpr)
            for w in up["warnings"]:
                print(f"note: {w}")
            print(f"{up['scale_note']}")
            model = infer.load_model(args.model)
            summary, files = pipeline.run_upload(model, up, fuse=fuse, quality=quality,
                                                 progress=report, title=Path(args.image).name)
        else:
            plan = _plan(args, ap)
            for w in plan["warnings"]:
                print(f"note: {w}")
            print(f"{plan['nx']} x {plan['ny']} tiles ({plan['calls']} Google Static Maps "
                  f"calls), {plan['extent_m'][0]:.0f} x {plan['extent_m'][1]:.0f} m")
            if not config.GOOGLE_MAPS_KEY:
                sys.exit("error: set GOOGLE_MAPS_KEY in .env to fetch imagery")
            model = infer.load_model(args.model)
            summary, files = pipeline.run_area(model, plan, fuse=fuse, quality=quality,
                                               progress=report)
    except (aoi.AOIError, uploads.UploadError) as ex:
        sys.exit(f"error: {ex}")
    print()

    pipeline.write_files(files, out)
    # stats.json as before: {class: {pixels, area_m2, share}} (+ units now)
    st = summary["stats"]["zone"] or summary["stats"]["extent"]
    (out / "stats.json").write_text(json.dumps(st["classes"], indent=2))
    for n in summary["notes"]:
        print(f"note: {n}")
    scope = "zone" if summary["stats"]["zone"] else "image"
    print(f"\nwrote {out}/   ({scope}: {st['total']['units']['ha']:.2f} ha, "
          f"{summary['elapsed_s']} s, {quality} quality, "
          f"{'fused with ' + ' + '.join(summary['fusion']['sources']) if summary['fusion']['active'] else 'model only'})")
    for name, v in st["classes"].items():
        print(f"  {name:<12} {v['share'] * 100:5.1f}%   {v['area_m2']:>12,.0f} m2   "
              f"{v['units']['acres']:>9,.2f} acres")


def _plan(args, ap):
    kw = {"allow_outside": args.allow_outside}
    if args.aoi:
        return aoi.plan(json.loads(Path(args.aoi).read_text()), margin_m=args.margin_m, **kw)
    if args.bbox:
        try:
            box = [float(v) for v in args.bbox.split(",")]
        except ValueError:
            ap.error("--bbox needs four numbers: west,south,east,north")
        return aoi.plan(box, margin_m=args.margin_m, **kw)
    lat, lng = args.lat, args.lng
    if args.location:
        lat, lng = aoi.parse_latlng(args.location)
    if lat is None or lng is None:
        ap.error("give --aoi, --bbox, --location, --lat/--lng or --image")
    return aoi.plan(lat=lat, lng=lng, grid=args.grid, **kw)


if __name__ == "__main__":
    main()
