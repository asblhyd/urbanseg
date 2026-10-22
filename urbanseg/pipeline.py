"""One analysis run, shared by the web API (jobs.py) and the command line.

run_area():   a planned area (aoi.plan) -> imagery -> model -> OSM -> fusion
              -> stats for the zone and the whole extent, layers, exports.
run_upload(): an uploaded image (uploads.load_upload) -> model -> fusion when
              it is georeferenced -> stats, layers, exports.

Both report progress per stage, can be cancelled between steps (Cancelled),
and return (summary, files): a JSON-ready summary and {name: File} with the
bytes a client downloads. Files are built in memory and never written to disk
unless the caller does so (the CLI writes them, the API keeps them in RAM).
"""
import csv
import io
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from rasterio.io import MemoryFile

from . import aoi, config, imagery, infer, osm_data, uploads

Cancelled = infer.Cancelled
LOW_CONFIDENCE = 0.6  # max softmax under this is shown hatched as low confidence
# share of the progress bar per stage (CPU: the model dominates)
STAGE_WEIGHTS = {"decode": 0.08, "imagery": 0.12, "model": 0.6, "osm": 0.1,
                 "fusion": 0.05, "render": 0.05}
STAGE_LABELS = {"decode": "Reading the upload", "imagery": "Fetching satellite imagery",
                "model": "Running the model", "osm": "Fetching OpenStreetMap",
                "fusion": "Fusing with map data", "render": "Preparing results"}
UNITS = ("m2", "ha", "acres", "sq_yd", "guntas")
# public Overpass mirrors answered our query in 12-35 s on 2026-10-22, when
# they answered at all; the fetch runs alongside the imagery and the model
OSM_DEADLINE_S = float(os.environ.get("URBANSEG_OSM_DEADLINE_S", "60"))
# after OSM_BACKOFF_AFTER outages in a row (every mirror failed), later runs
# use cached OSM only for OSM_BACKOFF_S instead of waiting out the deadline
OSM_BACKOFF_AFTER = int(os.environ.get("URBANSEG_OSM_BACKOFF_AFTER", "2"))
OSM_BACKOFF_S = float(os.environ.get("URBANSEG_OSM_BACKOFF_S", "120"))
_osm_down_until = 0.0
_osm_outages = 0


@dataclass
class File:
    """A downloadable result: bytes, or a function that builds them on request."""
    media_type: str
    data: bytes | None = None
    build: Callable[[], bytes] | None = None
    filename: str | None = None

    def bytes(self):
        if self.data is None:
            self.data = self.build()
        return self.data


class Progress:
    """Turns per-stage (done, total) into one 0..1 fraction for a progress bar."""

    def __init__(self, report, stages):
        self.report = report or (lambda **kw: None)
        total, acc, self.span = sum(STAGE_WEIGHTS[s] for s in stages), 0.0, {}
        for s in stages:
            self.span[s] = (acc / total, STAGE_WEIGHTS[s] / total)
            acc += STAGE_WEIGHTS[s]

    def __call__(self, stage, done=0, total=1):
        start, width = self.span[stage]
        self.report(stage=stage, label=STAGE_LABELS[stage], done=done, total=total,
                    fraction=round(start + width * min(1.0, done / max(total, 1)), 4))


def _check(cancel):
    if cancel is not None and cancel():
        raise Cancelled("cancelled")


def run_area(model, plan, *, fuse=True, quality="standard", progress=None, cancel=None,
             title=None):
    """Analyse a planned area. plan: aoi.plan()'s dict (its "meta" included)."""
    t0 = time.time()
    timings = {}
    stages = ["imagery", "model"] + (["osm", "fusion"] if fuse else []) + ["render"]
    prog = Progress(progress, stages)

    osm = _OsmFetch(imagery.bbox_latlng(plan["meta"])) if fuse else None
    prog("imagery", 0, plan["calls"])
    lat, lng = plan["center"]
    pil, meta = imagery.fetch_area(lat, lng, plan["nx"], plan["ny"], cancel=cancel,
                                   progress=lambda d, n: prog("imagery", d, n))
    img = np.asarray(pil)
    timings["imagery"] = time.time() - t0

    classmap, probs, raw, fusion, notes = _model_and_fusion(
        model, img, meta, fuse=fuse, quality=quality, prog=prog, cancel=cancel,
        timings=timings, osm=osm)

    t = time.time()
    prog("render")
    zone = aoi.aoi_mask(plan["aoi"], meta, classmap.shape) if plan.get("aoi") else None
    if not plan.get("in_coverage", True):
        notes.append("This location is outside greater Hyderabad, where the model was "
                     "trained: treat the result as indicative.")
    summary = _summary("area", title, meta, classmap, zone, fusion, quality, notes,
                       list(plan.get("warnings", [])))
    summary["model"] = getattr(model, "info", None)  # id, file, sha256
    summary["plan"] = {k: v for k, v in plan.items() if k != "meta"}
    summary["aoi"] = plan.get("aoi")
    summary["stats_model_only"] = _model_only_stats(raw, meta, zone, fusion)
    files = _files(img, classmap, raw, probs, meta, zone, summary,
                   geo=True, aoi_geom=plan.get("aoi"))
    timings["render"] = time.time() - t
    summary["timings"] = {k: round(v, 2) for k, v in timings.items()}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    files["result.json"] = File("application/json", json.dumps(summary, indent=1).encode())
    return summary, files


def run_upload(model, upload, *, fuse=True, quality="standard", progress=None, cancel=None,
               title=None, timings=None):
    """Analyse an uploaded image. upload: uploads.load_upload()'s dict."""
    t0 = time.time()
    timings = dict(timings or {})
    geo = upload["georeferenced"]
    fuse = fuse and geo  # no location, no map data
    stages = ["model"] + (["osm", "fusion"] if fuse else []) + ["render"]
    prog = Progress(progress, stages)
    meta = upload["meta"]
    osm = _OsmFetch(imagery.bbox_latlng(meta)) if fuse else None
    classmap, probs, raw, fusion, notes = _model_and_fusion(
        model, upload["img"], meta, fuse=fuse, quality=quality, prog=prog, cancel=cancel,
        timings=timings, valid=upload.get("valid"), osm=osm)

    t = time.time()
    prog("render")
    if not geo:
        notes.append("An image without coordinates is analysed by the model alone: no "
                     "OpenStreetMap or footprint data can be matched to it.")
    elif not _in_coverage(meta["center"]):
        notes.append("This image is outside greater Hyderabad, where the model was "
                     "trained: treat the result as indicative.")
    summary = _summary("upload", title, meta, classmap, None, fusion, quality, notes,
                       list(upload.get("warnings", [])))
    summary["model"] = getattr(model, "info", None)
    summary["stats_model_only"] = _model_only_stats(raw, meta, None, fusion)
    summary["upload"] = {k: upload.get(k) for k in (
        "georeferenced", "source_mpp", "scale_source", "scale_note", "original_size",
        "native_size", "crs", "format")}
    files = _files(upload["img"], classmap, raw, probs, meta, None, summary, geo=geo)
    timings["render"] = time.time() - t
    summary["timings"] = {k: round(v, 2) for k, v in timings.items()}
    summary["elapsed_s"] = round(time.time() - t0 + timings.get("decode", 0), 1)
    files["result.json"] = File("application/json", json.dumps(summary, indent=1).encode())
    return summary, files


def _model_and_fusion(model, img, meta, *, fuse, quality, prog, cancel, timings,
                      valid=None, osm=None):
    t = time.time()
    prog("model")
    classmap, probs = infer.predict_quality(model, img, quality, cancel=cancel,
                                            progress=lambda d, n: prog("model", d, n))
    timings["model"] = time.time() - t
    if valid is not None:  # no data is never a class: not in fusion, stats or exports
        classmap[~valid] = infer.NODATA
    raw = classmap.copy()
    notes = []
    fusion = {"active": False, "sources": [], "mode": None, "osm": None, "changed_px": None}
    if not fuse:
        return classmap, probs, raw, fusion, notes
    _check(cancel)
    t = time.time()
    prog("osm")
    if osm is None:
        osm = _OsmFetch(imagery.bbox_latlng(meta))
    layers, osm_info = osm.result(notes, cancel)
    timings["osm"] = time.time() - t
    _check(cancel)
    t = time.time()
    prog("fusion")
    classmap = infer.fuse_footprints(classmap, probs, meta, use_osm=layers is not None,
                                     layers=layers, osm_info=osm_info)
    timings["fusion"] = time.time() - t
    info = meta.get("fusion_info") or {}
    fusion = {"active": bool(meta.get("fusion")), "sources": list(meta.get("fusion", [])),
              "mode": info.get("mode"), "osm": _osm_public(osm_info),
              "changed_px": info.get("changed_px")}
    if info.get("error"):
        notes.append("Fusion with map data failed for this area, so the model's output is "
                     "shown unchanged.")
        fusion["active"] = False
    elif "footprints" not in fusion["sources"]:
        notes.append("No building-footprint index is installed: building classes come from "
                     "the model and OpenStreetMap tags only (./run.sh --with-footprints "
                     "adds Google Open Buildings).")
    return classmap, probs, raw, fusion, notes


def _live_osm(bbox, notes):
    """(layers, info) from osm_data.get_live_osm, or (None, info) with a note."""
    global _osm_down_until, _osm_outages
    backing_off = time.time() < _osm_down_until
    try:  # deadline 0 still serves an area's cached answer
        out = osm_data.get_live_osm(bbox, deadline_s=0 if backing_off else OSM_DEADLINE_S)
        _osm_outages = 0
        return out
    except osm_data.OverpassError as ex:
        if backing_off:
            mins = max(1, round((_osm_down_until - time.time()) / 60))
            notes.append("OpenStreetMap was unreachable just before, so it was skipped: roads, "
                         f"water and building tags were not used (retried in ~{mins} min).")
        else:
            if ex.reason in ("timeout", "unavailable"):
                _osm_outages += 1
                if _osm_outages >= OSM_BACKOFF_AFTER:
                    _osm_down_until = time.time() + OSM_BACKOFF_S
            notes.append("OpenStreetMap could not be reached, so roads, water and building "
                         "tags were not used. Try again in a few minutes.")
        return None, {"source": None, "reason": ex.reason}


class _OsmFetch:
    """Live OSM fetched in a background thread while imagery and the model run."""

    def __init__(self, bbox):
        self._notes, self._out, self._done = [], (None, None), threading.Event()
        threading.Thread(target=self._run, args=(bbox,), daemon=True, name="osm").start()

    def _run(self, bbox):
        try:
            self._out = _live_osm(bbox, self._notes)
        except Exception:  # noqa: BLE001 - fuse without OSM rather than fail the run
            self._notes.append("OpenStreetMap data could not be used for this area.")
        finally:
            self._done.set()

    def result(self, notes, cancel=None):
        while not self._done.wait(0.2):
            _check(cancel)
        notes.extend(self._notes)
        return self._out


def _osm_public(info):
    if not info:
        return None
    return {k: info.get(k) for k in ("source", "osm_base", "fetched_at", "elements", "reason")
            if info.get(k) is not None}


def _in_coverage(center):
    w, s, e, n = config.HYD_BBOX
    return s <= center[0] <= n and w <= center[1] <= e


def _stats(classmap, meta, mask):
    st = infer.area_stats(classmap, meta, mask)
    for v in st.values():
        v["units"] = aoi.area_units(v["area_m2"])
    total = sum(v["area_m2"] for v in st.values())
    return {"classes": st, "total": {"pixels": int(sum(v["pixels"] for v in st.values())),
                                     "area_m2": round(total, 1),
                                     "units": aoi.area_units(total)}}


def _model_only_stats(raw, meta, zone, fusion):
    """The model's own figures, to compare with the fused ones (None without fusion)."""
    if not fusion["active"]:
        return None
    return {"zone": _stats(raw, meta, zone) if zone is not None else None,
            "extent": _stats(raw, meta, None)}


def _summary(kind, title, meta, classmap, zone, fusion, quality, notes, warnings):
    h, w = classmap.shape
    out = {
        "kind": kind, "title": title,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "quality": quality,
        "classes": [{"id": k, "name": n, "color": list(config.CLASS_COLORS[k])}
                    for k, n in enumerate(config.CLASS_NAMES)],
        "georeferenced": bool(meta.get("georeferenced", kind == "area")),
        "image_size": [w, h],
        "grid": {k: meta[k] for k in ("zoom", "scale", "x0", "y0")},
        "bbox_wsen": list(imagery.bbox_latlng(meta)),
        "center": list(meta["center"]),
        "stats": {"zone": _stats(classmap, meta, zone) if zone is not None else None,
                  "extent": _stats(classmap, meta, None)},
        "fusion": fusion,
        "notes": notes, "warnings": warnings,
    }
    if not out["georeferenced"]:  # a pseudo grid: no position to show
        out["bbox_wsen"] = out["center"] = None
    return out


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _jpeg(img, quality=85):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def lowconf_png(probs, mask=None):
    """RGBA PNG hatching the pixels the model is unsure of (max prob < LOW_CONFIDENCE)."""
    low = probs.max(0) < LOW_CONFIDENCE
    if mask is not None:
        low &= mask
    h, w = low.shape
    yy, xx = np.ogrid[:h, :w]
    stripe = ((xx + yy) // 3) % 4 == 0  # diagonal hatch, 1 px in 4
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[low & stripe] = (255, 255, 255, 220)
    rgba[low & ~stripe] = (0, 0, 0, 40)
    return _png(rgba)


def geotiff(classmap, meta, mask=None):
    """Class raster as a GeoTIFF on the image's Web-Mercator grid (EPSG:3857),
    with a colour table; 255 = no data (outside the zone, or invalid)."""
    data = classmap if mask is None else np.where(mask, classmap, infer.NODATA)
    h, w = data.shape
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", width=w, height=h, count=1, dtype="uint8",
                     crs="EPSG:3857", transform=uploads.grid_transform(meta),
                     nodata=infer.NODATA, compress="deflate", tiled=True) as dst:
            dst.write(data.astype(np.uint8), 1)
            dst.write_colormap(1, {k: tuple(c) + (255,) for k, c in config.CLASS_COLORS.items()})
            dst.update_tags(1, **{f"class_{k}": n for k, n in enumerate(config.CLASS_NAMES)})
        return mf.read()


def stats_csv(summary):
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["scope", "class", "share", "pixels", *("area_" + u for u in UNITS)])
    for scope in ("zone", "extent"):
        st = summary["stats"][scope]
        if st is None:
            continue
        for name, v in st["classes"].items():
            wr.writerow([scope, name, v["share"], v["pixels"], *(v["units"][u] for u in UNITS)])
        wr.writerow([scope, "total", 1.0 if st["total"]["pixels"] else 0, st["total"]["pixels"],
                     *(st["total"]["units"][u] for u in UNITS)])
    return buf.getvalue().encode()


def _overlay_from(files, legend_stats=None):
    """Full-resolution overlay rebuilt from a run's own imagery.jpg and mask.png,
    so a kept job holds compressed bytes, not the arrays (~30-70 MB per run)."""
    img = np.asarray(Image.open(io.BytesIO(files["imagery.jpg"].bytes())).convert("RGB"))
    cm = np.asarray(Image.open(io.BytesIO(files["mask.png"].bytes())))
    ov = infer.render_overlay(img, cm)  # NODATA (outside the zone) shows dimmed
    return _png(ov if legend_stats is None else infer.add_legend(ov, legend_stats))


def _files(img, classmap, raw, probs, meta, zone, summary, *, geo, aoi_geom=None):
    data = classmap if zone is None else np.where(zone, classmap, infer.NODATA)
    files = {
        "classes.png": File("image/png", infer.render_class_png(classmap, zone)),
        "raw.png": File("image/png", infer.render_class_png(raw, zone)),
        "lowconf.png": File("image/png", lowconf_png(probs, zone)),
        "imagery.jpg": File("image/jpeg", _jpeg(img)),
        # class ids 0-3, 255 = no data / outside the zone (as the CLI's mask.png)
        "mask.png": File("image/png", _png(data.astype(np.uint8))),
        # the same for the model alone (the gold editor's unbiased seed)
        "raw_mask.png": File("image/png", _png(
            (raw if zone is None else np.where(zone, raw, infer.NODATA)).astype(np.uint8))),
        "stats.csv": File("text/csv", stats_csv(summary)),
    }
    if zone is not None:  # the result around the zone too, for context
        files["classes_full.png"] = File("image/png", infer.render_class_png(classmap))
    stats = summary["stats"]["zone"] or summary["stats"]["extent"]
    # full-resolution overlays are large (MBs): built only when downloaded
    files["overlay.png"] = File("image/png", build=lambda: _overlay_from(
        files, {n: v for n, v in stats["classes"].items()}))
    files["overlay_plain.png"] = File("image/png", build=lambda: _overlay_from(files))
    if geo:
        files["classes.tif"] = File("image/tiff", geotiff(classmap, meta, zone))
        fc = infer.to_geojson(classmap, meta)
        if aoi_geom is not None:
            fc = aoi.clip_geojson(fc, aoi_geom)
        files["result.geojson"] = File("application/geo+json", json.dumps(fc).encode())
    summary["files"] = sorted(files) + ["result.json"]
    return files


def write_files(files, out_dir):
    """Write a run's files into out_dir (the CLI's output folder)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, f in files.items():
        (out_dir / name).write_bytes(f.bytes())

