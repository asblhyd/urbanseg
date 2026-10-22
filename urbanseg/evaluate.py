"""Evaluate held-out areas: the model alone, and fusion scored for what it measures.

Run: python -m urbanseg.evaluate --split val [--model PATH] [--tta {1,4,8}]
         [--blend B] [--label-version V] [--use-corrected] [--mask-overlap]
         [--no-fuse] [--run-name NAME]
     (or the old form: python -m urbanseg.evaluate SPLIT [MODEL] [options])

The split has no default, so the test split is only scored when asked for.

Each run writes config.REPORTS_DIR/eval_runs/<run_id>/ (eval.json and
previews) and never reuses a run id, so no earlier result is overwritten;
nothing is written under data/. Fusion reads OSM from the data/osm snapshots
(never fetched) and footprints from the local index. Label pixels equal to
dataset.IGNORE are never scored.

eval.json, per area and pooled:
  model                   the model alone: pixel accuracy, per-class IoU/F1,
                          mIoU, a 3-class (other/building/free_land) score,
                          boundary-tolerant accuracy and building BF@1m;
  agreement_with_vectors  the fused map, scored the same way. The labels were
                          rasterized from the very OSM/Open Buildings vectors
                          that fusion pastes, so this is agreement with those
                          vectors, not accuracy;
  fusion_untouched        model and fused maps on the pixels outside every
                          fusion vector (footprints, road/rail/water buffers),
                          where fusion cannot copy the labels; scored as if
                          all other pixels were ignored.
"""
import argparse
import contextlib
import hashlib
import importlib
import inspect
import itertools
import json
import platform
import sys
import time

import numpy as np
import torch
from PIL import Image
from rasterio import features as rfeatures
from scipy import ndimage as ndi
from rasterio.transform import Affine
from shapely.ops import transform as shp_transform

from . import config, dataset, imagery, infer, labels, open_buildings, osm_data
from .train import (check_name, confusion, discard_on_failure, file_sha256, git_info,
                    plain_args)

TOLERANCE_M = 1.0  # boundary tolerance of tolerant accuracy and BF
# IGNORE narrower than ~2 x this (labels_v2's bands along building and road
# edges) is an uncertain edge, not a cut: building BF scores edges inside it
EDGE_BAND_PX = 4
THREE_CLASS = ["other", "building", "free_land"]
_TO_THREE = np.eye(3, dtype=np.int64)[[0, 1, 1, 2]]  # residential + commercial -> building
# The live OSM caches the published reports/eval_*.json were fused with.
# fuse_footprints keys them by hash(bbox), and lbnagar's bbox comes out 1 ulp
# different on macOS than on the Linux server that ran that evaluation, so
# the keys are recorded rather than recomputed.
EVAL_OSM_CACHES = {"lbnagar": "live_a9f296", "kompally": "live_19fb95",
                   "madhapur": "live_884c2a", "begumpet": "live_c2628e",
                   "kondapur": "live_6f2dc6"}
NOTES = {
    "labels": "auto-labels rasterized from Open Buildings footprints and OSM, not ground "
              "truth; footprints are ground-referenced and sit 1-8 m off leaning roofs, so "
              "output drawn on the roofs loses agreement near buildings even when right",
    "model": "the model alone against the labels",
    "agreement_with_vectors": "the fused map against the labels. Labels and fusion use the "
                              "same OSM/Open Buildings vectors, so this measures agreement with "
                              "those vectors, not accuracy",
    "fusion_untouched": "model and fused maps on the pixels outside every fusion vector "
                        "(footprints, road/rail/water buffers), where fusion cannot copy the "
                        "labels; every other pixel is treated as ignored",
    "boundary_tolerant_acc": "a pixel counts as correct when its predicted class occurs in the "
                             "labels within tolerance_px (1 m)",
    "building_bf": "boundary F1 of the building (residential + commercial) mask: its 1-px inner "
                   "edges matched within tolerance_px (1 m)",
    "train_overlap_scored_px": "scored pixels that lie inside a train area (val/test; "
                               "--mask-overlap ignores them)",
    "three_class": "residential and commercial merged into building",
}


# ---- metrics -----------------------------------------------------------------

def dilate(mask, r):
    """Binary dilation of a bool HxW array by a disk of radius r px (edges do not wrap)."""
    out = mask.copy()
    H, W = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dy or dx) and dy * dy + dx * dx <= r * r:
                out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] |= \
                    mask[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


def boundaries(m, valid):
    """bool HxW: inner edge of bool mask m (its valid pixels with a valid 4-neighbour outside m).

    Neighbours that are invalid (ignored) or off the image do not count, so
    an ignore region cuts no edges into the maps. The edge is one pixel
    thick, so a tolerance of r px forgives exactly r px of displacement
    (marking both sides of each change would forgive r + 1).
    """
    inside, outside = m & valid, ~m & valid
    b = np.zeros(m.shape, dtype=bool)
    b[:, 1:] |= inside[:, 1:] & outside[:, :-1]
    b[:, :-1] |= inside[:, :-1] & outside[:, 1:]
    b[1:] |= inside[1:] & outside[:-1]
    b[:-1] |= inside[:-1] & outside[1:]
    return b


def tolerance_px(meta, tol_m=TOLERANCE_M):
    """tol_m in whole image pixels: 1 m = 4 px at zoom 18 / scale 2."""
    mpp = imagery.meters_per_map_px(meta["center"][0], meta["zoom"]) / meta["scale"]
    return max(1, round(tol_m / mpp))


def _r(v):
    return None if v is None else round(float(v), 4)


def scores(cm, names=config.CLASS_NAMES):
    """Pixel accuracy, per-class IoU and F1, and mIoU of a confusion matrix (rows = labels).

    A class absent from both labels and prediction has no IoU/F1 (None) and
    is left out of mIoU.
    """
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = [tp[k] / union[k] if union[k] else None for k in range(len(names))]
    f1 = [2 * tp[k] / (union[k] + tp[k]) if union[k] else None for k in range(len(names))]
    defined = [v for v in iou if v is not None]
    return {"pixel_acc": _r(tp.sum() / cm.sum()) if cm.sum() else None,
            "mIoU": _r(np.mean(defined)) if defined else None,
            "iou": dict(zip(names, map(_r, iou))),
            "f1": dict(zip(names, map(_r, f1)))}


def summarize(c):
    """Metrics dict from (summed) Scorer.counts."""
    cm = c["cm"]
    out = scores(cm)
    out["three_class"] = scores(_TO_THREE.T @ cm @ _TO_THREE, THREE_CLASS)
    ok, n = c["tol"]
    out["boundary_tolerant_acc"] = _r(ok / n) if n else None
    hit_p, n_p, hit_g, n_g = c["bf"]
    p = hit_p / n_p if n_p else None
    rc = hit_g / n_g if n_g else None
    if not n_p and not n_g:
        f = 1.0  # neither map has a building edge
    else:
        p0, r0 = p or 0.0, rc or 0.0
        f = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    out["building_bf"] = {"f1": _r(f), "precision": _r(p), "recall": _r(rc)}
    # predicted / labelled building pixels (residential + commercial), on scored
    # pixels: above 1, buildings come out fatter than labelled (or there are more)
    labelled = cm[1:3, :].sum()
    out["building_area_ratio"] = _r(cm[:, 1:3].sum() / labelled) if labelled else None
    out["confusion"] = cm.tolist()
    return out


def edge_bands(gt):
    """(gt for edges, bool zone where edges are scored) of a label map.

    Thin IGNORE (what an opening with a disk of radius EDGE_BAND_PX removes:
    labels_v2 marks a few px on each side of building and road edges) takes
    the nearest scored label, so a building edge lies mid-band, where the
    outline is. Wider IGNORE (cuts, masked overlap, dry water) stays out.
    Labels without thin IGNORE (v1) come back unchanged.
    """
    ignore = gt >= config.N_CLASSES
    if not ignore.any() or ignore.all():
        return gt, ~ignore
    r = EDGE_BAND_PX
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    # padded with the edge values, so a region running off the image keeps its width
    wide = ndi.binary_opening(np.pad(ignore, r, mode="edge"), structure=yy * yy + xx * xx <= r * r)
    thin = ignore & ~wide[r:-r, r:-r]
    if not thin.any():
        return gt, ~ignore
    _, (iy, ix) = ndi.distance_transform_edt(ignore, return_indices=True)
    filled = gt.copy()
    filled[thin] = gt[iy[thin], ix[thin]]
    return filled, ~ignore | thin


class Scorer:
    """Scores class maps against one area's labels, on the whole area and on named regions.

    A region (bool HxW) is scored as if every pixel outside it were IGNORE,
    so its numbers depend on its own pixels only. The label-side work is done
    once per region. Building BF also scores edges inside thin IGNORE bands
    (edge_bands); every other metric skips all IGNORE pixels.
    """

    def __init__(self, gt, meta, regions=None, tol_m=TOLERANCE_M):
        self.gt = gt
        self.valid = gt < config.N_CLASSES
        self.tol_px = r = tolerance_px(meta, tol_m)
        gt_e, zone = edge_bands(gt)
        self._regions = {}
        for key, reg in {"all": None, **(regions or {})}.items():
            v = self.valid if reg is None else self.valid & reg
            ez = zone if reg is None else zone & reg
            gt_b = boundaries(np.isin(gt_e, (1, 2)), ez)
            near = [dilate((gt == k) & v, r) for k in range(config.N_CLASSES)]
            self._regions[key] = (v, near, gt_b, dilate(gt_b, r), ez)

    def counts(self, pred):
        """{region: counts} of pred ("all" = every scored pixel).

        Counts add up across areas; summarize(summed counts) pools them.
        """
        n = np.count_nonzero
        out = {}
        for key, (v, near, gt_b, near_gt_b, ez) in self._regions.items():
            tol_ok = np.zeros(pred.shape, dtype=bool)
            for k, near_k in enumerate(near):
                tol_ok |= (pred == k) & near_k
            pr_b = boundaries(np.isin(pred, (1, 2)), ez)
            out[key] = {"cm": confusion(self.gt[v], pred[v]),
                        "tol": np.array([n(tol_ok & v), n(v)]),
                        "bf": np.array([n(pr_b & near_gt_b), n(pr_b),
                                        n(gt_b & dilate(pr_b, self.tol_px)), n(gt_b)])}
        return out


def _add(acc, key, c):
    acc[key] = c if key not in acc else {k: acc[key][k] + v for k, v in c.items()}


# ---- inputs: predictions, OSM snapshots, fusion -------------------------------

def _predictor(tta, blend):
    """(model, img) -> (classmap, probs), passing only options infer supports."""
    if tta not in (1, 4, 8):
        raise ValueError(f"tta must be 1, 4 or 8, got {tta!r}")
    kw = {}
    if blend is not None:
        if "blend" in inspect.signature(infer.predict).parameters:
            kw["blend"] = blend
        elif blend != "uniform":  # an infer.predict without the option blends uniformly
            raise ValueError(f"this infer.predict has no blend option (got blend={blend!r})")
    if tta == 1:
        return lambda model, img: infer.predict(model, img, **kw)
    if "n" in inspect.signature(infer.predict_tta).parameters:
        return lambda model, img: infer.predict_tta(model, img, n=tta, **kw)
    if tta == 4:  # the original predict_tta always averages 4 flips/rotations
        return lambda model, img: infer.predict_tta(model, img, **kw)
    raise ValueError(f"tta={tta} needs an infer.predict_tta that takes n")


def _blend_used(blend):
    p = inspect.signature(infer.predict).parameters.get("blend")
    return "uniform" if p is None else (blend or p.default)


def osm_snapshot(name, meta):
    """(layers, provenance) of an area's OSM from data/osm, or (None, None).

    Read-only and never fetched. Prefers the live cache the published
    evaluation was fused with, then the one fuse_footprints would name for
    this bbox, then the label snapshot <name>_v2.json (in data_v2/osm for
    the version-2 areas).
    """
    bbox = imagery.bbox_latlng(meta)
    stems = [EVAL_OSM_CACHES.get(name), f"live_{hash(bbox) & 0xffffff:x}", name]
    paths = [config.DATA_DIR / "osm" / f"{s}_v2.json" for s in stems if s]
    paths.append(config.DATA_V2_DIR / "osm" / f"{name}_v2.json")
    for p in paths:
        if p.is_file():
            raw = p.read_bytes()
            data = json.loads(raw)
            return osm_data.parse_overpass(data), {
                "file": p.name, "osm_base": data.get("osm3s", {}).get("timestamp_osm_base"),
                "sha256": hashlib.sha256(raw).hexdigest()}
    return None, None


def fusion_vectors(footprints, layers, meta, shape):
    """bool HxW: pixels under any vector fusion can paste (footprints, road/rail/water buffers)."""
    zoom, s = meta["zoom"], meta["scale"]
    geoms = []
    if layers is not None:
        footprints = labels.merge_footprints(footprints, layers["buildings"])
        geoms += labels.other_geoms_world(layers, zoom)
    geoms += [shp_transform(lambda x, y: labels._world_arrays(x, y, zoom), g)
              for g in footprints.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool)
    tr = Affine(1.0 / s, 0, meta["x0"], 0, 1.0 / s, meta["y0"])
    return rfeatures.rasterize([(g, 1) for g in geoms], out_shape=shape, transform=tr,
                               fill=0, dtype="uint8").astype(bool)


def _fusion_module():
    """The fusion module when present (it supersedes infer.fuse_footprints), else None."""
    try:
        return importlib.import_module(".fusion", __package__)
    except ModuleNotFoundError as ex:
        if ex.name != f"{__package__}.fusion":
            raise
        return None


def check_fusion_mode(mode):
    """ValueError unless fuse_area can run mode (None: fusion's default).

    fusion.fuse never raises, so an unknown mode would otherwise only show
    as a fusion error after the first area was predicted.
    """
    fusion = _fusion_module()
    modes = getattr(fusion, "MODES", None) if fusion else ("legacy",)
    if mode is not None and modes is not None and mode not in modes:
        raise ValueError(f"fusion mode {mode!r} is not one of {tuple(modes)}"
                         + ("" if fusion else " (no fusion module: infer.fuse_footprints)"))


@contextlib.contextmanager
def _served(layers, footprints):
    """Make infer.fuse_footprints use these vectors instead of loading or fetching its own."""
    saved = osm_data.get_area_osm, open_buildings.load_for_bbox
    osm_data.get_area_osm = lambda *a, **k: layers
    open_buildings.load_for_bbox = lambda bbox: footprints
    try:
        yield
    finally:
        osm_data.get_area_osm, open_buildings.load_for_bbox = saved


def fuse_area(name, classmap, probs, meta, mode=None):
    """(fused classmap, untouched bool HxW, fusion info) for one area, without network.

    mode is passed to fusion.fuse (None = its default). Until the fusion
    module exists, infer.fuse_footprints (the "legacy" rules) is used.
    """
    footprints = open_buildings.load_for_bbox(imagery.bbox_latlng(meta))
    layers, osm = osm_snapshot(name, meta)
    m = dict(meta)
    fusion = _fusion_module()
    if fusion is not None:
        kw = {} if mode is None else {"mode": mode}
        fused = fusion.fuse(classmap, probs, m, use_osm=layers is not None, layers=layers,
                            footprints=footprints, osm_info=osm, **kw)
    elif mode not in (None, "legacy"):
        raise ValueError(f"fusion mode {mode!r} needs the fusion module")
    else:
        err = None
        with _served(layers, footprints):
            try:
                fused = infer.fuse_footprints(classmap, probs, m, use_osm=layers is not None)
            except Exception as ex:  # noqa: BLE001  a fusion failure must not end the run
                fused, err = classmap, f"{type(ex).__name__}: {ex}"
        m["fusion_info"] = {"mode": "legacy", "sources": list(m.get("fusion", [])), "osm": osm,
                            "changed_px": {"total": int(np.count_nonzero(fused != classmap))},
                            "error": err}
    untouched = ~fusion_vectors(footprints, layers, meta, classmap.shape)
    return fused, untouched, m.get("fusion_info") or {"sources": m.get("fusion", []), "osm": osm}


# ---- runs ----------------------------------------------------------------------

def run_path(run_name):
    """config.REPORTS_DIR/eval_runs/<run_name>; FileExistsError if that run exists."""
    run = config.REPORTS_DIR / "eval_runs" / check_name(run_name)
    if run.exists():
        raise FileExistsError(f"{run} already exists; choose a new --run-name")
    return run


def new_run_dir(split, run_name=None):
    """Create config.REPORTS_DIR/eval_runs/<run_id>/; an existing run id is never reused."""
    root = config.REPORTS_DIR / "eval_runs"
    root.mkdir(parents=True, exist_ok=True)
    if run_name:
        run = run_path(run_name)
        try:
            run.mkdir()  # atomic: also loses cleanly to a run started meanwhile
        except FileExistsError:
            raise FileExistsError(f"{run} already exists; choose a new --run-name") from None
        return run
    stem = f"{split}_{time.strftime('%Y%m%d_%H%M%S')}"
    for i in itertools.count(1):
        try:
            (run := root / (stem if i == 1 else f"{stem}_{i}")).mkdir()
            return run
        except FileExistsError:
            continue


def _json_default(v):
    return v.item() if isinstance(v, np.generic) else str(v)


def eval_split(split="test", model_path=None, fuse=True, save_previews=True, tta=1, *,
               blend=None, label_version="v1", use_corrected=False, mask_overlap=False,
               fusion_mode=None, names=None, run_name=None, model=None, areas=None):
    """Evaluate a split into a new eval_runs/<run_id>/ and return the summary dict.

    tta is the number of averaged passes (1, 4 or 8; True/False mean 4/1 as
    before); blend is passed to infer.predict when it takes it. names limits
    the split to those areas. model (a loaded network) and areas (tuples as
    from dataset.load_split) can be given instead of being loaded; labels are
    then taken as given. Arguments are checked before anything is loaded or
    created, and a run that fails before writing eval.json removes its
    directory.
    """
    params = dict(locals())  # a copy first, as in train.main
    args = plain_args(params, skip=("model", "areas"))
    if isinstance(tta, bool):
        tta = 4 if tta else 1
    predict = _predictor(tta, blend)
    if fuse:
        check_fusion_mode(fusion_mode)
    if run_name:
        run_path(run_name)  # an existing run fails before the slow loads
    loaded = areas is None
    if loaded:
        areas = dataset.load_split(split, label_version=label_version,
                                   use_corrected=use_corrected, mask_overlap=mask_overlap)
    if names:
        missing = set(names) - {a[0] for a in areas}
        if missing:
            raise ValueError(f"not in split {split!r}: {sorted(missing)}")
        areas = [a for a in areas if a[0] in names]
    if loaded and not areas:
        raise ValueError(f"split {split!r} has no areas")
    # a model id (v1, v2), a checkpoint path, or the default model
    mp = infer.resolve_model(model_path)[1] if (model is None or model_path) else None
    sha = file_sha256(mp) if mp else None
    if model is None:
        model = infer.load_model(mp)
    print(f"evaluating {split} ({', '.join(a[0] for a in areas)}) with "
          + (f"{mp} (sha256 {sha[:12]})" if mp else "the given model"), flush=True)

    fp_index = open_buildings.PARQUET
    prov = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": args, "git": git_info(),
        "model": {"path": str(mp) if mp else None, "sha256": sha},
        "labels": {"version": label_version, "use_corrected": use_corrected,
                   "mask_overlap": mask_overlap, "files": {}},
        "predict": {"tta": tta, "blend": _blend_used(blend)},
        "fusion": None if not fuse else {
            "requested_mode": fusion_mode,
            "implementation": "fusion.fuse" if _fusion_module() else "infer.fuse_footprints",
            "footprint_index": {"file": fp_index.name, "bytes": fp_index.stat().st_size,
                                "modified": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                          time.localtime(fp_index.stat().st_mtime))}
            if fp_index.exists() else None,
            "osm": {}},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform()},
    }
    run = new_run_dir(split, run_name)
    with discard_on_failure(run, "eval.json"):
        per_area, pooled = {}, {}
        for name, img, gt, meta in areas:
            t0 = time.time()
            p = dataset.mask_path(name, label_version, use_corrected)
            if p.exists():
                prov["labels"]["files"][name] = {"file": f"{p.parent.name}/{p.name}",
                                                 "sha256": file_sha256(p)}
            if loaded and use_corrected and not p.name.endswith("_gt.png"):
                _warn(f"{name}: no {name}_gt.png, so its auto-labels {p.name} are scored")
            inside = _train_overlap(meta, gt.shape) if split != "train" else None
            classmap, probs = predict(model, img)
            out_map = classmap
            if fuse:
                fused, untouched, info = fuse_area(name, classmap, probs, meta, fusion_mode)
                prov["fusion"]["osm"][name] = info.get("osm")
                if info.get("error"):
                    _warn(f"{name}: fusion failed, its fused map is the model's: {info['error']}")
                sc = Scorer(gt, meta, {"untouched": untouched})
                c_model, c_fused = sc.counts(classmap), sc.counts(fused)
                out_map = fused
            else:
                sc = Scorer(gt, meta)
                c_model = sc.counts(classmap)
            del probs
            n_scored = int(np.count_nonzero(sc.valid))
            entry = {"scored_px": n_scored, "ignored_px": int(gt.size - n_scored),
                     "tolerance_px": sc.tol_px, "model": summarize(c_model["all"])}
            if inside is not None:
                entry["train_overlap_scored_px"] = n_in = int(np.count_nonzero(inside & sc.valid))
                if n_in:
                    _warn(f"{name}: {n_in / n_scored:.1%} of the scored pixels lie inside train "
                          "areas (--mask-overlap ignores them)")
            _add(pooled, "model", c_model["all"])
            if fuse:
                n_u = int(c_model["untouched"]["tol"][1])
                changed = np.count_nonzero((fused != classmap) & untouched & sc.valid)
                entry["agreement_with_vectors"] = summarize(c_fused["all"])
                entry["fusion_untouched"] = {
                    "scored_px": n_u, "share": _r(n_u / max(n_scored, 1)),
                    "fused_changed_px": int(changed),
                    "model": summarize(c_model["untouched"]),
                    "fused": summarize(c_fused["untouched"])}
                entry["fusion"] = info
                for key, c in (("agreement_with_vectors", c_fused["all"]),
                               ("untouched_model", c_model["untouched"]),
                               ("untouched_fused", c_fused["untouched"])):
                    _add(pooled, key, c)
            del sc
            if save_previews:
                overlay = infer.render_overlay(img, out_map)
                Image.fromarray(overlay).save(run / f"{name}_overlay.png")
                Image.fromarray(classmap).save(run / f"{name}_model_classes.png")
                if fuse:
                    Image.fromarray(fused).save(run / f"{name}_fused_classes.png")
            entry["seconds"] = round(time.time() - t0, 1)
            per_area[name] = entry
            print(f"{name}: {_brief(entry)}", flush=True)

        out = {"model": summarize(pooled["model"])} if pooled else {}
        if fuse and pooled:
            n_u = int(pooled["untouched_model"]["tol"][1])
            out["agreement_with_vectors"] = summarize(pooled["agreement_with_vectors"])
            out["fusion_untouched"] = {
                "scored_px": n_u, "share": _r(n_u / max(int(pooled["model"]["tol"][1]), 1)),
                "model": summarize(pooled["untouched_model"]),
                "fused": summarize(pooled["untouched_fused"])}
        summary = {"split": split, "run_id": run.name, "notes": NOTES, "provenance": prov,
                   "per_area": per_area, "pooled": out}
        (run / "eval.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    if out:
        print(f"pooled: {_brief(out)}", flush=True)
    print(f"wrote {run}", flush=True)
    return {**summary, "run_dir": str(run)}


def _warn(msg):
    print(f"warning: {msg}", file=sys.stderr, flush=True)


def _train_overlap(meta, shape):
    """Pixels inside train areas (bool HxW), or None when the train areas' metas are not at hand."""
    try:
        return dataset.overlap_with_train(meta, shape)
    except (OSError, KeyError, ValueError):  # e.g. a release copy shipped without train tiles
        return None


def _brief(e):
    """Headline numbers of an area or pooled entry, on one line."""
    m = e["model"]
    s = (f"model acc={m['pixel_acc']} mIoU={m['mIoU']} tol1m={m['boundary_tolerant_acc']} "
         f"bldBF={m['building_bf']['f1']}")
    if "agreement_with_vectors" in e:
        u = e["fusion_untouched"]
        s += (f" | agreement_with_vectors acc={e['agreement_with_vectors']['pixel_acc']}"
              f" | untouched {u['share']:.0%}: model={u['model']['pixel_acc']}"
              f" fused={u['fused']['pixel_acc']}")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog=f"python -m {__package__}.evaluate",
        description="Evaluate a split into reports/eval_runs/<run_id>/ (never overwrites).")
    ap.add_argument("legacy", nargs="*", metavar="SPLIT [MODEL]",
                    help="old positional form of --split and --model")
    ap.add_argument("--split", choices=("train", "val", "test", "test2"))
    ap.add_argument("--model", help="model id (v1, v2) or checkpoint (default: the default model)")
    ap.add_argument("--tta", type=int, nargs="?", const=4, default=1, choices=(1, 4, 8),
                    help="averaged passes (a bare --tta means 4)")
    ap.add_argument("--blend", help="window blending, when infer.predict supports it")
    ap.add_argument("--label-version", default="v1",
                    help="v1 = data/masks (data_v2/masks_v1 for new areas), else data_v2/masks_<v>")
    ap.add_argument("--use-corrected", action="store_true",
                    help="prefer hand-corrected <area>_gt.png labels")
    ap.add_argument("--mask-overlap", action="store_true",
                    help="ignore val/test pixels that lie inside train areas")
    ap.add_argument("--no-fuse", action="store_true", help="model only")
    ap.add_argument("--fusion-mode", help="passed to fusion.fuse (default: its own)")
    ap.add_argument("--areas", nargs="+", metavar="NAME", help="only these areas of the split")
    ap.add_argument("--run-name", help="run id; must be new (default <split>_<time>)")
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    a = ap.parse_args(argv)
    if a.legacy:
        # mixing the forms is ambiguous: "--split val PATH" must not drop PATH
        # and silently evaluate the shipped model instead
        if a.split or a.model:
            ap.error("give SPLIT [MODEL] either positionally or as --split/--model, not both")
        if len(a.legacy) > 2:
            ap.error("give at most SPLIT and MODEL positionally")
        a.split, a.model = (a.legacy + [None])[:2]
    if a.split is None:  # no default: test is scored once, at the end
        ap.error("name the split: --split val to tune, --split test to report")
    if a.split not in ("train", "val", "test", "test2"):
        ap.error(f"unknown split {a.split!r}")
    if a.threads:
        torch.set_num_threads(a.threads)
    return eval_split(a.split, a.model, fuse=not a.no_fuse, save_previews=not a.no_previews,
                      tta=a.tta, blend=a.blend, label_version=a.label_version,
                      use_corrected=a.use_corrected, mask_overlap=a.mask_overlap,
                      fusion_mode=a.fusion_mode, names=a.areas, run_name=a.run_name)


if __name__ == "__main__":
    main(sys.argv[1:])
