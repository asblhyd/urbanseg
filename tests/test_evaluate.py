"""Evaluation: honest metrics, fusion scored as agreement, runs that never overwrite."""
import inspect
import json
import sys

import geopandas as gpd
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from shapely.geometry import Polygon

import conftest
import helpers
from urbanseg import config, dataset, evaluate, imagery

META = {"zoom": 18, "scale": 2, "center": [17.4, 78.4]}  # 1 m = 4 px here


def _score(gt, pred, regions=None):
    counts = evaluate.Scorer(gt, META, regions).counts(pred)
    out = {k: evaluate.summarize(c) for k, c in counts.items()}
    return out if regions else out["all"]


def test_tolerance_is_four_pixels_per_metre():
    assert evaluate.tolerance_px(META) == 4


def test_dilate_is_a_disk_that_does_not_wrap():
    m = np.zeros((11, 11), bool)
    m[5, 5] = True
    yy, xx = np.mgrid[:11, :11]
    assert (evaluate.dilate(m, 2) == ((yy - 5) ** 2 + (xx - 5) ** 2 <= 4)).all()
    m = np.zeros((5, 5), bool)
    m[0, 0] = True
    d = evaluate.dilate(m, 1)
    assert d.sum() == 3 and not d[-1].any() and not d[:, -1].any()


def test_boundary_tolerant_accuracy_forgives_one_metre():
    gt = np.full((40, 60), 3, np.uint8)
    gt[:, :20] = 1
    for shift, tolerant in ((4, 1.0), (5, 1 - 40 / 2400)):
        pred = np.full_like(gt, 3)
        pred[:, :20 + shift] = 1
        m = _score(gt, pred)
        assert m["pixel_acc"] == pytest.approx(1 - 40 * shift / 2400, abs=1e-4)
        assert m["boundary_tolerant_acc"] == pytest.approx(tolerant, abs=1e-4)


def test_building_bf_at_one_metre():
    gt = np.full((80, 80), 3, np.uint8)
    gt[20:60, 20:60] = 2
    m = _score(gt, gt.copy())
    assert m["building_bf"] == {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    near = np.full_like(gt, 3)
    near[23:63, 20:60] = 1  # 3 px off, other building class: BF looks at extent only
    m = _score(gt, near)
    assert m["building_bf"]["f1"] == 1.0
    far = np.full_like(gt, 3)
    far[30:70, 30:70] = 2  # 10 px off
    m = _score(gt, far)
    assert m["building_bf"]["f1"] < 0.25
    empty = np.full_like(gt, 3)
    m = _score(gt, empty)
    assert m["building_bf"] == {"f1": 0.0, "precision": None, "recall": 0.0}
    m = _score(empty, empty)
    assert m["building_bf"]["f1"] == 1.0


def test_building_bf_forgives_exactly_the_tolerance():
    """1-px inner edges: an edge 4 px (1 m) off still matches, 5 px off no longer does."""
    gt = np.full((40, 60), 3, np.uint8)
    gt[:, :20] = 1
    for shift, f1 in ((4, 1.0), (5, 0.0), (-4, 1.0), (-5, 0.0)):
        pred = np.full_like(gt, 3)
        pred[:, :20 + shift] = 1
        assert _score(gt, pred)["building_bf"] == {"f1": f1, "precision": f1, "recall": f1}
    edge = evaluate.boundaries(gt == 1, np.ones(gt.shape, bool))
    assert edge.sum() == 40 and edge[:, 19].all()  # one pixel thick, inside the building


def test_three_class_merges_residential_and_commercial():
    gt = np.full((20, 20), 1, np.uint8)
    gt[:, 10:] = 0
    pred = gt.copy()
    pred[gt == 1] = 2
    m = _score(gt, pred)
    assert m["pixel_acc"] == 0.5 and m["three_class"]["pixel_acc"] == 1.0
    assert m["three_class"]["iou"] == {"other": 1.0, "building": 1.0, "free_land": None}
    # a class absent from labels and prediction has no IoU instead of a zero in mIoU
    assert m["iou"]["free_land"] is None and m["mIoU"] == pytest.approx(1 / 3, abs=1e-4)


def test_ignored_pixels_are_never_scored():
    gt = np.full((40, 40), 3, np.uint8)
    gt[10:30, 10:30] = 1
    pred = gt.copy()
    gt[:, 20:] = dataset.IGNORE  # cuts the building in half
    pred[:, 20:] = 0  # whatever is predicted there does not count
    m = _score(gt, pred)
    assert m["pixel_acc"] == 1.0 and m["boundary_tolerant_acc"] == 1.0
    assert m["building_bf"]["f1"] == 1.0  # the cut is no building edge
    assert np.sum(m["confusion"]) == 800


def _banded(size=80, lo=20, hi=60, band=2):
    """A labels_v2-style label: one building, band px of IGNORE on each side of its edge."""
    from scipy import ndimage as ndi
    gt = np.full((size, size), 3, np.uint8)
    bld = np.zeros(gt.shape, bool)
    bld[lo:hi, lo:hi] = True
    gt[bld] = 1
    gt[ndi.binary_dilation(bld, iterations=band) & ~ndi.binary_erosion(bld, iterations=band)] = \
        dataset.IGNORE
    return gt, bld


def test_building_bf_scores_edges_inside_thin_ignore_bands():
    gt, bld = _banded()
    exact = np.where(bld, 1, 3).astype(np.uint8)
    m = _score(gt, exact)
    assert m["building_bf"] == {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    assert m["building_area_ratio"] == 1.0
    # 1.5 m (6 px) too fat on every side: past the 1 m tolerance
    fat = np.full(gt.shape, 3, np.uint8)
    fat[14:66, 14:66] = 1
    m = _score(gt, fat)
    assert m["building_bf"]["f1"] == 0.0 and m["building_area_ratio"] > 1.2
    # 0.5 m (2 px) too fat: within it
    near = np.full(gt.shape, 3, np.uint8)
    near[18:62, 18:62] = 1
    assert _score(gt, near)["building_bf"]["f1"] == 1.0
    # the band itself is still never scored for the other metrics
    assert np.sum(m["confusion"]) == np.count_nonzero(gt != dataset.IGNORE)


def test_edge_bands_leave_wide_ignore_and_v1_labels_alone():
    gt = np.full((60, 60), 3, np.uint8)
    gt[10:30, 10:30] = 1
    same, zone = evaluate.edge_bands(gt)
    assert same is gt and zone.all()
    gt[:, 30:] = dataset.IGNORE  # 30 px wide: a cut, not a band
    same, zone = evaluate.edge_bands(gt)
    assert (same == gt).all() and (zone == (gt != dataset.IGNORE)).all()


def test_regions_are_scored_as_if_the_rest_were_ignored():
    gt = np.full((40, 40), 3, np.uint8)
    gt[10:30, 4:16] = 1  # a building straddling the region edge at x = 10
    pred = gt.copy()
    pred[:, :5] = 0
    right = np.zeros(gt.shape, bool)
    right[:, 10:] = True
    res = _score(gt, pred, {"right": right})
    assert res["all"]["pixel_acc"] < 1.0 and res["right"]["pixel_acc"] == 1.0
    assert np.sum(res["right"]["confusion"]) == 40 * 30
    # nothing outside the region (not even a building edge next to it) matters
    other = pred.copy()
    other[:, :10] = 2
    assert _score(gt, other, {"right": right})["right"] == res["right"]
    assert res["right"]["building_bf"]["f1"] == 1.0


def test_counts_pool_across_areas():
    gt = np.full((20, 20), 3, np.uint8)
    pred = gt.copy()
    pred[:, :5] = 0
    sc = evaluate.Scorer(gt, META)
    pooled = {}
    for c in (sc.counts(pred)["all"], sc.counts(gt)["all"]):
        evaluate._add(pooled, "m", c)
    m = evaluate.summarize(pooled["m"])
    assert m["pixel_acc"] == 0.875 and np.sum(m["confusion"]) == 800


def test_fusion_vectors_cover_footprints():
    x0, y0 = imagery.latlng_to_world(17.4, 78.4)
    meta = {**META, "x0": x0, "y0": y0}
    corners = [(30, 20), (60, 20), (60, 40), (30, 40)]  # image px (x, y)
    poly = Polygon([imagery.img_px_to_latlng(x, y, meta)[::-1] for x, y in corners])
    fp = gpd.GeoDataFrame({"area_in_meters": [0.0], "confidence": [1.0]}, geometry=[poly], crs=4326)
    want = np.zeros((60, 80), bool)
    want[20:40, 30:60] = True
    assert (evaluate.fusion_vectors(fp, None, meta, want.shape) == want).all()
    assert not evaluate.fusion_vectors(fp.iloc[:0], None, meta, want.shape).any()


def test_predict_options_are_feature_detected(monkeypatch):
    calls = []

    def old_predict(model, img, window=1024, stride=768, batch=1):
        calls.append(("predict", {}))

    def old_tta(model, img, **kw):
        calls.append(("tta", kw))

    monkeypatch.setattr(evaluate.infer, "predict", old_predict)
    monkeypatch.setattr(evaluate.infer, "predict_tta", old_tta)
    evaluate._predictor(1, "uniform")(None, None)  # what the old predict does anyway
    evaluate._predictor(4, None)(None, None)
    assert calls == [("predict", {}), ("tta", {})] and evaluate._blend_used(None) == "uniform"
    with pytest.raises(ValueError):
        evaluate._predictor(1, "gaussian")
    with pytest.raises(ValueError):
        evaluate._predictor(8, None)

    def new_predict(model, img, window=1024, stride=768, batch=1, *, blend="gaussian",
                    progress=None, cancel=None):
        calls.append(("predict", {"blend": blend}))

    def new_tta(model, img, n=8, **kw):
        calls.append(("tta", {"n": n, **kw}))

    monkeypatch.setattr(evaluate.infer, "predict", new_predict)
    monkeypatch.setattr(evaluate.infer, "predict_tta", new_tta)
    evaluate._predictor(1, "uniform")(None, None)
    evaluate._predictor(8, None)(None, None)
    assert calls[2:] == [("predict", {"blend": "uniform"}), ("tta", {"n": 8})]
    assert evaluate._blend_used(None) == "gaussian"


def test_cli_old_and_new_forms(monkeypatch):
    # the old positional API still works
    sig = inspect.signature(evaluate.eval_split).parameters
    assert list(sig)[:5] == ["split", "model_path", "fuse", "save_previews", "tta"]
    seen = {}
    monkeypatch.setattr(evaluate, "eval_split", lambda *a, **k: seen.update(a=a, k=k))
    evaluate.main(["test", "m.pt", "--no-fuse", "--tta"])
    assert seen["a"] == ("test", "m.pt") and seen["k"]["fuse"] is False and seen["k"]["tta"] == 4
    evaluate.main(["--split", "val", "--tta", "8", "--blend", "uniform", "--mask-overlap",
                   "--label-version", "v2", "--use-corrected", "--run-name", "r1",
                   "--areas", "kompally"])
    k = seen["k"]
    assert seen["a"] == ("val", None) and k["tta"] == 8 and k["blend"] == "uniform"
    assert k["mask_overlap"] and k["use_corrected"] and k["label_version"] == "v2"
    assert k["run_name"] == "r1" and k["names"] == ["kompally"] and k["fuse"] is True


def test_cli_rejects_mixed_forms_and_needs_a_split(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate, "eval_split", lambda *a, **k: calls.append(a))
    for argv in (["--split", "val", "models/runs/new/best.pt"],  # used to drop the checkpoint
                 ["--model", "a.pt", "val", "b.pt"], ["--split", "val", "test"],
                 ["val", "a.pt", "b.pt"], ["dev"],
                 [], ["--model", "a.pt"]):  # no split: test is never scored by default
        with pytest.raises(SystemExit):
            evaluate.main(argv)
    assert calls == []
    evaluate.main(["--split", "val", "--model", "a.pt"])
    evaluate.main(["val", "a.pt"])
    assert calls == [("val", "a.pt")] * 2


class StubNet(torch.nn.Module):
    """Deterministic stand-in for the U-Net: classes from pixel brightness."""

    def forward(self, x):
        v = x.mean(1)
        k = (v > -0.6).long() + (v > 0.0).long() + (v > 0.8).long()
        return F.one_hot(k, 4).permute(0, 3, 1, 2).float()


def _areas(n=2, size=64):
    """In-memory georeferenced areas (far from every train area) with random labels."""
    rng = np.random.default_rng(0)
    return [(f"synth{i}", rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
             rng.integers(0, 4, (size, size)).astype(np.uint8),
             {**META, "x0": 48_000_000.0 + 1000 * i, "y0": 30_000_000.0}) for i in range(n)]


def test_arguments_are_recorded_under_a_tracer():
    """A tracer that reads f_locals (pdb, coverage) must not break the argument log."""
    def tracer(frame, event, arg):
        frame.f_locals  # re-syncs the frame's locals dict, as debuggers do
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        with pytest.raises(ValueError):  # the intended error, not "dict changed size"
            evaluate.eval_split("val", tta=3, model=StubNet(), areas=[])
    finally:
        sys.settrace(old)


def test_bad_arguments_fail_before_anything_is_loaded_or_created(monkeypatch):
    monkeypatch.setattr(evaluate.dataset, "load_split", lambda *a, **k: pytest.fail("data loaded"))
    monkeypatch.setattr(evaluate.infer, "load_model", lambda *a, **k: pytest.fail("model loaded"))
    # as the predict package's predict_tta(n), which would only reject n=3 on the first area
    monkeypatch.setattr(evaluate.infer, "predict_tta", lambda model, img, n=8, **kw: None)
    for bad in ({"tta": 3}, {"fusion_mode": "no-such-mode"}):
        with pytest.raises(ValueError):
            evaluate.eval_split("val", **bad)
    (config.REPORTS_DIR / "eval_runs" / "taken").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        evaluate.eval_split("val", run_name="taken")
    assert [p.name for p in (config.REPORTS_DIR / "eval_runs").iterdir()] == ["taken"]


class FailingNet(StubNet):
    """StubNet that fails on its second forward pass (the second 64 px area)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("boom")
        return super().forward(x)


def test_a_failed_run_leaves_no_directory():
    with pytest.raises(RuntimeError, match="boom"):  # after the first area's previews
        evaluate.eval_split("val", fuse=False, model=FailingNet(), areas=_areas(), run_name="r")
    assert not (config.REPORTS_DIR / "eval_runs" / "r").exists()  # the name is free again
    s = evaluate.eval_split("val", fuse=False, model=StubNet(), areas=_areas(), run_name="r")
    assert sorted(p.name for p in (config.REPORTS_DIR / "eval_runs" / "r").iterdir()) == sorted(
        ["eval.json"] + [f"synth{i}_{k}.png" for i in (0, 1) for k in ("overlay", "model_classes")])
    assert s["pooled"]["model"]["pixel_acc"] is not None


def _write_area(root, name, x0, y0=30_000_000.0, size=64, mask=3):
    """A georeferenced zoom-18 / scale-2 area on disk, laid out like data/."""
    meta = {"zoom": 18, "scale": 2, "x0": x0, "y0": y0,
            "width_map_px": size / 2, "height_map_px": size / 2,
            "center": list(imagery.world_to_latlng(x0 + size / 4, y0 + size / 4)), "nx": 1, "ny": 1}
    (root / "tiles" / name).mkdir(parents=True)
    (root / "tiles" / name / "meta.json").write_text(json.dumps(meta))
    img = np.full((size, size, 3), 128, np.uint8)
    Image.fromarray(img).save(root / "tiles" / f"{name}_full.png")
    (root / "masks").mkdir(exist_ok=True)
    Image.fromarray(np.full((size, size), mask, np.uint8)).save(root / "masks" / f"{name}.png")


@pytest.fixture
def synthetic_val(tmp_path, monkeypatch):
    """Val areas va (with a corrected va_gt.png) and vb, whose left half lies in train area tr."""
    root = tmp_path / "data"
    for key, path in (("DATA_DIR", root), ("TILES_DIR", root / "tiles"),
                      ("MASKS_DIR", root / "masks")):
        monkeypatch.setattr(config, key, path)
    for name, x0 in (("tr", 48_000_000.0), ("va", 48_001_000.0), ("vb", 48_000_016.0)):
        _write_area(root, name, x0)
    Image.fromarray(np.full((64, 64), 1, np.uint8)).save(root / "masks" / "va_gt.png")
    monkeypatch.setattr(config, "AREAS", [("tr", 0, 0, 1, 1, "train"), ("va", 0, 0, 1, 1, "val"),
                                          ("vb", 0, 0, 1, 1, "val")])
    return root


def test_a_split_without_areas_is_an_error(synthetic_val):
    with pytest.raises(ValueError, match="no areas"):  # instead of an empty eval.json
        evaluate.eval_split("test", model=StubNet())
    assert not (config.REPORTS_DIR / "eval_runs").exists()


def test_runs_name_their_model_and_warn_about_what_they_score(synthetic_val, tmp_path, capsys,
                                                               monkeypatch):
    ckpt = tmp_path / "candidate.pt"
    ckpt.write_bytes(b"weights")
    s = evaluate.eval_split("val", ckpt, fuse=False, save_previews=False, use_corrected=True,
                            model=StubNet())
    out, err = capsys.readouterr()
    assert str(ckpt) in out and evaluate.file_sha256(ckpt)[:12] in out  # which model was scored
    assert "vb: no vb_gt.png" in err and "va: no" not in err  # a silent fallback is named
    assert s["provenance"]["labels"]["files"]["va"]["file"] == "masks/va_gt.png"
    assert s["per_area"]["va"]["train_overlap_scored_px"] == 0
    assert s["per_area"]["vb"]["train_overlap_scored_px"] == 64 * 32
    assert "vb: 50.0% of the scored pixels lie inside train areas" in err
    s = evaluate.eval_split("val", ckpt, fuse=False, save_previews=False, mask_overlap=True,
                            model=StubNet())
    assert s["per_area"]["vb"]["train_overlap_scored_px"] == 0
    assert "inside train areas" not in capsys.readouterr().err

    # a fusion failure (fusion.fuse records it instead of raising) is reported too
    monkeypatch.setattr(evaluate, "fuse_area", lambda name, cm, probs, meta, mode: (
        cm, np.ones(cm.shape, bool), {"error": "ValueError: boom", "osm": None, "sources": []}))
    evaluate.eval_split("val", ckpt, save_previews=False, names=["va"], model=StubNet())
    assert "va: fusion failed" in capsys.readouterr().err


@pytest.mark.data
def test_eval_run_on_cropped_val_areas(tmp_path):
    areas = []
    for name, y0, x0 in (("lbnagar", 2166, 1049), ("kompally", 1500, 1500)):
        img, mask, meta = helpers.crop(*helpers.load_area(name), y0, y0 + 512, x0, x0 + 512)
        mask[dataset.overlap_with_train(meta, mask.shape)] = dataset.IGNORE
        areas.append((name, img, mask, meta))
    s = evaluate.eval_split("val", model=StubNet(), areas=areas, run_name="stub")

    run = config.REPORTS_DIR / "eval_runs" / "stub"
    assert s["run_dir"] == str(run)
    assert sorted(p.name for p in run.iterdir()) == sorted(
        ["eval.json"] + [f"{n}_{k}.png" for n in ("lbnagar", "kompally")
                         for k in ("overlay", "model_classes", "fused_classes")])
    assert all(run in p.parents for p in tmp_path.rglob("*") if p.is_file())  # nothing else
    saved = json.loads((run / "eval.json").read_text())
    assert saved["run_id"] == "stub" and set(saved["pooled"]) == {
        "model", "agreement_with_vectors", "fusion_untouched"}

    lb = s["per_area"]["lbnagar"]
    assert lb["ignored_px"] == np.count_nonzero(areas[0][2] == dataset.IGNORE) > 0
    assert lb["scored_px"] + lb["ignored_px"] == 512 * 512 and lb["tolerance_px"] == 4
    assert {"model", "agreement_with_vectors", "fusion_untouched", "fusion"} <= set(lb)
    assert {"footprints", "osm"} <= set(lb["fusion"]["sources"])
    u = lb["fusion_untouched"]
    assert 0 < u["share"] < 1 and u["scored_px"] < lb["scored_px"]
    if lb["fusion"].get("mode") == "legacy":  # those rules only touch vector pixels
        assert u["fused_changed_px"] == 0 and u["model"] == u["fused"]
    for key in ("model", "agreement_with_vectors"):
        assert np.array_equal(np.add(*(s["per_area"][n][key]["confusion"]
                                       for n in ("lbnagar", "kompally"))),
                              s["pooled"][key]["confusion"])

    prov = s["provenance"]
    osm = prov["fusion"]["osm"]
    assert (osm["lbnagar"]["file"], osm["lbnagar"]["osm_base"]) == (
        "live_a9f296_v2.json", "2026-08-16T05:51:35Z")
    assert osm["kompally"]["file"] == "live_19fb95_v2.json"
    assert prov["labels"]["files"]["lbnagar"]["file"] == "masks/lbnagar.png"
    assert prov["predict"]["tta"] == 1 and prov["model"]["sha256"] is None
    assert prov["date"] and "git" in prov

    with pytest.raises(FileExistsError):  # a run id is never reused
        evaluate.eval_split("val", model=StubNet(), areas=areas, run_name="stub")
    assert not (config.REPORTS_DIR / "eval_val.json").exists() and not config.PREVIEW_DIR.exists()


@pytest.mark.data
def test_osm_comes_from_the_read_only_snapshots():
    meta = json.loads((config.TILES_DIR / "kphb" / "meta.json").read_text())
    layers, prov = evaluate.osm_snapshot("kphb", meta)
    assert prov["file"] == "kphb_v2.json" and len(layers["roads"])
    assert evaluate.osm_snapshot("nowhere", meta) == (None, None)


@pytest.mark.data
@pytest.mark.model
@pytest.mark.slow
def test_shipped_model_single_pass_on_kompally(model):
    """Regression anchor: the shipped checkpoint, one uniform-blended pass, no fusion."""
    s = evaluate.eval_split("val", conftest.MODEL_PATH, fuse=False, save_previews=False,
                            blend="uniform", names=["kompally"], model=model, run_name="anchor")
    assert s["per_area"]["kompally"]["model"]["pixel_acc"] == pytest.approx(0.7270, abs=5e-4)
    assert s["provenance"]["model"]["sha256"] == evaluate.file_sha256(conftest.MODEL_PATH)
