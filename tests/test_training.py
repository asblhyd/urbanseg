"""Training data and runs: fresh reproducible crops, complete validation, no overwrites."""
import hashlib
import json
import re
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import helpers
from urbanseg import config, dataset, imagery, train


def _areas(n=2, h=200, w=230):
    """In-memory areas whose mask value encodes (area, y, x), so a crop reveals its origin."""
    rng = np.random.default_rng(0)
    return [(f"synth{a}", rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
             (a * h * w + np.arange(h * w).reshape(h, w)).astype(np.int32), {})
            for a in range(n)]


def _fingerprint(x, y):
    return hashlib.sha1(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("workers", [0, 2])
def test_crops_are_fresh_every_epoch_and_independent_of_workers(workers, monkeypatch):
    pickled = []  # every worker start pickles the dataset, in this process

    def getstate(self):
        pickled.append(1)
        return self.__dict__

    monkeypatch.setattr(dataset.RandomCropDataset, "__getstate__", getstate)
    ds = dataset.RandomCropDataset(_areas(), crop=64, samples_per_epoch=24, seed=3)
    loader = train.train_loader(ds, batch=5, workers=workers)  # as train.main builds it
    seen = set()
    for ep in range(3):
        ds.set_epoch(ep)
        got = [_fingerprint(x, y) for xb, yb in loader for x, y in zip(xb, yb)]
        assert len(set(got)) == len(got) == 24 and not seen & set(got)
        # the same samples in-process: no dependence on worker id or assignment
        assert got == [_fingerprint(*ds[i]) for i in range(24)]
        assert got == [_fingerprint(*ds[(ep, i)]) for i in range(24)]
        seen |= set(got)
    # workers persist, yet see each new epoch: one copy of the data per worker and run
    assert len(pickled) == workers


def test_seed_and_epoch_must_fit_one_seed_word():
    areas = _areas()
    for bad in (-1, 2 ** 32, 1.5, "0", True):
        with pytest.raises(ValueError):
            dataset.RandomCropDataset(areas, crop=64, samples_per_epoch=4, seed=bad)
    ds = dataset.RandomCropDataset(areas, crop=64, samples_per_epoch=4, seed=2 ** 32 - 1)
    with pytest.raises(ValueError):
        ds.set_epoch(-1)
    for bad in ("-1", str(2 ** 32), "1.5"):
        with pytest.raises(SystemExit):
            train.parse_args(["--out", "x", "--seed", bad])
    assert train.parse_args(["--out", "x", "--seed", "7"]).seed == 7


def test_crops_depend_only_on_seed_epoch_and_index():
    areas = _areas()
    a = dataset.RandomCropDataset(areas, crop=64, samples_per_epoch=50, seed=1)
    b = dataset.RandomCropDataset(areas, crop=64, samples_per_epoch=50, seed=1)
    torch.manual_seed(123)  # the old sampler keyed every crop on torch's seed
    draws = [a.draw(i) for i in range(50)]
    torch.manual_seed(456)
    assert [b.draw(i) for i in range(50)] == draws and b.digest() == a.digest()
    b.set_epoch(1)
    assert b.digest() != a.digest() and not set(b.draw(i) for i in range(50)) & set(draws)
    c = dataset.RandomCropDataset(areas, crop=64, samples_per_epoch=50, seed=2)
    assert not set(c.draw(i) for i in range(50)) & set(draws)
    # draw() describes exactly what __getitem__ returns
    h, w = areas[0][1].shape[:2]
    for i in range(10):
        d = a.draw(i)
        _, y = a[i]
        assert int(y.min()) == d.area * h * w + d.y * w + d.x


def test_weight_zero_areas_are_never_sampled():
    areas = _areas(3)
    areas[1] = ("nizampet",) + areas[1][1:]  # AREA_WEIGHTS = 0 (label audit)
    ds = dataset.RandomCropDataset(areas, crop=32, samples_per_epoch=200)
    assert [a[0] for a in ds.areas] == ["synth0", "synth2"]


def test_grid_dataset_scores_every_pixel_once():
    h, w, c = 1100, 1300, 512
    mask = np.zeros((h, w), np.uint8)
    mask[:10] = dataset.IGNORE  # pixels ignored by the labels stay ignored
    ds = dataset.GridDataset([("a", np.zeros((h, w, 3), np.uint8), mask, {})], crop=c)
    cover = np.zeros((h, w), int)
    for i, (_, y, x, _, _) in enumerate(ds.items):
        xt, yt = ds[i]
        assert xt.shape == (3, c, c) and yt.shape == (c, c)
        cover[y:y + c, x:x + c] += yt.numpy() != dataset.IGNORE
    assert (cover[10:] == 1).all() and (cover[:10] == 0).all()
    with pytest.raises(ValueError):
        dataset.GridDataset([("small", np.zeros((300, 600, 3), np.uint8),
                              np.zeros((300, 600), np.uint8), {})], crop=c)


class PixelNet(torch.nn.Module):
    """Classes from each pixel alone, so crop placement cannot change the prediction."""

    def forward(self, x):
        k = (x[:, 0] > 0).long() + 2 * (x[:, 1] > 0).long()
        return F.one_hot(k, 4).permute(0, 3, 1, 2).float()


def test_grid_validation_equals_whole_area_scoring():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (600, 700, 3), dtype=np.uint8)
    mask = rng.integers(0, 4, (600, 700)).astype(np.uint8)
    mask[:50, :80] = dataset.IGNORE
    with torch.no_grad():
        pred = PixelNet()(dataset.normalize(img)[None]).argmax(1)[0].numpy()
    want = train.confusion(mask, pred)
    assert want.sum() == np.count_nonzero(mask != dataset.IGNORE)
    loader = DataLoader(dataset.GridDataset([("a", img, mask, {})], crop=256), batch_size=4)
    got = train.evaluate_loader(PixelNet(), loader)
    assert got == train.metrics_from_confusion(want)


def test_loss_ignores_255():
    torch.manual_seed(0)
    loss = train.build_loss()
    target = torch.randint(0, 4, (2, 16, 16))
    target[:, :8] = dataset.IGNORE
    a = torch.randn(2, 4, 16, 16)
    b = a.clone()
    b[:, :, :8] = torch.randn(2, 4, 8, 16) * 10  # only where the target is ignored
    assert torch.isfinite(loss(a, target))
    assert torch.allclose(loss(a, target), loss(b, target))


# ---- label sources ------------------------------------------------------------

def _write_area(root, name, x0=48_000_000.0, y0=30_000_000.0, size=64, mask=3, img=None):
    """A georeferenced zoom-18 / scale-2 area on disk, laid out like data/."""
    meta = {"zoom": 18, "scale": 2, "x0": x0, "y0": y0,
            "width_map_px": size / 2, "height_map_px": size / 2,
            "center": list(imagery.world_to_latlng(x0 + size / 4, y0 + size / 4)), "nx": 1, "ny": 1}
    (root / "tiles" / name).mkdir(parents=True)
    (root / "tiles" / name / "meta.json").write_text(json.dumps(meta))
    if img is None:
        img = np.full((size, size, 3), 128, np.uint8)
    Image.fromarray(img).save(root / "tiles" / f"{name}_full.png")
    (root / "masks").mkdir(exist_ok=True)
    mask = np.full((size, size), mask, np.uint8) if np.isscalar(mask) else mask
    Image.fromarray(mask).save(root / "masks" / f"{name}.png")
    return meta


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A synthetic data/ in tmp_path (the research data is never touched)."""
    root = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "TILES_DIR", root / "tiles")
    monkeypatch.setattr(config, "MASKS_DIR", root / "masks")
    monkeypatch.setattr(config, "DATA_V2_DIR", tmp_path / "data_v2")
    monkeypatch.setattr(config, "AREAS_V2", [])
    return root


def test_corrected_masks_only_on_request(data_root, monkeypatch):
    for name in ("tr", "va"):
        _write_area(data_root, name, mask=3)
        Image.fromarray(np.full((64, 64), 1, np.uint8)).save(data_root / "masks" / f"{name}_gt.png")
    monkeypatch.setattr(config, "AREAS", [("tr", 0, 0, 1, 1, "train"), ("va", 0, 0, 1, 1, "val")])
    for split in ("train", "val"):
        ((_, _, mask, _),) = dataset.load_split(split)
        assert (mask == 3).all()
        ((_, _, mask, _),) = dataset.load_split(split, use_corrected=True)
        assert (mask == 1).all()


def test_label_versions_live_in_their_own_folder(data_root, monkeypatch):
    # every version after v1 lives in data_v2/, so data/ stays the frozen v1 set
    _write_area(data_root, "va", mask=3)
    v2 = config.DATA_V2_DIR / "masks_v2"
    v2.mkdir(parents=True)
    Image.fromarray(np.full((64, 64), 2, np.uint8)).save(v2 / "va.png")
    monkeypatch.setattr(config, "AREAS", [("va", 0, 0, 1, 1, "val")])
    ((_, _, mask, _),) = dataset.load_split("val", label_version="v2")
    assert (mask == 2).all()
    assert dataset.mask_path("va", "v2") == v2 / "va.png"
    assert not (data_root / "masks_v2").exists()
    with pytest.raises(ValueError):
        dataset.mask_path("va", "../masks")


def test_mask_overlap_ignores_val_pixels_inside_train_areas(data_root, monkeypatch):
    x0, y0 = 48_000_000.0, 30_000_000.0
    _write_area(data_root, "tr", x0, y0)  # 32 x 32 map px
    _write_area(data_root, "va", x0 + 10.25, y0 + 20)
    _write_area(data_root, "te", x0 + 100, y0)  # apart from the train area
    monkeypatch.setattr(config, "AREAS", [("tr", 0, 0, 1, 1, "train"), ("va", 0, 0, 1, 1, "val"),
                                          ("te", 0, 0, 1, 1, "test")])
    ((_, _, va, _),) = dataset.load_split("val", mask_overlap=True)
    # pixel centres inside the train area: rows < (32 - 20) * 2, cols < (32 - 10.25) * 2
    want = np.full((64, 64), 3, np.uint8)
    want[:24, :43] = dataset.IGNORE
    assert (va == want).all()
    ((_, _, te, _),) = dataset.load_split("test", mask_overlap=True)
    ((_, _, tr, _),) = dataset.load_split("train", mask_overlap=True)
    assert (te == 3).all() and (tr == 3).all()


@pytest.mark.data
def test_lbnagar_overlap_with_vanasthalipuram_is_masked():
    areas = {a[0]: a for a in dataset.load_split("val", mask_overlap=True)}
    ignored = areas["lbnagar"][2] == dataset.IGNORE
    assert ignored.mean() == pytest.approx(0.216, abs=0.001)  # 21.6% of lbnagar
    rows, cols = np.nonzero(ignored.any(1))[0], np.nonzero(ignored.any(0))[0]
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (2422, 3599, 1305, 3839)
    assert not (areas["kompally"][2] == dataset.IGNORE).any()
    _, mask, _ = helpers.load_area("lbnagar")
    assert (areas["lbnagar"][2][~ignored] == mask[~ignored]).all()


# ---- training runs ---------------------------------------------------------------

@pytest.fixture
def synthetic_split(data_root, tmp_path, monkeypatch):
    """One train and one val area of 1024 x 1024 px; checkpoints go to tmp_path/models."""
    rng = np.random.default_rng(0)
    for i, name in enumerate(("synth_train", "synth_val")):
        mask = np.repeat(np.repeat(rng.integers(0, 4, (16, 16)), 64, 0), 64, 1).astype(np.uint8)
        img = (mask[..., None] * 60 + rng.integers(0, 40, (1024, 1024, 3))).astype(np.uint8)
        _write_area(data_root, name, 48_000_000.0 + 2000 * i, 30_000_000.0, 1024, mask, img)
    monkeypatch.setattr(config, "AREAS", [("synth_train", 0, 0, 1, 1, "train"),
                                          ("synth_val", 0, 0, 1, 1, "val")])
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _record_batches(monkeypatch):
    """Patch train.main's model and loss to record every training batch (inputs, targets)."""
    batches = []
    build_model, build_loss = train.build_model, train.build_loss

    def record_input(module, inputs):
        if module.training:  # not the validation passes
            batches.append([inputs[0].detach().clone()])

    def model(**kw):
        m = build_model(**kw)
        m.register_forward_pre_hook(record_input)
        return m

    def loss(*a, **kw):
        fn = build_loss(*a, **kw)

        def recorded(out, target):
            batches[-1].append(target.clone())
            return fn(out, target)
        return recorded

    monkeypatch.setattr(train, "build_model", model)
    monkeypatch.setattr(train, "build_loss", loss)
    return batches


def test_train_smoke_run(synthetic_split, monkeypatch):
    torch.set_num_threads(2)
    shipped = config.MODELS_DIR / train.SHIPPED
    shipped.parent.mkdir(parents=True)
    shipped.write_bytes(b"shipped weights")
    init = synthetic_split / "init.pt"
    torch.manual_seed(0)
    torch.save(train.build_model(pretrained=False).state_dict(), init)

    batches = _record_batches(monkeypatch)
    res = train.main(2, 8, crop=256, batch=2, out_name="smoke", init=init, workers=0)
    run = config.MODELS_DIR / "runs" / "smoke"
    assert res["run_dir"] == str(run)
    assert sorted(p.name for p in run.iterdir()) == ["best.pt", "config.json", "last.pt",
                                                      "train_log.txt"]
    # the model trained on epoch e's own crops: new ones in epoch 1, each as the dataset defines it
    ds = dataset.RandomCropDataset(dataset.load_split("train"), crop=256, samples_per_epoch=8)
    assert len(batches) == 2 * 4
    for ep in range(2):
        xs, ys = (torch.cat(t) for t in zip(*batches[4 * ep:4 * ep + 4]))
        want = [ds[(ep, i)] for i in range(8)]
        assert torch.equal(xs, torch.stack([w[0] for w in want]))
        assert torch.equal(ys, torch.stack([w[1] for w in want]))
    assert not torch.equal(torch.cat([b[1] for b in batches[:4]]),
                           torch.cat([b[1] for b in batches[4:]]))
    digests = re.findall(r"crops=(\w+)", (run / "train_log.txt").read_text())
    assert digests == [h["crops"] for h in res["history"]] and digests[0] != digests[1]
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["init"]["sha256"] == train.file_sha256(init)
    assert cfg["data"]["label_version"] == "v1"
    assert cfg["data"]["masks"]["synth_val"]["file"] == "masks/synth_val.png"
    assert cfg["loader"] == {"workers": 0, "persistent_workers": False, "sampler": "EpochSampler",
                             "start_method": None, "iterations_per_epoch": 4, "val_crops": 16}
    assert cfg["args"]["out_name"] == "smoke" and cfg["result"]["best_epoch"] in (0, 1)
    assert "git" in cfg and len(cfg["result"]["history"]) == 2
    model = train.build_model(pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))

    # never overwrite: neither this run nor the shipped checkpoint
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    kw = dict(crop=256, batch=2, workers=0, encoder_weights="none")
    with pytest.raises(FileExistsError):
        train.main(1, 2, out_name="smoke", **kw)
    for name in ("unet_r34.pt", "unet_r34", None, "../smoke2", "runs/x"):
        with pytest.raises(ValueError):
            train.main(1, 2, out_name=name, **kw)
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    assert shipped.read_bytes() == b"shipped weights"
    assert sorted(p.name for p in (config.MODELS_DIR / "runs").iterdir()) == ["smoke"]


def test_bad_arguments_fail_before_anything_is_created(synthetic_split):
    kw = dict(crop=256, batch=2, workers=0, encoder_weights="none")
    for bad in ({"seed": -1}, {"seed": 2 ** 32}, {"seed": 1.5}, {"epochs": 0}, {"samples": 0},
                {"crop": 100}, {"batch": 0}):
        args = {"epochs": 1, "samples": 2, **kw, **bad}
        with pytest.raises(ValueError):
            train.main(out_name="bad", **args)
        assert not (config.MODELS_DIR / "runs").exists()
    # a torch.device is logged as text instead of breaking config.json
    res = train.main(1, 2, out_name="dev", device=torch.device("cpu"), **kw)
    cfg = json.loads((config.MODELS_DIR / "runs" / "dev" / "config.json").read_text())
    assert cfg["args"]["device"] == "cpu" and res["best_epoch"] == 0


def test_a_run_that_fails_before_its_first_checkpoint_leaves_nothing(synthetic_split,
                                                                      monkeypatch):
    kw = dict(crop=256, batch=2, workers=0, encoder_weights="none")  # 1 step per epoch
    build_loss = train.build_loss

    def failing_loss(ok_steps):
        """build_loss whose loss raises after ok_steps training steps."""
        steps = []

        def make(*a, **k):
            fn = build_loss(*a, **k)

            def loss(out, target):
                steps.append(1)
                if len(steps) > ok_steps:
                    raise RuntimeError("boom")
                return fn(out, target)
            return loss
        return make

    monkeypatch.setattr(train, "build_loss", failing_loss(0))  # fails in epoch 0
    with pytest.raises(RuntimeError, match="boom"):
        train.main(2, 2, out_name="retry", **kw)
    assert not (config.MODELS_DIR / "runs" / "retry").exists()  # the name is free again

    monkeypatch.setattr(train, "build_loss", failing_loss(1))  # fails in epoch 1
    with pytest.raises(RuntimeError, match="boom"):
        train.main(2, 2, out_name="retry", **kw)
    run = config.MODELS_DIR / "runs" / "retry"  # epoch 0's checkpoints are kept
    assert sorted(p.name for p in run.iterdir()) == ["best.pt", "config.json", "last.pt",
                                                      "train_log.txt"]


def test_arguments_are_recorded_under_a_tracer():
    """A tracer that reads f_locals (pdb, coverage) must not break the argument log."""
    def tracer(frame, event, arg):
        frame.f_locals  # re-syncs the frame's locals dict, as debuggers do
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        with pytest.raises(ValueError):  # the intended error, not "dict changed size"
            train.main(1, 2, out_name=None)
    finally:
        sys.settrace(old)


def test_cli_requires_out_and_keeps_the_positional_form():
    a = train.parse_args(["22", "1200", "--out", "fresh"])
    assert (a.epochs, a.samples, a.out, a.select) == (22, 1200, "fresh", "miou")
    with pytest.raises(SystemExit):
        train.parse_args(["22", "1200"])


# ---- version-2 data and training options -----------------------------------------

def _v2_area(name, mask_v1, mask_v2=None, x0=48_100_000.0, size=256):
    """A config.AREAS_V2 area laid out as scripts/build_dataset_v2.py writes it."""
    root = config.DATA_V2_DIR
    _write_area(root, name, x0, 30_000_000.0, size, mask_v1)
    (root / "masks").rename(root / "masks_v1") if not (root / "masks_v1").exists() else [
        p.rename(root / "masks_v1" / p.name) for p in (root / "masks").iterdir()]
    if mask_v2 is not None:
        (root / "masks_v2").mkdir(exist_ok=True)
        Image.fromarray(np.full((size, size), mask_v2, np.uint8)).save(root / "masks_v2" / f"{name}.png")


def test_new_areas_load_from_data_v2(data_root, monkeypatch):
    _write_area(data_root, "old_tr", mask=1, size=256)
    _v2_area("new_tr", mask_v1=3, mask_v2=2)
    _v2_area("new_te", mask_v1=3, mask_v2=0, x0=48_200_000.0)
    monkeypatch.setattr(config, "AREAS", [("old_tr", 0, 0, 1, 1, "train")])
    monkeypatch.setattr(config, "AREAS_V2", [("new_tr", 0, 0, 1, 1, "train", "plots"),
                                             ("new_te", 0, 0, 1, 1, "test2", "water")])
    assert [a[0] for a in dataset.load_split("train")] == ["old_tr"]
    both = dataset.load_split("train", new_areas=True)
    assert [a[0] for a in both] == ["old_tr", "new_tr"] and (both[1][2] == 3).all()
    ((name, img, mask, meta),) = dataset.load_split("test2", label_version="v2")
    assert name == "new_te" and (mask == 0).all() and img.shape == (256, 256, 3)
    assert dataset.mask_path("new_tr", "v1") == config.DATA_V2_DIR / "masks_v1" / "new_tr.png"
    assert not (data_root / "masks_v1").exists() and not (data_root / "masks_v2").exists()


def _areas_for_aug(size=512):
    rng = np.random.default_rng(0)
    mask = np.repeat(np.repeat(rng.integers(0, 4, (8, 8)), size // 8, 0), size // 8, 1).astype(np.uint8)
    mask[:20, :20] = dataset.IGNORE
    img = (mask[..., None] % 4 * 50 + rng.integers(0, 40, (size, size, 3))).astype(np.uint8)
    return [("a", img, mask, {})]


def test_aug_v2_is_reproducible_and_keeps_labels_valid():
    ds = dataset.RandomCropDataset(_areas_for_aug(), crop=128, samples_per_epoch=24, aug="v2")
    lo, hi = ds.SCALE_RANGE
    sizes = [ds.draw(i).size for i in range(24)]
    assert all(round(128 / hi) <= s <= round(128 / lo) for s in sizes) and len(set(sizes)) > 5
    for i in range(24):
        x1, y1 = ds[(0, i)]
        x2, y2 = ds[(0, i)]
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
        assert x1.shape == (3, 128, 128) and y1.shape == (128, 128)
        assert set(y1.unique().tolist()) <= {0, 1, 2, 3, dataset.IGNORE}
    assert sum(not torch.equal(ds[(0, i)][0], ds[(1, i)][0]) for i in range(24)) >= 22
    assert any(ds.draw(i).jpeg for i in range(24)) and any(ds.draw(i).blur for i in range(24))
    # the default keeps the original draws: no v2 fields set
    v1 = dataset.RandomCropDataset(_areas_for_aug(), crop=128, samples_per_epoch=24)
    assert all(v1.draw(i)[7:] == (0, 1.0, 1.0, 1.0, 1.0, 0.0, 0) for i in range(24))
    with pytest.raises(ValueError):
        dataset.RandomCropDataset(_areas_for_aug(), aug="v3")


def test_class_aware_crops_centre_on_commercial_pixels():
    size = 1024
    mask = np.full((size, size), 3, np.uint8)
    mask[900:960, 40:100] = 2  # one small commercial block in a corner
    areas = [("a", np.zeros((size, size, 3), np.uint8), mask, {})]
    hit = lambda ds: np.mean([(ds[(0, i)][1] == 2).any().item() for i in range(40)])  # noqa: E731
    always = dataset.RandomCropDataset(areas, crop=256, samples_per_epoch=40, class_aware=1.0)
    never = dataset.RandomCropDataset(areas, crop=256, samples_per_epoch=40)
    assert hit(always) == 1.0 and hit(never) < 0.3
    with pytest.raises(ValueError):
        dataset.RandomCropDataset(areas, class_aware=1.5)


def test_train_smoke_run_with_new_data_and_options(synthetic_split, monkeypatch):
    torch.set_num_threads(2)
    rng = np.random.default_rng(1)
    m = np.repeat(np.repeat(rng.integers(0, 4, (8, 8)), 64, 0), 64, 1).astype(np.uint8)
    _v2_area("synth_new", mask_v1=m, size=512)
    monkeypatch.setattr(config, "AREAS_V2", [("synth_new", 0, 0, 1, 1, "train", "plots")])
    res = train.main(1, 4, crop=256, batch=2, out_name="smoke_v2", workers=0,
                     encoder_weights="none", new_areas=True, aug="v2", class_aware=0.5,
                     amp="bf16", accum=2)
    cfg = json.loads((config.MODELS_DIR / "runs" / "smoke_v2" / "config.json").read_text())
    assert cfg["data"]["train_areas"] == ["synth_train", "synth_new"]
    assert cfg["args"]["aug"] == "v2" and cfg["args"]["amp"] == "bf16" and cfg["args"]["accum"] == 2
    assert np.isfinite(res["history"][0]["train_loss"])
    with pytest.raises(ValueError):
        train.main(1, 4, crop=256, batch=3, out_name="bad", accum=2)
    with pytest.raises(ValueError):
        train.main(1, 4, crop=256, batch=2, out_name="bad", amp="fp8")
