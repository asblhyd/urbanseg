"""Train the 4-class U-Net into a new run directory, never over an existing checkpoint.

Run: python -m urbanseg.train [epochs] [samples/epoch] --out NAME [--init CKPT] [options]

Writes only under config.MODELS_DIR/runs/NAME/: best.pt (best val --select
metric), last.pt, train_log.txt, and config.json (arguments, git commit,
label version and mask hashes, sha256 of --init). NAME must be new, and the
shipped name unet_r34.pt is refused. A run that fails before its first
checkpoint removes its directory, so the name can be used again. Data is
read, never written.
"""
import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from . import config, dataset

SHIPPED = "unet_r34.pt"  # the deployed checkpoint: never a training output
CLASS_WEIGHTS = [1.0, 1.0, 2.0, 0.8]


def build_model(pretrained=True):
    # ImageNet encoder weights only matter when training; inference loads a
    # full checkpoint over them, so skip the download there
    return smp.Unet(encoder_name="resnet34",
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=config.N_CLASSES)


def metrics_from_confusion(cm):
    acc = np.trace(cm) / max(cm.sum(), 1)
    ious, f1s = {}, {}
    for k in range(config.N_CLASSES):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        ious[config.CLASS_NAMES[k]] = float(tp / max(tp + fp + fn, 1))
        f1s[config.CLASS_NAMES[k]] = float(2 * tp / max(2 * tp + fp + fn, 1))
    return acc, ious, f1s


def build_loss(device="cpu"):
    """loss(logits, target): class-weighted CE + multiclass Dice, both skipping IGNORE pixels."""
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=device),
                                   ignore_index=dataset.IGNORE)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=dataset.IGNORE)
    return lambda out, target: ce(out, target) + dice(out, target)


def confusion(gt, pred, n=config.N_CLASSES):
    """n x n counts of (label row, prediction column); labels >= n (IGNORE) are skipped."""
    gt = np.asarray(gt).ravel()
    keep = gt < n
    idx = gt[keep].astype(np.int64) * n + np.asarray(pred).ravel()[keep]
    return np.bincount(idx, minlength=n * n).reshape(n, n)


@torch.no_grad()
def evaluate_loader(model, loader, device="cpu"):
    model.eval()
    cm = np.zeros((config.N_CLASSES, config.N_CLASSES), dtype=np.int64)
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(1).cpu().numpy()
        cm += confusion(yb.numpy(), pred)
    return metrics_from_confusion(cm)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    """{"commit", "dirty"} of the code checkout, or None outside git."""
    try:
        git = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], cwd=config.ROOT, capture_output=True, text=True,
            timeout=10, check=True).stdout.strip()
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return None


def check_name(name):
    """name if it is a plain file name (no path parts), else ValueError."""
    if not name or not re.fullmatch(r"[A-Za-z0-9][\w.-]*", name):
        raise ValueError(f"expected a plain name like 'fresh_crops_s0', got {name!r}")
    return name


def run_dir(name):
    """config.MODELS_DIR/runs/<name>; refused if it exists or is a shipped model's name."""
    shipped = [SHIPPED] + [m["file"] for m in config.MODELS.values()]
    if name in {f for s in shipped for f in (s, s.rsplit(".", 1)[0])}:
        raise ValueError(f"refusing --out {name!r}: that is a shipped checkpoint's name")
    run = config.MODELS_DIR / "runs" / check_name(name)
    if run.exists():
        raise FileExistsError(f"{run} already exists; choose a new --out name")
    return run


def new_run_dir(name):
    """Create config.MODELS_DIR/runs/<name>/; an existing run is never reused."""
    run = run_dir(name)
    run.parent.mkdir(parents=True, exist_ok=True)
    try:
        run.mkdir()  # atomic: also loses cleanly to a run started meanwhile
    except FileExistsError:
        raise FileExistsError(f"{run} already exists; choose a new --out name") from None
    return run


@contextlib.contextmanager
def discard_on_failure(run, result):
    """Remove run, a directory this process just created, if the body fails before writing result.

    result is a glob (e.g. "*.pt"). A run that fails before producing it
    has nothing worth keeping and would only block its name; once it exists
    the run is kept as it is.
    """
    try:
        yield
    except BaseException:
        if not any(run.glob(result)):
            with contextlib.suppress(OSError):  # best effort: the run's own error is what matters
                for p in run.iterdir():
                    if p.is_file():
                        p.unlink()
                run.rmdir()  # fails, and keeps the run, if a subdirectory appeared in it
                print(f"removed {run}: the run failed before writing {result}", file=sys.stderr)
        raise


def plain_args(params, skip=()):
    """A function's arguments for JSON logs: values JSON cannot hold (paths, devices) as str."""
    return {k: v if v is None or isinstance(v, (bool, int, float, str, list, tuple)) else str(v)
            for k, v in params.items() if k not in skip}


def default_workers():
    # macOS starts DataLoader workers fresh (no fork) and each unpickles its
    # own copy of the training arrays (~1.1 GB for the v1 train split)
    return 2 if sys.platform == "darwin" else min(4, os.cpu_count() or 1)


def train_loader(ds, batch, workers):
    """DataLoader over a RandomCropDataset, fresh crops every epoch.

    EpochSampler sends the epoch from the main process with every index, so
    call ds.set_epoch(e) before iterating epoch e. Workers then hold no
    epoch of their own and persist: each copies the dataset once per run,
    where restarting them every epoch re-pickled the whole train split each
    time (seconds per epoch, and macOS keeps the freed pickle buffers).
    """
    # Spawned workers (the macOS default) that returned tensors stall ~5 s
    # each at exit (torch 2.13); forkserver children exit at once.
    ctx = "forkserver" if workers and sys.platform == "darwin" else None
    return DataLoader(ds, batch_size=batch, sampler=dataset.EpochSampler(ds), num_workers=workers,
                      persistent_workers=workers > 0, multiprocessing_context=ctx)


def _save(model, path):
    # write-then-rename: an interrupted save never leaves a truncated checkpoint
    tmp = path.with_name(path.name + ".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, path)


def _write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _check_positive(**kw):
    for k, v in kw.items():
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{k} must be a positive integer, got {v!r}")


def main(epochs=20, samples=1200, crop=512, batch=12, lr=3e-4, out_name=None,
         init=None, *, seed=0, select="miou", workers=None, threads=None,
         device="cpu", label_version="v1", encoder_weights="imagenet",
         keep_val_overlap=False, resume=None, new_areas=False, aug="v1", class_aware=0.0,
         amp="none", accum=1):
    """Train into config.MODELS_DIR/runs/<out_name>/ and return a summary dict.

    init (old name: resume) is a checkpoint to start from; it is only read.
    Without it the encoder starts from ImageNet weights unless
    encoder_weights="none". Validation scores every pixel of the val areas
    except those inside train areas (keep_val_overlap=True keeps them);
    select ("miou" or "acc") picks best.pt. new_areas adds the training areas
    of config.AREAS_V2; aug ("v1" or "v2") and class_aware (share of crops
    centred near commercial pixels) are dataset.RandomCropDataset's options;
    amp="bf16" runs forward passes in bfloat16 (the loss stays float32);
    accum splits each batch into that many micro-batches (memory). Arguments
    are checked before anything is loaded or created.
    """
    # a snapshot first: on 3.12 locals() is this frame's own dict, which a tracer
    # reading f_locals (pdb, coverage) re-syncs, resizing it under any iteration
    params = dict(locals())
    args = plain_args(params)
    init = init or resume
    if select not in ("miou", "acc"):
        raise ValueError(f"select must be 'miou' or 'acc', got {select!r}")
    if encoder_weights not in ("imagenet", "none"):
        raise ValueError(f"encoder_weights must be 'imagenet' or 'none', got {encoder_weights!r}")
    _check_positive(epochs=epochs, samples=samples, crop=crop, batch=batch, accum=accum)
    if amp not in ("none", "bf16"):
        raise ValueError(f"amp must be 'none' or 'bf16', got {amp!r}")
    if batch % accum:
        raise ValueError(f"batch {batch} is not a multiple of accum {accum}")
    if crop % 32:
        raise ValueError(f"crop must be a multiple of 32 (the U-Net's stride), got {crop}")
    seed = dataset.check_seed(seed)
    run_dir(out_name)  # fail before the slow data load, create only once all is built
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    workers = default_workers() if workers is None else workers

    train_areas = dataset.load_split("train", label_version=label_version, new_areas=new_areas)
    val_areas = dataset.load_split("val", label_version=label_version,
                                   mask_overlap=not keep_val_overlap)
    tr_ds = dataset.RandomCropDataset(train_areas, crop=crop, samples_per_epoch=samples, seed=seed,
                                      aug=aug, class_aware=class_aware)
    va_ds = dataset.GridDataset(val_areas, crop=crop)
    tr = train_loader(tr_ds, batch, workers)
    # validation only slices arrays; workers would merely copy them
    va = DataLoader(va_ds, batch_size=batch)

    model = build_model(pretrained=encoder_weights == "imagenet" and not init)
    if init:
        model.load_state_dict(torch.load(init, map_location="cpu"))
    model.to(device)
    loss_fn = build_loss(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    masks = {}
    for name, *_ in train_areas + val_areas:
        p = dataset.mask_path(name, label_version)
        masks[name] = {"file": f"{p.parent.name}/{p.name}", "sha256": file_sha256(p)}
    train_names = [a[0] for a in tr_ds.areas]
    cfg = {
        "args": args, "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_info(),
        "init": {"path": str(init), "sha256": file_sha256(init)} if init else None,
        "data": {
            "label_version": label_version, "use_corrected": False, "masks": masks,
            "train_areas": train_names,
            "sampling_weights": dict(zip(train_names, tr_ds.p.round(4).tolist())),
            "val_areas": [a[0] for a in val_areas],
            "val_ignored_share": {a[0]: round(float((a[2] == dataset.IGNORE).mean()), 4)
                                  for a in val_areas},
        },
        "loader": {"workers": workers, "persistent_workers": tr.persistent_workers,
                   "sampler": type(tr.sampler).__name__,
                   "start_method": tr.multiprocessing_context.get_start_method()
                   if tr.multiprocessing_context else None,
                   "iterations_per_epoch": len(tr), "val_crops": len(va_ds)},
        "loss": {"ce_class_weights": CLASS_WEIGHTS, "dice": "multiclass",
                 "ignore_index": dataset.IGNORE},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "smp": smp.__version__, "platform": platform.platform(), "device": str(device)},
    }
    run = new_run_dir(out_name)
    with discard_on_failure(run, "*.pt"):
        _write_json(run / "config.json", cfg)

        best, best_ep, history = float("-inf"), None, []
        with open(run / "train_log.txt", "a", buffering=1) as log:
            log.write(f"run {run.name}: {json.dumps(args, default=str)}\n")
            for ep in range(epochs):
                tr_ds.set_epoch(ep)
                digest = tr_ds.digest()
                model.train()
                t0 = time.time()
                tot = 0.0
                for i, (xb, yb) in enumerate(tr):
                    opt.zero_grad()
                    li = 0.0
                    for xm, ym in zip(xb.chunk(accum), yb.chunk(accum)):
                        xm, ym = xm.to(device), ym.to(device)
                        with torch.autocast(device_type=torch.device(device).type,
                                            dtype=torch.bfloat16, enabled=amp == "bf16"):
                            out = model(xm)
                        loss = loss_fn(out.float(), ym) * (len(xm) / len(xb))
                        loss.backward()
                        li += float(loss.detach())
                    opt.step()
                    tot += li
                    if i % 10 == 0:
                        print(f"ep{ep} it{i}/{len(tr)} loss={li:.4f}", flush=True)
                lr_ep = opt.param_groups[0]["lr"]
                sched.step()
                acc, ious, _ = evaluate_loader(model, va, device)
                miou = float(np.mean(list(ious.values())))
                score = miou if select == "miou" else float(acc)
                msg = (f"epoch {ep}: train_loss={tot/len(tr):.4f} val_acc={acc:.4f} "
                       f"val_mIoU={miou:.4f} ious={ {k: round(v, 3) for k, v in ious.items()} } "
                       f"lr={lr_ep:.2e} crops={digest} ({time.time()-t0:.0f}s)")
                print(msg, flush=True)
                log.write(msg + "\n")
                history.append({"epoch": ep, "train_loss": round(tot / len(tr), 4),
                                "val_acc": round(float(acc), 4), "val_mIoU": round(miou, 4),
                                "crops": digest})
                _save(model, run / "last.pt")
                if score > best:
                    best, best_ep = score, ep
                    _save(model, run / "best.pt")
                    log.write(f"  saved best.pt (val_{select}={score:.4f})\n")
        best = round(best, 4) if best_ep is not None else None
        cfg["result"] = {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "select": select,
                         "best_epoch": best_ep, f"best_val_{select}": best, "history": history}
        _write_json(run / "config.json", cfg)
        return {"run_dir": str(run), "best_epoch": best_ep, "best": best, "history": history}


def _seed_arg(text):
    try:
        return dataset.check_seed(int(text))
    except ValueError as ex:
        raise argparse.ArgumentTypeError(str(ex)) from None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog=f"python -m {__package__}.train",
        description="Train the U-Net into models/runs/NAME/ (never over an existing file).")
    ap.add_argument("epochs", nargs="?", type=int, default=20)
    ap.add_argument("samples", nargs="?", type=int, default=1200, help="random crops per epoch")
    ap.add_argument("--out", required=True, metavar="NAME",
                    help="new run name; writes models/runs/NAME/")
    ap.add_argument("--init", metavar="CKPT", help="start from these weights (read only)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--seed", type=_seed_arg, default=0,
                    help="crop sampling seed, 0 <= seed < 2**32")
    ap.add_argument("--select", choices=("miou", "acc"), default="miou",
                    help="val metric that picks best.pt")
    ap.add_argument("--workers", type=int,
                    help="DataLoader workers (default: 2 on macOS, else min(4, CPUs))")
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    ap.add_argument("--label-version", default="v1",
                    help="v1 = data/masks (data_v2/masks_v1 for new areas), "
                         "else data_v2/masks_<version> (v2, v2t: see labels_v2)")
    ap.add_argument("--encoder-weights", choices=("imagenet", "none"), default="imagenet",
                    help="encoder start when there is no --init (imagenet needs a download)")
    ap.add_argument("--keep-val-overlap", action="store_true",
                    help="also score val pixels that lie inside train areas")
    ap.add_argument("--new-areas", action="store_true",
                    help="also train on the config.AREAS_V2 training areas (data_v2/)")
    ap.add_argument("--aug", choices=("v1", "v2"), default="v1",
                    help="v2 adds scale, colour, blur and JPEG jitter")
    ap.add_argument("--class-aware", type=float, default=0.0,
                    help="share of crops centred near commercial pixels (0-1)")
    ap.add_argument("--amp", choices=("none", "bf16"), default="none",
                    help="bfloat16 forward passes (MPS / CUDA speed-up)")
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per batch (lower memory, same result)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    main(a.epochs, a.samples, crop=a.crop, batch=a.batch, lr=a.lr, out_name=a.out,
         init=a.init, seed=a.seed, select=a.select, workers=a.workers, threads=a.threads,
         device=a.device, label_version=a.label_version, encoder_weights=a.encoder_weights,
         keep_val_overlap=a.keep_val_overlap, new_areas=a.new_areas, aug=a.aug,
         class_aware=a.class_aware, amp=a.amp, accum=a.accum)
