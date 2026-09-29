"""Train the 4-class U-Net. Run: python -m urbanseg.train [epochs] [samples/epoch]"""
import sys
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from . import config, dataset


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


@torch.no_grad()
def evaluate_loader(model, loader):
    model.eval()
    cm = np.zeros((config.N_CLASSES, config.N_CLASSES), dtype=np.int64)
    for xb, yb in loader:
        pred = model(xb).argmax(1)
        idx = yb.numpy().ravel() * config.N_CLASSES + pred.numpy().ravel()
        cm += np.bincount(idx, minlength=config.N_CLASSES ** 2).reshape(
            config.N_CLASSES, config.N_CLASSES)
    return metrics_from_confusion(cm)


def main(epochs=20, samples=1200, crop=512, batch=12, lr=3e-4,
         out_name="unet_r34.pt", resume=None):
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
    torch.manual_seed(0)
    train_areas = dataset.load_split("train")
    val_areas = dataset.load_split("val")
    tr_ds = dataset.RandomCropDataset(train_areas, crop=crop, samples_per_epoch=samples)
    va_ds = dataset.GridDataset(val_areas, crop=crop)
    tr = DataLoader(tr_ds, batch_size=batch, num_workers=8, persistent_workers=True)
    va = DataLoader(va_ds, batch_size=batch, num_workers=4)

    model = build_model()
    if resume:
        model.load_state_dict(torch.load(resume, map_location="cpu"))
    w = torch.tensor([1.0, 1.0, 2.0, 0.8])
    ce = torch.nn.CrossEntropyLoss(weight=w)
    dice = smp.losses.DiceLoss(mode="multiclass")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 0.0
    log = open(config.MODELS_DIR / "train_log.txt", "a", buffering=1)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for i, (xb, yb) in enumerate(tr):
            opt.zero_grad()
            out = model(xb)
            loss = ce(out, yb) + dice(out, yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            if i % 10 == 0:
                print(f"ep{ep} it{i}/{len(tr)} loss={float(loss):.4f}", flush=True)
        sched.step()
        acc, ious, f1s = evaluate_loader(model, va)
        miou = np.mean(list(ious.values()))
        msg = (f"epoch {ep}: train_loss={tot/len(tr):.4f} val_acc={acc:.4f} "
               f"val_mIoU={miou:.4f} ious={ {k: round(v,3) for k,v in ious.items()} } "
               f"({time.time()-t0:.0f}s)")
        print(msg, flush=True)
        log.write(msg + "\n")
        torch.save(model.state_dict(), config.MODELS_DIR / f"last_{out_name}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), config.MODELS_DIR / out_name)
            log.write(f"  saved best (val_acc={acc:.4f})\n")
    log.close()


if __name__ == "__main__":
    kw = {}
    if len(sys.argv) > 1:
        kw["epochs"] = int(sys.argv[1])
    if len(sys.argv) > 2:
        kw["samples"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        kw["resume"] = sys.argv[3]
    if len(sys.argv) > 4:
        kw["lr"] = float(sys.argv[4])
    if len(sys.argv) > 5:
        kw["out_name"] = sys.argv[5]
    main(**kw)
