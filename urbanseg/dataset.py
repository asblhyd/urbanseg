"""PyTorch datasets over the stitched area images + masks."""
import hashlib
import io
import json
import math
import re
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset, Sampler

from . import config, imagery

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IGNORE = 255  # mask value that losses and metrics skip


def check_seed(v, what="seed"):
    """v as an int in [0, 2**32), else ValueError.

    SeedSequence splits larger ints into several 32-bit words and pads short
    keys with zeros, so (2**32, 0, i) would replay (0, 1, i); one word per
    key part keeps every (seed, epoch, idx) stream distinct.
    """
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v < 2 ** 32:
        raise ValueError(f"{what} must be an integer in [0, 2**32), got {v!r}")
    return int(v)


def normalize(img_u8):
    x = img_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def is_v2_area(name):
    return any(a[0] == name for a in config.AREAS_V2)


def area_splits(new_areas=False):
    """[(name, split)] of config.AREAS, plus config.AREAS_V2 when new_areas."""
    out = [(a[0], a[5]) for a in config.AREAS]
    if new_areas:
        out += [(a[0], a[5]) for a in config.AREAS_V2]
    return out


def mask_path(name, label_version="v1", use_corrected=False):
    """Label file of an area (read-only).

    v1 is data/masks/<area>.png for the original areas and
    data_v2/masks_v1/<area>.png for config.AREAS_V2; any later version is
    data_v2/masks_<version>/<area>.png, so data/ stays the frozen v1 set.
    A hand-corrected <area>_gt.png next to it is used only with use_corrected.
    """
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*", label_version):
        raise ValueError(f"bad label version {label_version!r}")
    if label_version == "v1":
        d = config.DATA_V2_DIR / "masks_v1" if is_v2_area(name) else config.MASKS_DIR
    else:
        d = config.DATA_V2_DIR / f"masks_{label_version}"
    gt = d / f"{name}_gt.png"
    return gt if use_corrected and gt.exists() else d / f"{name}.png"


def overlap_with_train(meta, shape):
    """bool HxW: pixels of a georeferenced area whose centre lies in a train area.

    Train-area footprints come from their tiles/<area>/meta.json (world map px),
    so no imagery is loaded. Areas with weight 0 count too: they are still
    train-split data.
    """
    s, H, W = meta["scale"], shape[0], shape[1]
    out = np.zeros((H, W), dtype=bool)

    def px(world, origin, n):
        # pixel i covers world [origin + i/s, origin + (i+1)/s); test its centre
        return int(np.clip(np.ceil((world - origin) * s - 0.5), 0, n))

    for name, split in area_splits(new_areas=True):
        if split != "train":
            continue
        meta_file = imagery.area_tiles_dir(name) / name / "meta.json"
        if not meta_file.exists():  # a v2 area not fetched yet cannot overlap anything
            continue
        m = json.loads(meta_file.read_text())
        f = 2.0 ** (meta["zoom"] - m["zoom"])
        c0, c1 = (px(v * f, meta["x0"], W) for v in (m["x0"], m["x0"] + m["width_map_px"]))
        r0, r1 = (px(v * f, meta["y0"], H) for v in (m["y0"], m["y0"] + m["height_map_px"]))
        out[r0:r1, c0:c1] = True
    return out


def load_split(split, *, label_version="v1", use_corrected=False, mask_overlap=False,
               new_areas=False):
    """[(name, img HxWx3 uint8, mask HxW uint8, meta)] for every area of a split.

    Labels come from mask_path(name, label_version, use_corrected), so a
    corrected *_gt.png is never picked up silently. mask_overlap=True sets the
    pixels of a val/test area that lie inside any train area to IGNORE (21.6%
    of lbnagar lies inside vanasthalipuram). new_areas adds config.AREAS_V2
    (the "test2" split always means those held-out areas).
    """
    out = []
    for name, sp in area_splits(new_areas or split == "test2"):
        if sp != split:
            continue
        img, meta = imagery.load_area(name)
        mask = np.asarray(Image.open(mask_path(name, label_version, use_corrected))).copy()
        if mask_overlap and split != "train":
            mask[overlap_with_train(meta, mask.shape)] = IGNORE
        out.append((name, img, mask, meta))
    return out


class Crop(NamedTuple):
    """crop px square of areas[area] at (y, x), rot90 by rot, h-flipped, then im * gain + bias.

    aug="v2" adds: a size px square read and resized to crop px (scale
    jitter), per-channel gains gr/gg/gb, saturation sat, Gaussian blur sigma
    blur and JPEG quality jpeg (0 = none).
    """
    area: int
    y: int
    x: int
    rot: int
    flip: bool
    gain: float
    bias: float
    size: int = 0
    gr: float = 1.0
    gg: float = 1.0
    gb: float = 1.0
    sat: float = 1.0
    blur: float = 0.0
    jpeg: int = 0


class RandomCropDataset(Dataset):
    """Random augmented crops; areas sampled by label-quality weight.

    Sample idx of epoch e is a pure function of (seed, e, idx): its RNG is
    seeded from SeedSequence([seed, e, idx]), so every epoch draws new crops
    and augmentations, reruns are identical, and the result does not depend
    on which DataLoader worker serves it. ds[i] is sample i of the epoch
    last given to set_epoch; ds[(e, i)] names the epoch itself, which is
    how EpochSampler hands it to DataLoader workers. Load it with
    EpochSampler (train.train_loader does): with plain indices a persistent
    worker would keep the epoch of the copy it was started with.
    """

    AUGS = ("v1", "v2")
    SCALE_RANGE = (0.8, 1.25)  # objects shown 0.8-1.25x their native size (uploads)

    def __init__(self, areas, crop=512, samples_per_epoch=1200, seed=0, aug="v1",
                 class_aware=0.0):
        if aug not in self.AUGS:
            raise ValueError(f"aug must be one of {self.AUGS}, got {aug!r}")
        if not 0 <= class_aware <= 1:
            raise ValueError(f"class_aware must be in [0, 1], got {class_aware!r}")
        self.areas = [a for a in areas
                      if config.AREA_WEIGHTS.get(a[0], 1.0) > 0]
        w = np.array([config.AREA_WEIGHTS.get(a[0], 1.0) for a in self.areas])
        self.p = w / w.sum()
        self.crop = crop
        self.n = samples_per_epoch
        self.seed = check_seed(seed)
        self.epoch = 0
        self.aug, self.class_aware = aug, float(class_aware)
        # class_aware: share of crops centred near a commercial pixel (7.7% of
        # label pixels, the weakest class); anchors are a sparse sample of them
        self.anchors = []
        if self.class_aware:
            for _, _, mask, _ in self.areas:
                ys, xs = np.nonzero(mask[::8, ::8] == 2)
                self.anchors.append(np.stack([ys * 8, xs * 8], 1))

    def set_epoch(self, epoch):
        self.epoch = check_seed(epoch, "epoch")

    def __len__(self):
        return self.n

    def draw(self, idx, epoch=None):
        """Crop and augmentation of sample idx of epoch (default: the current); no pixels read."""
        ep = self.epoch if epoch is None else check_seed(epoch, "epoch")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, ep, int(idx)]))
        a = int(rng.choice(len(self.areas), p=self.p))
        h, w = self.areas[a][1].shape[:2]
        c = self.crop
        if self.aug == "v1" and not self.class_aware:  # exactly the original draws
            y = int(rng.integers(0, h - c + 1))
            x = int(rng.integers(0, w - c + 1))
            rot = int(rng.integers(4))
            flip = bool(rng.random() < 0.5)
            gain = 1.0 + rng.uniform(-0.2, 0.2)
            bias = rng.uniform(-20, 20)
            return Crop(a, y, x, rot, flip, float(gain), float(bias))
        size = c
        if self.aug == "v2":
            lo, hi = self.SCALE_RANGE
            size = min(int(round(c / math.exp(rng.uniform(math.log(lo), math.log(hi))))), h, w)
        anchors = self.anchors[a] if self.class_aware else ()
        if len(anchors) and rng.random() < self.class_aware:
            cy, cx = anchors[rng.integers(len(anchors))]
            j = size // 4  # the anchor lands anywhere in the central half
            y = int(np.clip(cy - size // 2 + rng.integers(-j, j + 1), 0, h - size))
            x = int(np.clip(cx - size // 2 + rng.integers(-j, j + 1), 0, w - size))
        else:
            y = int(rng.integers(0, h - size + 1))
            x = int(rng.integers(0, w - size + 1))
        rot = int(rng.integers(4))
        flip = bool(rng.random() < 0.5)
        gain = float(1.0 + rng.uniform(-0.2, 0.2))
        bias = float(rng.uniform(-20, 20))
        if self.aug == "v1":
            return Crop(a, y, x, rot, flip, gain, bias, size)
        gr, gg, gb = (float(v) for v in 1.0 + rng.uniform(-0.08, 0.08, 3))
        sat = float(rng.uniform(0.75, 1.25))
        blur = float(rng.uniform(0.3, 1.2)) if rng.random() < 0.3 else 0.0
        jpeg = int(rng.integers(55, 96)) if rng.random() < 0.3 else 0
        return Crop(a, y, x, rot, flip, gain, bias, size, gr, gg, gb, sat, blur, jpeg)

    def digest(self):
        """Short fingerprint of this epoch's crops, for logs: a repeat shows as an equal digest."""
        draws = np.array([self.draw(i) for i in range(self.n)], dtype=np.float64)
        return hashlib.sha1(draws.tobytes()).hexdigest()[:12]

    def __getitem__(self, idx):
        ep, i = idx if isinstance(idx, tuple) else (None, idx)
        d = self.draw(i, ep)
        _, img, mask, _ = self.areas[d.area]
        c = self.crop
        size = d.size or c
        im = img[d.y:d.y + size, d.x:d.x + size].copy()
        mk = mask[d.y:d.y + size, d.x:d.x + size].copy()
        if size != c:  # scale jitter: the model sees objects c/size times larger
            im = np.asarray(Image.fromarray(im).resize((c, c), Image.BILINEAR))
            mk = np.asarray(Image.fromarray(mk).resize((c, c), Image.NEAREST))
        if d.rot:
            im = np.rot90(im, d.rot).copy()
            mk = np.rot90(mk, d.rot).copy()
        if d.flip:
            im = im[:, ::-1].copy()
            mk = mk[:, ::-1].copy()
        # photometric jitter
        f = im.astype(np.float32) * d.gain + d.bias
        if self.aug == "v2":  # Google's captures differ in colour, sharpness and compression
            f *= np.array([d.gr, d.gg, d.gb], np.float32)
            grey = f.mean(2, keepdims=True)
            f = grey + d.sat * (f - grey)
            if d.blur:
                f = ndi.gaussian_filter(f, (d.blur, d.blur, 0))
        im = np.clip(f, 0, 255).astype(np.uint8)
        if d.jpeg:
            buf = io.BytesIO()
            Image.fromarray(im).save(buf, format="JPEG", quality=d.jpeg)
            im = np.asarray(Image.open(buf).convert("RGB"))
        return normalize(im), torch.from_numpy(mk.astype(np.int64))


class EpochSampler(Sampler):
    """Indices (epoch, i) of a RandomCropDataset, the epoch read as each epoch's iteration starts.

    The epoch travels with every index from the main process, so DataLoader
    workers hold no epoch of their own and may persist: the dataset is then
    copied into each worker once per run instead of once per epoch.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        ep = self.ds.epoch
        return iter([(ep, i) for i in range(len(self.ds))])


def _grid(n, c):
    """[(start, overlap)] of crops of size c covering range(n) completely.

    A side that is not a multiple of c gets one more crop flush with its end;
    overlap is how many of its leading rows/columns the previous crop scored.
    """
    if n < c:
        raise ValueError(f"area side of {n} px is smaller than the {c} px crop")
    out = [(s, 0) for s in range(0, n - c + 1, c)]
    end = out[-1][0] + c
    if end < n:
        out.append((n - c, end - (n - c)))
    return out


class GridDataset(Dataset):
    """Deterministic crops that score every pixel of each area exactly once (validation).

    The last row/column of crops is placed flush with the far edge, and the
    pixels it shares with the previous crop are set to IGNORE in its target.
    """

    def __init__(self, areas, crop=512):
        self.items = []
        self.areas = areas
        for ai, (_, img, _, _) in enumerate(areas):
            for y, oy in _grid(img.shape[0], crop):
                for x, ox in _grid(img.shape[1], crop):
                    self.items.append((ai, y, x, oy, ox))
        self.crop = crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ai, y, x, oy, ox = self.items[idx]
        _, img, mask, _ = self.areas[ai]
        c = self.crop
        im = img[y:y + c, x:x + c]
        mk = mask[y:y + c, x:x + c].astype(np.int64)
        mk[:oy] = IGNORE
        mk[:, :ox] = IGNORE
        return normalize(im), torch.from_numpy(mk)
