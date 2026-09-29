"""PyTorch datasets over the stitched area images + masks."""
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config, imagery

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img_u8):
    x = img_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def load_split(split):
    from PIL import Image
    out = []
    for name, _lat, _lng, _nx, _ny, sp in config.AREAS:
        if sp != split:
            continue
        img, meta = imagery.load_area(name)
        p = config.MASKS_DIR / f"{name}_gt.png"   # corrected GT if present
        if not p.exists():
            p = config.MASKS_DIR / f"{name}.png"
        mask = np.asarray(Image.open(p))
        out.append((name, img, mask.copy(), meta))
    return out


class RandomCropDataset(Dataset):
    """Random augmented crops; areas sampled by label-quality weight."""

    def __init__(self, areas, crop=512, samples_per_epoch=1200):
        self.areas = [a for a in areas
                      if config.AREA_WEIGHTS.get(a[0], 1.0) > 0]
        w = np.array([config.AREA_WEIGHTS.get(a[0], 1.0) for a in self.areas])
        self.p = w / w.sum()
        self.crop = crop
        self.n = samples_per_epoch

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(torch.initial_seed() % (2**31) + idx)
        _, img, mask, _ = self.areas[rng.choice(len(self.areas), p=self.p)]
        c = self.crop
        y = rng.integers(0, img.shape[0] - c + 1)
        x = rng.integers(0, img.shape[1] - c + 1)
        im = img[y:y + c, x:x + c].copy()
        mk = mask[y:y + c, x:x + c].copy()
        k = rng.integers(4)
        if k:
            im = np.rot90(im, k).copy()
            mk = np.rot90(mk, k).copy()
        if rng.random() < 0.5:
            im = im[:, ::-1].copy()
            mk = mk[:, ::-1].copy()
        # photometric jitter
        f = 1.0 + rng.uniform(-0.2, 0.2)
        b = rng.uniform(-20, 20)
        im = np.clip(im.astype(np.float32) * f + b, 0, 255).astype(np.uint8)
        return normalize(im), torch.from_numpy(mk.astype(np.int64))


class GridDataset(Dataset):
    """Deterministic non-overlapping crops (for validation)."""

    def __init__(self, areas, crop=512):
        self.items = []
        self.areas = areas
        for ai, (_, img, _, _) in enumerate(areas):
            for y in range(0, img.shape[0] - crop + 1, crop):
                for x in range(0, img.shape[1] - crop + 1, crop):
                    self.items.append((ai, y, x))
        self.crop = crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ai, y, x = self.items[idx]
        _, img, mask, _ = self.areas[ai]
        c = self.crop
        im = img[y:y + c, x:x + c]
        mk = mask[y:y + c, x:x + c]
        return normalize(im), torch.from_numpy(mk.astype(np.int64))
