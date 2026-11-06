from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import maximum_filter
import torch


@dataclass(frozen=True)
class DatasetSpec:
    cube: str
    label: str
    classes: int

DATASETS = {
    "IP": DatasetSpec("IP/indian_pines_corrected.mat", "IP/indian_pines_gt.mat", 16),
    "SA": DatasetSpec("salinas/salinas_corrected.mat", "salinas/salinas_gt.mat", 16),
    "UP": DatasetSpec("UP/paviaU/PaviaU.mat", "UP/paviaU/PaviaU_gt.mat", 9),
}

def read_mat(path: Path) -> np.ndarray:
    """Each provided MAT file has one user array; reject ambiguous files."""
    values = [value for key, value in sio.loadmat(path).items() if not key.startswith("__")]
    if len(values) != 1:
        raise ValueError(f"Expected one array in {path}, found {len(values)}")
    return values[0]

class PatchScene:
    def __init__(self, root: Path, name: str, patch_size: int):
        spec = DATASETS[name]
        cube = read_mat(root / spec.cube).astype(np.float32)
        if cube.ndim != 3:
            raise ValueError(f"{name}: expected H x W x bands cube")
        # Standardize each band with unlabeled whole-scene statistics.
        mean = cube.mean(axis=(0, 1), keepdims=True)
        std = cube.std(axis=(0, 1), keepdims=True)
        cube = (cube - mean) / np.maximum(std, 1e-6)
        radius = patch_size // 2
        self.cube = np.pad(cube, ((radius, radius), (radius, radius), (0, 0)))
        self.shape = cube.shape[:2]
        self.bands = cube.shape[-1]
        self.patch_size = patch_size
        self.classes = spec.classes
        self.labels = read_mat(root / spec.label).astype(np.int64)
        self.train_labels = self.labels
        self.test_labels = self.labels
        if self.train_labels.shape != self.shape or self.test_labels.shape != self.shape:
            raise ValueError(f"{name}: cube and label image sizes differ")
        if self.train_labels.max() != self.classes or self.test_labels.max() != self.classes:
            raise ValueError(f"{name}: unexpected class count in labels")

    def patches(self, centers: np.ndarray) -> torch.Tensor:
        """Centers are original image row/column coordinates."""
        offsets = np.arange(self.patch_size)
        rows = centers[:, 0, None, None] + offsets[None, :, None]
        cols = centers[:, 1, None, None] + offsets[None, None, :]
        x = self.cube[rows, cols, :].transpose(0, 3, 1, 2).copy()
        return torch.from_numpy(x)

    def split(self, shots: int, rng: np.random.RandomState, guard_radius: int = 0):
        support = []
        for cls in range(1, self.classes + 1):
            coords = np.argwhere(self.train_labels == cls)
            if len(coords) < shots:
                raise ValueError(f"Class {cls} has {len(coords)} train pixels, needs {shots}")
            support.append(coords[rng.permutation(len(coords))[:shots]])
        support = np.concatenate(support, axis=0)
        support_y = self.train_labels[support[:, 0], support[:, 1]] - 1
        test_mask = self.test_labels > 0
        if self.labels is not None:
            test_mask[support[:, 0], support[:, 1]] = False
        if guard_radius:
            selected = np.zeros(self.shape, dtype=bool)
            selected[support[:, 0], support[:, 1]] = True
            forbidden = maximum_filter(selected, size=2 * guard_radius + 1)
            test_mask &= ~forbidden
        test = np.argwhere(test_mask)
        test_y = self.test_labels[test[:, 0], test[:, 1]] - 1
        if len(test) == 0:
            raise ValueError("No test pixels remain after applying spatial guard")
        missing = sorted(set(range(self.classes)) - set(np.unique(test_y).tolist()))
        if missing:
            raise ValueError(f"Spatial guard leaves no test pixels for classes {missing}; "
                             "choose a different split or reduce guard radius")
        return support, support_y, test, test_y

class SourcePatches:
    def __init__(self, root: Path):
        with (root / "Chikusei_imdb_128_7_7.pickle").open("rb") as handle:
            data = pickle.load(handle)
        x = np.asarray(data["data"], dtype=np.float32)
        y = np.asarray(data["Labels"], dtype=np.int64)
        if x.ndim != 4 or x.shape[1:] != (7, 7, 128):
            raise ValueError(f"Unexpected Chikusei patch shape: {x.shape}")
        self.x = x
        self.by_class = {int(c): np.flatnonzero(y == c) for c in np.unique(y)}

    def episode(self, ways: int, shots: int, queries: int, rng: np.random.Generator):
        classes = rng.choice(list(self.by_class), ways, replace=False)
        chosen_s, chosen_q = [], []
        for c in classes:
            ids = rng.choice(self.by_class[int(c)], shots + queries, replace=False)
            chosen_s.extend(ids[:shots])
            chosen_q.extend(ids[shots:])
        s = torch.from_numpy(self.x[chosen_s].transpose(0, 3, 1, 2).copy())
        q = torch.from_numpy(self.x[chosen_q].transpose(0, 3, 1, 2).copy())
        sy = torch.arange(ways).repeat_interleave(shots)
        qy = torch.arange(ways).repeat_interleave(queries)
        return s, sy, q, qy


