"""Dataset discovery and leakage-aware train/test splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Folder names that hold each class, in label order.
DATASET_CLASSES: dict[str, list[str]] = {
    "bach": ["Benign", "InSitu", "Invasive", "Normal"],
    "breakhis": ["benign", "malignant"],
    "lc25000": ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"],
}

# BreakHis file names encode the patient: SOB_B_A-14-22549AB-40-001.png -> "14-22549AB"
_BREAKHIS_PATIENT = re.compile(r"SOB_[BM]_[A-Z]+-(\d+-[0-9A-Z]+)-\d+-\d+", re.IGNORECASE)
_BREAKHIS_MAGNIFICATION = re.compile(r"-(\d+)-\d+\.\w+$")


@dataclass(frozen=True)
class DataSplit:
    train_paths: list[Path]
    train_labels: np.ndarray
    test_paths: list[Path]
    test_labels: np.ndarray
    class_names: list[str]


def _images_under(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def discover_images(dataset: str, root: str | Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """Find every image of `dataset` below `root`.

    Each class is located by its folder name (case-insensitive), so the official
    archive layouts work as-is, e.g. ``ICIAR2018_BACH_Challenge/Photos/Benign/``,
    ``BreaKHis_v1/histology_slides/breast/benign/`` or ``lung_image_sets/lung_aca/``.
    """
    if dataset not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATASET_CLASSES)}")
    root = Path(root)
    classes = DATASET_CLASSES[dataset]
    all_dirs = [d for d in root.rglob("*") if d.is_dir()] + [root]

    paths: list[Path] = []
    labels: list[int] = []
    for label, class_name in enumerate(classes):
        class_dirs = [d for d in all_dirs if d.name.lower() == class_name.lower()]
        files = sorted({f for d in class_dirs for f in _images_under(d)})
        if not files:
            raise FileNotFoundError(f"No images found for class {class_name!r} under {root}")
        paths.extend(files)
        labels.extend([label] * len(files))
    return paths, np.asarray(labels), classes


def breakhis_patient_id(path: Path) -> str:
    match = _BREAKHIS_PATIENT.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse BreakHis patient id from {path.name!r}")
    return match.group(1)


def breakhis_magnification(path: Path) -> int | None:
    match = _BREAKHIS_MAGNIFICATION.search(path.name)
    return int(match.group(1)) if match else None


def make_split(dataset: str, root: str | Path, test_size: float = 0.2, seed: int = 42,
               magnification: int | None = None) -> DataSplit:
    """Stratified train/test split.

    BreakHis is split **by patient** so that images of the same patient (and the same
    tissue region at other magnifications) never appear in both train and test.
    """
    paths, labels, classes = discover_images(dataset, root)

    if dataset == "breakhis":
        if magnification is not None:
            keep = [i for i, p in enumerate(paths) if breakhis_magnification(p) == magnification]
            paths, labels = [paths[i] for i in keep], labels[keep]
        groups = np.asarray([breakhis_patient_id(p) for p in paths])
        n_splits = round(1 / test_size)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(splitter.split(paths, labels, groups))
    else:
        indices = np.arange(len(paths))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=labels,
                                               random_state=seed)

    return DataSplit(
        train_paths=[paths[i] for i in train_idx],
        train_labels=labels[train_idx],
        test_paths=[paths[i] for i in test_idx],
        test_labels=labels[test_idx],
        class_names=classes,
    )
