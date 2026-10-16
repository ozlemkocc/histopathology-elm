"""StandardScaler -> PCA -> ELM pipelines for deep features and (streamed) raw pixels."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .elm import ELMClassifier
from .features import iter_raw_pixel_batches


def _elm(n_neurons: int, ufunc: str, alpha: float, seed: int) -> ELMClassifier:
    return ELMClassifier(n_neurons=n_neurons, ufunc=ufunc, alpha=alpha, random_state=seed)


def fit_feature_pipeline(X_train: np.ndarray, y_train: np.ndarray, *, n_components: int = 512,
                         n_neurons: int = 2048, ufunc: str = "sigm", alpha: float = 1e-3,
                         seed: int = 42) -> Pipeline:
    """Fit scaler + PCA + ELM on in-memory features. PCA is fitted on training data only."""
    n_components = min(n_components, X_train.shape[0], X_train.shape[1])
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("elm", _elm(n_neurons, ufunc, alpha, seed)),
    ])
    return pipeline.fit(X_train, y_train)


def _chunks(n_items: int, chunk_size: int, min_size: int) -> list[slice]:
    """Index chunks of `chunk_size`; a too-small tail is merged into the previous chunk
    (IncrementalPCA needs every batch to have at least n_components samples)."""
    bounds = list(range(0, n_items, chunk_size)) + [n_items]
    slices = [slice(a, b) for a, b in zip(bounds, bounds[1:])]
    if len(slices) > 1 and slices[-1].stop - slices[-1].start < min_size:
        last = slices.pop()
        slices[-1] = slice(slices[-1].start, last.stop)
    return slices


def fit_raw_pixel_pipeline(train_paths: Sequence[Path], y_train: np.ndarray, *, image_size: int = 224,
                           n_components: int = 512, n_neurons: int = 2048, ufunc: str = "sigm",
                           alpha: float = 1e-3, seed: int = 42,
                           chunk_size: int = 2048) -> Pipeline:
    """Raw-pixel baseline that never holds the full pixel matrix in memory.

    224x224x3 images give 150,528 features per image (~12 GB for LC25000 train), so the
    scaler and PCA are fitted incrementally over chunks of images.
    """
    n_components = min(n_components, len(train_paths))
    chunk_size = max(chunk_size, n_components)
    slices = _chunks(len(train_paths), chunk_size, n_components)

    def batches() -> Iterator[np.ndarray]:
        for s in slices:
            yield next(iter_raw_pixel_batches(train_paths[s], image_size, batch_size=s.stop - s.start))

    scaler = StandardScaler()
    for batch in batches():
        scaler.partial_fit(batch)
    ipca = IncrementalPCA(n_components=n_components)
    for batch in batches():
        ipca.partial_fit(scaler.transform(batch))

    reduced = np.concatenate([ipca.transform(scaler.transform(b)) for b in batches()])
    elm = _elm(n_neurons, ufunc, alpha, seed).fit(reduced, y_train)
    return Pipeline([("scaler", scaler), ("pca", ipca), ("elm", elm)])


def predict_by_voting(pipeline: Pipeline, patch_features: np.ndarray, n_images: int) -> np.ndarray:
    """Image-level prediction from patch features: average the ELM scores of each image's
    patches (soft voting) and take the best class."""
    scores = pipeline.decision_function(patch_features)
    scores = scores.reshape(n_images, -1, scores.shape[1]).mean(axis=1)
    return pipeline.classes_[scores.argmax(axis=1)]


def predict_raw_pixels(pipeline: Pipeline, paths: Sequence[Path], image_size: int = 224,
                       batch_size: int = 512) -> np.ndarray:
    return np.concatenate([pipeline.predict(b) for b in iter_raw_pixel_batches(paths, image_size, batch_size)])
