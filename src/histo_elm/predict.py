"""Classify one or more images with a pipeline saved by `run.py --save-models`.

    python -m histo_elm.predict results/lc25000/models/deit-b.joblib img1.jpeg img2.jpeg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from .features import extract_deep_features, iter_raw_pixel_batches
from .pipeline import predict_by_voting


def predict(artifact_path: Path, image_paths: list[Path], device: str = "cuda") -> list[str]:
    artifact = joblib.load(artifact_path)
    pipeline = artifact["pipeline"]
    if artifact["backbone"] is None:
        features = next(iter_raw_pixel_batches(image_paths, artifact["raw_size"], len(image_paths)))
        labels = pipeline.predict(features)
    elif artifact.get("patches"):
        features = extract_deep_features(artifact["backbone"], image_paths, device, num_workers=0, patches=True)
        labels = predict_by_voting(pipeline, features, len(image_paths))
    else:
        features = extract_deep_features(artifact["backbone"], image_paths, device, num_workers=0)
        labels = pipeline.predict(features)
    return [artifact["class_names"][i] for i in np.asarray(labels)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    for path, label in zip(args.images, predict(args.model, args.images, args.device)):
        print(f"{path.name}: {label}")


if __name__ == "__main__":
    main()
