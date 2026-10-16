"""Train and evaluate every feature extractor + ELM combination on one dataset.

Example (Colab, GPU):
    python -m histo_elm.run --dataset lc25000 --data-dir /content/lc25000 --out-dir results/lc25000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import DATASET_CLASSES, make_split
from .evaluate import compute_metrics, save_confusion_matrix, save_summary_plot, text_report
from .features import BACKBONES, RAW_PIXELS, extract_deep_features
from .pipeline import (fit_feature_pipeline, fit_raw_pixel_pipeline, predict_by_voting,
                       predict_raw_pixels)


def _slug(name: str) -> str:
    return name.lower().replace("/", "-").replace(" ", "_")


def _cached_features(cache: Path, name: str, compute) -> np.ndarray:
    """Feature extraction is the slow step; cache it so ELM/PCA settings can be tuned freely."""
    path = cache / f"{name}.npy"
    if path.exists():
        return np.load(path)
    features = compute()
    np.save(path, features)
    return features


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CLASSES))
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--models", nargs="+", default=[RAW_PIXELS, *BACKBONES],
                   choices=[RAW_PIXELS, *BACKBONES], metavar="MODEL",
                   help=f"Subset of: {RAW_PIXELS!r}, " + ", ".join(map(repr, BACKBONES)))
    p.add_argument("--magnification", type=int, choices=[40, 100, 200, 400],
                   help="BreakHis only: use a single magnification level")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-components", type=int, default=512)
    p.add_argument("--elm-neurons", type=int, default=2048)
    p.add_argument("--elm-ufunc", default="sigm", choices=["sigm", "tanh", "relu"])
    p.add_argument("--elm-alpha", type=float, default=1e-3)
    p.add_argument("--raw-size", type=int, default=224, help="Image side length for the raw-pixel baseline")
    p.add_argument("--patches", action="store_true",
                   help="Deep models: 5 patches per image (corners + centre); test images are "
                        "classified by soft voting over their patches")
    p.add_argument("--augment-copies", type=int, default=1,
                   help="With --patches: augmented copies of each training image (brightness, "
                        "contrast, 90° rotation / flip). Test images are never augmented.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-models", action="store_true", help="Store fitted pipelines for predict.py")
    p.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)  # offline tests
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    out = args.out_dir
    cache = out / "features"
    for d in (out, cache, out / "confusion_matrices", out / "reports"):
        d.mkdir(parents=True, exist_ok=True)

    split = make_split(args.dataset, args.data_dir, args.test_size, args.seed, args.magnification)
    print(f"{args.dataset}: {len(split.train_paths)} train / {len(split.test_paths)} test images, "
          f"{len(split.class_names)} classes")
    elm_kwargs = dict(n_components=args.pca_components, n_neurons=args.elm_neurons,
                      ufunc=args.elm_ufunc, alpha=args.elm_alpha, seed=args.seed)

    rows = []
    for name in args.models:
        print(f"\n=== {name} ===")
        if name == RAW_PIXELS:
            start = time.perf_counter()
            pipeline = fit_raw_pixel_pipeline(split.train_paths, split.train_labels,
                                              image_size=args.raw_size, **elm_kwargs)
            fit_seconds = time.perf_counter() - start
            y_pred = predict_raw_pixels(pipeline, split.test_paths, args.raw_size)
        else:
            timm_id = BACKBONES[name]

            def extract(paths, copies: int):
                return extract_deep_features(timm_id, paths, args.device, args.batch_size, args.num_workers,
                                             pretrained=not args.no_pretrained, patches=args.patches,
                                             augment_copies=copies, seed=args.seed)

            # The split is made on whole images first, so patches of one image never cross train/test.
            train_copies = args.augment_copies if args.patches else 0
            tag = f"_patches_aug{train_copies}" if args.patches else ""
            X_train = _cached_features(cache, f"{_slug(name)}_train{tag}",
                                       lambda: extract(split.train_paths, train_copies))
            X_test = _cached_features(cache, f"{_slug(name)}_test{'_patches' if args.patches else ''}",
                                      lambda: extract(split.test_paths, 0))
            y_train = split.train_labels
            if args.patches:
                y_train = np.repeat(y_train, len(X_train) // len(split.train_paths))

            start = time.perf_counter()
            pipeline = fit_feature_pipeline(X_train, y_train, **elm_kwargs)
            fit_seconds = time.perf_counter() - start  # scaler + PCA + ELM, excluding the frozen backbone
            y_pred = (predict_by_voting(pipeline, X_test, len(split.test_paths)) if args.patches
                      else pipeline.predict(X_test))

        metrics = compute_metrics(split.test_labels, y_pred)
        rows.append({"model": name, **metrics, "fit_seconds": fit_seconds})
        print(" | ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + f" | fit={fit_seconds:.1f}s")

        slug = _slug(name)
        (out / "reports" / f"{slug}.txt").write_text(text_report(split.test_labels, y_pred, split.class_names))
        save_confusion_matrix(split.test_labels, y_pred, split.class_names,
                              f"{name} + ELM ({args.dataset})", out / "confusion_matrices" / f"{slug}.png")
        if args.save_models:
            models_dir = out / "models"
            models_dir.mkdir(exist_ok=True)
            joblib.dump({"pipeline": pipeline, "backbone": None if name == RAW_PIXELS else BACKBONES[name],
                         "raw_size": args.raw_size, "class_names": split.class_names,
                         "patches": args.patches and name != RAW_PIXELS},
                        models_dir / f"{slug}.joblib", compress=3)

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    summary.to_csv(out / "summary.csv", index=False)
    save_summary_plot(summary, f"Feature extractor + ELM — {args.dataset}", out / "summary.png")
    (out / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    print("\n" + summary.round(4).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
