from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split

from histo_elm.data import breakhis_patient_id, make_split
from histo_elm.elm import ELMClassifier
from histo_elm.pipeline import _chunks, fit_feature_pipeline
from histo_elm.run import main


# ---------- ELM ----------

def test_elm_learns_multiclass_digits():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X / 16.0, y, test_size=0.25, stratify=y, random_state=0)
    clf = ELMClassifier(n_neurons=1000, alpha=1e-2, random_state=0).fit(X_tr, y_tr)
    assert clf.score(X_te, y_te) > 0.93


def test_elm_binary_and_string_labels():
    X, y = make_classification(n_samples=400, n_features=10, random_state=0)
    labels = np.where(y == 1, "malignant", "benign")
    clf = ELMClassifier(n_neurons=200, random_state=0).fit(X, labels)
    assert set(clf.predict(X)) <= {"benign", "malignant"}
    assert clf.score(X, labels) > 0.9


def test_elm_is_deterministic_with_seed():
    X, y = make_classification(n_samples=200, n_features=8, random_state=1)
    a = ELMClassifier(n_neurons=50, random_state=7).fit(X, y).decision_function(X)
    b = ELMClassifier(n_neurons=50, random_state=7).fit(X, y).decision_function(X)
    np.testing.assert_allclose(a, b)


def test_elm_rejects_unknown_activation():
    with pytest.raises(ValueError):
        ELMClassifier(ufunc="softplus").fit(np.zeros((4, 2)), [0, 1, 0, 1])


# ---------- pipeline ----------

def test_pca_components_are_clipped_to_small_training_sets():
    X, y = make_classification(n_samples=60, n_features=30, n_informative=10, n_classes=3, random_state=0)
    pipe = fit_feature_pipeline(X, y, n_components=512, n_neurons=64)
    assert pipe.named_steps["pca"].n_components_ == 30


def test_chunks_merge_small_tail():
    slices = _chunks(n_items=1050, chunk_size=500, min_size=100)
    assert [(s.start, s.stop) for s in slices] == [(0, 500), (500, 1050)]
    assert sum(s.stop - s.start for s in _chunks(1300, 500, 100)) == 1300


# ---------- data ----------

def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    pixels = np.clip(np.array(color) + rng.normal(0, 20, (32, 32, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


@pytest.fixture
def breakhis_root(tmp_path: Path) -> Path:
    root = tmp_path / "BreaKHis_v1" / "histology_slides" / "breast"
    for cls, code, color in (("benign", "B", (200, 120, 200)), ("malignant", "M", (90, 40, 140))):
        for patient in range(10):
            pid = f"{patient}-1000{code}"
            for mag in (40, 100):
                for k in range(3):
                    name = f"SOB_{code}_A-{pid}-{mag}-{k:03d}.png"
                    _write_image(root / cls / "SOB" / "adenosis" / f"SOB_{code}_A_{pid}" / f"{mag}X" / name, color)
    return tmp_path


def test_breakhis_patient_id():
    assert breakhis_patient_id(Path("SOB_B_A-14-22549AB-40-001.png")) == "14-22549AB"


def test_breakhis_split_has_no_patient_overlap(breakhis_root: Path):
    split = make_split("breakhis", breakhis_root, test_size=0.2, seed=0)
    train_ids = {breakhis_patient_id(p) for p in split.train_paths}
    test_ids = {breakhis_patient_id(p) for p in split.test_paths}
    assert train_ids and test_ids and not train_ids & test_ids
    assert set(split.test_labels) == {0, 1}


def test_breakhis_magnification_filter(breakhis_root: Path):
    split = make_split("breakhis", breakhis_root, magnification=40)
    assert all("-40-" in p.name for p in split.train_paths + split.test_paths)


@pytest.fixture
def lc25000_root(tmp_path: Path) -> Path:
    colors = [(220, 60, 60), (60, 220, 60), (60, 60, 220), (220, 220, 60), (60, 220, 220)]
    groups = ["colon_image_sets"] * 2 + ["lung_image_sets"] * 3
    for (cls, color), group in zip(zip(["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"], colors), groups):
        for k in range(20):
            _write_image(tmp_path / "lung_colon_image_set" / group / cls / f"{cls}{k}.jpeg", color)
    return tmp_path


def test_lc25000_stratified_split(lc25000_root: Path):
    split = make_split("lc25000", lc25000_root)
    assert len(split.train_paths) == 80 and len(split.test_paths) == 20
    assert np.bincount(split.test_labels).tolist() == [4] * 5


# ---------- end to end ----------

def test_raw_pixel_run_end_to_end(lc25000_root: Path, tmp_path: Path):
    out = tmp_path / "results"
    summary = main(["--dataset", "lc25000", "--data-dir", str(lc25000_root), "--out-dir", str(out),
                    "--models", "Raw Pixels", "--raw-size", "16", "--pca-components", "20",
                    "--elm-neurons", "100", "--save-models"])
    assert summary.loc[0, "accuracy"] > 0.9
    for name in ("summary.csv", "summary.png", "config.json", "reports/raw_pixels.txt",
                 "confusion_matrices/raw_pixels.png", "models/raw_pixels.joblib"):
        assert (out / name).exists(), name


# ---------- patches ----------

def test_five_crops_and_deterministic_augmentation(tmp_path: Path):
    from histo_elm.features import N_CROPS, five_crops, patch_views

    img_path = tmp_path / "img.png"
    _write_image(img_path, (120, 60, 160))
    crops = five_crops(Image.open(img_path), size=16)
    assert len(crops) == N_CROPS and all(c.size == (16, 16) for c in crops)
    a = [np.asarray(v) for v in patch_views(img_path, size=16, augmented=True, seed=3)]
    b = [np.asarray(v) for v in patch_views(img_path, size=16, augmented=True, seed=3)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_soft_voting_returns_one_label_per_image():
    from histo_elm.pipeline import predict_by_voting

    X, y = make_classification(n_samples=300, n_features=12, n_informative=6, n_classes=3, random_state=0)
    pipe = fit_feature_pipeline(X, y, n_neurons=100)
    pred = predict_by_voting(pipe, X[:50], n_images=10)  # 10 images x 5 patches
    assert pred.shape == (10,)


def test_patch_run_end_to_end_with_fake_backbone(lc25000_root: Path, tmp_path: Path):
    """Exercises the --patches code path in run.py without torch: the fake backbone returns
    mean colour per patch and the same row layout as extract_deep_features."""
    import histo_elm.run as run_module
    from histo_elm.features import patch_views

    def fake_extract(timm_id, paths, device, batch_size, num_workers, pretrained, patches,
                     augment_copies, seed):
        rows = []
        for i, p in enumerate(paths):
            for copy in range(1 + augment_copies):
                for v in patch_views(p, size=8, augmented=copy > 0, seed=seed + 1000 * i + copy):
                    rows.append(np.asarray(v, dtype=np.float32).reshape(-1, 3).mean(0))
        return np.asarray(rows)

    original = run_module.extract_deep_features
    run_module.extract_deep_features = fake_extract
    try:
        summary = main(["--dataset", "lc25000", "--data-dir", str(lc25000_root), "--out-dir",
                        str(tmp_path / "r"), "--models", "ResNet50", "--patches", "--augment-copies", "1",
                        "--elm-neurons", "100"])
    finally:
        run_module.extract_deep_features = original
    assert summary.loc[0, "accuracy"] > 0.9
    train = np.load(tmp_path / "r" / "features" / "resnet50_train_patches_aug1.npy")
    test = np.load(tmp_path / "r" / "features" / "resnet50_test_patches.npy")
    assert train.shape[0] == 80 * 5 * 2 and test.shape[0] == 20 * 5
