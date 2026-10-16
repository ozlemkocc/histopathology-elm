# Histopathology Image Classification with Extreme Learning Machines

Senior project, Yıldız Technical University, Computer Engineering (June 2025)
**Authors:** Özlem Koç, Ömer Aşkın · **Advisor:** Prof. Dr. Gökhan Bilgin

Histopathological images are classified by combining **frozen pretrained CNN / Vision Transformer
backbones** with an **Extreme Learning Machine (ELM)** classifier. The backbone is never fine-tuned:
its features are standardised, reduced with PCA and fed to an ELM, whose training is a single
linear solve. The approach is evaluated on three datasets covering breast, lung and colon tissue
and compared against a raw-pixel ELM baseline.

## Pipeline

```
image ─► resize + ImageNet normalisation ─► frozen backbone ─► StandardScaler ─► PCA ─► ELM ─► class
                                            ResNet50 · DenseNet121 · EfficientNet-B3
                                            ViT-B/16 · Swin-T · DeiT-B
         baseline: flattened raw pixels ───────────────────────► StandardScaler ─► IncrementalPCA ─► ELM
```

**Extreme Learning Machine.** A single-hidden-layer network whose input weights `W` and biases `b`
are random and fixed. Only the output weights are learned, in closed form:

```
H = g(XW + b)            β = (HᵀH + αI)⁻¹ HᵀT
```

No backpropagation and no epochs are needed: on precomputed features the classifier trains in seconds.
The ELM is implemented from scratch in [`src/histo_elm/elm.py`](src/histo_elm/elm.py) as a
scikit-learn compatible estimator.

## Datasets

| Dataset | Tissue | Classes | Images | Split |
|---|---|---|---|---|
| [BACH](https://iciar2018-challenge.grand-challenge.org/Dataset/) (ICIAR 2018) | Breast | Normal, Benign, In situ, Invasive | 400 (2048×1536) | stratified 80/20 |
| [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) | Breast | Benign, Malignant | 7,909 (700×460, 40×–400×) | **patient-level** stratified 80/20 |
| [LC25000](https://arxiv.org/abs/1912.12142) | Lung + colon | 5 | 25,000 (768×768) | stratified 80/20 |

The datasets are not included because of their size. Images are discovered by class folder name,
so the official archives work after unzipping.

## Results (from the project report)

Test-set accuracy of each feature extractor + ELM:

| Features | BACH | BreakHis | LC25000 |
|---|---|---|---|
| Raw pixels (baseline) | 0.4000 | 0.8021 | 0.7518 |
| ResNet50 | 0.6250 | 0.8906 | 0.9846 |
| DenseNet121 | 0.4625 | 0.8982 | 0.9782 |
| EfficientNet-B3 | 0.6625 | 0.8679 | 0.9830 |
| ViT-B/16 | 0.8000 | 0.9185 | 0.9888 |
| Swin-T | **0.8125** | 0.9115 | 0.9862 |
| DeiT-B | 0.7875 | **0.9381** | **0.9942** |

- Transformer backbones outperformed CNN backbones on all three datasets.
- Deep features improved on the raw-pixel baseline by 41 (BACH), 14 (BreakHis) and 24 (LC25000) points.
- BACH scores are noisy: the test set has only 80 images, so one error ≈ 1.25 points.

### Caveats

These numbers come from the original project notebooks. The code in this repository is a cleaned-up
re-implementation and **fixes two issues** of the original runs, so re-running it can give
different (more honest) numbers:

1. **ImageNet normalisation.** Pretrained backbones expect ImageNet mean/std-normalised input; the
   original runs fed `[0, 1]`-scaled pixels. Each backbone's own transform is now used.
2. **BreakHis patient leakage.** BreakHis contains several images per patient (and the same region
   at four magnifications). A random image-level split puts the same patient in train and test.
   This repo splits by patient ID.

LC25000 also has a known limitation: it was produced by augmenting 1,250 original images, so
near-duplicate images can fall on both sides of any split. Its ~99 % scores are likely optimistic.

## Usage

```bash
pip install -e ".[dev]"

# Train + evaluate every model on one dataset (GPU recommended)
python -m histo_elm.run --dataset lc25000 --data-dir /path/to/lc25000 --out-dir results/lc25000 --save-models

# Subset of models, BreakHis at a single magnification
python -m histo_elm.run --dataset breakhis --data-dir /path/to/BreaKHis_v1 \
    --models "Swin-T" "DeiT-B" --magnification 40

# Patch-based variant (useful for the large 2048×1536 BACH images)
python -m histo_elm.run --dataset bach --data-dir /path/to/bach --patches --augment-copies 1

# Classify new images with a saved pipeline
python -m histo_elm.predict results/lc25000/models/deit-b.joblib image1.jpeg image2.jpeg
```

**Patch mode** (`--patches`) cuts each image into its four corner patches and a centre patch.
Training images additionally get augmented copies (brightness / contrast jitter, 90° rotation or
flip). The train/test split is made on whole images *before* patching, and each test image is
classified by averaging the ELM scores of its five patches, so patches of one image never leak
across the split.

Each run writes `summary.csv`, `summary.png`, per-model classification reports and confusion
matrices to the output folder. Deep features are cached, so changing the PCA / ELM settings
(`--pca-components`, `--elm-neurons`, `--elm-alpha`, `--elm-ufunc`) does not re-run the backbone.

To run in Google Colab, open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb).

```bash
pytest   # ELM, splits (incl. patient-level leakage check) and an end-to-end run on synthetic images
```

## Repository layout

```
src/histo_elm/
├── elm.py        # ELM classifier (NumPy, scikit-learn API)
├── data.py       # dataset discovery, stratified / patient-level splits
├── features.py   # timm backbones and raw-pixel loading
├── pipeline.py   # scaler → PCA → ELM; memory-safe raw-pixel baseline
├── evaluate.py   # metrics, reports, plots
├── run.py        # experiment CLI
└── predict.py    # inference CLI
notebooks/colab_runner.ipynb
tests/
docs/             # poster
```

## Tech stack

Python · PyTorch · timm · scikit-learn · NumPy · pandas · Matplotlib
