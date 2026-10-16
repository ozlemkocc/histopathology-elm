"""Feature extraction: pretrained CNN / ViT backbones (timm) and raw pixels."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

# Display name -> timm model id
BACKBONES: dict[str, str] = {
    "ResNet50": "resnet50",
    "DenseNet121": "densenet121",
    "EfficientNet-B3": "efficientnet_b3",
    "ViT-B/16": "vit_base_patch16_224",
    "Swin-T": "swin_tiny_patch4_window7_224",
    "DeiT-B": "deit_base_distilled_patch16_224",
}
RAW_PIXELS = "Raw Pixels"


N_CROPS = 5  # 4 corners + centre


def load_rgb(path: str | Path, size: int | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.BILINEAR) if size else image


def five_crops(image: Image.Image, size: int = 224) -> list[Image.Image]:
    """Resize to 2*size and cut the four corner patches and the centre patch."""
    image = image.resize((2 * size, 2 * size), Image.Resampling.BILINEAR)
    s, c = size, size // 2
    boxes = [(0, 0, s, s), (s, 0, 2 * s, s), (0, s, s, 2 * s), (s, s, 2 * s, 2 * s), (c, c, c + s, c + s)]
    return [image.crop(box) for box in boxes]


def augment(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Brightness / contrast jitter plus a random 90° rotation or flip.

    Tissue has no canonical orientation, so 90° rotations and flips are label-preserving
    and, unlike small-angle rotations, add no black borders.
    """
    from PIL import ImageEnhance

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    transposes = [Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
                  Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM]
    return image.transpose(transposes[rng.integers(len(transposes))])


def patch_views(path: str | Path, size: int = 224, augmented: bool = False, seed: int = 0) -> list[Image.Image]:
    """The five patches of one image, optionally augmented (deterministic for a given seed)."""
    crops = five_crops(load_rgb(path), size)
    if not augmented:
        return crops
    rng = np.random.default_rng(seed)
    return [augment(crop, rng) for crop in crops]


def iter_raw_pixel_batches(paths: Sequence[Path], size: int, batch_size: int) -> Iterator[np.ndarray]:
    """Yield flattened, [0, 1]-scaled pixel vectors without loading the whole set into memory."""
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        yield np.stack([np.asarray(load_rgb(p, size), dtype=np.float32).ravel() / 255.0 for p in batch])


def build_backbone(timm_id: str, pretrained: bool = True):
    """Return a headless timm model and its matching eval transform (incl. ImageNet normalisation)."""
    import timm

    # DeiT (distilled) only supports token pooling; the others use global average pooling.
    pool_kwargs = {} if "deit" in timm_id else {"global_pool": "avg"}
    model = timm.create_model(timm_id, pretrained=pretrained, num_classes=0, **pool_kwargs).eval()
    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**config)
    return model, transform


def extract_deep_features(timm_id: str, paths: Sequence[Path], device: str = "cuda",
                          batch_size: int = 64, num_workers: int = 2, pretrained: bool = True,
                          patches: bool = False, augment_copies: int = 0, seed: int = 42) -> np.ndarray:
    """Run a frozen backbone over `paths`.

    Returns (n_images, n_features), or with ``patches=True``
    (n_images * N_CROPS * (1 + augment_copies), n_features), ordered image by image:
    the 5 original patches first, then 5 per augmented copy.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    model, transform = build_backbone(timm_id, pretrained)
    model.to(device)
    views_per_image = 1 + augment_copies

    class _Images(Dataset):
        def __len__(self) -> int:
            return len(paths) * (views_per_image if patches else 1)

        def __getitem__(self, idx: int):
            if not patches:
                return transform(load_rgb(paths[idx]))
            image_idx, copy = divmod(idx, views_per_image)
            views = patch_views(paths[image_idx], augmented=copy > 0, seed=seed + 1000 * image_idx + copy)
            return torch.stack([transform(v) for v in views])  # (5, C, H, W)

    def _forward(images):
        if patches:  # (B, 5, C, H, W) -> (B*5, C, H, W)
            images = images.flatten(0, 1)
        return model(images.to(device))

    loader = DataLoader(_Images(), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.startswith("cuda"))
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=timm_id, leave=False):
            chunks.append(_forward(images).float().cpu().numpy())

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(chunks)
