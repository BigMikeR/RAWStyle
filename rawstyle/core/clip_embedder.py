"""
CLIP-based content embedding using open-clip-torch.

Singleton pattern: the model is loaded once per process and reused.
Automatically uses Apple MPS (Apple Silicon) when available, otherwise CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image as PILImage

_model = None
_preprocess = None
_device = None


def _load_model(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    global _model, _preprocess, _device
    if _model is not None:
        return _model, _preprocess, _device

    import torch
    import open_clip

    if torch.backends.mps.is_available():
        _device = "mps"
    elif torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    _model = _model.to(_device)
    _model.eval()
    return _model, _preprocess, _device


def embed_image(img: "PILImage.Image", model_name: str = "ViT-B-32") -> np.ndarray:
    """
    Embed a single PIL image.  Returns a L2-normalised float32 vector of shape (512,).
    """
    import torch

    model, preprocess, device = _load_model(model_name)
    tensor = preprocess(img).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).cpu().numpy().astype(np.float32)


def embed_batch(
    images: list["PILImage.Image"],
    batch_size: int = 64,
    model_name: str = "ViT-B-32",
) -> np.ndarray:
    """
    Embed a list of PIL images in batches.
    Returns float32 array of shape (N, 512), L2-normalised.
    """
    import torch

    model, preprocess, device = _load_model(model_name)
    all_features = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            features = model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0).astype(np.float32)
