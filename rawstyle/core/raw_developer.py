"""
Develop a Sony ARW file into:
  - a full-resolution linear float32 array for style application
  - a small sRGB uint8 thumbnail suitable for CLIP embedding
"""
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image


# Target size for CLIP thumbnail (model expects 224x224 minimum)
_THUMB_SIZE = (336, 336)


def develop_linear(arw_path: Path) -> np.ndarray:
    """
    Develop ARW to a scene-referred linear float32 RGB array, shape (H, W, 3),
    values in [0.0, 1.0].

    Settings:
    - Camera white balance preserved (use_camera_wb=True)
    - No auto-brightness (no_auto_bright=True)
    - No gamma (gamma=(1,1)) — pure linear output
    - 16-bit output, then normalised to [0,1]
    - sRGB colour space (demosaic to sRGB primaries)
    """
    with rawpy.imread(str(arw_path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(1, 1),
        )
    # uint16 → float32 in [0, 1]
    return rgb16.astype(np.float32) / 65535.0


def develop_thumbnail(arw_path: Path, size: tuple[int, int] = _THUMB_SIZE) -> Image.Image:
    """
    Develop ARW to a small sRGB JPEG-ready PIL image for CLIP embedding.
    Applies standard sRGB gamma so the image looks natural to CLIP.
    """
    with rawpy.imread(str(arw_path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=False,     # allow auto-brightness for thumbnail only
            bright=1.0,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            # Default gamma (2.222, 4.5) gives standard sRGB-like output
        )
    img = Image.fromarray(rgb16)
    img.thumbnail(size, Image.LANCZOS)
    return img
