"""
Blend K StyleFeature objects into a single StyleFeature using
distance-weighted averaging (softmax over inverse distances).
"""
from __future__ import annotations

import numpy as np

from rawstyle.core.style_extractor import StyleFeature, _identity_lut


# Scalar fields blended by weighted average
_SCALAR_FIELDS = [
    "contrast", "shadow_lift", "highlight_comp", "color_temp_delta",
    "sat_r", "sat_g", "sat_b", "vibrancy",
    "hue_shift_r", "hue_shift_o", "hue_shift_y", "hue_shift_g",
    "hue_shift_c", "hue_shift_b", "hue_shift_p", "hue_shift_m",
    "lum_hue_r", "lum_hue_o", "lum_hue_y", "lum_hue_g",
    "lum_hue_c", "lum_hue_b", "lum_hue_p", "lum_hue_m",
    "vignette_strength", "clarity", "grain_strength",
]

# Array (LUT) fields blended element-wise
_ARRAY_FIELDS = ["lum_curve", "curve_r", "curve_g", "curve_b"]


def blend(
    matches: list[tuple[StyleFeature, float]],
    temperature: float = 0.15,
) -> StyleFeature:
    """
    Blend a list of (StyleFeature, distance) pairs.

    distance  — cosine distance in [0, 2]; 0 = identical
    temperature — softmax sharpness; lower = closer match dominates more
    """
    if not matches:
        raise ValueError("No matches to blend")

    if len(matches) == 1:
        return matches[0][0]

    distances = np.array([d for _, d in matches], dtype=np.float32)
    log_w = -distances / max(temperature, 1e-6)
    log_w -= log_w.max()
    weights = np.exp(log_w)
    weights /= weights.sum()

    features = [f for f, _ in matches]

    def wavg(attr: str) -> float:
        return float(sum(w * getattr(f, attr) for w, f in zip(weights, features)))

    def arr_avg(attr: str) -> np.ndarray:
        blended = sum(w * getattr(f, attr) for w, f in zip(weights, features))
        return _ensure_monotonic(blended.astype(np.float32))

    scalars = {field: wavg(field) for field in _SCALAR_FIELDS}
    arrays  = {field: arr_avg(field) for field in _ARRAY_FIELDS}

    return StyleFeature(**scalars, **arrays)


def _ensure_monotonic(curve: np.ndarray) -> np.ndarray:
    out = curve.copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return np.clip(out, 0.0, 1.0)
