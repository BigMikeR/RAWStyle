"""
Blend K StyleFeature objects into a single StyleFeature using
distance-weighted averaging (softmax over inverse distances).
"""
from __future__ import annotations

import numpy as np

from rawstyle.core.style_extractor import StyleFeature


def blend(
    matches: list[tuple[StyleFeature, float]],
    temperature: float = 0.15,
) -> StyleFeature:
    """
    Blend a list of (StyleFeature, distance) pairs.

    `distance` is cosine distance in [0, 2] where 0 = identical.
    `temperature` controls how sharply the closest match dominates:
      - low T  (0.05) → winner-takes-almost-all
      - high T (0.5)  → near-equal weighting

    Returns a single blended StyleFeature.
    """
    if not matches:
        raise ValueError("No matches to blend")

    if len(matches) == 1:
        return matches[0][0]

    distances = np.array([d for _, d in matches], dtype=np.float32)
    # Softmax with temperature: w_i = exp(-d_i / T)
    log_weights = -distances / max(temperature, 1e-6)
    log_weights -= log_weights.max()  # numerical stability
    weights = np.exp(log_weights)
    weights /= weights.sum()

    features = [f for f, _ in matches]

    # Weighted average of each scalar field
    def wavg(attr):
        return float(sum(w * getattr(f, attr) for w, f in zip(weights, features)))

    # Weighted average of lum_curve arrays, then validate monotonicity
    lum_curve = sum(w * f.lum_curve for w, f in zip(weights, features))
    lum_curve = _ensure_monotonic(lum_curve.astype(np.float32))

    return StyleFeature(
        lum_curve=lum_curve,
        shadow_lift=wavg("shadow_lift"),
        highlight_comp=wavg("highlight_comp"),
        sat_r=wavg("sat_r"),
        sat_g=wavg("sat_g"),
        sat_b=wavg("sat_b"),
        vibrancy=wavg("vibrancy"),
    )


def _ensure_monotonic(curve: np.ndarray) -> np.ndarray:
    """
    Enforce non-decreasing values in the LUT using a cumulative max pass.
    This is a safety net; PCHIP-extracted curves should already be monotonic.
    """
    out = curve.copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return np.clip(out, 0.0, 1.0)
