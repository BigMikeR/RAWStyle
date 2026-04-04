"""
Extract photographic style features from a processed JPEG.

Works without the original RAW file by characterising the JPEG's
tonal and colour distribution directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator


# Neutral saturation baseline — approximate mean S for a well-exposed,
# colour-rich outdoor photo with no saturation adjustment.
_NEUTRAL_SAT = 0.45

# Hue ranges (in degrees 0-360) per colour group
_HUE_RANGES = {
    "r": [(-30, 30), (330, 360)],   # reds, oranges (wrap around)
    "g": [(75, 165)],               # greens, cyans
    "b": [(195, 285)],              # blues, magentas
}


@dataclass
class StyleFeature:
    lum_curve: np.ndarray   # float32[256]: luminance LUT, values in [0, 1]
    shadow_lift: float      # mean luminance in bottom 10th pct, [0, 1]
    highlight_comp: float   # 1 - mean luminance in top 5th pct, [0, 1]
    sat_r: float            # saturation multiplier for red/orange/yellow
    sat_g: float            # saturation multiplier for green/cyan
    sat_b: float            # saturation multiplier for blue/magenta
    vibrancy: float         # selective saturation boost metric


def extract(img: Image.Image) -> StyleFeature:
    """Extract a StyleFeature from a processed JPEG PIL image."""
    img_rgb = img.convert("RGB")
    return StyleFeature(
        lum_curve=_extract_lum_curve(img_rgb),
        **_extract_shadow_highlight(img_rgb),
        **_extract_saturation(img_rgb),
        vibrancy=_extract_vibrancy(img_rgb),
    )


# ---------------------------------------------------------------------------
# Luminance curve
# ---------------------------------------------------------------------------

def _reference_cdf() -> np.ndarray:
    """
    CDF of a gamma-2.2 encoded image with a uniform linear scene distribution.
    Maps 256 output levels → cumulative fraction.
    """
    levels = np.arange(256, dtype=np.float32) / 255.0
    # Inverse gamma: linear values for each encoded level
    linear = np.where(levels <= 0.04045, levels / 12.92, ((levels + 0.055) / 1.055) ** 2.4)
    # CDF of the linearised values assuming uniform distribution in linear space
    cdf = linear / linear[-1]
    return cdf


_REF_CDF = _reference_cdf()


def _extract_lum_curve(img: Image.Image) -> np.ndarray:
    """
    Extract a 256-point tone curve by comparing the empirical luminance CDF
    to a reference gamma-2.2 CDF.  Enforces monotonicity via PCHIP.
    """
    ycbcr = img.convert("YCbCr")
    y = np.array(ycbcr)[:, :, 0].ravel().astype(np.float32)

    # Empirical CDF of Y channel
    hist, _ = np.histogram(y, bins=256, range=(0, 255))
    cdf_actual = np.cumsum(hist).astype(np.float32)
    cdf_actual /= cdf_actual[-1]

    # For each output level out, find the input level where cdf_actual ≈ ref_cdf[out]
    # i.e. invert: ref_cdf[inp] = cdf_actual[out]
    out_levels = np.arange(256, dtype=np.float32)
    inp_levels = np.interp(_REF_CDF, cdf_actual, out_levels / 255.0)

    # inp_levels[i] gives: "what input fraction maps to output fraction i/255?"
    # We want a LUT: given input level i (0-255), what is the output in [0,1]?
    x_ctrl = inp_levels          # input fractions
    y_ctrl = out_levels / 255.0  # output fractions

    # Deduplicate and sort (required by PCHIP)
    order = np.argsort(x_ctrl)
    x_ctrl = x_ctrl[order]
    y_ctrl = y_ctrl[order]
    _, unique = np.unique(x_ctrl, return_index=True)
    x_ctrl = x_ctrl[unique]
    y_ctrl = y_ctrl[unique]

    spline = PchipInterpolator(x_ctrl, y_ctrl, extrapolate=True)
    lut_x = np.arange(256, dtype=np.float32) / 255.0
    lut_y = spline(lut_x).astype(np.float32)
    lut_y = np.clip(lut_y, 0.0, 1.0)

    return lut_y


# ---------------------------------------------------------------------------
# Shadow / highlight
# ---------------------------------------------------------------------------

def _extract_shadow_highlight(img: Image.Image) -> dict:
    ycbcr = img.convert("YCbCr")
    y = np.array(ycbcr)[:, :, 0].astype(np.float32) / 255.0

    p10 = np.percentile(y, 10)
    shadow_mask = y <= p10
    shadow_lift = float(np.mean(y[shadow_mask])) if shadow_mask.any() else 0.0

    p95 = np.percentile(y, 95)
    hi_mask = y >= p95
    highlight_comp = float(1.0 - np.mean(y[hi_mask])) if hi_mask.any() else 0.0

    return {"shadow_lift": shadow_lift, "highlight_comp": highlight_comp}


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

def _extract_saturation(img: Image.Image) -> dict:
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    H = hsv[:, :, 0] / 255.0 * 360.0   # 0-360
    S = hsv[:, :, 1] / 255.0           # 0-1

    def sat_for_group(ranges):
        mask = np.zeros(H.shape, dtype=bool)
        for lo, hi in ranges:
            if lo < 0:
                mask |= (H >= (lo + 360)) | (H < hi)
            else:
                mask |= (H >= lo) & (H < hi)
        mask &= S > 0.15  # exclude near-neutral pixels
        if not mask.any():
            return 1.0
        return float(np.mean(S[mask]) / _NEUTRAL_SAT)

    return {
        "sat_r": sat_for_group(_HUE_RANGES["r"]),
        "sat_g": sat_for_group(_HUE_RANGES["g"]),
        "sat_b": sat_for_group(_HUE_RANGES["b"]),
    }


# ---------------------------------------------------------------------------
# Vibrancy
# ---------------------------------------------------------------------------

def _extract_vibrancy(img: Image.Image) -> float:
    """
    Estimate vibrancy as the skewness of the saturation distribution toward
    lower values (vibrancy boosts low-S pixels selectively).

    Positive value = low-S pixels are boosted (high vibrancy).
    Zero = flat saturation adjustment (no vibrancy).
    """
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    S = hsv[:, :, 1] / 255.0
    S_mid = S[(S > 0.05) & (S < 0.95)]  # exclude clipped values
    if len(S_mid) == 0:
        return 0.0
    # Measure how much weight sits in low-saturation territory
    low_frac = float(np.mean(S_mid < 0.4))  # fraction of pixels with S < 0.4
    # A vibrancy-boosted image will have fewer very-low-S pixels than neutral
    # Neutral expectation: ~0.5 of non-extreme pixels below 0.4
    vibrancy = float((0.5 - low_frac) * 2.0)  # range roughly [-1, +1]
    return np.clip(vibrancy, -0.5, 1.5).item()
