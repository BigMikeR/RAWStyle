"""
Extract photographic style features from a processed JPEG.

Two modes:
  extract(jpeg)            — JPEG-only; estimates style from the image alone.
  extract_from_pair(arw, jpeg) — Pair mode; directly compares the neutral ARW
                                 development to the processed JPEG for accurate
                                 before/after measurements.

Pair mode is significantly more accurate because it measures the actual
adjustments applied rather than inferring them from the output alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator


# Neutral saturation baseline used in JPEG-only mode
_NEUTRAL_SAT = 0.45

# Comparison resolution for pair-based extraction
_CMP_SIZE = (512, 512)

# Hue ranges (degrees 0-360) per colour group
_HUE_RANGES = {
    "r": [(-30, 30), (330, 360)],
    "g": [(75, 165)],
    "b": [(195, 285)],
}


@dataclass
class StyleFeature:
    lum_curve: np.ndarray   # float32[256]: luminance LUT, values in [0, 1]
    shadow_lift: float      # additive lift in shadow regions
    highlight_comp: float   # pull-down in highlight regions
    sat_r: float            # saturation multiplier for red/orange/yellow
    sat_g: float            # saturation multiplier for green/cyan
    sat_b: float            # saturation multiplier for blue/magenta
    vibrancy: float         # selective saturation boost metric


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(img: Image.Image) -> StyleFeature:
    """
    Estimate style from a processed JPEG alone (no original RAW).
    Less accurate than extract_from_pair — use when ARW files are unavailable.
    """
    img_rgb = img.convert("RGB")
    return StyleFeature(
        lum_curve=_lum_curve_from_jpeg(img_rgb),
        **_shadow_highlight_from_jpeg(img_rgb),
        **_saturation_from_jpeg(img_rgb),
        vibrancy=_vibrancy_from_jpeg(img_rgb),
    )


def extract_from_pair(arw_path: Path, jpeg_path: Path) -> StyleFeature:
    """
    Accurately extract style by directly comparing the neutrally-developed ARW
    to the processed JPEG.

    Steps:
    1. Develop the ARW to a neutral sRGB uint8 image (no tone curve, no
       auto-brightness — just demosaic + camera WB + sRGB gamma).
    2. Resize both images to a common size for comparison.
    3. Measure tone curve, shadow/highlight, saturation, and vibrancy as the
       actual difference between source (ARW) and destination (JPEG).
    """
    from rawstyle.core.raw_developer import develop_thumbnail

    arw_img = develop_thumbnail(arw_path, size=_CMP_SIZE)
    arw_img = _to_square(arw_img, _CMP_SIZE)

    jpeg_img = Image.open(jpeg_path).convert("RGB")
    jpeg_img = _to_square(jpeg_img, _CMP_SIZE)

    return StyleFeature(
        lum_curve=_lum_curve_from_pair(arw_img, jpeg_img),
        **_shadow_highlight_from_pair(arw_img, jpeg_img),
        **_saturation_from_pair(arw_img, jpeg_img),
        vibrancy=_vibrancy_from_pair(arw_img, jpeg_img),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_square(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize to exactly `size`, ignoring aspect ratio, for pixel comparison."""
    return img.resize(size, Image.LANCZOS)


def _luminance_array(img: Image.Image) -> np.ndarray:
    """Return float32 luminance in [0, 1] from a PIL RGB image."""
    y = np.array(img.convert("YCbCr"))[:, :, 0].astype(np.float32) / 255.0
    return y


def _hsv_arrays(img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (H_deg, S, V) float32 arrays from a PIL RGB image."""
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    H = hsv[:, :, 0] / 255.0 * 360.0
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0
    return H, S, V


def _hue_mask(H: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(H.shape, dtype=bool)
    for lo, hi in ranges:
        if lo < 0:
            mask |= (H >= lo + 360) | (H < hi)
        else:
            mask |= (H >= lo) & (H < hi)
    return mask


def _fit_monotonic_lut(x_ctrl: np.ndarray, y_ctrl: np.ndarray) -> np.ndarray:
    """Fit a 256-point monotonic LUT through control points using PCHIP."""
    order = np.argsort(x_ctrl)
    x_ctrl, y_ctrl = x_ctrl[order], y_ctrl[order]
    _, unique = np.unique(x_ctrl, return_index=True)
    x_ctrl, y_ctrl = x_ctrl[unique], y_ctrl[unique]
    spline = PchipInterpolator(x_ctrl, y_ctrl, extrapolate=True)
    lut = spline(np.arange(256, dtype=np.float32) / 255.0).astype(np.float32)
    return np.clip(lut, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pair-based extraction  (accurate — uses before/after comparison)
# ---------------------------------------------------------------------------

def _lum_curve_from_pair(src: Image.Image, dst: Image.Image) -> np.ndarray:
    """
    Derive the tone curve by histogram specification:
    for each input level, find the output level that has the same CDF value.

    This directly measures "what luminance mapping did the photographer apply?"
    """
    y_src = (_luminance_array(src) * 255.0).astype(np.uint8).ravel()
    y_dst = (_luminance_array(dst) * 255.0).astype(np.uint8).ravel()

    hist_src, _ = np.histogram(y_src, bins=256, range=(0, 255))
    hist_dst, _ = np.histogram(y_dst, bins=256, range=(0, 255))

    cdf_src = np.cumsum(hist_src).astype(np.float32)
    cdf_src /= cdf_src[-1]
    cdf_dst = np.cumsum(hist_dst).astype(np.float32)
    cdf_dst /= cdf_dst[-1]

    # For each src level x, find dst level y where cdf_dst[y] ≈ cdf_src[x]
    out_levels = np.arange(256, dtype=np.float32) / 255.0
    lut = np.interp(cdf_src, cdf_dst, out_levels)

    # Re-fit with PCHIP to smooth quantisation noise
    x_ctrl = np.arange(256, dtype=np.float32) / 255.0
    return _fit_monotonic_lut(x_ctrl, lut.astype(np.float32))


def _shadow_highlight_from_pair(src: Image.Image, dst: Image.Image) -> dict:
    """
    Measure shadow lift and highlight compression as the direct luminance
    difference between source and destination in each tonal region.
    """
    y_src = _luminance_array(src)
    y_dst = _luminance_array(dst)

    # Shadow: pixels in the bottom 10th percentile of the SOURCE luminance
    p10 = np.percentile(y_src, 10)
    shadow_mask = y_src <= p10
    if shadow_mask.any():
        shadow_lift = float(np.mean(y_dst[shadow_mask] - y_src[shadow_mask]))
    else:
        shadow_lift = 0.0

    # Highlight: pixels in the top 5th percentile of the SOURCE luminance
    p95 = np.percentile(y_src, 95)
    hi_mask = y_src >= p95
    if hi_mask.any():
        highlight_comp = float(np.mean(y_src[hi_mask] - y_dst[hi_mask]))
    else:
        highlight_comp = 0.0

    return {
        "shadow_lift": float(np.clip(shadow_lift, -0.5, 0.5)),
        "highlight_comp": float(np.clip(highlight_comp, -0.5, 0.5)),
    }


def _saturation_from_pair(src: Image.Image, dst: Image.Image) -> dict:
    """
    Measure per-hue saturation multiplier as mean(S_dst) / mean(S_src)
    for pixels in each hue group.  Uses SOURCE hue to determine group
    membership so the denominator is stable.
    """
    H_src, S_src, _ = _hsv_arrays(src)
    _, S_dst, _ = _hsv_arrays(dst)

    def sat_ratio(ranges):
        mask = _hue_mask(H_src, ranges) & (S_src > 0.10)
        if not mask.any():
            return 1.0
        mean_src = float(np.mean(S_src[mask]))
        mean_dst = float(np.mean(S_dst[mask]))
        if mean_src < 1e-4:
            return 1.0
        return float(np.clip(mean_dst / mean_src, 0.1, 5.0))

    return {
        "sat_r": sat_ratio(_HUE_RANGES["r"]),
        "sat_g": sat_ratio(_HUE_RANGES["g"]),
        "sat_b": sat_ratio(_HUE_RANGES["b"]),
    }


def _vibrancy_from_pair(src: Image.Image, dst: Image.Image) -> float:
    """
    Measure vibrancy as the differential saturation boost applied to
    low-saturation pixels vs high-saturation pixels.

    Vibrancy > 0 means low-S pixels were boosted more than high-S pixels
    (characteristic of a vibrancy adjustment rather than plain saturation).
    """
    _, S_src, _ = _hsv_arrays(src)
    _, S_dst, _ = _hsv_arrays(dst)

    delta_S = S_dst - S_src  # positive = boosted

    low_mask  = (S_src > 0.05) & (S_src < 0.35)
    high_mask = (S_src > 0.55) & (S_src < 0.95)

    boost_low  = float(np.mean(delta_S[low_mask]))  if low_mask.any()  else 0.0
    boost_high = float(np.mean(delta_S[high_mask])) if high_mask.any() else 0.0

    # Vibrancy = how much MORE low-S pixels were boosted vs high-S pixels
    vibrancy = boost_low - boost_high
    return float(np.clip(vibrancy, -0.5, 1.5))


# ---------------------------------------------------------------------------
# JPEG-only extraction  (fallback — no original RAW available)
# ---------------------------------------------------------------------------

def _reference_cdf() -> np.ndarray:
    levels = np.arange(256, dtype=np.float32) / 255.0
    linear = np.where(levels <= 0.04045, levels / 12.92, ((levels + 0.055) / 1.055) ** 2.4)
    return linear / linear[-1]


_REF_CDF = _reference_cdf()


def _lum_curve_from_jpeg(img: Image.Image) -> np.ndarray:
    y = np.array(img.convert("YCbCr"))[:, :, 0].ravel().astype(np.float32)
    hist, _ = np.histogram(y, bins=256, range=(0, 255))
    cdf_actual = np.cumsum(hist).astype(np.float32)
    cdf_actual /= cdf_actual[-1]

    out_levels = np.arange(256, dtype=np.float32)
    inp_levels = np.interp(_REF_CDF, cdf_actual, out_levels / 255.0)

    return _fit_monotonic_lut(inp_levels, out_levels / 255.0)


def _shadow_highlight_from_jpeg(img: Image.Image) -> dict:
    y = _luminance_array(img)
    p10 = np.percentile(y, 10)
    shadow_mask = y <= p10
    shadow_lift = float(np.mean(y[shadow_mask])) if shadow_mask.any() else 0.0
    p95 = np.percentile(y, 95)
    hi_mask = y >= p95
    highlight_comp = float(1.0 - np.mean(y[hi_mask])) if hi_mask.any() else 0.0
    return {"shadow_lift": shadow_lift, "highlight_comp": highlight_comp}


def _saturation_from_jpeg(img: Image.Image) -> dict:
    H, S, _ = _hsv_arrays(img)

    def sat_for_group(ranges):
        mask = _hue_mask(H, ranges) & (S > 0.15)
        if not mask.any():
            return 1.0
        return float(np.mean(S[mask]) / _NEUTRAL_SAT)

    return {
        "sat_r": sat_for_group(_HUE_RANGES["r"]),
        "sat_g": sat_for_group(_HUE_RANGES["g"]),
        "sat_b": sat_for_group(_HUE_RANGES["b"]),
    }


def _vibrancy_from_jpeg(img: Image.Image) -> float:
    _, S, _ = _hsv_arrays(img)
    S_mid = S[(S > 0.05) & (S < 0.95)]
    if len(S_mid) == 0:
        return 0.0
    low_frac = float(np.mean(S_mid < 0.4))
    vibrancy = float((0.5 - low_frac) * 2.0)
    return float(np.clip(vibrancy, -0.5, 1.5))
