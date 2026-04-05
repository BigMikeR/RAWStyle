"""
Extract photographic style features from a processed JPEG.

Two modes:
  extract(jpeg)                — JPEG-only; estimates style from the image alone.
  extract_from_pair(arw, jpeg) — Pair mode; directly compares the neutral ARW
                                 development to the processed JPEG for accurate
                                 before/after measurements.

StyleFeature fields:
  Tone:        lum_curve, curve_r, curve_g, curve_b, contrast
  Tonal range: shadow_lift, highlight_comp
  Colour:      color_temp_delta, sat_r/g/b, vibrancy
  HSL:         hue_shift_*, lum_hue_*  (8 colour groups each)
  Local:       clarity, vignette_strength, grain_strength
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEUTRAL_SAT = 0.45          # baseline saturation for JPEG-only mode
_CMP_SIZE    = (512, 512)    # comparison resolution for pair mode

# 8 hue groups matching Lightroom's HSL panel (lo, hi in degrees)
# Groups that wrap around 0° are handled specially.
HSL_GROUPS = {
    "r": (345.0, 15.0),   # red     (wraps)
    "o": (15.0,  45.0),   # orange
    "y": (45.0,  75.0),   # yellow
    "g": (75.0, 135.0),   # green
    "c": (135.0, 195.0),  # cyan
    "b": (195.0, 255.0),  # blue
    "p": (255.0, 285.0),  # purple
    "m": (285.0, 345.0),  # magenta
}
HSL_GROUP_KEYS = list(HSL_GROUPS.keys())  # stable ordering

# Legacy 3-group sat ranges kept for sat_r/g/b
_SAT3_RANGES = {
    "r": [(-30.0, 30.0), (330.0, 360.0)],
    "g": [(75.0, 165.0)],
    "b": [(195.0, 285.0)],
}


# ---------------------------------------------------------------------------
# StyleFeature dataclass
# ---------------------------------------------------------------------------

def _identity_lut() -> np.ndarray:
    return np.linspace(0.0, 1.0, 256, dtype=np.float32)


@dataclass
class StyleFeature:
    # --- Tone curves ---
    lum_curve: np.ndarray        # float32[256]: overall luminance LUT
    curve_r:   np.ndarray        # float32[256]: red channel LUT
    curve_g:   np.ndarray        # float32[256]: green channel LUT
    curve_b:   np.ndarray        # float32[256]: blue channel LUT

    # --- Contrast ---
    contrast: float              # midtone slope (1.0 = neutral)

    # --- Tonal range ---
    shadow_lift:    float        # delta lift applied to shadows
    highlight_comp: float        # delta compression applied to highlights

    # --- Global colour ---
    color_temp_delta: float      # positive = warmer, negative = cooler
    sat_r: float                 # saturation multiplier, red/orange hues
    sat_g: float                 # saturation multiplier, green/cyan hues
    sat_b: float                 # saturation multiplier, blue/magenta hues
    vibrancy: float              # selective low-saturation boost

    # --- HSL: hue shifts (8 groups, degrees) ---
    hue_shift_r: float
    hue_shift_o: float
    hue_shift_y: float
    hue_shift_g: float
    hue_shift_c: float
    hue_shift_b: float
    hue_shift_p: float
    hue_shift_m: float

    # --- HSL: luminance per hue group (8 groups, multiplier) ---
    lum_hue_r: float
    lum_hue_o: float
    lum_hue_y: float
    lum_hue_g: float
    lum_hue_c: float
    lum_hue_b: float
    lum_hue_p: float
    lum_hue_m: float

    # --- Local / finishing ---
    vignette_strength: float     # negative = darken edges (typical vignette)
    clarity:           float     # midtone local contrast boost
    grain_strength:    float     # added noise std dev (0 = none)


def neutral_feature() -> StyleFeature:
    """Return a fully neutral StyleFeature (no adjustments)."""
    return StyleFeature(
        lum_curve=_identity_lut(), curve_r=_identity_lut(),
        curve_g=_identity_lut(),   curve_b=_identity_lut(),
        contrast=1.0, shadow_lift=0.0, highlight_comp=0.0,
        color_temp_delta=0.0, sat_r=1.0, sat_g=1.0, sat_b=1.0, vibrancy=0.0,
        hue_shift_r=0.0, hue_shift_o=0.0, hue_shift_y=0.0, hue_shift_g=0.0,
        hue_shift_c=0.0, hue_shift_b=0.0, hue_shift_p=0.0, hue_shift_m=0.0,
        lum_hue_r=1.0, lum_hue_o=1.0, lum_hue_y=1.0, lum_hue_g=1.0,
        lum_hue_c=1.0, lum_hue_b=1.0, lum_hue_p=1.0, lum_hue_m=1.0,
        vignette_strength=0.0, clarity=0.0, grain_strength=0.0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(img: Image.Image) -> StyleFeature:
    """Estimate style from a processed JPEG alone (no original RAW)."""
    rgb = img.convert("RGB")
    hs = _shadow_highlight_from_jpeg(rgb)
    sat = _saturation_from_jpeg(rgb)
    return StyleFeature(
        lum_curve=_lum_curve_from_jpeg(rgb),
        curve_r=_identity_lut(), curve_g=_identity_lut(), curve_b=_identity_lut(),
        contrast=_contrast_from_jpeg(rgb),
        shadow_lift=hs["shadow_lift"], highlight_comp=hs["highlight_comp"],
        color_temp_delta=0.0,
        sat_r=sat["sat_r"], sat_g=sat["sat_g"], sat_b=sat["sat_b"],
        vibrancy=_vibrancy_from_jpeg(rgb),
        hue_shift_r=0.0, hue_shift_o=0.0, hue_shift_y=0.0, hue_shift_g=0.0,
        hue_shift_c=0.0, hue_shift_b=0.0, hue_shift_p=0.0, hue_shift_m=0.0,
        lum_hue_r=1.0, lum_hue_o=1.0, lum_hue_y=1.0, lum_hue_g=1.0,
        lum_hue_c=1.0, lum_hue_b=1.0, lum_hue_p=1.0, lum_hue_m=1.0,
        vignette_strength=_vignette_from_jpeg(rgb),
        clarity=0.0, grain_strength=0.0,
    )


def extract_from_pair(arw_path: Path, jpeg_path: Path) -> StyleFeature:
    """
    Accurately extract style by directly comparing the neutrally-developed ARW
    to the processed JPEG.  Both are resized to _CMP_SIZE for comparison.
    """
    from rawstyle.core.raw_developer import develop_thumbnail

    src = _to_square(develop_thumbnail(arw_path, size=_CMP_SIZE), _CMP_SIZE)
    dst = _to_square(Image.open(jpeg_path).convert("RGB"), _CMP_SIZE)

    src_arr = _img_to_f32(src)
    dst_arr = _img_to_f32(dst)

    hs   = _shadow_highlight_from_pair(src_arr, dst_arr)
    sat  = _saturation_from_pair(src_arr, dst_arr)
    hue_shifts = _hue_shifts_from_pair(src_arr, dst_arr)
    lum_hues   = _lum_hue_from_pair(src_arr, dst_arr)

    return StyleFeature(
        lum_curve=_lum_curve_from_pair(src_arr, dst_arr),
        curve_r=_channel_curve_from_pair(src_arr, dst_arr, 0),
        curve_g=_channel_curve_from_pair(src_arr, dst_arr, 1),
        curve_b=_channel_curve_from_pair(src_arr, dst_arr, 2),
        contrast=_contrast_from_pair(src_arr, dst_arr),
        shadow_lift=hs["shadow_lift"], highlight_comp=hs["highlight_comp"],
        color_temp_delta=_color_temp_from_pair(src_arr, dst_arr),
        sat_r=sat["sat_r"], sat_g=sat["sat_g"], sat_b=sat["sat_b"],
        vibrancy=_vibrancy_from_pair(src_arr, dst_arr),
        **hue_shifts,
        **lum_hues,
        vignette_strength=_vignette_from_pair(src_arr, dst_arr),
        clarity=_clarity_from_pair(src_arr, dst_arr),
        grain_strength=_grain_from_pair(src_arr, dst_arr),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_square(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.LANCZOS)


def _img_to_f32(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def _luminance(arr: np.ndarray) -> np.ndarray:
    """BT.709 luminance from float32 RGB array."""
    return 0.2126 * arr[:,:,0] + 0.7152 * arr[:,:,1] + 0.0722 * arr[:,:,2]


def _hsv_from_arr(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (H_deg 0-360, S 0-1, V 0-1) from float32 RGB array."""
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(r)
    m = delta > 1e-7
    mr = m & (cmax == r)
    mg = m & (cmax == g)
    mb = m & (cmax == b)
    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6.0
    h[mg] = (b[mg] - r[mg]) / delta[mg] + 2.0
    h[mb] = (r[mb] - g[mb]) / delta[mb] + 4.0
    h = (h / 6.0) * 360.0

    s = np.where(cmax > 1e-7, delta / cmax, 0.0)
    return h, s, cmax


def _hue_mask(H: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if lo > hi:  # wraps around 0°
        return (H >= lo) | (H < hi)
    return (H >= lo) & (H < hi)


def _fit_lut(x_ctrl: np.ndarray, y_ctrl: np.ndarray) -> np.ndarray:
    """Fit a 256-point monotonic LUT via PCHIP interpolation."""
    order = np.argsort(x_ctrl)
    x_ctrl, y_ctrl = x_ctrl[order], y_ctrl[order]
    _, unique = np.unique(x_ctrl, return_index=True)
    x_ctrl, y_ctrl = x_ctrl[unique], y_ctrl[unique]
    spline = PchipInterpolator(x_ctrl, y_ctrl, extrapolate=True)
    lut = spline(np.linspace(0.0, 1.0, 256, dtype=np.float32)).astype(np.float32)
    return np.clip(lut, 0.0, 1.0)


def _cdf_lut(arr_src: np.ndarray, arr_dst: np.ndarray) -> np.ndarray:
    """
    Derive a 256-point LUT by histogram specification:
    map input CDF → output CDF so that dst's distribution is reproduced.
    """
    src_u8 = np.clip(arr_src * 255.0, 0, 255).astype(np.uint8).ravel()
    dst_u8 = np.clip(arr_dst * 255.0, 0, 255).astype(np.uint8).ravel()

    hist_s, _ = np.histogram(src_u8, bins=256, range=(0, 255))
    hist_d, _ = np.histogram(dst_u8, bins=256, range=(0, 255))

    cdf_s = np.cumsum(hist_s).astype(np.float32); cdf_s /= cdf_s[-1]
    cdf_d = np.cumsum(hist_d).astype(np.float32); cdf_d /= cdf_d[-1]

    levels = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.interp(cdf_s, cdf_d, levels)
    return _fit_lut(np.linspace(0.0, 1.0, 256, dtype=np.float32), lut.astype(np.float32))


# ---------------------------------------------------------------------------
# Pair-based extraction
# ---------------------------------------------------------------------------

def _lum_curve_from_pair(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return _cdf_lut(_luminance(src), _luminance(dst))


def _channel_curve_from_pair(src: np.ndarray, dst: np.ndarray, ch: int) -> np.ndarray:
    return _cdf_lut(src[:,:,ch], dst[:,:,ch])


def _contrast_from_pair(src: np.ndarray, dst: np.ndarray) -> float:
    y_src = _luminance(src)
    y_dst = _luminance(dst)
    mid = (y_src >= 0.25) & (y_src <= 0.75)
    if not mid.any():
        return 1.0
    xs = y_src[mid] - 0.5
    ys = y_dst[mid] - 0.5
    slope = float(np.sum(xs * ys) / (np.sum(xs * xs) + 1e-8))
    return float(np.clip(slope, 0.1, 4.0))


def _shadow_highlight_from_pair(src: np.ndarray, dst: np.ndarray) -> dict:
    y_src = _luminance(src)
    y_dst = _luminance(dst)

    p10 = np.percentile(y_src, 10)
    shadow_mask = y_src <= p10
    shadow_lift = float(np.mean(y_dst[shadow_mask] - y_src[shadow_mask])) if shadow_mask.any() else 0.0

    p95 = np.percentile(y_src, 95)
    hi_mask = y_src >= p95
    highlight_comp = float(np.mean(y_src[hi_mask] - y_dst[hi_mask])) if hi_mask.any() else 0.0

    return {
        "shadow_lift":    float(np.clip(shadow_lift,    -0.5, 0.5)),
        "highlight_comp": float(np.clip(highlight_comp, -0.5, 0.5)),
    }


def _color_temp_from_pair(src: np.ndarray, dst: np.ndarray) -> float:
    """Positive = photographer pushed warmer; negative = cooler."""
    def rg_ratio(arr):
        g = np.mean(arr[:,:,1]) + 1e-6
        return np.mean(arr[:,:,0]) / g

    def bg_ratio(arr):
        g = np.mean(arr[:,:,1]) + 1e-6
        return np.mean(arr[:,:,2]) / g

    r_shift = rg_ratio(dst) - rg_ratio(src)
    b_shift = bg_ratio(dst) - bg_ratio(src)
    temp_delta = r_shift - b_shift   # warm = R up or B down
    return float(np.clip(temp_delta, -1.0, 1.0))


def _saturation_from_pair(src: np.ndarray, dst: np.ndarray) -> dict:
    H_src, S_src, _ = _hsv_from_arr(src)
    _, S_dst, _      = _hsv_from_arr(dst)

    def ratio(ranges):
        mask = np.zeros(H_src.shape, dtype=bool)
        for lo, hi in ranges:
            mask |= _hue_mask(H_src, lo, hi)
        mask &= S_src > 0.10
        if not mask.any():
            return 1.0
        mean_s = float(np.mean(S_src[mask]))
        mean_d = float(np.mean(S_dst[mask]))
        return float(np.clip(mean_d / (mean_s + 1e-6), 0.1, 5.0))

    return {
        "sat_r": ratio(_SAT3_RANGES["r"]),
        "sat_g": ratio(_SAT3_RANGES["g"]),
        "sat_b": ratio(_SAT3_RANGES["b"]),
    }


def _vibrancy_from_pair(src: np.ndarray, dst: np.ndarray) -> float:
    _, S_src, _ = _hsv_from_arr(src)
    _, S_dst, _ = _hsv_from_arr(dst)
    delta = S_dst - S_src
    low  = (S_src > 0.05) & (S_src < 0.35)
    high = (S_src > 0.55) & (S_src < 0.95)
    boost_low  = float(np.mean(delta[low]))  if low.any()  else 0.0
    boost_high = float(np.mean(delta[high])) if high.any() else 0.0
    return float(np.clip(boost_low - boost_high, -0.5, 1.5))


def _hue_shifts_from_pair(src: np.ndarray, dst: np.ndarray) -> dict:
    """Measure circular mean hue rotation per 8 HSL colour groups (degrees)."""
    H_src, S_src, _ = _hsv_from_arr(src)
    H_dst, _, _      = _hsv_from_arr(dst)

    # Circular difference: map to [-180, 180]
    diff = ((H_dst - H_src) + 180.0) % 360.0 - 180.0

    result = {}
    for key in HSL_GROUP_KEYS:
        lo, hi = HSL_GROUPS[key]
        mask = _hue_mask(H_src, lo, hi) & (S_src > 0.15)
        if mask.any():
            shift = float(np.mean(diff[mask]))
        else:
            shift = 0.0
        result[f"hue_shift_{key}"] = float(np.clip(shift, -60.0, 60.0))

    return result


def _lum_hue_from_pair(src: np.ndarray, dst: np.ndarray) -> dict:
    """Measure luminance multiplier per 8 HSL colour groups."""
    H_src, S_src, V_src = _hsv_from_arr(src)
    _,     _,     V_dst = _hsv_from_arr(dst)

    result = {}
    for key in HSL_GROUP_KEYS:
        lo, hi = HSL_GROUPS[key]
        mask = _hue_mask(H_src, lo, hi) & (S_src > 0.10)
        if mask.any():
            mean_v_src = float(np.mean(V_src[mask]))
            mean_v_dst = float(np.mean(V_dst[mask]))
            ratio = float(np.clip(mean_v_dst / (mean_v_src + 1e-6), 0.1, 5.0))
        else:
            ratio = 1.0
        result[f"lum_hue_{key}"] = ratio

    return result


def _vignette_from_pair(src: np.ndarray, dst: np.ndarray) -> float:
    H, W = src.shape[:2]
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((y - H/2) / (H/2))**2 + ((x - W/2) / (W/2))**2)

    center_mask = r < 0.3
    edge_mask   = r > 0.8

    y_src = _luminance(src)
    y_dst = _luminance(dst)

    def mean_ratio(mask):
        valid = mask & (y_src > 0.05)
        if not valid.any():
            return 1.0
        return float(np.mean(y_dst[valid] / (y_src[valid] + 1e-6)))

    ratio_center = mean_ratio(center_mask)
    ratio_edge   = mean_ratio(edge_mask)
    # Negative = edges darkened relative to centre (classic vignette)
    return float(np.clip(ratio_edge - ratio_center, -0.5, 0.5))


def _clarity_from_pair(src: np.ndarray, dst: np.ndarray) -> float:
    def local_contrast_energy(arr, mask):
        grey = _luminance(arr)
        hp = grey - gaussian_filter(grey, sigma=8.0)
        if not mask.any():
            return 1e-6
        return float(np.mean(np.abs(hp[mask])))

    y_src = _luminance(src)
    mid_mask = (y_src >= 0.25) & (y_src <= 0.75)
    e_src = local_contrast_energy(src, mid_mask)
    e_dst = local_contrast_energy(dst, mid_mask)
    clarity = (e_dst / (e_src + 1e-8)) - 1.0
    return float(np.clip(clarity, -0.5, 2.0))


def _grain_from_pair(src: np.ndarray, dst: np.ndarray) -> float:
    y_src = _luminance(src)
    shadow_mask = y_src < 0.2
    if not shadow_mask.any():
        return 0.0

    def noise_std(arr, mask):
        grey = _luminance(arr)
        residual = grey - gaussian_filter(grey, sigma=1.5)
        return float(np.std(residual[mask]))

    src_noise = noise_std(src, shadow_mask)
    dst_noise = noise_std(dst, shadow_mask)
    added = max(0.0, dst_noise - src_noise)
    return float(np.clip(added, 0.0, 0.08))


# ---------------------------------------------------------------------------
# JPEG-only extraction (fallback)
# ---------------------------------------------------------------------------

def _reference_cdf() -> np.ndarray:
    levels = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    linear = np.where(levels <= 0.04045, levels / 12.92,
                      ((levels + 0.055) / 1.055) ** 2.4)
    return linear / linear[-1]


_REF_CDF = _reference_cdf()


def _lum_curve_from_jpeg(img: Image.Image) -> np.ndarray:
    y = np.array(img.convert("YCbCr"))[:,:,0].ravel().astype(np.float32)
    hist, _ = np.histogram(y, bins=256, range=(0, 255))
    cdf = np.cumsum(hist).astype(np.float32); cdf /= cdf[-1]
    levels = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    inp = np.interp(_REF_CDF, cdf, levels)
    return _fit_lut(inp, levels)


def _contrast_from_jpeg(img: Image.Image) -> float:
    arr = _img_to_f32(img)
    y = _luminance(arr)
    std = float(np.std(y))
    # Neutral std for a well-exposed image is ~0.22; scale relative to that
    return float(np.clip(std / 0.22, 0.3, 3.0))


def _shadow_highlight_from_jpeg(img: Image.Image) -> dict:
    arr = _img_to_f32(img)
    y = _luminance(arr)
    p10 = np.percentile(y, 10)
    shadow_lift = float(np.mean(y[y <= p10]))
    p95 = np.percentile(y, 95)
    highlight_comp = float(1.0 - np.mean(y[y >= p95]))
    return {"shadow_lift": shadow_lift, "highlight_comp": highlight_comp}


def _saturation_from_jpeg(img: Image.Image) -> dict:
    arr = _img_to_f32(img)
    H, S, _ = _hsv_from_arr(arr)

    def sat_for(ranges):
        mask = np.zeros(H.shape, dtype=bool)
        for lo, hi in ranges:
            mask |= _hue_mask(H, lo, hi)
        mask &= S > 0.15
        if not mask.any():
            return 1.0
        return float(np.mean(S[mask]) / _NEUTRAL_SAT)

    return {
        "sat_r": sat_for(_SAT3_RANGES["r"]),
        "sat_g": sat_for(_SAT3_RANGES["g"]),
        "sat_b": sat_for(_SAT3_RANGES["b"]),
    }


def _vibrancy_from_jpeg(img: Image.Image) -> float:
    arr = _img_to_f32(img)
    _, S, _ = _hsv_from_arr(arr)
    S_mid = S[(S > 0.05) & (S < 0.95)]
    if len(S_mid) == 0:
        return 0.0
    low_frac = float(np.mean(S_mid < 0.4))
    return float(np.clip((0.5 - low_frac) * 2.0, -0.5, 1.5))


def _vignette_from_jpeg(img: Image.Image) -> float:
    arr = _img_to_f32(img)
    H, W = arr.shape[:2]
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((y - H/2)/(H/2))**2 + ((x - W/2)/(W/2))**2)
    lum = _luminance(arr)
    center = float(np.mean(lum[r < 0.3]))
    edge   = float(np.mean(lum[r > 0.8]))
    # Ratio below 1 suggests vignette was applied
    ratio = edge / (center + 1e-6)
    return float(np.clip(ratio - 1.0, -0.5, 0.5))
