"""
Apply a StyleFeature to a linearly-developed float32 RGB array.

Pipeline (all display-space operations after initial gamma encode):

  Linear domain  →  gamma encode
  Display domain →  lum_curve → per-channel curves → contrast
                 →  shadow_lift → highlight_comp → color_temp
                 →  HSV: hue_shifts, saturation, lum_per_hue, vibrancy
                 →  clarity → vignette → grain
                 →  uint8
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from rawstyle.core.style_extractor import StyleFeature, HSL_GROUPS, HSL_GROUP_KEYS


def apply(linear_rgb: np.ndarray, style: StyleFeature) -> np.ndarray:
    """
    Apply `style` to `linear_rgb` and return a display-ready uint8 sRGB image.

    Input:  float32[H, W, 3] in [0.0, 1.0], scene-referred linear light
    Output: uint8[H, W, 3]  sRGB
    """
    img = np.clip(linear_rgb, 0.0, 1.0).astype(np.float32)

    # ---- 1. Gamma encode (linear → display-referred) ----
    img = _srgb_gamma(img)

    # ---- 2. Luminance tone curve ----
    img = _apply_lum_curve(img, style.lum_curve)

    # ---- 3. Per-channel R/G/B curves (colour grade / split toning) ----
    img = _apply_channel_curves(img, style.curve_r, style.curve_g, style.curve_b)

    # ---- 4. Contrast (midtone slope) ----
    img = _apply_contrast(img, style.contrast)

    # ---- 5. Shadow lift ----
    img = _apply_shadow_lift(img, style.shadow_lift)

    # ---- 6. Highlight compression ----
    img = _apply_highlight_comp(img, style.highlight_comp)

    # ---- 7. Colour temperature ----
    img = _apply_color_temp(img, style.color_temp_delta)

    img = np.clip(img, 0.0, 1.0)

    # ---- 8–11. HSV adjustments ----
    img = _apply_hsv(img, style)

    # ---- 12. Clarity (midtone local contrast) ----
    img = _apply_clarity(img, style.clarity)

    # ---- 13. Vignette ----
    img = _apply_vignette(img, style.vignette_strength)

    img = np.clip(img, 0.0, 1.0)

    # ---- 14. Convert to uint8 ----
    result = (img * 255.0).round().astype(np.uint8)

    # ---- 15. Grain (applied on uint8 as film grain) ----
    result = _apply_grain(result, style.grain_strength)

    return result


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _luminance(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]


def _srgb_gamma(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.4) - 0.055,
    )


def _apply_lum_curve(img: np.ndarray, lum_curve: np.ndarray) -> np.ndarray:
    lum = _luminance(img)
    idx = np.clip((lum * 255.0).astype(np.int32), 0, 255)
    lum_out = lum_curve[idx]
    scale = np.where(lum > 1e-6, lum_out / lum, 1.0)
    return img * scale[:,:, np.newaxis]


def _apply_channel_curves(
    img: np.ndarray,
    curve_r: np.ndarray,
    curve_g: np.ndarray,
    curve_b: np.ndarray,
) -> np.ndarray:
    result = img.copy()
    for c, curve in enumerate([curve_r, curve_g, curve_b]):
        idx = np.clip((img[:,:,c] * 255.0).astype(np.int32), 0, 255)
        result[:,:,c] = curve[idx]
    return result


def _apply_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    if abs(contrast - 1.0) < 0.02:
        return img
    lum = _luminance(img)
    lum_new = np.clip(0.5 + (lum - 0.5) * contrast, 0.0, 1.0)
    scale = np.where(lum > 1e-6, lum_new / lum, 1.0)
    return img * scale[:,:, np.newaxis]


def _apply_shadow_lift(img: np.ndarray, shadow_lift: float) -> np.ndarray:
    if abs(shadow_lift) < 1e-4:
        return img
    lum = _luminance(img)
    mask = np.clip(1.0 - lum / 0.2, 0.0, 1.0)
    return img + (shadow_lift * mask)[:,:, np.newaxis]


def _apply_highlight_comp(img: np.ndarray, highlight_comp: float) -> np.ndarray:
    if abs(highlight_comp) < 1e-4:
        return img
    lum = _luminance(img)
    mask = np.clip((lum - 0.85) / 0.15, 0.0, 1.0)
    return img - (highlight_comp * mask)[:,:, np.newaxis]


def _apply_color_temp(img: np.ndarray, delta: float) -> np.ndarray:
    if abs(delta) < 0.005:
        return img
    result = img.copy()
    result[:,:,0] = np.clip(img[:,:,0] * (1.0 + delta * 0.5), 0.0, 1.0)  # R
    result[:,:,2] = np.clip(img[:,:,2] * (1.0 - delta * 0.5), 0.0, 1.0)  # B
    return result


def _apply_hsv(img: np.ndarray, style: StyleFeature) -> np.ndarray:
    hsv = _rgb_to_hsv(img)
    H, S, V = hsv[:,:,0].copy(), hsv[:,:,1].copy(), hsv[:,:,2].copy()

    hue_shifts = [
        style.hue_shift_r, style.hue_shift_o, style.hue_shift_y, style.hue_shift_g,
        style.hue_shift_c, style.hue_shift_b, style.hue_shift_p, style.hue_shift_m,
    ]
    lum_muls = [
        style.lum_hue_r, style.lum_hue_o, style.lum_hue_y, style.lum_hue_g,
        style.lum_hue_c, style.lum_hue_b, style.lum_hue_p, style.lum_hue_m,
    ]
    sat_muls = {
        "r": style.sat_r, "o": style.sat_r, "y": style.sat_r,   # red group
        "g": style.sat_g, "c": style.sat_g,                      # green group
        "b": style.sat_b, "p": style.sat_b, "m": style.sat_b,   # blue group
    }

    for key, hs, lm in zip(HSL_GROUP_KEYS, hue_shifts, lum_muls):
        lo, hi = HSL_GROUPS[key]
        if lo > hi:
            mask = (H >= lo) | (H < hi)
        else:
            mask = (H >= lo) & (H < hi)

        if not mask.any():
            continue

        # Hue shift
        if abs(hs) > 0.1:
            H[mask] = (H[mask] + hs) % 360.0

        # Saturation multiplier (mapped from 3-group to 8-group)
        sm = sat_muls[key]
        if abs(sm - 1.0) > 0.01:
            S[mask] = np.clip(S[mask] * sm, 0.0, 1.0)

        # Luminance per hue group
        if abs(lm - 1.0) > 0.01:
            V[mask] = np.clip(V[mask] * lm, 0.0, 1.0)

    # Vibrancy: boost low-saturation pixels selectively
    if abs(style.vibrancy) > 1e-4:
        S = np.clip(S + style.vibrancy * (1.0 - S), 0.0, 1.0)

    hsv[:,:,0] = H
    hsv[:,:,1] = S
    hsv[:,:,2] = V
    return _hsv_to_rgb(hsv)


def _apply_clarity(img: np.ndarray, clarity: float) -> np.ndarray:
    if abs(clarity) < 0.005:
        return img
    blurred = np.stack(
        [gaussian_filter(img[:,:,c], sigma=8.0) for c in range(3)], axis=-1
    )
    highpass = img - blurred
    lum = _luminance(img)
    midtone_w = np.exp(-8.0 * (lum - 0.5) ** 2)[:,:, np.newaxis]
    return np.clip(img + clarity * highpass * midtone_w, 0.0, 1.0)


def _apply_vignette(img: np.ndarray, vignette_strength: float) -> np.ndarray:
    if abs(vignette_strength) < 0.005:
        return img
    H, W = img.shape[:2]
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((y - H/2) / (H/2))**2 + ((x - W/2) / (W/2))**2)
    radial = np.clip(1.0 - r**2, 0.0, 1.0)
    scale = np.clip(1.0 + vignette_strength * (1.0 - radial), 0.0, 2.0)
    return img * scale[:,:, np.newaxis]


def _apply_grain(img: np.ndarray, grain_strength: float) -> np.ndarray:
    if grain_strength < 0.005:
        return img
    sigma = grain_strength * 255.0
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Vectorised RGB ↔ HSV
# ---------------------------------------------------------------------------

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
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
    h = (h / 6.0) * 360.0  # convert to degrees

    s = np.where(cmax > 1e-7, delta / cmax, 0.0)
    return np.stack([h, s, cmax], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    H_deg, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    h = (H_deg / 360.0) * 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = V * (1.0 - S)
    q = V * (1.0 - f * S)
    t = V * (1.0 - (1.0 - f) * S)

    rgb = np.zeros(hsv.shape, dtype=np.float32)
    for idx, (rv, gv, bv) in enumerate(
        [(V, t, p), (q, V, p), (p, V, t), (p, q, V), (t, p, V), (V, p, q)]
    ):
        mask = i == idx
        rgb[:,:,0][mask] = rv[mask]
        rgb[:,:,1][mask] = gv[mask]
        rgb[:,:,2][mask] = bv[mask]

    return rgb
