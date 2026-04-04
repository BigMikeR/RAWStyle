"""
Apply a StyleFeature to a linearly-developed float32 RGB array.

Input:  float32[H, W, 3] in [0.0, 1.0], scene-referred, linear light
Output: uint8[H, W, 3]  sRGB gamma-encoded, ready for JPEG encoding
"""
from __future__ import annotations

import numpy as np

from rawstyle.core.style_extractor import StyleFeature


def apply(linear_rgb: np.ndarray, style: StyleFeature) -> np.ndarray:
    """
    Apply `style` to `linear_rgb` and return a display-ready uint8 sRGB image.
    """
    img = np.clip(linear_rgb, 0.0, 1.0).astype(np.float32)

    # -- Step 1: Luminance-preserving tone curve --
    img = _apply_lum_curve(img, style.lum_curve)

    # -- Step 2: Shadow lift --
    img = _apply_shadow_lift(img, style.shadow_lift)

    # -- Step 3: Highlight compression --
    img = _apply_highlight_comp(img, style.highlight_comp)

    img = np.clip(img, 0.0, 1.0)

    # -- Steps 4 & 5: Saturation + vibrancy in HSV space --
    img = _apply_sat_vibrancy(img, style)

    # -- Step 6: sRGB gamma encode --
    img = _srgb_gamma(img)

    # -- Step 7: → uint8 --
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Perceptual luminance (BT.709), shape (H, W)."""
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def _apply_lum_curve(img: np.ndarray, lum_curve: np.ndarray) -> np.ndarray:
    lum = _luminance(img)
    lum_idx = np.clip((lum * 255.0).astype(np.int32), 0, 255)
    lum_out = lum_curve[lum_idx]                          # (H, W)
    scale = np.where(lum > 1e-6, lum_out / lum, 1.0)     # (H, W)
    return img * scale[:, :, np.newaxis]


def _apply_shadow_lift(img: np.ndarray, shadow_lift: float) -> np.ndarray:
    if abs(shadow_lift) < 1e-4:
        return img
    lum = _luminance(img)
    # Smooth mask: full effect at lum=0, zero effect at lum=0.2+
    mask = np.clip(1.0 - lum / 0.2, 0.0, 1.0)
    # shadow_lift is the mean luminance in shadows of the reference image;
    # we add the difference between that and a "pure black" shadow baseline.
    lift = (shadow_lift - 0.0) * mask
    return img + lift[:, :, np.newaxis]


def _apply_highlight_comp(img: np.ndarray, highlight_comp: float) -> np.ndarray:
    if abs(highlight_comp) < 1e-4:
        return img
    lum = _luminance(img)
    # Smooth mask: zero effect at lum=0.85, full effect at lum=1.0
    mask = np.clip((lum - 0.85) / 0.15, 0.0, 1.0)
    pull = highlight_comp * mask
    return img - pull[:, :, np.newaxis]


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV.  Input/output float32 [0,1]."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(r)
    mask = delta > 1e-7
    m_r = mask & (cmax == r)
    m_g = mask & (cmax == g)
    m_b = mask & (cmax == b)
    h[m_r] = ((g[m_r] - b[m_r]) / delta[m_r]) % 6.0
    h[m_g] = (b[m_g] - r[m_g]) / delta[m_g] + 2.0
    h[m_b] = (r[m_b] - g[m_b]) / delta[m_b] + 4.0
    h = h / 6.0  # normalise to [0, 1]

    # Saturation
    s = np.where(cmax > 1e-7, delta / cmax, 0.0)

    return np.stack([h, s, cmax], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB.  Input/output float32 [0,1]."""
    h, s, v = hsv[:, :, 0] * 6.0, hsv[:, :, 1], hsv[:, :, 2]
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    rgb = np.zeros(hsv.shape, dtype=np.float32)
    for idx, (r_val, g_val, b_val) in enumerate(
        [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    ):
        mask = i == idx
        rgb[:, :, 0][mask] = r_val[mask]
        rgb[:, :, 1][mask] = g_val[mask]
        rgb[:, :, 2][mask] = b_val[mask]

    return rgb


def _apply_sat_vibrancy(img: np.ndarray, style: StyleFeature) -> np.ndarray:
    hsv = _rgb_to_hsv(img)
    H = hsv[:, :, 0] * 360.0  # degrees
    S = hsv[:, :, 1]

    # Per-hue saturation multipliers
    sat_mul = np.ones_like(S)

    # Reds / oranges / yellows: hue 0-30 and 330-360 (i.e. 0-0.083 and 0.917-1.0)
    r_mask = (H < 30.0) | (H >= 330.0)
    sat_mul[r_mask] *= style.sat_r

    # Greens / cyans: 75-165 degrees
    g_mask = (H >= 75.0) & (H < 165.0)
    sat_mul[g_mask] *= style.sat_g

    # Blues / magentas: 195-285 degrees
    b_mask = (H >= 195.0) & (H < 285.0)
    sat_mul[b_mask] *= style.sat_b

    S_new = np.clip(S * sat_mul, 0.0, 1.0)

    # Vibrancy: boost low-saturation pixels more
    if abs(style.vibrancy) > 1e-4:
        S_new = np.clip(S_new + style.vibrancy * (1.0 - S_new), 0.0, 1.0)

    hsv[:, :, 1] = S_new
    return _hsv_to_rgb(hsv)


def _srgb_gamma(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB transfer function."""
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.4) - 0.055,
    )
