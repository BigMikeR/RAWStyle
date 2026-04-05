import sqlite3
import numpy as np

from rawstyle.core.style_extractor import StyleFeature, _identity_lut


def _blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _curve_blob(curve) -> bytes:
    """Safely serialise a curve — fall back to identity if None."""
    if curve is None:
        return _blob(_identity_lut())
    return _blob(curve)


def upsert_image(
    conn: sqlite3.Connection,
    path: str,
    mtime: float,
    embedding: np.ndarray,
    style: StyleFeature,
) -> None:
    cur = conn.execute(
        """
        INSERT INTO images (path, mtime, embedding)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            mtime     = excluded.mtime,
            embedding = excluded.embedding
        RETURNING id
        """,
        (path, mtime, _blob(embedding)),
    )
    image_id = cur.fetchone()[0]

    conn.execute(
        """
        INSERT INTO styles (
            image_id,
            lum_curve, curve_r, curve_g, curve_b,
            contrast, shadow_lift, highlight_comp,
            color_temp_delta, sat_r, sat_g, sat_b, vibrancy,
            hs_r, hs_o, hs_y, hs_g, hs_c, hs_b, hs_p, hs_m,
            lh_r, lh_o, lh_y, lh_g, lh_c, lh_b, lh_p, lh_m,
            vignette_strength, clarity, grain_strength
        ) VALUES (
            ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?
        )
        ON CONFLICT(image_id) DO UPDATE SET
            lum_curve         = excluded.lum_curve,
            curve_r           = excluded.curve_r,
            curve_g           = excluded.curve_g,
            curve_b           = excluded.curve_b,
            contrast          = excluded.contrast,
            shadow_lift       = excluded.shadow_lift,
            highlight_comp    = excluded.highlight_comp,
            color_temp_delta  = excluded.color_temp_delta,
            sat_r             = excluded.sat_r,
            sat_g             = excluded.sat_g,
            sat_b             = excluded.sat_b,
            vibrancy          = excluded.vibrancy,
            hs_r = excluded.hs_r, hs_o = excluded.hs_o,
            hs_y = excluded.hs_y, hs_g = excluded.hs_g,
            hs_c = excluded.hs_c, hs_b = excluded.hs_b,
            hs_p = excluded.hs_p, hs_m = excluded.hs_m,
            lh_r = excluded.lh_r, lh_o = excluded.lh_o,
            lh_y = excluded.lh_y, lh_g = excluded.lh_g,
            lh_c = excluded.lh_c, lh_b = excluded.lh_b,
            lh_p = excluded.lh_p, lh_m = excluded.lh_m,
            vignette_strength = excluded.vignette_strength,
            clarity           = excluded.clarity,
            grain_strength    = excluded.grain_strength
        """,
        (
            image_id,
            _curve_blob(style.lum_curve),
            _curve_blob(style.curve_r),
            _curve_blob(style.curve_g),
            _curve_blob(style.curve_b),
            style.contrast, style.shadow_lift, style.highlight_comp,
            style.color_temp_delta, style.sat_r, style.sat_g, style.sat_b, style.vibrancy,
            style.hue_shift_r, style.hue_shift_o, style.hue_shift_y, style.hue_shift_g,
            style.hue_shift_c, style.hue_shift_b, style.hue_shift_p, style.hue_shift_m,
            style.lum_hue_r, style.lum_hue_o, style.lum_hue_y, style.lum_hue_g,
            style.lum_hue_c, style.lum_hue_b, style.lum_hue_p, style.lum_hue_m,
            style.vignette_strength, style.clarity, style.grain_strength,
        ),
    )
    conn.commit()


def needs_reindex(conn: sqlite3.Connection, path: str, mtime: float) -> bool:
    row = conn.execute(
        "SELECT mtime FROM images WHERE path = ?", (path,)
    ).fetchone()
    return row is None or row[0] != mtime


def prune_missing(conn: sqlite3.Connection) -> int:
    from pathlib import Path
    rows = conn.execute("SELECT id, path FROM images").fetchall()
    removed = 0
    for image_id, path in rows:
        if not Path(path).exists():
            conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
            removed += 1
    conn.commit()
    return removed
