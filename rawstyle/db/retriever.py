import sqlite3
from dataclasses import dataclass

import numpy as np

from rawstyle.core.style_extractor import StyleFeature, _identity_lut


@dataclass
class Match:
    path: str
    distance: float
    style: StyleFeature


def _blob_to_curve(blob) -> np.ndarray:
    if blob is None:
        return _identity_lut()
    return np.frombuffer(blob, dtype=np.float32).copy()


def _row_to_style(row: sqlite3.Row) -> StyleFeature:
    return StyleFeature(
        lum_curve=_blob_to_curve(row["lum_curve"]),
        curve_r=_blob_to_curve(row["curve_r"]),
        curve_g=_blob_to_curve(row["curve_g"]),
        curve_b=_blob_to_curve(row["curve_b"]),
        contrast=row["contrast"]          or 1.0,
        shadow_lift=row["shadow_lift"]    or 0.0,
        highlight_comp=row["highlight_comp"] or 0.0,
        color_temp_delta=row["color_temp_delta"] or 0.0,
        sat_r=row["sat_r"] or 1.0,
        sat_g=row["sat_g"] or 1.0,
        sat_b=row["sat_b"] or 1.0,
        vibrancy=row["vibrancy"] or 0.0,
        hue_shift_r=row["hs_r"] or 0.0,
        hue_shift_o=row["hs_o"] or 0.0,
        hue_shift_y=row["hs_y"] or 0.0,
        hue_shift_g=row["hs_g"] or 0.0,
        hue_shift_c=row["hs_c"] or 0.0,
        hue_shift_b=row["hs_b"] or 0.0,
        hue_shift_p=row["hs_p"] or 0.0,
        hue_shift_m=row["hs_m"] or 0.0,
        lum_hue_r=row["lh_r"] or 1.0,
        lum_hue_o=row["lh_o"] or 1.0,
        lum_hue_y=row["lh_y"] or 1.0,
        lum_hue_g=row["lh_g"] or 1.0,
        lum_hue_c=row["lh_c"] or 1.0,
        lum_hue_b=row["lh_b"] or 1.0,
        lum_hue_p=row["lh_p"] or 1.0,
        lum_hue_m=row["lh_m"] or 1.0,
        vignette_strength=row["vignette_strength"] or 0.0,
        clarity=row["clarity"]           or 0.0,
        grain_strength=row["grain_strength"] or 0.0,
    )


def find_similar(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    k: int = 5,
) -> list[Match]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT i.path, i.embedding,
               s.lum_curve, s.curve_r, s.curve_g, s.curve_b,
               s.contrast, s.shadow_lift, s.highlight_comp,
               s.color_temp_delta, s.sat_r, s.sat_g, s.sat_b, s.vibrancy,
               s.hs_r, s.hs_o, s.hs_y, s.hs_g, s.hs_c, s.hs_b, s.hs_p, s.hs_m,
               s.lh_r, s.lh_o, s.lh_y, s.lh_g, s.lh_c, s.lh_b, s.lh_p, s.lh_m,
               s.vignette_strength, s.clarity, s.grain_strength
        FROM images i
        JOIN styles s ON s.image_id = i.id
        """
    ).fetchall()
    conn.row_factory = None

    if not rows:
        return []

    paths = [r["path"] for r in rows]
    embeddings = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )

    query = query_embedding.astype(np.float32)
    query /= np.linalg.norm(query) + 1e-8
    sims = embeddings @ query

    top_k = min(k, len(rows))
    indices = np.argpartition(sims, -top_k)[-top_k:]
    indices = indices[np.argsort(sims[indices])[::-1]]

    return [
        Match(
            path=paths[i],
            distance=float(1.0 - sims[i]),
            style=_row_to_style(rows[i]),
        )
        for i in indices
    ]


def count_images(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
