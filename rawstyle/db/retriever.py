import sqlite3
from dataclasses import dataclass

import numpy as np

from rawstyle.core.style_extractor import StyleFeature


@dataclass
class Match:
    path: str
    distance: float  # cosine distance (0 = identical, 2 = opposite)
    style: StyleFeature


def _row_to_style(row) -> StyleFeature:
    lum_curve = np.frombuffer(row["lum_curve"], dtype=np.float32).copy()
    return StyleFeature(
        lum_curve=lum_curve,
        shadow_lift=row["shadow_lift"],
        highlight_comp=row["highlight_comp"],
        sat_r=row["sat_r"],
        sat_g=row["sat_g"],
        sat_b=row["sat_b"],
        vibrancy=row["vibrancy"],
    )


def find_similar(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    k: int = 5,
) -> list[Match]:
    """
    Return the K most content-similar reference images as Match objects.
    Uses brute-force cosine similarity — fast enough for personal libraries.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT i.path, i.embedding,
               s.lum_curve, s.shadow_lift, s.highlight_comp,
               s.sat_r, s.sat_g, s.sat_b, s.vibrancy
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
    )  # (N, 512)

    # Cosine similarity: query is already L2-normalised from clip_embedder
    query = query_embedding.astype(np.float32)
    query /= np.linalg.norm(query) + 1e-8
    sims = embeddings @ query  # (N,)

    top_k = min(k, len(rows))
    indices = np.argpartition(sims, -top_k)[-top_k:]
    indices = indices[np.argsort(sims[indices])[::-1]]  # descending similarity

    results = []
    for idx in indices:
        # distance = 1 - similarity (range [0, 2])
        distance = float(1.0 - sims[idx])
        results.append(
            Match(
                path=paths[idx],
                distance=distance,
                style=_row_to_style(rows[idx]),
            )
        )
    return results


def count_images(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
