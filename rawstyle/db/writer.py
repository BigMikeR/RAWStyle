import sqlite3
import numpy as np

from rawstyle.core.style_extractor import StyleFeature


def upsert_image(
    conn: sqlite3.Connection,
    path: str,
    mtime: float,
    embedding: np.ndarray,
    style: StyleFeature,
) -> None:
    """Insert or replace a reference image and its style features."""
    emb_bytes = embedding.astype(np.float32).tobytes()

    cur = conn.execute(
        """
        INSERT INTO images (path, mtime, embedding)
        VALUES (?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            mtime     = excluded.mtime,
            embedding = excluded.embedding
        RETURNING id
        """,
        (path, mtime, emb_bytes),
    )
    row = cur.fetchone()
    image_id = row[0]

    conn.execute(
        """
        INSERT INTO styles
            (image_id, lum_curve, shadow_lift, highlight_comp,
             sat_r, sat_g, sat_b, vibrancy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(image_id) DO UPDATE SET
            lum_curve      = excluded.lum_curve,
            shadow_lift    = excluded.shadow_lift,
            highlight_comp = excluded.highlight_comp,
            sat_r          = excluded.sat_r,
            sat_g          = excluded.sat_g,
            sat_b          = excluded.sat_b,
            vibrancy       = excluded.vibrancy
        """,
        (
            image_id,
            style.lum_curve.astype(np.float32).tobytes(),
            style.shadow_lift,
            style.highlight_comp,
            style.sat_r,
            style.sat_g,
            style.sat_b,
            style.vibrancy,
        ),
    )
    conn.commit()


def needs_reindex(conn: sqlite3.Connection, path: str, mtime: float) -> bool:
    """Return True if the file is not in the DB or its mtime has changed."""
    row = conn.execute(
        "SELECT mtime FROM images WHERE path = ?", (path,)
    ).fetchone()
    return row is None or row[0] != mtime


def prune_missing(conn: sqlite3.Connection) -> int:
    """Remove DB entries whose files no longer exist on disk."""
    from pathlib import Path

    rows = conn.execute("SELECT id, path FROM images").fetchall()
    removed = 0
    for image_id, path in rows:
        if not Path(path).exists():
            conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
            removed += 1
    conn.commit()
    return removed
