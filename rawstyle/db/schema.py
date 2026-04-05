import sqlite3
from pathlib import Path


_BASE_DDL = """
CREATE TABLE IF NOT EXISTS images (
    id        INTEGER PRIMARY KEY,
    path      TEXT UNIQUE NOT NULL,
    mtime     REAL NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS styles (
    image_id       INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    -- Tone curves
    lum_curve      BLOB    NOT NULL,
    curve_r        BLOB,
    curve_g        BLOB,
    curve_b        BLOB,
    -- Contrast / tonal range
    contrast       REAL    NOT NULL DEFAULT 1.0,
    shadow_lift    REAL    NOT NULL DEFAULT 0.0,
    highlight_comp REAL    NOT NULL DEFAULT 0.0,
    -- Global colour
    color_temp_delta  REAL NOT NULL DEFAULT 0.0,
    sat_r             REAL NOT NULL DEFAULT 1.0,
    sat_g             REAL NOT NULL DEFAULT 1.0,
    sat_b             REAL NOT NULL DEFAULT 1.0,
    vibrancy          REAL NOT NULL DEFAULT 0.0,
    -- Hue shifts (8 colour groups, degrees)
    hs_r  REAL NOT NULL DEFAULT 0.0,
    hs_o  REAL NOT NULL DEFAULT 0.0,
    hs_y  REAL NOT NULL DEFAULT 0.0,
    hs_g  REAL NOT NULL DEFAULT 0.0,
    hs_c  REAL NOT NULL DEFAULT 0.0,
    hs_b  REAL NOT NULL DEFAULT 0.0,
    hs_p  REAL NOT NULL DEFAULT 0.0,
    hs_m  REAL NOT NULL DEFAULT 0.0,
    -- Luminance per hue group (8 groups, multiplier)
    lh_r  REAL NOT NULL DEFAULT 1.0,
    lh_o  REAL NOT NULL DEFAULT 1.0,
    lh_y  REAL NOT NULL DEFAULT 1.0,
    lh_g  REAL NOT NULL DEFAULT 1.0,
    lh_c  REAL NOT NULL DEFAULT 1.0,
    lh_b  REAL NOT NULL DEFAULT 1.0,
    lh_p  REAL NOT NULL DEFAULT 1.0,
    lh_m  REAL NOT NULL DEFAULT 1.0,
    -- Local / finishing
    vignette_strength REAL NOT NULL DEFAULT 0.0,
    clarity           REAL NOT NULL DEFAULT 0.0,
    grain_strength    REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
"""

# Columns added after the initial schema — used for migration of existing DBs.
_MIGRATION_COLUMNS = [
    ("curve_r",           "BLOB"),
    ("curve_g",           "BLOB"),
    ("curve_b",           "BLOB"),
    ("contrast",          "REAL NOT NULL DEFAULT 1.0"),
    ("color_temp_delta",  "REAL NOT NULL DEFAULT 0.0"),
    ("hs_r",  "REAL NOT NULL DEFAULT 0.0"),
    ("hs_o",  "REAL NOT NULL DEFAULT 0.0"),
    ("hs_y",  "REAL NOT NULL DEFAULT 0.0"),
    ("hs_g",  "REAL NOT NULL DEFAULT 0.0"),
    ("hs_c",  "REAL NOT NULL DEFAULT 0.0"),
    ("hs_b",  "REAL NOT NULL DEFAULT 0.0"),
    ("hs_p",  "REAL NOT NULL DEFAULT 0.0"),
    ("hs_m",  "REAL NOT NULL DEFAULT 0.0"),
    ("lh_r",  "REAL NOT NULL DEFAULT 1.0"),
    ("lh_o",  "REAL NOT NULL DEFAULT 1.0"),
    ("lh_y",  "REAL NOT NULL DEFAULT 1.0"),
    ("lh_g",  "REAL NOT NULL DEFAULT 1.0"),
    ("lh_c",  "REAL NOT NULL DEFAULT 1.0"),
    ("lh_b",  "REAL NOT NULL DEFAULT 1.0"),
    ("lh_p",  "REAL NOT NULL DEFAULT 1.0"),
    ("lh_m",  "REAL NOT NULL DEFAULT 1.0"),
    ("vignette_strength", "REAL NOT NULL DEFAULT 0.0"),
    ("clarity",           "REAL NOT NULL DEFAULT 0.0"),
    ("grain_strength",    "REAL NOT NULL DEFAULT 0.0"),
]


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_BASE_DDL)
    conn.commit()
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any columns that were introduced after the initial schema."""
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(styles)").fetchall()
    }
    for col_name, col_def in _MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE styles ADD COLUMN {col_name} {col_def}")
    conn.commit()
