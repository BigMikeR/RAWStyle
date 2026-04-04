import sqlite3
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS images (
    id        INTEGER PRIMARY KEY,
    path      TEXT UNIQUE NOT NULL,
    mtime     REAL NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS styles (
    image_id       INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    lum_curve      BLOB NOT NULL,
    shadow_lift    REAL NOT NULL,
    highlight_comp REAL NOT NULL,
    sat_r          REAL NOT NULL,
    sat_g          REAL NOT NULL,
    sat_b          REAL NOT NULL,
    vibrancy       REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(DDL)
    conn.commit()
    return conn
