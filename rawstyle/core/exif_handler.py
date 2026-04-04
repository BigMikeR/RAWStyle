"""
Copy EXIF metadata from a Sony ARW file into an output JPEG.

Primary method: exiftool (install via `brew install exiftool`).
Fallback: piexif-based copy for environments without exiftool.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def copy_exif(arw_path: Path, jpeg_path: Path) -> None:
    """
    Copy all EXIF/IPTC/XMP metadata from `arw_path` into `jpeg_path` in-place.

    Prefers exiftool when available; falls back to piexif.
    """
    if shutil.which("exiftool"):
        _copy_with_exiftool(arw_path, jpeg_path)
    else:
        _copy_with_piexif(arw_path, jpeg_path)


def _copy_with_exiftool(arw_path: Path, jpeg_path: Path) -> None:
    """
    Use exiftool to copy all metadata tags from ARW → JPEG, overwriting
    any existing metadata in the JPEG.  The original JPEG backup created
    by exiftool is deleted automatically.
    """
    result = subprocess.run(
        [
            "exiftool",
            "-TagsFromFile", str(arw_path),
            "-All:All",
            "-overwrite_original",
            str(jpeg_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"exiftool failed for {jpeg_path.name}: {result.stderr.strip()}"
        )


def _copy_with_piexif(arw_path: Path, jpeg_path: Path) -> None:
    """
    Fallback: read EXIF from the ARW using rawpy + piexif and write into the
    output JPEG.  Less comprehensive than exiftool (no XMP/IPTC), but covers
    the core camera metadata.
    """
    import io

    import piexif
    import rawpy
    from PIL import Image

    # rawpy exposes the raw EXIF bytes as a TIFF-structured blob
    with rawpy.imread(str(arw_path)) as raw:
        exif_bytes: bytes | None = getattr(raw, "raw_image", None) and None
        # Use the ARW file itself as the EXIF source via piexif's TIFF loader
        try:
            exif_dict = piexif.load(str(arw_path))
        except Exception:
            return  # silently skip if piexif can't parse the ARW

    # Remove stale thumbnail from the EXIF dict to avoid size mismatches
    if "1st" in exif_dict:
        exif_dict["1st"] = {}
    if "thumbnail" in exif_dict:
        exif_dict["thumbnail"] = None

    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception:
        return

    # Load the JPEG, insert EXIF, save back
    img = Image.open(jpeg_path)
    out_buf = io.BytesIO()
    img.save(out_buf, format="JPEG", exif=exif_bytes, quality="keep")
    jpeg_path.write_bytes(out_buf.getvalue())


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None
