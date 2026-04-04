from pathlib import Path

from PIL import Image

ARW_SUFFIXES = {".arw", ".ARW"}
JPEG_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def is_arw(path: Path) -> bool:
    return path.suffix in ARW_SUFFIXES


def is_jpeg(path: Path) -> bool:
    return path.suffix in JPEG_SUFFIXES


def open_jpeg(path: Path) -> Image.Image:
    """Open a JPEG and ensure it's in RGB mode."""
    return Image.open(path).convert("RGB")


def find_arw_files(folder: Path, pattern: str = "*.ARW") -> list[Path]:
    files = list(folder.glob(pattern))
    if not files:
        # Try lowercase as well
        files = list(folder.glob(pattern.lower()))
    return sorted(files)


def find_jpeg_files(folder: Path) -> list[Path]:
    files = []
    for suffix in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        files.extend(folder.glob(suffix))
    return sorted(set(files))
