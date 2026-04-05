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
        files.extend(folder.rglob(suffix))
    return sorted(set(files))


def find_jpeg_arw_pairs(
    jpeg_dir: Path,
    arw_dir: Path,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """
    Walk `jpeg_dir` recursively and match each JPEG to a same-named ARW in
    the mirrored subfolder structure of `arw_dir`.

    Returns:
        pairs    — list of (jpeg_path, arw_path) for every matched pair
        unmatched — list of jpeg_path entries with no corresponding ARW
    """
    pairs: list[tuple[Path, Path]] = []
    unmatched: list[Path] = []

    for jpeg_path in sorted(jpeg_dir.rglob("*")):
        if not is_jpeg(jpeg_path):
            continue

        rel = jpeg_path.relative_to(jpeg_dir)

        # Try both upper and lower case ARW extension
        matched = False
        for ext in (".ARW", ".arw"):
            arw_path = arw_dir / rel.with_suffix(ext)
            if arw_path.exists():
                pairs.append((jpeg_path, arw_path))
                matched = True
                break

        if not matched:
            unmatched.append(jpeg_path)

    return pairs, unmatched
