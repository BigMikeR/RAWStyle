# RAWStyle

RAWStyle is a macOS CLI tool that automatically applies your personal photographic style to Sony ARW RAW files.

It analyses your library of previously processed JPEG photos, identifies which ones are visually similar to each new RAW file (by subject matter, lighting, and composition), and applies a blended version of your style — tone curves, shadows, highlights, saturation, and vibrancy — to produce a high-resolution JPEG output.

---

## How it works

1. **Index** — RAWStyle scans your processed JPEG library and builds a searchable index. Each image is embedded using a CLIP vision model (ViT-B/32), which understands the semantic content of a photo (portrait, landscape, seascape, golden hour, etc.). Style features are extracted from each JPEG — tone curve, shadow lift, highlight compression, per-hue saturation, and vibrancy.

2. **Match** — When processing a new ARW file, RAWStyle develops a preview and finds the most content-similar images in your library using cosine similarity on the CLIP embeddings. A portrait will be matched against your processed portraits; a seascape against your seascapes.

3. **Blend** — The style features of the top K matches are blended together using a distance-weighted average, so closer matches have more influence on the final result.

4. **Apply** — The blended style is applied to the full-resolution linear RAW data: tone curve, shadow/highlight treatment, saturation per colour range, and vibrancy. The result is gamma-encoded to sRGB and saved as a high-quality JPEG.

5. **EXIF** — All original camera metadata from the ARW file is copied into the output JPEG.

---

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.11+
- Homebrew

---

## Installation

### 1. Install system dependencies

```bash
brew install python@3.11 exiftool
```

`exiftool` is used for reliable EXIF metadata transfer from ARW to JPEG output.

### 2. Install RAWStyle

```bash
pip3.11 install git+https://github.com/BigMikeR/RAWStyle.git
```

Or clone and install locally:

```bash
git clone https://github.com/BigMikeR/RAWStyle.git
cd RAWStyle
python3.11 -m pip install -e .
```

> **Note:** The first run will download the CLIP ViT-B/32 model (~350 MB). This is cached automatically and only happens once.

---

## Usage

### Step 1 — Index your processed JPEG library

Run this once (and again whenever you add new photos to your library):

```bash
rawstyle index ~/Pictures/Processed/
```

This scans all JPEGs in the folder, extracts style features and CLIP embeddings, and stores them in a local SQLite database at `~/.rawstyle/library.db`.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--db PATH` | `~/.rawstyle/library.db` | Path to the library database |
| `--force` | off | Re-index all files, ignoring the modification time cache |
| `--batch-size INT` | `64` | Number of images per CLIP embedding batch |
| `--model TEXT` | `ViT-B-32` | CLIP model variant (`ViT-B-32` or `ViT-L-14` for higher accuracy) |
| `--verbose` | off | Show per-file progress |

---

### Step 2 — Process ARW files

```bash
rawstyle process ~/Shoots/2026-04/ ~/Shoots/2026-04/output/
```

RAWStyle reads every `.ARW` file in the input folder, finds matching styles from your library, and writes styled JPEGs to the output folder.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--db PATH` | `~/.rawstyle/library.db` | Path to the library database |
| `--k INT` | `5` | Number of reference images to blend |
| `--temperature FLOAT` | `0.15` | Blend sharpness — lower values favour the closest match more strongly |
| `--quality INT` | `95` | Output JPEG quality (1–100) |
| `--pattern TEXT` | `*.ARW` | Input file glob pattern |
| `--dry-run` | off | Print matches without writing any output |
| `--verbose` | off | Print matched reference filenames and similarity scores |

---

### Inspect a single file

See which reference images would be matched for a given ARW, and what style would be applied:

```bash
rawstyle inspect DSC00001.ARW --verbose
```

---

### View library info

```bash
rawstyle info
```

---

### Remove missing entries from the index

If you have deleted photos from your library, clean up the index with:

```bash
rawstyle reindex --prune
```

---

## Tips

- **More accurate matching** — Use `--model ViT-L-14` when indexing for better semantic discrimination. Re-index with `--force` if you switch models.
- **Stronger style** — Lower `--temperature` (e.g. `0.05`) makes the closest match dominate. Higher values (e.g. `0.4`) give a more averaged result.
- **More variety** — Increase `--k` to blend from more references, producing a more generalised look.
- **White balance** — RAWStyle preserves the camera's recorded white balance from the ARW. White balance adjustments made in post are not transferred (they cannot be reliably extracted without the original RAW).

---

## Example workflow

```bash
# Index your library
rawstyle index ~/Pictures/Processed/ --verbose

# Preview what would be applied to a single shot
rawstyle inspect ~/Shoots/2026-04/DSC00001.ARW --verbose

# Process the full shoot
rawstyle process ~/Shoots/2026-04/ ~/Shoots/2026-04/output/ --k 5 --quality 95

# Check the output
open ~/Shoots/2026-04/output/
```
