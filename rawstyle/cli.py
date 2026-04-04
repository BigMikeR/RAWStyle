"""
rawstyle — apply your photographic style to Sony ARW files.

Commands:
  index    Build / update the reference library index
  process  Process ARW files using the indexed style library
  inspect  Show which references match a given ARW file
  info     Display library statistics
  reindex  Re-index all or prune missing files
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

DEFAULT_DB = Path.home() / ".rawstyle" / "library.db"


def _require_db(db_path: Path) -> None:
    if not db_path.exists():
        click.echo(
            f"Library not found at {db_path}.\n"
            "Run `rawstyle index <LIBRARY_DIR>` first.",
            err=True,
        )
        sys.exit(1)


@click.group()
def main():
    """rawstyle: content-aware RAW photo style transfer."""


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--db", default=DEFAULT_DB, type=click.Path(path_type=Path), show_default=True, help="Path to SQLite library DB")
@click.option("--force", is_flag=True, help="Re-index all files, ignoring mtime cache")
@click.option("--batch-size", default=64, show_default=True, help="CLIP embedding batch size")
@click.option("--model", default="ViT-B-32", show_default=True, help="CLIP model variant")
@click.option("--verbose", is_flag=True)
def index(library_dir, db, force, batch_size, model, verbose):
    """Build or update the reference JPEG library index."""
    from rawstyle.db.schema import open_db
    from rawstyle.db.writer import needs_reindex, upsert_image
    from rawstyle.core.clip_embedder import embed_batch
    from rawstyle.core.style_extractor import extract as extract_style
    from rawstyle.utils.image_utils import find_jpeg_files, open_jpeg
    from rawstyle.utils.progress import bar

    db.parent.mkdir(parents=True, exist_ok=True)
    conn = open_db(db)

    jpegs = find_jpeg_files(library_dir)
    if not jpegs:
        click.echo("No JPEG files found.", err=True)
        sys.exit(1)

    click.echo(f"Found {len(jpegs)} JPEG files in {library_dir}")

    # Filter to files that need (re-)indexing
    to_index = []
    for path in jpegs:
        mtime = path.stat().st_mtime
        if force or needs_reindex(conn, str(path), mtime):
            to_index.append((path, mtime))

    if not to_index:
        click.echo("All files already indexed and up to date.")
        return

    click.echo(f"Indexing {len(to_index)} files...")

    # Extract style features and compute embeddings in batches
    paths_batch = [p for p, _ in to_index]
    mtimes_batch = [m for _, m in to_index]

    images = []
    styles = []
    failed = []
    for path in bar(paths_batch, desc="Loading JPEGs", verbose=verbose):
        try:
            img = open_jpeg(path)
            images.append(img)
            styles.append(extract_style(img))
        except Exception as exc:
            click.echo(f"  Skipping {path.name}: {exc}", err=True)
            failed.append(path)

    # Remove failed paths from the batch
    valid = [(p, m, s) for (p, m), s, _ in zip(to_index, styles, range(len(styles)))
             if p not in failed]
    valid_paths = [p for p, _, _ in valid]
    valid_mtimes = [m for _, m, _ in valid]
    valid_styles = [s for _, _, s in valid]
    valid_images = [img for path, img in zip(paths_batch, images) if path not in failed]

    if not valid_images:
        click.echo("No files could be loaded.", err=True)
        sys.exit(1)

    click.echo(f"Computing CLIP embeddings (model={model})...")
    embeddings = embed_batch(valid_images, batch_size=batch_size, model_name=model)

    click.echo("Writing to database...")
    for path, mtime, style, embedding in bar(
        zip(valid_paths, valid_mtimes, valid_styles, embeddings),
        desc="Saving",
        total=len(valid_paths),
        verbose=verbose,
    ):
        upsert_image(conn, str(path), mtime, embedding, style)

    conn.close()
    click.echo(f"Done. Indexed {len(valid_paths)} images into {db}")


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--db", default=DEFAULT_DB, type=click.Path(path_type=Path), show_default=True)
@click.option("--k", default=5, show_default=True, help="Number of reference images to blend")
@click.option("--temperature", default=0.15, show_default=True, help="Softmax blend temperature")
@click.option("--quality", default=95, show_default=True, help="Output JPEG quality (1-100)")
@click.option("--pattern", default="*.ARW", show_default=True, help="Input file glob pattern")
@click.option("--dry-run", is_flag=True, help="Show matches without writing output")
@click.option("--verbose", is_flag=True)
def process(input_dir, output_dir, db, k, temperature, quality, pattern, dry_run, verbose):
    """Process ARW files and apply matched style from the reference library."""
    from PIL import Image

    from rawstyle.core.clip_embedder import embed_image
    from rawstyle.core.exif_handler import copy_exif, exiftool_available
    from rawstyle.core.raw_developer import develop_linear, develop_thumbnail
    from rawstyle.core.style_applier import apply as apply_style
    from rawstyle.core.style_blender import blend
    from rawstyle.db.retriever import find_similar
    from rawstyle.db.schema import open_db
    from rawstyle.utils.image_utils import find_arw_files
    from rawstyle.utils.progress import bar

    _require_db(db)
    conn = open_db(db)

    arw_files = find_arw_files(input_dir, pattern)
    if not arw_files:
        click.echo(f"No ARW files found in {input_dir} matching '{pattern}'.", err=True)
        sys.exit(1)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not exiftool_available():
            click.echo(
                "Warning: exiftool not found. Install with `brew install exiftool` "
                "for full EXIF preservation. Falling back to piexif.",
                err=True,
            )

    click.echo(f"Processing {len(arw_files)} ARW files...")

    for arw_path in bar(arw_files, desc="Processing", verbose=not verbose):
        try:
            _process_one(
                arw_path=arw_path,
                output_dir=output_dir,
                conn=conn,
                k=k,
                temperature=temperature,
                quality=quality,
                dry_run=dry_run,
                verbose=verbose,
            )
        except Exception as exc:
            click.echo(f"  Error processing {arw_path.name}: {exc}", err=True)

    conn.close()
    if not dry_run:
        click.echo(f"Done. Output written to {output_dir}")


def _process_one(arw_path, output_dir, conn, k, temperature, quality, dry_run, verbose):
    from rawstyle.core.clip_embedder import embed_image
    from rawstyle.core.exif_handler import copy_exif
    from rawstyle.core.raw_developer import develop_linear, develop_thumbnail
    from rawstyle.core.style_applier import apply as apply_style
    from rawstyle.core.style_blender import blend
    from rawstyle.db.retriever import find_similar
    from PIL import Image

    # 1. Develop thumbnail for CLIP embedding
    thumb = develop_thumbnail(arw_path)
    embedding = embed_image(thumb)

    # 2. Find K most similar reference images
    matches = find_similar(conn, embedding, k=k)
    if not matches:
        click.echo(f"  {arw_path.name}: no matches in library, skipping.", err=True)
        return

    if verbose:
        click.echo(f"\n  {arw_path.name} — top matches:")
        for m in matches:
            sim = 1.0 - m.distance
            click.echo(f"    {Path(m.path).name}  similarity={sim:.3f}")

    if dry_run:
        return

    # 3. Blend styles
    style_matches = [(m.style, m.distance) for m in matches]
    blended = blend(style_matches, temperature=temperature)

    # 4. Develop full-resolution linear image
    linear = develop_linear(arw_path)

    # 5. Apply style
    result_uint8 = apply_style(linear, blended)

    # 6. Save JPEG
    out_path = output_dir / (arw_path.stem + ".jpg")
    Image.fromarray(result_uint8).save(str(out_path), format="JPEG", quality=quality, subsampling=0)

    # 7. Copy EXIF from ARW
    copy_exif(arw_path, out_path)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@main.command()
@click.argument("arw_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--db", default=DEFAULT_DB, type=click.Path(path_type=Path), show_default=True)
@click.option("--k", default=5, show_default=True)
@click.option("--verbose", is_flag=True, help="Show style feature values")
def inspect(arw_file, db, k, verbose):
    """Show which reference images would be matched for a given ARW file."""
    from rawstyle.core.clip_embedder import embed_image
    from rawstyle.core.raw_developer import develop_thumbnail
    from rawstyle.core.style_blender import blend
    from rawstyle.db.retriever import find_similar
    from rawstyle.db.schema import open_db

    _require_db(db)
    conn = open_db(db)

    click.echo(f"Developing thumbnail for {arw_file.name}...")
    thumb = develop_thumbnail(arw_file)
    embedding = embed_image(thumb)

    click.echo(f"Finding top {k} matches...\n")
    matches = find_similar(conn, embedding, k=k)

    if not matches:
        click.echo("No matches found — is the library indexed?")
        conn.close()
        return

    for i, m in enumerate(matches, 1):
        sim = 1.0 - m.distance
        click.echo(f"  {i}. {Path(m.path).name}  (similarity={sim:.3f})")
        if verbose:
            s = m.style
            click.echo(f"       shadow_lift={s.shadow_lift:.3f}  highlight_comp={s.highlight_comp:.3f}")
            click.echo(f"       sat_r={s.sat_r:.3f}  sat_g={s.sat_g:.3f}  sat_b={s.sat_b:.3f}")
            click.echo(f"       vibrancy={s.vibrancy:.3f}")

    if verbose:
        # Show blended style
        style_matches = [(m.style, m.distance) for m in matches]
        blended = blend(style_matches)
        click.echo("\n  Blended style (what will be applied):")
        click.echo(f"    shadow_lift={blended.shadow_lift:.3f}  highlight_comp={blended.highlight_comp:.3f}")
        click.echo(f"    sat_r={blended.sat_r:.3f}  sat_g={blended.sat_g:.3f}  sat_b={blended.sat_b:.3f}")
        click.echo(f"    vibrancy={blended.vibrancy:.3f}")
        click.echo(f"    lum_curve: min={blended.lum_curve.min():.3f}  max={blended.lum_curve.max():.3f}  "
                   f"midpoint(128)={blended.lum_curve[128]:.3f}")

    conn.close()


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@main.command()
@click.option("--db", default=DEFAULT_DB, type=click.Path(path_type=Path), show_default=True)
def info(db):
    """Display library statistics."""
    if not db.exists():
        click.echo(f"No library found at {db}")
        return

    from rawstyle.db.retriever import count_images
    from rawstyle.db.schema import open_db

    conn = open_db(db)
    n = count_images(conn)
    conn.close()

    import os
    size_mb = db.stat().st_size / 1024 / 1024
    click.echo(f"Library:  {db}")
    click.echo(f"Images:   {n}")
    click.echo(f"DB size:  {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------

@main.command()
@click.option("--db", default=DEFAULT_DB, type=click.Path(path_type=Path), show_default=True)
@click.option("--prune", is_flag=True, help="Remove entries for files that no longer exist")
def reindex(db, prune):
    """Re-index all files or prune missing entries."""
    _require_db(db)

    from rawstyle.db.schema import open_db
    from rawstyle.db.writer import prune_missing

    conn = open_db(db)

    if prune:
        removed = prune_missing(conn)
        click.echo(f"Pruned {removed} missing entries.")
    else:
        click.echo("Use `rawstyle index <LIBRARY_DIR> --force` to re-index all files.")

    conn.close()
