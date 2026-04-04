from tqdm import tqdm


def bar(iterable, desc: str = "", total: int | None = None, verbose: bool = True):
    """Wrap an iterable with a tqdm progress bar, suppressed when not verbose."""
    if not verbose:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="file")
