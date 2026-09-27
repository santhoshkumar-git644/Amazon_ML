"""Normalize every raw source file once and cache the result as parquet.

Re-running the pipeline (during development, or to regenerate a submission)
should not pay the ~1-2 minutes of per-row text cleaning again for every
multi-million-row file — this script does it once and caches to
``work/cache/<name>.parquet``, keyed off the raw file's mtime/size so a
changed input is automatically re-normalized.
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_frame  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_DIR = os.path.join(ROOT, "dataset")
CACHE_DIR = os.path.join(ROOT, "work", "cache")

FILES = {
    "train_source1": os.path.join(DATASET_DIR, "train", "train_source1.tsv"),
    "train_source2": os.path.join(DATASET_DIR, "train", "train_source2.tsv"),
    "train_source3": os.path.join(DATASET_DIR, "train", "train_source3.tsv"),
    "test_source1": os.path.join(DATASET_DIR, "test", "test_source1.tsv"),
    "test_source2": os.path.join(DATASET_DIR, "test", "test_source2.tsv"),
    "test_source3": os.path.join(DATASET_DIR, "test", "test_source3.tsv"),
}


def _cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def load_normalized(name, force=False):
    """Return the normalized frame for ``name`` (see ``FILES`` for valid keys)."""
    raw_path = FILES[name]
    cpath = _cache_path(name)
    if not force and os.path.isfile(cpath):
        raw_mtime = os.path.getmtime(raw_path)
        if os.path.getmtime(cpath) >= raw_mtime:
            return pd.read_parquet(cpath)

    t0 = time.time()
    df = pd.read_csv(raw_path, sep="\t", dtype=str, keep_default_na=False)
    norm = normalize_frame(df).reset_index(drop=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    norm.to_parquet(cpath, index=False)
    print(f"[prepare] {name}: {len(norm)} rows normalized in {time.time() - t0:.1f}s", flush=True)
    return norm


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name in FILES:
        load_normalized(name)


if __name__ == "__main__":
    main()
