"""GPU I/O helpers: loading source TSVs via cuDF's GPU-accelerated CSV
parser, and writing the two submission files.

cudf.read_csv is a genuine, well-documented GPU-accelerated parser (parsing
runs as a CUDA kernel, not on the CPU) -- higher confidence than the
list-column operations elsewhere in this branch, since read_csv's `sep`/
`dtype`/`na_filter` kwargs mirror pandas closely.

Output writing stays plain Python file I/O: writing a few million short text
lines is not a GPU-shaped workload (it's dominated by disk/OS calls, not
compute), and cuDF's own to_csv doesn't produce this exact
tab-then-comma-joined-list format directly, so results are pulled back to
host memory (.to_pandas() / values_host) for this one small step.
"""
from __future__ import annotations

import cudf


def load_source(path: str) -> cudf.DataFrame:
    df = cudf.read_csv(path, sep="\t", dtype="str", na_filter=False)
    expected = {"entity_id", "business_name", "business_address", "country"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    return df


def load_ground_truth(path: str) -> cudf.DataFrame:
    return cudf.read_csv(path, sep="\t", dtype="str", na_filter=False)


def write_id_list_file(path: str, s1_to_ids: dict, all_s1_ids, id_col: str, header_id_col: str):
    """Write a {source1_entity_id \t comma,separated,ids} TSV covering every id
    in ``all_s1_ids`` (empty string when a given s1 id has no entries)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_id_col}\n")
        for s1 in all_s1_ids:
            ids = s1_to_ids.get(s1, [])
            f.write(f"{s1}\t{','.join(ids)}\n")
