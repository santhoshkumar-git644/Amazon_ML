"""Shared I/O helpers: loading source TSVs and writing the two submission files."""
from __future__ import annotations

import pandas as pd


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    expected = {"entity_id", "business_name", "business_address", "country"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    return df


def write_id_list_file(path: str, s1_to_ids: dict, all_s1_ids, id_col: str, header_id_col: str):
    """Write a {source1_entity_id \t comma,separated,ids} TSV covering every id
    in ``all_s1_ids`` (empty string when a given s1 id has no entries)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_id_col}\n")
        for s1 in all_s1_ids:
            ids = s1_to_ids.get(s1, [])
            f.write(f"{s1}\t{','.join(ids)}\n")


def pairs_df_to_dict(pairs: pd.DataFrame, group_col: str, id_col: str) -> dict:
    """Group a (s1_id, cand_id[, ...]) dataframe into {s1_id: [cand_id, ...]}."""
    grouped = pairs.groupby(group_col)[id_col].apply(list)
    return grouped.to_dict()
