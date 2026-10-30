"""Data loading and aggregation helpers used by the plotting scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from common.sql_utils import get_tables_from_comment


ESTIMATION_FILE = "estimation.csv"
EXECUTION_FILE = "end2end.csv"
STATISTICS_FILE = "statistics.json"

QUERY_SIZE_GROUPS = {
    "job_light": [[1], [2], [3], [4, 5]],
    "stats_ceb": [[1], [2], [3], [4], [5, 6, 7]],
    "ergastf1": [[1], [2], [3], [4], [5, 6, 7]],
    "job_m_ours": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10, 11]],
    "job_large_ours": [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18, 19],
    ],
}


def _experiment_file(experiment_dir, filename: str) -> Path:
    return Path(experiment_dir) / filename


def _read_column(csv_path: Path, column: str) -> np.ndarray:
    dataframe = pd.read_csv(csv_path)
    return dataframe[column].to_numpy(dtype=np.float64)


def _query_table_counts(dataframe: pd.DataFrame) -> pd.Series:
    return dataframe["sql"].map(
        lambda sql: len(get_tables_from_comment(sql))
    )


def _read_dynamic_pair(
    experiment_dir,
    filename_pairs: Sequence[tuple[str, str]],
    column: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read the first complete stale/updated file pair in priority order."""
    directory = Path(experiment_dir)
    for stale_filename, updated_filename in filename_pairs:
        stale_path = directory / stale_filename
        updated_path = directory / updated_filename
        if stale_path.exists() and updated_path.exists():
            return (
                _read_column(stale_path, column),
                _read_column(updated_path, column),
            )
    return None, None


def read_estimation_csv(experiment_dir) -> pd.DataFrame:
    return pd.read_csv(_experiment_file(experiment_dir, ESTIMATION_FILE))


def read_execution_csv(experiment_dir) -> pd.DataFrame:
    return pd.read_csv(_experiment_file(experiment_dir, EXECUTION_FILE))


def read_statistics_json(experiment_dir) -> dict:
    with _experiment_file(experiment_dir, STATISTICS_FILE).open(
        "r", encoding="utf-8"
    ) as statistics_file:
        return json.load(statistics_file)


def get_q_error_and_query_size(
    experiment_dir,
) -> tuple[np.ndarray, pd.Series]:
    dataframe = read_estimation_csv(experiment_dir)
    estimated_cards = dataframe["estimated_card"].to_numpy(
        dtype=np.float64, copy=True
    )
    true_cards = dataframe["true_card"].to_numpy(dtype=np.float64, copy=True)

    # Cardinalities at or below one are normalized before computing the ratio
    # used by the existing figures.
    estimated_cards[estimated_cards <= 1] = 1.0
    true_cards[true_cards <= 1] = 1.0
    q_errors = estimated_cards / true_cards

    query_sizes = _query_table_counts(dataframe).astype(np.float64)
    return q_errors, query_sizes


def get_real_q_error(experiment_dir) -> np.ndarray:
    return read_estimation_csv(experiment_dir)["q_error"].to_numpy(
        dtype=np.float64
    )


def get_real_q_error_wo_subquery_from_df(
    df: pd.DataFrame,
) -> pd.Series:
    """Return one q-error per query, selecting its largest-table subquery."""
    dataframe = df
    table_counts = _query_table_counts(dataframe)
    largest_subquery_indices = table_counts.groupby(dataframe["index"]).idxmax()
    return dataframe.loc[largest_subquery_indices, "q_error"]


def get_real_q_error_wo_subquery(experiment_dir) -> pd.Series:
    return get_real_q_error_wo_subquery_from_df(
        read_estimation_csv(experiment_dir)
    )


def get_p_error(experiment_dir) -> np.ndarray:
    """Read p-error, preferring end-to-end results over the legacy file."""
    try:
        return read_execution_csv(experiment_dir)["p_error"].to_numpy(
            dtype=np.float64
        )
    except (FileNotFoundError, KeyError):
        return _read_column(
            _experiment_file(experiment_dir, "p_error.csv"), "p_error"
        )


def get_dynamic_p_errors(
    experiment_dir,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    return _read_dynamic_pair(
        experiment_dir,
        (
            ("end2end_stale.csv", "end2end_updated.csv"),
            ("p_error_stale.csv", "p_error_updated.csv"),
            ("p_error_stale.csv", "end2end_updated.csv"),
        ),
        "p_error",
    )


def get_dynamic_execution_times(
    experiment_dir,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    return _read_dynamic_pair(
        experiment_dir,
        (("end2end_stale.csv", "end2end_updated.csv"),),
        "execution_time",
    )


def get_execution_time(experiment_dir) -> np.ndarray:
    return read_execution_csv(experiment_dir)["execution_time"].to_numpy(
        dtype=np.float64
    )


def get_cardinality_estimation_time(experiment_dir) -> np.ndarray:
    return read_execution_csv(experiment_dir)["planning_time"].to_numpy(
        dtype=np.float64
    )
    
def get_dace_inference_time(queryset) -> np.ndarray:
    path = f"experiments/competitors_results/dace/{queryset}/{queryset}_execution_results.csv"
    return pd.read_csv(path)["infer_time"].to_numpy(dtype=np.float64)

def get_dace_execution_time(queryset) -> np.ndarray:
    path = f"experiments/competitors_results/dace/{queryset}/{queryset}_execution_results.csv"
    return pd.read_csv(path)["execution_time"].to_numpy(dtype=np.float64)

def get_genjoin_inference_time(queryset) -> np.ndarray:
    path = f"experiments/competitors_results/genjoin/{queryset}/full_execution_time_{queryset}_test.csv"
    return pd.read_csv(path)["inference_time"].to_numpy(dtype=np.float64) / 1000.0

def get_genjoin_execution_time(queryset) -> np.ndarray:
    path = f"experiments/competitors_results/genjoin/{queryset}/full_execution_time_{queryset}_test.csv"
    return pd.read_csv(path)["execution_time"].to_numpy(dtype=np.float64) / 1000.0

def get_end_to_end_time(experiment_dir) -> np.ndarray:
    return read_execution_csv(experiment_dir)["e2e_time"].to_numpy(
        dtype=np.float64
    )


def get_training_time(experiment_dir):
    statistics = read_statistics_json(experiment_dir)
    return statistics["training_time"] or statistics["pretraining_time"]


def get_updating_time(experiment_dir):
    return read_statistics_json(experiment_dir)["updating_time"]


def get_model_size(experiment_dir):
    return read_statistics_json(experiment_dir)["model_size"]


def get_query_size_group(queryset: str) -> list[list[int]] | None:
    groups = QUERY_SIZE_GROUPS.get(queryset)
    return [group.copy() for group in groups] if groups is not None else None


def group_q_error_by_query_size(
    q_error: Iterable[float],
    query_size: Iterable[float],
    group: Iterable[Iterable[int]],
) -> list[list[float]]:
    q_errors = np.asarray(q_error)
    query_sizes = np.asarray(query_size)
    return [
        q_errors[np.isin(query_sizes, list(sizes))].tolist()
        for sizes in group
    ]


def get_q_error_by_group_by_method(
    methods: Iterable[str],
    experiment_dir: Mapping[str, str | Path],
    query_size_group: Iterable[Iterable[int]],
) -> dict[str, list[list[float]]]:
    grouped_errors = {}
    for method in methods:
        if method not in experiment_dir:
            continue
        q_errors, query_sizes = get_q_error_and_query_size(
            experiment_dir[method]
        )
        grouped_errors[method] = group_q_error_by_query_size(
            q_errors, query_sizes, query_size_group
        )
    return grouped_errors


def format_query_size_group(group: Sequence[int]) -> str:
    if len(group) == 1:
        return str(group[0])
    return rf"$\geq${min(group)}"


# Backward-compatible names for external scripts that used the original API.
get_p_error_dyn = get_dynamic_p_errors
get_execution_time_dyn = get_dynamic_execution_times
query_size_group_name_mapping = format_query_size_group
