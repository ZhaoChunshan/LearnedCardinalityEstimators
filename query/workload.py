"""File-backed workload access used by the experiment runner.

The workload files are the source of truth.  Query identifiers are deliberately
local to a test workload: the first outer query is 1, the second is 2, etc.
Subqueries are identified by their SQL text and do not have numeric identifiers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


TEST_WORKLOAD_FILES = {
    "job_light": "job_light_all.sql",
    "stats_ceb": "stats_ceb_all.sql",
    "ergastf1": "ergastf1_queries_all.sql",
    "job_m_ours": "job_m_ours_all.sql",
    "job_large_ours": "job_large_ours_all.sql",
}


@dataclass(frozen=True)
class Subquery:
    sql: str
    postgres_cardinality: float
    true_cardinality: Optional[float] = None


@dataclass(frozen=True)
class WorkloadQuery:
    query_id: int
    sql: str
    true_cardinality: float
    subqueries: Tuple[Subquery, ...]
    postgres_cardinality: Optional[float] = None
    execution_time: Optional[float] = None


def workload_file(
    query_set: str,
    data_role: str,
    workload_root: Path = Path("workloads"),
) -> Path:
    """Return the source workload file for a query set and role."""
    if query_set not in TEST_WORKLOAD_FILES:
        raise ValueError("Unknown query set: %s" % query_set)

    query_set_dir = Path(workload_root) / query_set
    if data_role == "test":
        return query_set_dir / TEST_WORKLOAD_FILES[query_set]
    if data_role in ("train", "valid"):
        return query_set_dir / (data_role + "_workload.sql")
    if data_role in ("train_small", "valid_small"):
        return query_set_dir / "train_small" / (
            data_role.split("_", 1)[0] + "_workload.sql"
        )
    if data_role in ("train_template", "valid_template"):
        return query_set_dir / "train_template" / (
            data_role.split("_", 1)[0] + "_workload.sql"
        )
    raise ValueError("Unknown workload role: %s" % data_role)


def training_estimates_file(source_file: Path) -> Path:
    """Return the PostgreSQL subquery-estimate sidecar for a training file."""
    return source_file.with_name(source_file.stem + "_with_sub_est.sql")


def _nonempty_lines(path: Path) -> Iterator[str]:
    with path.open("r") as workload:
        for line in workload:
            line = line.strip()
            if line:
                yield line


def _is_subquery(line: str) -> bool:
    return line.startswith("/*")


def _grouped_lines(path: Path) -> Iterator[Tuple[str, List[str]]]:
    outer_line = None
    subquery_lines = []
    for line in _nonempty_lines(path):
        if _is_subquery(line):
            if outer_line is None:
                raise ValueError("Subquery before outer query in %s" % path)
            subquery_lines.append(line)
            continue
        if outer_line is not None:
            yield outer_line, subquery_lines
        outer_line = line
        subquery_lines = []
    if outer_line is not None:
        yield outer_line, subquery_lines


def load_test_workload(
    query_set: str,
    workload_root: Path = Path("workloads"),
) -> List[WorkloadQuery]:
    """Load test queries and all stored true/PostgreSQL subquery cards."""
    path = workload_file(query_set, "test", workload_root)
    result = []
    for query_id, (outer_line, subquery_lines) in enumerate(
        _grouped_lines(path), start=1
    ):
        sql, true_card, postgres_card = outer_line.rsplit("||", 2)
        subqueries = []
        for subquery_line in subquery_lines:
            subquery_sql, subquery_true, subquery_postgres = subquery_line.rsplit(
                "||", 2
            )
            subqueries.append(
                Subquery(
                    sql=subquery_sql.strip(),
                    true_cardinality=float(subquery_true),
                    postgres_cardinality=float(subquery_postgres),
                )
            )
        if not subqueries:
            raise ValueError("Test query %d has no subqueries in %s" % (query_id, path))
        result.append(
            WorkloadQuery(
                query_id=query_id,
                sql=sql.strip(),
                true_cardinality=float(true_card),
                postgres_cardinality=float(postgres_card),
                subqueries=tuple(subqueries),
            )
        )
    return result


def load_training_workload(
    query_set: str,
    data_role: str,
    workload_root: Path = Path("workloads"),
) -> List[WorkloadQuery]:
    """Load training/validation queries and PostgreSQL subquery estimates."""
    source_path = workload_file(query_set, data_role, workload_root)
    estimates_path = training_estimates_file(source_path)
    if not estimates_path.exists():
        raise FileNotFoundError(
            "Missing PostgreSQL subquery estimates for %s/%s: %s"
            % (query_set, data_role, estimates_path)
        )

    estimates_by_sql = {}
    for estimate_outer_line, subquery_lines in _grouped_lines(estimates_path):
        estimate_sql, postgres_card = estimate_outer_line.rsplit("||", 1)
        estimates_by_sql[estimate_sql.strip()] = (postgres_card, subquery_lines)

    result = []
    for query_id, source_line in enumerate(_nonempty_lines(source_path), start=1):
        sql, true_card, execution_time = source_line.rsplit("||", 2)
        normalized_sql = sql.strip()
        if normalized_sql not in estimates_by_sql:
            raise ValueError(
                "Query %d from %s is missing in %s"
                % (query_id, source_path, estimates_path)
            )
        postgres_card, subquery_lines = estimates_by_sql[normalized_sql]

        subqueries = []
        for subquery_line in subquery_lines:
            subquery_sql, subquery_postgres = subquery_line.rsplit("||", 1)
            subqueries.append(
                Subquery(
                    sql=subquery_sql.strip(),
                    postgres_cardinality=float(subquery_postgres),
                )
            )
        if not subqueries:
            raise ValueError(
                "Training query %d has no subquery estimates in %s"
                % (query_id, estimates_path)
            )
        result.append(
            WorkloadQuery(
                query_id=query_id,
                sql=normalized_sql,
                true_cardinality=float(true_card),
                postgres_cardinality=float(postgres_card),
                execution_time=float(execution_time),
                subqueries=tuple(subqueries),
            )
        )
    return result


def load_workload(
    query_set: str,
    data_role: str,
    workload_root: Path = Path("workloads"),
) -> List[WorkloadQuery]:
    if data_role == "test":
        return load_test_workload(query_set, workload_root)
    return load_training_workload(query_set, data_role, workload_root)
