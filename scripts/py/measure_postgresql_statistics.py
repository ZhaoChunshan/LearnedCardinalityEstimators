#!/usr/bin/env python3
"""Build PostgreSQL statistics and record their size and build time.

The reported size is the logical size of the workload tables' rows in
``pg_statistic``.  In extended mode it also includes the corresponding rows in
``pg_statistic_ext`` and ``pg_statistic_ext_data``.  Shared catalog indexes are
not included because their space cannot be attributed to individual tables.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.constants import DB_CONNECTION_ARGS  # noqa: E402


WORKLOADS = (
    "job_light",
    "stats_ceb",
    "ergastf1",
    "job_m_ours",
    "job_large_ours",
)

WORKLOAD_DATABASES = {
    "job_light": "imdb",
    "stats_ceb": "stats",
    "ergastf1": "ergastf1",
    "job_m_ours": "imdb",
    "job_large_ours": "imdb",
}

EXPERIMENT_TYPES = {
    "job_light": "overall",
    "stats_ceb": "overall",
    "ergastf1": "overall",
    "job_m_ours": "scalability",
    "job_large_ours": "scalability",
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected a boolean value: true/false, yes/no, or 1/0"
    )


def load_workload_tables(workload: str) -> List[str]:
    predicates_path = REPO_ROOT / "workloads" / workload / "predicates.json"
    with predicates_path.open("r", encoding="utf-8") as file:
        predicates = json.load(file)

    tables = predicates.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError(f"{predicates_path} must contain a non-empty 'tables' list")
    if not all(isinstance(table, str) and table for table in tables):
        raise ValueError(f"{predicates_path} contains an invalid table name")
    if len(tables) != len(set(tables)):
        raise ValueError(f"{predicates_path} contains duplicate table names")
    return tables


def load_extended_statistics_statements(workload: str) -> List[str]:
    sql_path = (
        REPO_ROOT
        / "scripts"
        / "sql"
        / f"create_{workload}_extended_statistics.sql"
    )
    sql_text = sql_path.read_text(encoding="utf-8")
    sql_without_comments = re.sub(r"--[^\n]*", "", sql_text)
    statements = [
        statement.strip()
        for statement in sql_without_comments.split(";")
        if statement.strip()
    ]

    create_statements = []
    for statement in statements:
        statement_type = statement.split(None, 1)[0].upper()
        if statement_type == "CREATE" and re.match(
            r"^CREATE\s+STATISTICS\b", statement, flags=re.IGNORECASE
        ):
            create_statements.append(statement)
        elif statement_type != "ANALYZE":
            raise ValueError(f"unsupported statement in {sql_path}: {statement[:80]}")

    if not create_statements:
        raise ValueError(f"no CREATE STATISTICS statements found in {sql_path}")
    return create_statements


def result_path(workload: str, extended: bool) -> Path:
    estimator = "postgresql_ext" if extended else "postgresql"
    return (
        REPO_ROOT
        / "experiments"
        / EXPERIMENT_TYPES[workload]
        / workload
        / estimator
        / "default"
        / "statistics.json"
    )


def resolve_table_oids(cursor, tables: Sequence[str]) -> List[int]:
    table_oids = []
    missing_tables = []
    for table in tables:
        cursor.execute("SELECT to_regclass(%s)::oid", (table,))
        oid = cursor.fetchone()[0]
        if oid is None:
            missing_tables.append(table)
        else:
            table_oids.append(oid)

    if missing_tables:
        raise RuntimeError(
            "tables not found in the database search_path: "
            + ", ".join(missing_tables)
        )
    return table_oids


def drop_extended_statistics(connection, table_oids: Sequence[int], sql_module) -> int:
    """Drop every extended-statistics object attached to the workload tables."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT namespace.nspname, statistics.stxname
                FROM pg_catalog.pg_statistic_ext AS statistics
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = statistics.stxnamespace
                WHERE statistics.stxrelid = ANY(%s::oid[])
                ORDER BY namespace.nspname, statistics.stxname
                """,
                (list(table_oids),),
            )
            statistics_objects: List[Tuple[str, str]] = cursor.fetchall()
            for schema_name, statistics_name in statistics_objects:
                cursor.execute(
                    sql_module.SQL("DROP STATISTICS {}.{}").format(
                        sql_module.Identifier(schema_name),
                        sql_module.Identifier(statistics_name),
                    )
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return len(statistics_objects)


def build_statistics(
    connection,
    tables: Sequence[str],
    create_statements: Sequence[str],
    sql_module,
) -> float:
    """Create requested definitions and ANALYZE every workload table once."""
    started_at = time.perf_counter()
    try:
        with connection.cursor() as cursor:
            for statement in create_statements:
                cursor.execute(statement)
            for table in tables:
                cursor.execute(
                    sql_module.SQL("ANALYZE {}").format(sql_module.Identifier(table))
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return time.perf_counter() - started_at


def statistics_size_bytes(connection, table_oids: Sequence[int]) -> int:
    """Return attributable logical catalog space for the selected tables."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COALESCE(SUM(pg_column_size(statistics)), 0)::bigint
            FROM pg_catalog.pg_statistic AS statistics
            WHERE statistics.starelid = ANY(%s::oid[])
            """,
            (list(table_oids),),
        )
        default_statistics_bytes = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COALESCE(SUM(pg_column_size(statistics)), 0)::bigint
            FROM pg_catalog.pg_statistic_ext AS statistics
            WHERE statistics.stxrelid = ANY(%s::oid[])
            """,
            (list(table_oids),),
        )
        extended_definitions_bytes = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COALESCE(SUM(pg_column_size(statistics_data)), 0)::bigint
            FROM pg_catalog.pg_statistic_ext_data AS statistics_data
            JOIN pg_catalog.pg_statistic_ext AS statistics
              ON statistics.oid = statistics_data.stxoid
            WHERE statistics.stxrelid = ANY(%s::oid[])
            """,
            (list(table_oids),),
        )
        extended_data_bytes = cursor.fetchone()[0]

    return (
        default_statistics_bytes
        + extended_definitions_bytes
        + extended_data_bytes
    )


def write_result(path: Path, statistics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=4)
        file.write("\n")
    temporary_path.replace(path)


def run(workload: str, extended: bool) -> Tuple[Dict[str, float], Path, int]:
    try:
        import psycopg2
        from psycopg2 import sql
    except ImportError as error:
        raise RuntimeError(
            "psycopg2 is required; run this script in the project's PostgreSQL "
            "environment"
        ) from error

    tables = load_workload_tables(workload)
    create_statements = (
        load_extended_statistics_statements(workload) if extended else []
    )
    connection = psycopg2.connect(
        database=WORKLOAD_DATABASES[workload],
        user=DB_CONNECTION_ARGS["user"],
        password=DB_CONNECTION_ARGS["password"],
        host=DB_CONNECTION_ARGS["host"],
        port=DB_CONNECTION_ARGS["port"],
    )

    try:
        with connection.cursor() as cursor:
            table_oids = resolve_table_oids(cursor, tables)
        connection.rollback()

        dropped_count = drop_extended_statistics(connection, table_oids, sql)
        training_time = build_statistics(
            connection, tables, create_statements, sql
        )
        model_size = statistics_size_bytes(connection, table_oids) / (1024 * 1024)

        statistics = {
            "model_size": model_size,
            "training_time": training_time,
        }
        output_path = result_path(workload, extended)
        write_result(output_path, statistics)
        return statistics, output_path, dropped_count
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build PostgreSQL statistics for a workload and record their size "
            "and build time. Existing extended statistics on the workload's "
            "tables are dropped before measurement."
        )
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=WORKLOADS,
        help="workload whose predicates.json supplies the table list",
    )
    parser.add_argument(
        "--extended",
        required=True,
        type=parse_bool,
        metavar="{true,false}",
        help="whether to build the workload's extended statistics",
    )
    args = parser.parse_args()

    try:
        statistics, output_path, dropped_count = run(
            args.workload, args.extended
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    mode = "extended" if args.extended else "default"
    print(f"Workload: {args.workload}")
    print(f"Mode: {mode}")
    print(f"Dropped extended statistics: {dropped_count}")
    print(f"Statistics size (MB): {statistics['model_size']:.6f}")
    print(f"Build time (sec): {statistics['training_time']:.6f}")
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
