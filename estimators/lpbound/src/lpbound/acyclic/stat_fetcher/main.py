from duckdb import DuckDBPyConnection

from lpbound.config.lpbound_config import LpBoundConfig
from lpbound.duckdb_adapter.duckdb_manager import DatabaseManager
from lpbound.config.benchmark_schema import load_benchmark_schema, BenchmarkSchema

from lpbound.acyclic.join_graph.sql_parser import parse_sql_query_to_join_graph
from lpbound.acyclic.join_graph.join_graph import JoinGraph
from lpbound.acyclic.stat_fetcher.join_pool_statistics import compute_join_pool_domain_sizes
from lpbound.acyclic.stat_fetcher.jg_pred_stats import fetch_groupby_domain_sizes, fetch_predicate_norms
from lpbound.acyclic.stat_fetcher.vertex_base_stats import fetch_base_norms
from lpbound.acyclic.stat_fetcher.vertex_local_pred_stats import compute_predicate_ids

from lpbound.utils.types import DomainSizeStats, Stats


_SINGLETON_BASE_COUNTS: dict[tuple[str, str, str], tuple[float, float]] = {}


def fetch_statistics_for_query(
    query: str,
    config: LpBoundConfig,
    connection: DuckDBPyConnection | None = None,
) -> tuple[Stats, DomainSizeStats, JoinGraph]:
    """
    Fetch statistics for a given query.
    """
    schema_data: BenchmarkSchema = load_benchmark_schema(config)
    jg = parse_sql_query_to_join_graph(query, schema_data)
    _add_singleton_anchor_if_needed(jg, schema_data)

    owns_connection = connection is None
    con: DuckDBPyConnection = (
        DatabaseManager(schema_data).create_or_load_db(read_only=True)
        if connection is None
        else connection
    )
    try:
        # Use the standalone fetch_statistics function
        statistics = _fetch_statistics(con, jg, config.p_max)
        _replace_singleton_anchor_statistics(con, jg, statistics, config)

        domain_size_statistics: DomainSizeStats = {}
        if jg.is_groupby:
            domain_size_statistics = fetch_groupby_domain_sizes(con, jg)
    finally:
        if owns_connection:
            con.close()

    # print("----> domain_size_statistics: ", domain_size_statistics)
    for key, value in statistics.items():
        # print(key, value)
        assert len(value) > 0

    return statistics, domain_size_statistics, jg


def _add_singleton_anchor_if_needed(jg: JoinGraph, schema_data: BenchmarkSchema) -> None:
    """Make a single-relation query estimable by the regular Berge LP.

    LpBound's statistics are indexed by join columns.  A singleton query has
    no edge and therefore no join column in its parsed graph, leaving the LP
    objective unconstrained.  We deterministically attach one schema join
    column as an anchor.  Its l1 degree norm bounds the filtered relation
    cardinality, and the resulting LP follows the same Python/C++ path as all
    other subqueries.
    """
    if len(jg.vertices) != 1 or jg.edges:
        return

    vertex = next(iter(jg.vertices.values()))
    candidates = sorted(schema_data["join_variables"].get(vertex.relation_name, []))
    if not candidates:
        raise ValueError(
            f"Single-relation query on {vertex.relation_name} has no configured join variable to use as an anchor"
        )

    anchor = candidates[0]
    vertex.add_join_column(anchor)
    jg.uf.make_set((vertex.alias, anchor))
    jg.build_transitive_closure_and_join_pools()


def _replace_singleton_anchor_statistics(
    con: DuckDBPyConnection,
    jg: JoinGraph,
    statistics: Stats,
    config: LpBoundConfig,
) -> None:
    """Turn the artificial singleton anchor into a synthetic unique key.

    Join-column statistics intentionally omit NULL join keys because NULLs
    cannot participate in an SQL equality join.  That is correct for every
    real join, but it can underestimate a standalone table when its chosen
    artificial anchor is a nullable foreign key.

    For a singleton only, estimate predicate selectivity from the existing
    join-column l1 statistic and scale it from the non-NULL anchor population
    to the whole relation.  We then express that estimated cardinality as the
    degree norms of a synthetic unique key.  An unfiltered singleton therefore
    gets the exact base-table row count; no true workload cardinality is read.
    Multi-relation queries never enter this function and retain the repository's
    original statistics unchanged.
    """
    if len(jg.vertices) != 1 or jg.edges:
        return

    vertex = next(iter(jg.vertices.values()))
    if len(vertex.join_columns) != 1:
        raise ValueError(
            f"Expected exactly one artificial anchor for singleton alias {vertex.alias}, "
            f"got {vertex.join_columns}"
        )

    anchor = next(iter(vertex.join_columns))
    key = (config.benchmark_name, vertex.relation_name, anchor)
    if key not in _SINGLETON_BASE_COUNTS:
        relation = vertex.relation_name.replace('"', '""')
        column = anchor.replace('"', '""')
        row = con.execute(
            f'SELECT COUNT(*), COUNT("{column}") FROM "{relation}"'
        ).fetchone()
        if row is None:
            raise ValueError(f"Could not count singleton base relation {vertex.relation_name}")
        _SINGLETON_BASE_COUNTS[key] = (float(row[0]), float(row[1]))

    total_count, nonnull_anchor_count = _SINGLETON_BASE_COUNTS[key]
    filtered_anchor_count = float(statistics[(vertex.alias, anchor)][1])
    if nonnull_anchor_count > 0:
        estimated_count = filtered_anchor_count * total_count / nonnull_anchor_count
    else:
        estimated_count = total_count
    estimated_count = max(1.0, min(total_count, estimated_count))

    unique_key_norms = [estimated_count, estimated_count]
    unique_key_norms.extend(
        estimated_count ** (1.0 / p) for p in range(2, config.p_max + 1)
    )
    unique_key_norms.append(1.0)
    statistics[(vertex.alias, anchor)] = unique_key_norms


def _fetch_statistics(con: DuckDBPyConnection, join_graph: JoinGraph, max_p: int) -> dict[tuple[str, str], list[float]]:
    """
    Main function to fetch all statistics for a join graph.
    Returns a dictionary mapping (alias, column) to statistics.
    """
    # # Print all vertices for debugging
    # for v in join_graph.vertices.values():
    #     print(v)

    # 1. Compute MCV and range IDs for all vertices
    compute_predicate_ids(con, join_graph)

    # 2. Fetch predicate norms for each join column
    statistics: dict[tuple[str, str], list[float]] = fetch_predicate_norms(con, join_graph, max_p)

    # 3. Compute join pool domain sizes
    compute_join_pool_domain_sizes(con, join_graph, statistics)

    # 4. Compute base norms
    statistics = fetch_base_norms(con, join_graph, statistics, max_p)

    return statistics
