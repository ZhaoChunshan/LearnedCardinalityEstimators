import argparse
import itertools
import logging
import os
import sys
import time
import json

import numpy as np
import polars as pl

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

# RDC parameters (aligned with DeepDB defaults)
RDC_K = 10
RDC_S = 1.0 / 6.0
RDC_NON_LINEARITY = np.sin
RDC_N_JOBS = -2
MAX_SAMPLING_THRESHOLD_ROWS = 10000

# Our parameters for determine highly correlated column groups
RDC_THRESHOLD = 0.3
MAX_COLUMN_GROUP_PER_SIZE = 200
MAX_STATS_COLUMNS = 6

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-5.5s] %(message)s")
logger = logging.getLogger(__name__)


def load_table_column_types(create_db_sql_path):
    tbls_cols_types = {}

    with open(create_db_sql_path, 'r') as file:
        for line in file:
            if "create table" in line:
                tbl = line.split("create table")[1].split()[0].replace('"', '')
                tbls_cols_types[tbl] = {}
            elif "integer" in line or "bigint" in line or "smallint" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Int64
            elif "character" in line or "varchar(" in line or "char(" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Utf8
            elif "decimal(" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Float64
            elif "double precision" in line or "float" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Float64
            elif "date" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Date
            elif "interval" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Duration
            else:
                pass

    return tbls_cols_types


def load_queryset_tables(csv_file_folder, predicate_json, tbls_cols_types):
    """Load CSV tables into polars DataFrames, keeping only predicate-relevant columns."""

    with open(predicate_json, 'r') as f:
        pred_data = json.load(f)

    tables = pred_data['tables']
    predicates = pred_data['predicates']

    table_columns = {}
    for key in predicates:
        tbl, col = key.split('.', 1)
        if tbl not in table_columns:
            table_columns[tbl] = set()
        table_columns[tbl].add(col)

    tbls_data = {}
    for tbl in tables:
        csv_path = os.path.join(csv_file_folder, f'{tbl}.csv')
        relevant_cols = table_columns.get(tbl, set())
        
        if len(relevant_cols) == 0:
            continue

        # Only specify dtypes for columns we actually need
        dtype_map = tbls_cols_types.get(tbl, {})

        schema_overrides = {col: dt for col, dt in dtype_map.items()
                            if col in relevant_cols}

        df = pl.read_csv(csv_path, separator='|', dtypes=schema_overrides,
                         null_values=['', 'NULL'], columns=list(relevant_cols))

        if relevant_cols:
            df = df.select(list(relevant_cols))

        tbls_data[tbl] = df

    return tbls_data


# ============================================================================
# RDC computation
# ============================================================================

def polars_dtype_to_meta_type(dtype):
    """Map polars dtype to SPFlow MetaType."""
    from spn.structure.StatisticalTypes import MetaType

    if dtype == pl.Utf8 or dtype == pl.Categorical:
        return MetaType.DISCRETE
    elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                   pl.Float32, pl.Float64):
        return MetaType.REAL
    elif dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
        return MetaType.REAL
    else:
        raise ValueError(f"Unsupported polars dtype '{dtype}' for MetaType mapping.")


def prepare_column_data(series, meta_type):
    """Prepare a single polars Series for RDC computation.

    Returns:
        data: numpy array (N, 1)
        domain: domain array for SPFlow Context
    """
    if meta_type.name == 'DISCRETE':
        # Convert to string numpy array
        raw_vals = series.cast(pl.Utf8).to_numpy()
        null_mask = series.is_null().to_numpy().astype(bool)

        non_null_vals = raw_vals[~null_mask]
        unique_vals, inverse = np.unique(non_null_vals, return_inverse=True)

        encoded = np.full(len(raw_vals), np.nan, dtype=np.float64)
        encoded[~null_mask] = inverse.astype(np.float64)

        domain = np.arange(0, len(unique_vals) + 1, 1)
        return encoded.reshape(-1, 1), domain
    else:
        # REAL: convert to float64
        data = series.cast(pl.Float64).to_numpy().astype(np.float64)
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        if np.isnan(min_val) or np.isnan(max_val):
            min_val, max_val = 0.0, 1.0
        domain = np.array([min_val, max_val])
        return data.reshape(-1, 1), domain


def build_meta_types_and_domains(df):
    """Build SPFlow meta_types, domains and a combined numpy array from a polars DataFrame."""
    meta_types = []
    domains = []
    columns_data = []

    for col in df.columns:
        meta_type = polars_dtype_to_meta_type(df[col].dtype)
        col_data, domain = prepare_column_data(df[col], meta_type)
        meta_types.append(meta_type)
        domains.append(domain)
        columns_data.append(col_data)

    encoded_data = np.concatenate(columns_data, axis=1)
    return meta_types, domains, encoded_data


def compute_pairwise_rdc(df, max_rows=MAX_SAMPLING_THRESHOLD_ROWS,
                         k=RDC_K, s=RDC_S, non_linearity=RDC_NON_LINEARITY,
                         n_jobs=RDC_N_JOBS):
    """Compute pairwise RDC for all column pairs in a polars DataFrame.

    Returns:
        list of (col_i, col_j, rdc_value) tuples sorted by RDC descending.
    """
    from spn.algorithms.splitting.RDC import rdc_transformer, rdc_cca
    from spn.structure.Base import Context
    from joblib import Parallel, delayed

    if len(df.columns) < 2:
        return []

    if len(df) > max_rows:
        df = df.sample(n=max_rows, seed=42)

    meta_types, domains, data = build_meta_types_and_domains(df)

    n_features = data.shape[1]
    if n_features < 2:
        return []

    # Build SPFlow Context
    ds_context = Context(meta_types=meta_types)
    ds_context.add_domains(data)

    # Compute RDC transformed features
    logger.debug(f"Computing RDC transform for {n_features} columns, {data.shape[0]} rows...")
    rdc_features = rdc_transformer(
        data, meta_types, ds_context.domains,
        k=k, s=s, non_linearity=non_linearity, return_matrix=False
    )

    # Parallel CCA for all column pairs
    pairwise_comparisons = list(itertools.combinations(range(n_features), 2))
    logger.debug(f"Computing RDC CCA for {len(pairwise_comparisons)} pairs...")

    rdc_vals = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(rdc_cca)((i, j, rdc_features)) for i, j in pairwise_comparisons
    )

    # Build results
    results = []
    columns = list(df.columns)
    for (i, j), rdc_val in zip(pairwise_comparisons, rdc_vals):
        if np.isnan(rdc_val):
            rdc_val = 0.0
        results.append((columns[i], columns[j], float(rdc_val)))

    results.sort(key=lambda x: -x[2])
    return results


def get_connected_column_groups(pairwise_rdc_results,
                                threshold=RDC_THRESHOLD):
    """Find all connected column subsets (size >= 2) in the RDC graph.

    Args:
        pairwise_rdc_results: list of (col_a, col_b, rdc_value) tuples.
        threshold: minimum RDC value for an edge to exist.

    Returns:
        Dict mapping group size (int) to list of ((col_a, col_b, ...), avg_rdc)
        sorted by avg_rdc descending.
    """
    from collections import deque

    if not pairwise_rdc_results:
        return {}

    rdc_dict = {}
    adjacency = {}
    for a, b, v in pairwise_rdc_results:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        rdc_dict[(a, b)] = v
        rdc_dict[(b, a)] = v

    column_set = set()
    for a, b, v in pairwise_rdc_results:
        v = rdc_dict[(a, b)]
        if v > threshold:
            column_set.add(a)
            column_set.add(b)
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    columns = sorted(column_set)
    n = len(columns)

    if n < 2:
        return {}

    assert n <= 15, f"Too many columns with #columns ({n}), enumeration may be very slow. "

    connected_groups = {}

    total_masks = 1 << n

    for mask in range(1, total_masks):
        popcount = bin(mask).count('1')
        if popcount < 2:
            continue

        subset_nodes = [columns[i] for i in range(n) if (mask >> i) & 1]
        subset_set = set(subset_nodes)

        start = subset_nodes[0]
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                if neighbor in subset_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(visited) != popcount:
            continue  # not connected

        total_rdc = 0.0
        pair_count = 0
        for i in range(len(subset_nodes)):
            for j in range(i + 1, len(subset_nodes)):
                a, b = subset_nodes[i], subset_nodes[j]
                rdc_val = rdc_dict.get((a, b), 0.0)
                total_rdc += rdc_val
                pair_count += 1

        avg_rdc = total_rdc / pair_count if pair_count > 0 else 0.0

        if popcount not in connected_groups:
            connected_groups[popcount] = []
        connected_groups[popcount].append((tuple(subset_nodes), avg_rdc))

    # Sort each size group by avg_rdc descending
    for size in connected_groups:
        connected_groups[size].sort(key=lambda x: -x[1])

    return connected_groups


def generate_create_stats_sql(all_table_results, output_path):
    """Generate CREATE STATISTICS SQL for connected column groups.

    Args:
        all_table_results: list of (tbl_name, connected_groups) tuples
            where connected_groups is a dict mapping group size (int) to
            list of ((col_a, col_b, ...), avg_rdc) sorted by avg_rdc descending.
    """
    lines = []

    for tbl_name, connected_groups in all_table_results:
        if not connected_groups:
            continue

        lines.append(f"-- Table: {tbl_name}")

        # Flatten all size groups into a single sequence for SQL generation
        idx = 0
        for size in sorted(connected_groups.keys()):
            for cols_tuple, _ in connected_groups[size]:
                idx += 1
                cols = ", ".join(cols_tuple)
                col_names = "_".join(cols_tuple)
                stat_name = f"stats_{tbl_name}_{col_names}"
                # If name exceeds PostgreSQL 63-byte limit, use an anonymous name
                if len(stat_name.encode('utf-8')) > 63:
                    stat_name = f"stats_{tbl_name}_group_{idx}"
                lines.append(
                    f"CREATE STATISTICS {stat_name} ON {cols} FROM {tbl_name};")

        lines.append(f"ANALYZE {tbl_name};")
        lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Wrote CREATE STATISTICS SQL to '{output_path}' "
                f"({len(lines)} lines)")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    db_dict = {
        'job_light': 'imdb',
        'stats_ceb': 'stats',
        'ergastf1': 'ergastf1',
        'job_m_ours': 'imdb',
        'job_large_ours': 'imdb',
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--queryset', required=True, choices=db_dict)
    args = parser.parse_args()
    queryset = args.queryset
    db = db_dict[queryset]

    csv_file_folder = f'datasets/{db}'
    predicate_json = f'workloads/{queryset}/predicates.json'
    create_db_sql_path = f'datasets/{db}/postgres_create_{db}.sql'
    result_create_stats_sql_path = f'scripts/sql/create_{queryset}_extended_statistics.sql'

    logger.info(f"Database: {db}, QuerySet: {queryset}")
    logger.info(f"RDC parameters: k={RDC_K}, s={RDC_S}, non_linearity={RDC_NON_LINEARITY.__name__}, "
                f"max_rows={MAX_SAMPLING_THRESHOLD_ROWS}")

    # Load schema info and table data
    logger.info("Loading table column types...")
    tbls_cols_types = load_table_column_types(create_db_sql_path)
    logger.info(f"Table column types loaded.")

    logger.info("Loading queryset tables...")
    tbls_data = load_queryset_tables(csv_file_folder, predicate_json, tbls_cols_types)
    logger.info(f"Loaded {len(tbls_data)} tables with data.")

    overall_start = time.perf_counter()

    all_table_results = []  # (tbl_name, connected_groups)

    for tbl_name, df in tbls_data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Table: {tbl_name}  |  {len(df.columns)} columns  |  {len(df)} rows")
        logger.info(f"{'='*60}")

        if len(df.columns) < 2:
            logger.info("  (skipped: fewer than 2 columns)")
            continue

        tbl_start = time.perf_counter()
        results = compute_pairwise_rdc(df)
        connected_groups = get_connected_column_groups(results)

        # Truncate each size group to top MAX_COLUMN_GROUP_PER_SIZE,
        # and drop groups larger than MAX_STATS_COLUMNS
        truncated_groups = {}
        total_groups = 0
        total_truncated = 0
        total_skipped = 0
        for size, groups in connected_groups.items():
            if size > MAX_STATS_COLUMNS:
                total_skipped += len(groups)
                continue
            truncated_groups[size] = groups[:MAX_COLUMN_GROUP_PER_SIZE]
            total_groups += len(groups)
            total_truncated += len(truncated_groups[size])

        all_table_results.append((tbl_name, truncated_groups))
        tbl_elapsed = time.perf_counter() - tbl_start

        # Print all pairwise RDC results
        print(f"\n{'─'*70}")
        print(f"  {'Column A':<35s} {'Column B':<35s} {'RDC':>8s}")
        print(f"  {'─'*35} {'─'*35} {'─'*8}")
        for col_a, col_b, rdc_val in results:
            print(f"  {col_a:<35s} {col_b:<35s} {rdc_val:>8.4f}")
        print(f"  {'─'*70}")

        print(f"\n  >> Connected column groups (rdc > {RDC_THRESHOLD}): "
              f"{total_groups} found, {total_truncated} kept "
              f"(max {MAX_COLUMN_GROUP_PER_SIZE} per size, size <= {MAX_STATS_COLUMNS})")
        if total_skipped > 0:
            print(f"     (skipped {total_skipped} groups with size > {MAX_STATS_COLUMNS})")
        for sz in sorted(truncated_groups.keys()):
            n_full = len(connected_groups[sz])
            n_kept = len(truncated_groups[sz])
            truncated_note = f" (truncated from {n_full})" if n_full > n_kept else ""
            print(f"     size={sz}: {n_kept} groups{truncated_note}")

        print(f"\n  Table '{tbl_name}': {len(results)} pairwise RDCs, "
              f"{total_truncated} connected groups "
              f"in {tbl_elapsed:.2f}s")

    overall_elapsed = time.perf_counter() - overall_start
    logger.info(f"\nAll tables processed in {overall_elapsed:.2f}s")

    # Write CREATE STATISTICS SQL
    logger.info(f"Generating CREATE STATISTICS SQL...")
    generate_create_stats_sql(all_table_results, result_create_stats_sql_path)
