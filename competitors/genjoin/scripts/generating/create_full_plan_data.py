# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: workload support, compact JSONL plan data, resumable generation,
# logging, and execution-time collection.

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import json
import csv
import argparse
import logging

import torch

from src.database.query_loader import QueryLoader
from src.encoding.database_graph import DatabaseGraph
from src.encoding.join_graph import JoinGraph, SimpleJoinGraph, JoinType, RedundantJoinException, JoinNotPossibleException
from src.database.selectivity_extractor import PostgresSelectivityExtractor,SelectivityExtractorCannotExtractException
from src.encoding.encoder import QueryEncoder, PlanEncoder
from src.database.postgres_executor import PostgresExecutor
from src.database.utils import PlanHinter, PlanParser, gather_join_order
from src.utils.query_helper import parse_sql_query
from src.utils.path_helper import DB_CONFIG, WORKLOADS
from src.utils.logging_config import setup_logging

from tqdm.auto import tqdm


DEMO_BASE_PATH = 'data/demonstrations/'
PROCESSED_BASE_PATH = 'data/processed/'
DEMO_TYPE = 'cvae'
N_RUNS = 5
N_PLANS_PER_QUERY = 200
JSONL_FILENAME = 'train_data.jsonl'
CHECKPOINT_FILENAME = 'checkpoint.txt'

np.random.seed(1234)
torch.manual_seed(1234)

NEW_WORKLOADS = ('job_light', 'stats_ceb', 'ergastf1')


def gather_exec_time(query, executor, n_runs=3, return_all=False, return_planning_time=False):
    """Execute EXPLAIN query and return timing results."""
    is_timeout = False
    try:
        execution_time, execution_times, planning_time = executor.get_exec_time(
            query, log_execution=False, return_all=True,
            n_runs=n_runs, return_planning_time=True
        )
    except Exception:
        is_timeout = True

    if is_timeout or execution_time is None:
        execution_time = executor.statement_timeout
        execution_times = [execution_time] * n_runs
        planning_time = 0

    if return_planning_time:
        if return_all:
            return execution_times, is_timeout, planning_time
        return execution_time, is_timeout, planning_time
    else:
        if return_all:
            return execution_times, is_timeout
        return execution_time, is_timeout


def generate_encoding(join_graph, hinter, executor=None, selectivity_extractor=None, n_runs=3):
    """Generate a random plan encoding and measure its execution time."""
    plan_enc = PlanEncoder.encode(join_graph)
    plan_query = hinter.vec_to_hint(join_graph, plan_enc.numpy())
    plan_exec_times, _ = gather_exec_time(plan_query, executor=executor,
                                          n_runs=n_runs, return_all=True)
    plan_exec_times = torch.tensor(plan_exec_times)
    return plan_enc, plan_exec_times


def encode_plan_to_compact(plan_enc_flat, join_graph):
    """
    Extract priority flat_idx from PlanEncoder output, sorted by rank ascending.

    In each participating edge's 3-cell row, the join type with the highest
    encoding value is Step 2's priority pick (lowest rank). Sorting by value
    descending yields rank-ascending order.

    Returns: list[int], length K = number of participating edges.
    """
    num_edges = join_graph.num_undirected_edges
    enc_2d = plan_enc_flat.reshape(num_edges, 3)

    priority = []
    for e in range(num_edges):
        row = enc_2d[e, :]
        if row.max() > 0:
            jt = int(row.argmax())
            priority.append((e, jt, row[jt]))

    priority.sort(key=lambda x: x[2], reverse=True)
    return [e * 3 + jt for e, jt, _ in priority]


def encode_query_to_compact(join_graph):
    """Extract alias_selectivities dict for compact storage."""
    return dict(join_graph.alias_selectivities)


def _qid_to_str(qid):
    """Normalize qid to string. Integer qids from WorkloadFileLoader, string qids from MultipleFileLoader."""
    if isinstance(qid, str):
        return qid.split('.')[0]
    return str(qid)


def main(args):
    WORKLOAD_NAME = args.workload
    SCHEMA_FOLDER = WORKLOADS[WORKLOAD_NAME]
    IS_NEW_WORKLOAD = WORKLOAD_NAME in NEW_WORKLOADS
    REQUIRED_QUERIES = args.num_training_queries_required

    setup_logging('gen_data', log_dir='logs')

    # Timeout per workload type
    if WORKLOAD_NAME == 'stack':
        TIMEOUT = 0.15 * 60 * 1000
    elif WORKLOAD_NAME in ('join_order_benchmark', 'join_order_benchmark_light'):
        TIMEOUT = 15 * 60 * 1000
    elif IS_NEW_WORKLOAD:
        TIMEOUT = 2 * 60 * 1000
    else:
        TIMEOUT = 15 * 60 * 1000

    logging.info(f"Workload: {WORKLOAD_NAME}, schema: {SCHEMA_FOLDER}, "
                 f"timeout: {TIMEOUT}ms, plans_per_query: {N_PLANS_PER_QUERY}")

    os.makedirs(os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER), exist_ok=True)
    pg_csv_path = os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, 'postgres_execution_time.csv')

    # ---- Output setup ----
    if IS_NEW_WORKLOAD:
        output_dir = os.path.join(DEMO_BASE_PATH, SCHEMA_FOLDER)
        os.makedirs(output_dir, exist_ok=True)
        jsonl_path = os.path.join(output_dir, JSONL_FILENAME)
        checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILENAME)

        completed_qids = set()
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                completed_qids = set(line.strip() for line in f)
            logging.info(f"Resuming: {len(completed_qids)} queries already done")
        results_written = 0
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                results_written = sum(1 for _ in f)
    else:
        base_path = os.path.join(DEMO_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE)
        completed_qids = set()
        if os.path.isdir(base_path):
            completed_qids = {
                qid for qid in os.listdir(base_path)
                if all(os.path.isfile(os.path.join(base_path, qid, filename))
                       for filename in ('query_enc.pt', 'plan_enc.pt', 'exec_times.pt'))
            }
        results_written = len(completed_qids)

    if results_written >= REQUIRED_QUERIES:
        logging.info(f"Already have {results_written} training queries, "
                     f"which meets required {REQUIRED_QUERIES}")
        return

    ql = QueryLoader(WORKLOAD_NAME)
    queries = ql.queries()
    incomplete_queries = [qid for qid in queries
                          if _qid_to_str(qid) not in completed_qids]
    if not incomplete_queries:
        logging.warning(f"No incomplete queries remain; have {results_written} "
                        f"training queries, below required {REQUIRED_QUERIES}")
        return

    remaining_needed = REQUIRED_QUERIES - results_written
    logging.info(f"Total: {len(queries)}, incomplete: {len(incomplete_queries)}, "
                 f"already have: {results_written}, need: {remaining_needed}")
    np.random.shuffle(incomplete_queries)

    executor = PostgresExecutor.create_postgres_engine(
        local_port=DB_CONFIG[WORKLOAD_NAME]['DB_PORT'],
        db_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
        statement_timeout=TIMEOUT
    )
    executor.execute("LOAD 'pg_hint_plan'; SET pg_hint_plan.enable_hint_table TO off;")
    selectivity_extractor = PostgresSelectivityExtractor(workload_name=WORKLOAD_NAME,
                                                         database_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
                                                         executor=executor)
    database_graph = DatabaseGraph(selectivity_extractor=selectivity_extractor,
                                   workload_name=WORKLOAD_NAME)
    hinter = PlanHinter()

    # Read existing PG baseline to skip already-collected queries
    pg_completed_qids = set()
    if os.path.exists(pg_csv_path):
        with open(pg_csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    pg_completed_qids.add(row[3])  # query_ident column
        logging.info(f"PG baseline already has {len(pg_completed_qids)} queries")
    else:
        with open(pg_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'run_id', 'experiment', 'query_ident',
                             'split', 'inference_time', 'planning_time', 'execution_time'])

    pbar = tqdm(total=min(remaining_needed, len(incomplete_queries)),
                desc='Generating training data...')
    for qid in incomplete_queries:
        qid_str = _qid_to_str(qid)

        # ---- Checkpoint / resume ----
        if not IS_NEW_WORKLOAD:
            if WORKLOAD_NAME == 'join_order_benchmark_light' and '21' in qid_str:
                logging.info(f"Skipping {qid_str} (known memory issue)")
                continue
            query_path = os.path.join(base_path, qid_str)

        query = ql.get_query(qid)

        # ---- Build join graph ----
        try:
            jg = SimpleJoinGraph(query, workload_name=WORKLOAD_NAME,
                                 selectivity_extractor=selectivity_extractor,
                                 database_graph=database_graph)
        except JoinNotPossibleException as e:
            logging.warning(f"[{qid_str}] Join not possible: {e}")
            if IS_NEW_WORKLOAD:
                with open(checkpoint_path, 'a') as cp:
                    cp.write(f"{qid_str}\n")
            continue
        except SelectivityExtractorCannotExtractException as e:
            logging.warning(f"[{qid_str}] Selectivity extraction failed: {e}")
            if IS_NEW_WORKLOAD:
                with open(checkpoint_path, 'a') as cp:
                    cp.write(f"{qid_str}\n")
            continue
          

        # For JOB/STACK legacy: prepare query_enc for saving
        if not IS_NEW_WORKLOAD:
            query_enc = QueryEncoder.encode(jg)
            all_plan_enc = []
            all_exec_times = []

        # ---- PG baseline ----
        if qid_str not in pg_completed_qids:
            try:
                joins, pg_times, pg_planning_time = gather_join_order(
                    query, executor=executor, for_postgres=True,
                    n_runs=N_RUNS, return_planning_time=True
                )
                with open(pg_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for run_id in range(3):
                        writer.writerow(['PostgreSQL', run_id, None, qid_str, None, 0,
                                         pg_planning_time, pg_times[-(3 - run_id)]])
                pg_completed_qids.add(qid_str)
                logging.info(f"[{qid_str}] PG avg: {sum(pg_times[-3:])/3:.2f}ms, "
                             f"planning: {pg_planning_time:.2f}ms")
            except Exception as e:
                logging.warning(
                    f"[{qid_str}] PG baseline failed; skipping training data and checkpoint: {e}"
                )
                if IS_NEW_WORKLOAD:
                    continue

        # ---- Generate random plans ----
        if IS_NEW_WORKLOAD:
            alias_sels = encode_query_to_compact(jg)
            plans_data = []

        counter = 0
        for _ in range(N_PLANS_PER_QUERY):
            try:
                plan_enc, exec_times = generate_encoding(
                    jg, hinter, executor=executor,
                    selectivity_extractor=selectivity_extractor,
                    n_runs=N_RUNS
                )
            except JoinNotPossibleException:
                continue

            if len(exec_times) == N_RUNS:
                counter += 1
                if IS_NEW_WORKLOAD:
                    flat_idx_list = encode_plan_to_compact(plan_enc.numpy(), jg)
                    plans_data.append({
                        'flat_idx': flat_idx_list,
                        'exec_times': exec_times[-3:].tolist()
                    })
                else:
                    all_plan_enc.append(plan_enc)
                    all_exec_times.append(exec_times[-3:])
                    # Overwrite intermediate results
                    os.makedirs(query_path, exist_ok=True)
                    torch.save(torch.stack([query_enc]),
                               os.path.join(query_path, 'query_enc.pt'))
                    torch.save(torch.stack(all_plan_enc),
                               os.path.join(query_path, 'plan_enc.pt'))
                    torch.save(torch.stack(all_exec_times),
                               os.path.join(query_path, 'exec_times.pt'))

        # ---- Save per-query results ----
        if IS_NEW_WORKLOAD:
            record = {
                'qid': int(qid_str),
                'alias_selectivities': alias_sels,
                'plans': plans_data
            }
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

            with open(checkpoint_path, 'a') as cp:
                cp.write(f"{qid_str}\n")

        logging.info(f"[{qid_str}] Generated {counter} plans")
        if IS_NEW_WORKLOAD or counter > 0:
            results_written += 1
            pbar.update(1)
            if results_written >= REQUIRED_QUERIES:
                logging.info(f"Reached required {REQUIRED_QUERIES} queries, stopping")
                break

    pbar.close()
    executor.close()
    if results_written < REQUIRED_QUERIES:
        logging.warning(f"Queries exhausted: have {results_written} training queries, "
                        f"below required {REQUIRED_QUERIES}")
    logging.info(f"PG baseline saved: {pg_csv_path}")
    if IS_NEW_WORKLOAD:
        logging.info(f"Training data saved: {jsonl_path} ({results_written} lines)")
    logging.info("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, required=True,
                        help='Workload name, e.g. join_order_benchmark, job_light, etc.')
    parser.add_argument('--num_training_queries_required', type=int, default=500,
                        help='Number of training queries required')
    args = parser.parse_args()
    main(args)

# python scripts/generating/create_full_plan_data.py --workload job_light --num_training_queries_required 500
