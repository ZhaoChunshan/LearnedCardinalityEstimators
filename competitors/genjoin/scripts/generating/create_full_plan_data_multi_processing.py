# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Addition: a multiprocessing variant of GenJoin plan-data generation with
# logging and resumable output.

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import json
import csv
import argparse
import logging
import multiprocessing as mp

import torch

from src.database.query_loader import QueryLoader
from src.encoding.encoder import PlanEncoder
from src.database.postgres_executor import PostgresExecutor
from src.database.utils import gather_join_order
from src.utils.path_helper import DB_CONFIG, WORKLOADS
from src.utils.logging_config import setup_logging

from tqdm.auto import tqdm


DEMO_BASE_PATH = 'data/demonstrations/'
PROCESSED_BASE_PATH = 'data/processed/'
N_RUNS = 5
N_PLANS_PER_QUERY = 200
JSONL_FILENAME = 'train_data.jsonl'
CHECKPOINT_FILENAME = 'checkpoint.txt'

np.random.seed(1234)
torch.manual_seed(1234)


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


def worker_process(worker_id, workload_name, task_queue, result_queue, stop_event,
                   timeout_ms, n_plans_per_query, n_runs):
    """
    Worker process: takes query IDs from task_queue, generates N_PLANS_PER_QUERY random
    plans with execution times, and puts results in result_queue.

    Each worker creates its own DB connections and objects (selectivity_extractor,
    database_graph, hinter) for process isolation.
    """
    # Re-import inside worker for clean process-local state
    from src.database.query_loader import QueryLoader
    from src.encoding.database_graph import DatabaseGraph
    from src.encoding.join_graph import SimpleJoinGraph, JoinNotPossibleException
    from src.database.selectivity_extractor import PostgresSelectivityExtractor, SelectivityExtractorCannotExtractException
    from src.database.postgres_executor import PostgresExecutor
    from src.database.utils import PlanHinter
    from src.utils.path_helper import DB_CONFIG

    import numpy as np
    import torch
    import logging

    # Per-worker seed for diverse random plans
    np.random.seed(1234 + worker_id)
    torch.manual_seed(1234 + worker_id)

    pid = os.getpid()
    logger = logging.getLogger(f'Worker-{pid}')
    logger.info(f"Worker {worker_id} (PID {pid}) started")

    # ---- Create per-process DB connection and objects ----
    executor = PostgresExecutor.create_postgres_engine(
        local_port=DB_CONFIG[workload_name]['DB_PORT'],
        db_name=DB_CONFIG[workload_name]['DB_NAME'],
        statement_timeout=timeout_ms
    )

    load_hinter = """
    LOAD 'pg_hint_plan';
    SET pg_hint_plan.enable_hint_table TO off;
    """
    executor.execute(load_hinter)

    selectivity_extractor = PostgresSelectivityExtractor(
        workload_name=workload_name,
        database_name=DB_CONFIG[workload_name]['DB_NAME'],
        executor=executor
    )
    database_graph = DatabaseGraph(
        selectivity_extractor=selectivity_extractor,
        workload_name=workload_name
    )
    hinter = PlanHinter()
    ql = QueryLoader(workload_name)

    while not stop_event.is_set():
        try:
            qid = task_queue.get(timeout=1)
        except:  # queue empty
            continue

        if qid is None:  # sentinel — no more tasks
            break

        qid_str = _qid_to_str(qid)
        logger.info(f"[{qid_str}] Processing...")

        # ---- Load query ----
        try:
            query = ql.get_query(qid)
        except Exception as e:
            logger.warning(f"[{qid_str}] Failed to load query: {e}")
            result_queue.put({
                'qid_str': qid_str, 'success': False,
                'query': None, 'data': None, 'error': f"load_error:{e}"
            })
            continue

        # ---- Build join graph ----
        try:
            jg = SimpleJoinGraph(
                query,
                workload_name=workload_name,
                selectivity_extractor=selectivity_extractor,
                database_graph=database_graph
            )
        except JoinNotPossibleException as e:
            logger.warning(f"[{qid_str}] Join not possible: {e}")
            result_queue.put({
                'qid_str': qid_str, 'success': False,
                'query': None, 'data': None, 'error': f"join_error:{e}"
            })
            continue
        except SelectivityExtractorCannotExtractException as e:
            logger.warning(f"[{qid_str}] Selectivity extraction failed: {e}")
            result_queue.put({
                'qid_str': qid_str, 'success': False,
                'query': None, 'data': None, 'error': f"selectivity_error:{e}"
            })
            continue

        # ---- Generate random plans ----
        alias_sels = encode_query_to_compact(jg)
        plans_data = []
        counter = 0

        for _ in range(n_plans_per_query):
            if stop_event.is_set():
                break
            try:
                plan_enc, exec_times = generate_encoding(
                    jg, hinter, executor=executor,
                    selectivity_extractor=selectivity_extractor,
                    n_runs=n_runs
                )
            except JoinNotPossibleException:
                continue

            if len(exec_times) == n_runs:
                counter += 1
                flat_idx_list = encode_plan_to_compact(plan_enc.numpy(), jg)
                plans_data.append({
                    'flat_idx': flat_idx_list,
                    'exec_times': exec_times[-3:].tolist()
                })

        logger.info(f"[{qid_str}] Generated {counter} plans")
        result_queue.put({
            'qid_str': qid_str,
            'success': True,
            'query': query,
            'data': {
                'alias_selectivities': alias_sels,
                'plans': plans_data
            },
            'error': None
        })

    executor.close()
    logger.info(f"Worker {worker_id} exiting")


def main(args):
    WORKLOAD_NAME = args.workload
    SCHEMA_FOLDER = WORKLOADS[WORKLOAD_NAME]
    NUM_WORKERS = args.num_workers
    REQUIRED_QUERIES = args.num_training_queries_required

    setup_logging('gen_data', log_dir='logs')

    # Timeout per workload type.
    # IS_NEW_WORKLOAD is always True — only the else branch is reachable for these workloads.
    if WORKLOAD_NAME == 'stack':
        TIMEOUT = 0.15 * 60 * 1000
    elif WORKLOAD_NAME in ('join_order_benchmark', 'join_order_benchmark_light'):
        TIMEOUT = 15 * 60 * 1000
    else:
        TIMEOUT = 2 * 60 * 1000

    logging.info(f"Workload: {WORKLOAD_NAME}, schema: {SCHEMA_FOLDER}, "
                 f"timeout: {TIMEOUT}ms, plans_per_query: {N_PLANS_PER_QUERY}")

    # ---- Output paths (IS_NEW_WORKLOAD is always True) ----
    output_dir = os.path.join(DEMO_BASE_PATH, SCHEMA_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, JSONL_FILENAME)
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILENAME)

    # ---- Load completed qids from checkpoint ----
    completed_qids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            completed_qids = set(line.strip() for line in f)
        logging.info(f"Resuming: {len(completed_qids)} queries already done")

    # ---- PG csv setup ----
    os.makedirs(os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER), exist_ok=True)
    pg_csv_path = os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER,
                               'postgres_execution_time.csv')

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
                             'split', 'inference_time', 'planning_time',
                             'execution_time'])

    # ---- Load queries ----
    ql = QueryLoader(WORKLOAD_NAME)
    queries = ql.queries()

    # Filter to incomplete queries only
    incomplete_queries = [qid for qid in queries
                          if _qid_to_str(qid) not in completed_qids]

    if not incomplete_queries:
        logging.info("All queries completed already")
        return

    # Count existing lines in jsonl
    jsonl_line_count = 0
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            jsonl_line_count = sum(1 for _ in f)
    if jsonl_line_count >= REQUIRED_QUERIES:
        logging.info(f"Already have {jsonl_line_count} lines in {jsonl_path}, "
                     f"which meets required {REQUIRED_QUERIES}")
        return

    remaining_needed = REQUIRED_QUERIES - jsonl_line_count
    logging.info(f"Total: {len(queries)}, incomplete: {len(incomplete_queries)}, "
                 f"already have: {jsonl_line_count}, need: {remaining_needed}")

    # Shuffle for random processing order
    np.random.shuffle(incomplete_queries)

    # ---- Set up multiprocessing queues ----
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()

    # Add all incomplete tasks to the queue, followed by one sentinel per worker
    for qid in incomplete_queries:
        task_queue.put(qid)
    for _ in range(NUM_WORKERS):
        task_queue.put(None)  # sentinel

    # Start worker processes
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=worker_process,
            args=(i, WORKLOAD_NAME, task_queue, result_queue, stop_event,
                  TIMEOUT, N_PLANS_PER_QUERY, N_RUNS)
        )
        p.start()
        processes.append(p)

    # ---- Main process: executor for PG baseline timing ----
    executor = PostgresExecutor.create_postgres_engine(
        local_port=DB_CONFIG[WORKLOAD_NAME]['DB_PORT'],
        db_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
        statement_timeout=TIMEOUT
    )

    load_hinter = """
    LOAD 'pg_hint_plan';
    SET pg_hint_plan.enable_hint_table TO off;
    """
    executor.execute(load_hinter)

    # ---- Collect results from workers ----
    results_written = jsonl_line_count
    workers_alive = NUM_WORKERS

    try:
        pbar = tqdm(
            total=min(remaining_needed, len(incomplete_queries)),
            desc='Generating training data...'
        )

        while workers_alive > 0 and results_written < REQUIRED_QUERIES:
            try:
                result = result_queue.get(timeout=2)
            except:  # timeout — check if workers are still alive
                workers_alive = sum(1 for p in processes if p.is_alive())
                continue

            qid_str = result['qid_str']

            # ---- Handle worker-reported failures ----
            if not result['success']:
                logging.warning(f"[{qid_str}] Failed: {result['error']}")
                with open(checkpoint_path, 'a') as cp:
                    cp.write(f"{qid_str}\n")
                continue

            # ---- PG baseline timing (main process) ----
            if qid_str not in pg_completed_qids:
                try:
                    query = result['query']
                    _, pg_times, pg_planning_time = gather_join_order(
                        query, executor=executor, for_postgres=True,
                        n_runs=N_RUNS, return_planning_time=True
                    )
                    with open(pg_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for run_id in range(3):
                            writer.writerow(
                                ['PostgreSQL', run_id, None, qid_str, None, 0,
                                 pg_planning_time, pg_times[-(3 - run_id)]]
                            )
                    pg_completed_qids.add(qid_str)
                    logging.info(f"[{qid_str}] PG avg: {sum(pg_times[-3:])/3:.2f}ms, "
                                 f"planning: {pg_planning_time:.2f}ms")
                except Exception as e:
                    logging.warning(
                        f"[{qid_str}] PG baseline failed; discarding worker result "
                        f"without writing JSONL or checkpoint: {e}"
                    )
                    continue

            # ---- Write result to jsonl ----
            record = {
                'qid': int(qid_str),
                'alias_selectivities': result['data']['alias_selectivities'],
                'plans': result['data']['plans']
            }
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

            # ---- Write checkpoint ----
            with open(checkpoint_path, 'a') as cp:
                cp.write(f"{qid_str}\n")

            results_written += 1
            pbar.update(1)

            # ---- Check if we have enough queries ----
            if results_written >= REQUIRED_QUERIES:
                logging.info(f"Reached required {REQUIRED_QUERIES} queries, stopping")
                stop_event.set()
                break

        pbar.close()

    finally:
        # Ensure all workers are stopped
        stop_event.set()
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        executor.close()

    final_line_count = 0
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            final_line_count = sum(1 for _ in f)
    logging.info(f"PG baseline saved: {pg_csv_path}")
    logging.info(f"Training data saved: {jsonl_path} ({final_line_count} lines)")
    logging.info("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, required=True,
                        help='Workload name, e.g. job_light, stats_ceb, ergastf1')
    parser.add_argument('--num_training_queries_required', type=int, default=500,
                        help='Number of training queries required')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of worker processes')
    args = parser.parse_args()
    main(args)

# python scripts/generating/create_full_plan_data_multi_processing.py --workload job_light --num_workers 16 --num_training_queries_required 500
