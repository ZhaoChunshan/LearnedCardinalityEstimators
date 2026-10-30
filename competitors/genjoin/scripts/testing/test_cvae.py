# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: workload loading, checkpoint selection, evaluation timeouts,
# logging, and result-file output.

import os
import sys
import gc
import time
import logging
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import networkx as nx
import argparse
import json
from collections import defaultdict
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from mo_sql_parsing import parse

from src.database.postgres_executor import PostgresExecutor
from src.database.query_loader import QueryLoader, WorkloadFileLoader
from src.encoding.database_graph import DatabaseGraph
from src.encoding.join_graph import JoinGraph, SimpleJoinGraph, JoinType, RedundantJoinException, JoinNotPossibleException
from src.encoding.encoder import PSTEncoder, PlanEncoder, QueryEncoder, positional_encoding
from src.encoding.filters import FiltersProcessor
from src.database.utils import PlanParser, PlanHinter, gather_join_order
from src.database.selectivity_extractor import PostgresSelectivityExtractor
from src.utils.query_helper import parse_sql_query
from src.utils.path_helper import WORKLOADS, DB_CONFIG
from src.utils.train_test_splits import get_train_test_split
from src.utils.logging_config import setup_logging
from scripts.generating.create_full_plan_data import generate_encoding, gather_exec_time

from tqdm.auto import tqdm
import wandb

LOGS_FILENAME = 'logs'
TIMEOUT = 2 * 60 * 60 * 1000 # (In milliseconds) For test queries, a larger timeout is used to avoid aborting queries that are slow due to bad plans
N_RUNS = 3
N_MODELS = 1
NEW_WORKLOADS = ('job_light', 'stats_ceb', 'ergastf1')

TEST_FILES = {
    'job_light': 'job_light.sql',
    'stats_ceb': 'stats_ceb.sql',
    'ergastf1': 'ergastf1_queries.sql',
}

EXPERIMENTS = {
    'LeaveOneOutSplit1': 'leave_one_out_split_1', 
    'LeaveOneOutSplit2': 'leave_one_out_split_2',
    'LeaveOneOutSplit3': 'leave_one_out_split_3',
    'RandomSplit1': 'random_split_1',
    'RandomSplit2': 'random_split_2',
    'RandomSplit3': 'random_split_3',
    'BaseQuerySplit1': 'base_query_split_1', 
    'BaseQuerySplit2': 'base_query_split_2',
    'BaseQuerySplit3': 'base_query_split_3',
    'LeaveOneOutSplitA': 'leave_one_out_split_a', 
    'LeaveOneOutSplitB': 'leave_one_out_split_b',
    'LeaveOneOutSplitC': 'leave_one_out_split_c',
    'RandomSplitA': 'random_split_a',
    'RandomSplitB': 'random_split_b',
    'RandomSplitC': 'random_split_c',
    'BaseQuerySplitA': 'base_query_split_a', 
    'BaseQuerySplitB': 'base_query_split_b',
    'BaseQuerySplitC': 'base_query_split_c',
    'SlowSplit': 'slow_split'
}


def load_test_queries(workload_name):
    """Load test queries from workloads/ directory."""
    schema_folder = WORKLOADS[workload_name]
    filename = TEST_FILES[workload_name]
    filepath = f'workloads/{schema_folder}/{filename}'
    queries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line.rstrip(';'))
    return queries


def main(args):

    torch.manual_seed(1234)
    np.random.seed(1234)

    WORKLOAD_NAME = args.workload
    SCHEMA_FOLDER = WORKLOADS[WORKLOAD_NAME]
    IS_NEW_WORKLOAD = WORKLOAD_NAME in NEW_WORKLOADS
    TEST_RESULTS = f"full_execution_time_{args.workload}_{args.mode}.csv"

    setup_logging('testing', log_dir='logs')
    logging.info(f"Testing start: workload={WORKLOAD_NAME}, experiment={args.experiment}, mode={args.mode}")

    # Norm per workload
    if WORKLOAD_NAME == 'join_order_benchmark':
        norm = 100
    elif WORKLOAD_NAME == 'stack':
        norm = 50
    elif WORKLOAD_NAME == 'job_light':
        norm = 23
    elif WORKLOAD_NAME == 'stats_ceb':
        norm = 24
    elif WORKLOAD_NAME == 'ergastf1':
        norm = 28
    else:
        norm = 100

    executor = PostgresExecutor.create_postgres_engine(local_port=DB_CONFIG[WORKLOAD_NAME]['DB_PORT'],
                                                       db_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
                                                       statement_timeout=TIMEOUT)
    executor.execute("LOAD 'pg_hint_plan'; SET pg_hint_plan.enable_hint_table TO off;")

    # Run warmup if warmup.sql exists (pre-warm buffer cache for consistent timings)
    warmup_path = f'workloads/{SCHEMA_FOLDER}/warmup.sql'
    if os.path.exists(warmup_path):
        logging.info(f"Running warmup from {warmup_path}")
        with open(warmup_path, 'r') as f:
            warmup_sql = f.read()
        executor.execute(warmup_sql)
        logging.info("Warmup completed")
    else:
        logging.info(f"No warmup file found at {warmup_path}, skipping.")

    selectivity_extractor = PostgresSelectivityExtractor(workload_name=WORKLOAD_NAME,
                                                         database_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
                                                         executor=executor)
    database_graph = DatabaseGraph(selectivity_extractor=selectivity_extractor, workload_name=WORKLOAD_NAME)
    hinter = PlanHinter()

    if IS_NEW_WORKLOAD:
        test_queries = load_test_queries(WORKLOAD_NAME)
    else:
        ql = QueryLoader(WORKLOAD_NAME)
        if args.mode == 'test':
            _, test_queries = get_train_test_split(args.experiment, workload_name=WORKLOAD_NAME)
        else:
            test_queries, _ = get_train_test_split(args.experiment, workload_name=WORKLOAD_NAME)

    logging.info(f'Experiment: {args.experiment}, Mode: {args.mode}, Test queries: {len(test_queries)}')

    model = dict()
    for run_id in range(N_MODELS):
        model_path = args.model_path or \
            f'./models/{SCHEMA_FOLDER}/cvae/run{run_id}/cvae_{args.experiment}.pt'
        model_path = os.path.expanduser(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'Model checkpoint not found: {os.path.abspath(model_path)}')
        logging.info(f'Loading model: {os.path.abspath(model_path)}')
        model[run_id] = torch.load(model_path, map_location='cpu')
        model[run_id].eval()
        model[run_id].to('cpu')

    # Load PG baseline for non-pg_live mode
    pg_csv_path = os.path.join('./data/processed', SCHEMA_FOLDER, 'postgres_execution_time.csv')
    if os.path.exists(pg_csv_path):
        pg_df = pd.read_csv(pg_csv_path)
    else:
        pg_df = pd.DataFrame(columns=['method', 'run_id', 'experiment', 'query_ident',
                                       'split', 'inference_time', 'planning_time', 'execution_time'])

    # Load ordered joins if available
    ordered_joins_path = f'./data/processed/{SCHEMA_FOLDER}/postgres16_ordered_joins.json'
    if os.path.exists(ordered_joins_path):
        with open(ordered_joins_path, 'r') as file:
            ordered_joins = json.loads(file.read())[0]
    else:
        ordered_joins = {}

    results_filename = os.path.join('./data/processed', SCHEMA_FOLDER, TEST_RESULTS)

    cvae_results = []
    pg_results = []
    softmax = torch.nn.Softmax(dim=1)
    confs_true = defaultdict(list)
    confs_pred = defaultdict(list)

    parse('SELECT 1')

    for qid_idx, qid in enumerate(test_queries):
        gc.collect()

        logging.info(f"[{qid_idx + 1}] {str(qid)}...")

        if IS_NEW_WORKLOAD:
            query = qid  # test queries are the SQL text itself
        else:
            query = ql.get_query(qid)

        inference_start = time.time()
        jg = SimpleJoinGraph(query, selectivity_extractor=selectivity_extractor,
                            workload_name=WORKLOAD_NAME, database_graph=database_graph)
        input_enc = PlanEncoder.encode(jg)
        query_enc = QueryEncoder.encode(jg)
        conf_dist = (torch.exp(torch.tensor([0.0])) - 1) / (np.exp(1) - 1)
        out_dist = (torch.exp(torch.tensor([0.0])) - 1) / (np.exp(0.025) - 1)
        inference_fixed = (time.time() - inference_start) * 1000
        for run_id in range(N_MODELS):
            if args.pg_live:
                joins, pg_times, pg_planning_time = gather_join_order(query, executor=executor, for_postgres=True, n_runs=N_RUNS, return_planning_time=True)
                ordered_joins[qid] = joins
                pg_results.append(('PostgreSQL', run_id, args.experiment, qid, args.mode, 0, pg_planning_time, pg_times[-1]))
            else:
                pg_match = pg_df[pg_df.query_ident == str(qid)]
                if len(pg_match) > 0:
                    pg_times = pg_match.execution_time.values
                    pg_planning_time = pg_match.planning_time.values[0]
                else:
                    pg_times = np.array([0])
                    pg_planning_time = 0
            inference_run = time.time()
            out, mu, logvar, z = model[run_id](input_enc.view(-1, len(input_enc)), query_enc.view(-1, len(query_enc)), conf_dist.view(-1, len(conf_dist)), out_dist.view(-1, len(out_dist)))
            from src.encoding.plan_decoder import model_output_to_hints
            plan_enc_pred = model_output_to_hints(out, jg, norm=norm, method='argmax')
            hinted = hinter.vec_to_hint(jg, plan_enc_pred)
            inference_time = inference_fixed + (time.time() - inference_run) * 1000
            # In our experiments, we only run the execution once, after warmup all pages.
            cvae_times, _, planning_time = gather_exec_time(hinted, executor, n_runs=1,
                                             return_all=True, return_planning_time=True)
            cvae_results.append(('cVAE', run_id, args.experiment, qid, args.mode, inference_time, planning_time, cvae_times[-1]))
            logging.info(f'PG={(np.mean(pg_times[-N_RUNS:]) + pg_planning_time):.2f} GenJoin={(cvae_times[-1] + planning_time + inference_time):.2f} infer={inference_time:.2f}ms')

    temp = pd.DataFrame(cvae_results, columns=['method', 'run_id', 'experiment',
                                               'query_ident', 'split', 'inference_time', 'planning_time', 'execution_time'])
    # pg_df_out = pd.DataFrame(pg_results, columns=['method', 'run_id', 'experiment',
                                            #    'query_ident', 'split', 'inference_time', 'planning_time', 'execution_time'])
    # logging.info(f"Total Postgres time (approx): {1/N_MODELS * (pg_df_out['execution_time'].sum() + pg_df_out['planning_time'].sum()):.2f}ms")
    logging.info(f"Total cVAE time: {1/N_MODELS * (temp[temp.method == 'cVAE']['execution_time'].sum() + temp[temp.method == 'cVAE']['planning_time'].sum() + temp[temp.method == 'cVAE']['inference_time'].sum()):.2f}ms")

    if args.dump:
        os.makedirs(os.path.dirname(f'./data/processed/{SCHEMA_FOLDER}/{TEST_RESULTS}'), exist_ok=True)
        temp.to_csv(f'./data/processed/{SCHEMA_FOLDER}/{TEST_RESULTS}', index=False)
        logging.info(f"Results saved: {TEST_RESULTS}")

    executor.close_connection()
    logging.info("Testing done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, required=True, help='Defines the worload to be used, i.e. as of now only `join_order_benchmark` for JOB or `stack` for STACK.')
    parser.add_argument('--experiment', type=str, default='default', required=False, help='Defines which train-test split to use (see #get_train_test_split method).')
    parser.add_argument('--dump', action='store_true', default=False, help='Saving the results')
    parser.add_argument('--pg_live', action='store_true', default=False, help='If to make live PostgreSQL runs for "fairness" of results comparison')
    parser.add_argument('--mode', type=str, default='test', help='Defines which dataset to predict on (`test` set is a default one).')
    parser.add_argument('--model_path', '--model-path', type=str, default=None,
                        help='Optional checkpoint path. Defaults to the model selected by workload, experiment, and run_id.')
    args = parser.parse_args()

    main(args)

# python scripts/testing/test_cvae.py --dump --workload job_light/stats_ceb/ergastf1
