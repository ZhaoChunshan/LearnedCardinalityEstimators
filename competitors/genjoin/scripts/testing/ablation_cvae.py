# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: disable an unavailable visualization import and use the local
# PostgreSQL baseline filename.

import os
import sys
import gc
import time
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
from src.database.query_loader import QueryLoader
from src.encoding.database_graph import DatabaseGraph
from src.encoding.join_graph import JoinGraph, SimpleJoinGraph, JoinType, RedundantJoinException, JoinNotPossibleException
from src.encoding.encoder import PSTEncoder, PlanEncoder, QueryEncoder, positional_encoding
from src.encoding.filters import FiltersProcessor
from src.database.utils import PlanParser, PlanHinter, gather_join_order
from src.database.selectivity_extractor import PostgresSelectivityExtractor
from src.utils.query_helper import parse_sql_query
from src.utils.path_helper import WORKLOADS, DB_CONFIG
from src.utils.train_test_splits import get_train_test_split
# from src.utils.viz import plot_exec_time_difference
from scripts.generating.create_full_plan_data import generate_encoding, gather_exec_time
# from src.models.cvae import CVAE_G, CVAE_D
from scripts.training.train_cvae import kld_gaussians

from tqdm.auto import tqdm
import wandb

LOGS_FILENAME = 'logs'
TIMEOUT = 1.5 * 60 * 1000
N_RUNS = 5
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


def main(args):
    
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    WORKLOAD_NAME = args.workload  #  'join_order_benchmark', 'stack'
    SCHEMA_FOLDER = WORKLOADS[WORKLOAD_NAME]
    TEST_RESULTS = f"ablation_chain_{args.mode}.csv"
    
    if WORKLOAD_NAME == 'join_order_benchmark':
        norm = 100
    elif WORKLOAD_NAME == 'stack':
        norm = 50
    
    executor = PostgresExecutor.create_postgres_engine(local_port=DB_CONFIG[WORKLOAD_NAME]['DB_PORT'],
                                                       db_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
                                                       statement_timeout=TIMEOUT)
    # Enable pg_hint_plan
    executor.execute("LOAD 'pg_hint_plan'; SET pg_hint_plan.enable_hint_table TO off;")

    selectivity_extractor = PostgresSelectivityExtractor(workload_name=WORKLOAD_NAME,
                                                         executor=executor)
    database_graph = DatabaseGraph(selectivity_extractor=selectivity_extractor, workload_name=WORKLOAD_NAME)
    hinter = PlanHinter()
    ql = QueryLoader(WORKLOAD_NAME)
    if args.mode == 'test':
        _, test_queries = get_train_test_split(args.experiment, workload_name=WORKLOAD_NAME)
    else:
        test_queries, _ = get_train_test_split(args.experiment, workload_name=WORKLOAD_NAME)
    print(f'Experiment: {args.experiment}, Mode: {args.mode}')
    if args.experiment == 'join_order_benchmark':
        norm = 100
    else:
        norm = 50
    
    model = torch.load(f'./models/{SCHEMA_FOLDER}/cvae/cvae_{args.experiment}.pt')
    model.eval()
    model.to('cpu')
    
    clf = torch.load(f'./models/{SCHEMA_FOLDER}/cvae/clf_{args.experiment}.pt')
    clf.eval()
    clf.to('cpu')
    
    # Reading already created files with execution times, orders of joins etc.
    with open(f'./data/processed/{SCHEMA_FOLDER}/postgres16_ordered_joins.json', 'r') as file:
        ordered_joins = json.loads(file.read())[0]

    pg_df = pd.read_csv(os.path.join('./data/processed', SCHEMA_FOLDER, 'postgres_execution_time.csv'))
    
    results_filename = os.path.join('./data/processed', SCHEMA_FOLDER, TEST_RESULTS)
    if os.path.exists(results_filename):
        df = pd.read_csv(results_filename)
    else:
        df = pg_df.copy()
    
    cvae_results = []
    pg_results = []
    # plot_data = []
    pg_decision_count = 0
    softmax = torch.nn.Softmax(dim=1)
    confs_true = defaultdict(list)
    confs_pred = defaultdict(list)

    # Parsing a query once to avoid the overhead of generating the Parser object initially
    # (which is stored globally, so it only will be created once)
    parse('SELECT 1')

    for qid in test_queries:
        # Set up initial confidences to compare against inside "chain-of-plans"
        conf_dist_prev = 1e+8
        out_dist_prev = 1e+8
        counter = 0

        # ensure the garbage is collected at the beginning of each query
        # to reduce the likelihood that garbage collection happens at unforunate times
        # during the inference, e.g. because mo_sql_parsing is generating a lot of objects
        gc.collect()

        print(qid)
        query = ql.get_query(qid)

        # TODO: manage `pg_results` if we do not have `pg_live`
        if args.pg_live:
            joins, pg_times, pg_planning_time = gather_join_order(query, executor=executor, for_postgres=True, n_runs=N_RUNS, return_planning_time=True)
            ordered_joins[qid] = joins
            pg_results.append(('PostgreSQL', 0, EXPERIMENTS[args.experiment], qid, args.mode, 0, pg_planning_time, pg_times[-3], None, None, None))
            pg_results.append(('PostgreSQL', 1, EXPERIMENTS[args.experiment], qid, args.mode, 0, pg_planning_time, pg_times[-2], None, None, None))
            pg_results.append(('PostgreSQL', 2, EXPERIMENTS[args.experiment], qid, args.mode, 0, pg_planning_time, pg_times[-1], None, None, None))
            # print(pg_times)
        else:
            pg_times = pg_df[pg_df.query_ident == qid].execution_time.values
            pg_planning_time = pg_df[pg_df.query_ident == qid].planning_time.values[0]
        
        inference_start = time.time()
        # input_enc = PlanEncoder.encode(jg, join_order=ordered_joins[qid])
        jg = SimpleJoinGraph(query, selectivity_extractor=selectivity_extractor, workload_name=WORKLOAD_NAME, database_graph=database_graph)
        # jg = JoinGraph(query, selectivity_extractor=selectivity_extractor, workload_name=WORKLOAD_NAME)
        input_enc = PlanEncoder.encode(jg)
        # print(input_enc.shape)
        query_enc = QueryEncoder.encode(jg)
        conf_dist = (torch.exp(torch.tensor([0.0])) - 1) / (np.exp(1) - 1)
        out_dist = (torch.exp(torch.tensor([0.0])) - 1) / (np.exp(0.025) - 1)
        # out, mu, logvar, z = model(input_enc.view(-1, len(input_enc)), query_enc.view(-1, len(query_enc)), conf_dist.view(-1, len(conf_dist)), out_dist.view(-1, len(out_dist)))
        # conf = clf(input_enc.view(-1, len(input_enc)), out, query_enc.view(-1, len(query_enc))) 
        # conf_dist_pred, out_dist_pred = conf.detach()[0][0].item(), conf.detach()[0][1].item()
        # hinted = hinter.vec_to_hint(jg, out.detach().numpy().flatten())
        for _ in range(25):
            out, mu, logvar, z = model(input_enc.view(-1, len(input_enc)), query_enc.view(-1, len(query_enc)), conf_dist.view(-1, len(conf_dist)), out_dist.view(-1, len(out_dist)))
            conf = clf(input_enc.view(-1, len(input_enc)), out, query_enc.view(-1, len(query_enc))) / 1e+6
            conf_dist_pred, out_dist_pred = conf.detach()[0][0].item(), conf.detach()[0][1].item()
            out_order = len(out) - (out.detach().flatten().numpy().argsort().argsort())
            input_enc = input_enc.bool() * positional_encoding(torch.tensor((1e+8*(1 - input_enc.bool().float()) + out_order).argsort().argsort()) + 1, norm=norm)
            counter += 1
            conf_dist_prev = conf_dist_pred
            out_dist_prev = out_dist_pred
            hinted = hinter.vec_to_hint(jg, input_enc.detach().numpy().flatten())
            inference_end = time.time()
            inference_time = (inference_end - inference_start) * 1000
            cvae_times, _, planning_time = gather_exec_time(hinted, executor, n_runs=N_RUNS,
                                             return_all=True, return_planning_time=True)
            cvae_results.append(('cVAE', 0, EXPERIMENTS[args.experiment], qid, args.mode, inference_time, planning_time, cvae_times[-3], conf_dist_pred, out_dist_pred, counter))
            cvae_results.append(('cVAE', 1, EXPERIMENTS[args.experiment], qid, args.mode, inference_time, planning_time, cvae_times[-2], conf_dist_pred, out_dist_pred, counter))
            cvae_results.append(('cVAE', 2, EXPERIMENTS[args.experiment], qid, args.mode, inference_time, planning_time, cvae_times[-1], conf_dist_pred, out_dist_pred, counter))
        
            print(f'{(np.mean(pg_times[-3:]) + pg_planning_time):.2f}', f'{(np.mean(cvae_times[-3:]) + planning_time + inference_time):.2f}',  f'{inference_time:.2f}', f'{conf_dist_pred:.2f}', f'{out_dist_pred:.2f}')

    temp = pd.DataFrame(cvae_results, columns=['method', 'run_id', 'experiment', 
                                               'query_ident', 'split', 'inference_time', 'planning_time', 'execution_time', 'conf_pred', 'out_pred', 'shot'])
    pg_df = pd.DataFrame(pg_results, columns=['method', 'run_id', 'experiment', 
                                               'query_ident', 'split', 'inference_time', 'planning_time', 'execution_time', 'conf_pred', 'out_pred', 'shot'])
    print(f'Number of decisions to keep a plan from PostgreSQL: {pg_decision_count}/{len(test_queries)}')
    print(f"Total Postgres time: {1/3 * (pg_df[pg_df.query_ident.isin(test_queries)]['execution_time'].sum() + pg_df[pg_df.query_ident.isin(test_queries)]['planning_time'].sum() + pg_df[pg_df.query_ident.isin(test_queries)]['inference_time'].sum()):.2f}ms")
    print(f"Total cVAE time: {1/3 * (temp[temp.method == 'cVAE']['execution_time'].sum() + temp[temp.method == 'cVAE']['planning_time'].sum() + temp[temp.method == 'cVAE']['inference_time'].sum()):.2f}ms")
    
    if args.dump:
        df = df[df.experiment != EXPERIMENTS[args.experiment]]
        df = pd.concat([df, pg_df, temp])
        df.to_csv(f'./data/processed/{SCHEMA_FOLDER}/{TEST_RESULTS}', index=False)
    
    executor.close_connection()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, required=True, help='Defines the worload to be used, i.e. as of now only `join_order_benchmark` for JOB or `stack` for STACK.')
    parser.add_argument('--experiment', type=str, required=True, help='Defines which train-test split to use (see #get_train_test_split method).')
    parser.add_argument('--dump', action='store_true', default=False, help='Saving the results')
    parser.add_argument('--pg_live', action='store_true', default=False, help='If to make live PostgreSQL runs for "fairness" of results comparison')
    parser.add_argument('--mode', type=str, default='test', help='Defines which dataset to predict on (`test` set is a default one).')
    args = parser.parse_args()

    main(args)
