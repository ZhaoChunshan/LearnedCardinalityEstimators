# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: disable an unavailable visualization import and use the local
# PostgreSQL baseline filename.

import os
import sys
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

from src.database.postgres_executor import PostgresExecutor
from src.database.query_loader import QueryLoader
from src.encoding.join_graph import JoinGraph, JoinType, RedundantJoinException, JoinNotPossibleException
from src.encoding.encoder import PSTEncoder, PlanEncoder, QueryEncoder, positional_encoding
from src.encoding.filters import FiltersProcessor
from src.database.utils import PlanParser, PlanHinter, gather_join_order
from src.database.selectivity_extractor import PostgresSelectivityExtractor
from src.utils.query_helper import parse_sql_query
from src.utils.path_helper import WORKLOADS, DB_CONFIG
from src.utils.train_test_splits import get_train_test_split
# from src.utils.viz import plot_exec_time_difference
from scripts.generating.create_full_plan_data import generate_encoding, gather_exec_time
from src.models.cvae import CVAE

from tqdm.auto import tqdm
import wandb

TEST_RESULTS = 'full_execution_time.csv' 
LOGS_FILENAME = 'logs'
TIMEOUT = 1.5 * 60 * 1000
EXPERIMENTS = {
    'LeaveOneOutSplit1': 'leave_one_out_split_1', 
    'LeaveOneOutSplit2': 'leave_one_out_split_2',
    'LeaveOneOutSplit3': 'leave_one_out_split_3',
    'RandomSplit1': 'random_split_1',
    'RandomSplit2': 'random_split_2',
    'RandomSplit3': 'random_split_3',
    'BaseQuerySplit1': 'base_query_split_1', 
    'BaseQuerySplit2': 'base_query_split_2',
    'BaseQuerySplit3': 'base_query_split_3'
}


def main(args):
    
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    WORKLOAD_NAME = args.workload  #  'join_order_benchmark', 'stack'
    SCHEMA_FOLDER = WORKLOADS[WORKLOAD_NAME]
    
    executor = PostgresExecutor.create_postgres_engine(local_port=DB_CONFIG[WORKLOAD_NAME]['DB_PORT'],
                                                       db_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
                                                       statement_timeout=TIMEOUT)
    selectivity_extractor = PostgresSelectivityExtractor(workload_name=WORKLOAD_NAME,
                                                         executor=executor)
    hinter = PlanHinter()
    ql = QueryLoader(WORKLOAD_NAME)
    _, test_queries = get_train_test_split(args.experiment, workload_name=WORKLOAD_NAME)
    print(args.experiment)
    
    model = torch.load(f'./models/{SCHEMA_FOLDER}/cvae/cvae_{args.experiment}.pt')
    model.eval()
    model.to('cpu')
    
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
    divs = dict()
    confs = dict()

    diff_test_data = []

    for qid in test_queries:
        print(qid)
        for diff in torch.linspace(0, 0.025, 20):
            query = ql.get_query(qid)
            jg = JoinGraph(query, selectivity_extractor=selectivity_extractor, workload_name=WORKLOAD_NAME)
            query_enc = QueryEncoder.encode(jg)
            # diff = torch.tensor([0])
            # diff = torch.rand(1) * 0.025
            if args.pg_live:
                joins, pg_times, pg_planning_time = gather_join_order(query, executor=executor, for_postgres=True,
                                                                    statement_timeout=TIMEOUT, n_runs=5, return_planning_time=True)
                ordered_joins[qid] = joins
                pg_results.append(('PostgreSQL', 0, EXPERIMENTS[args.experiment], qid, 'test', 0, pg_planning_time, pg_times[-3]))
                pg_results.append(('PostgreSQL', 1, EXPERIMENTS[args.experiment], qid, 'test', 0, pg_planning_time, pg_times[-2]))
                pg_results.append(('PostgreSQL', 2, EXPERIMENTS[args.experiment], qid, 'test', 0, pg_planning_time, pg_times[-1]))
                # print(pg_times)
            else:
                pg_times = pg_df[pg_df.query_ident == qid].execution_time.values
                pg_planning_time = pg_df[pg_df.query_ident == qid].planning_time.values[0]

            input_enc = PlanEncoder.encode(jg, join_order=ordered_joins[qid])
            # input_enc = PlanEncoder.encode(jg)
            # input_enc = torch.where(input_enc < 0, 0.0, input_enc)
            input_enc = input_enc.view(-1, len(input_enc))
            inference_start = time.time()
            for _ in range(1):
                test_data = torch.utils.data.TensorDataset(input_enc, 
                                                        query_enc.view(-1, len(query_enc)),
                                                        diff.view(-1, len(diff)))
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1)
                outs = []
                # pvalues = torch.rand(100) * 0.0025
                # consts = torch.tensor(np.linspace(-6, 6, 50))  # 50
                # indicators = torch.tensor(np.linspace(0, 1, 20))  # 20
                for batch_idx, (i_enc, q_enc, diff) in enumerate(test_dataloader):
                    # vec, _ = consts.repeat(len(indicators)).sort()
                    # eps = (torch.ones([model.latent_size, len(consts) * len(indicators)]) * vec).T
                    # margin = indicators.repeat(len(consts)).float().view(-1, 1)
                    # i_enc = i_enc.repeat(pvalues.size(0), 1)
                    # q_enc = q_enc.repeat(pvalues.size(0), 1)
                    out_mask = torch.where(q_enc == 0, 0.0, 1.0)
                    # diff = diff.repeat(eps.size(0), 1)
                    out, lambda_ = model(i_enc, q_enc, diff)  # pvalues.view(-1, 1)
                    # conf = torch.sum(out * out_mask, axis=1).detach().numpy()
                    # max_ = torch.max(out[out_mask.bool()].view(len(diff), -1), axis=1).values
                    # min_ = torch.min(out[out_mask.bool()].view(len(diff), -1), axis=1).values
                    # div = (max_ - min_).detach().numpy()
                    # divs[qid] = div
                    # confs[qid] = conf
                    # outs = [(out_, conf_) for out_, conf_ in zip(out, conf)]
                # outs.sort(key=lambda x: x[1], reverse=True)
                # out = outs[0][0]
                # input_enc = out * i_enc.bool()
            inference_end = time.time()
            inference_time = (inference_end - inference_start) * 1000
            # if False:
            # if torch.sum(out).detach().item() < 0:
            # # if stats.ttest_1samp(divs[qid], np.median(divs[qid]), alternative='less').pvalue < 0.1:
            #     pg_decision_count += 1
            #     cvae_times = pg_times
            #     planning_time = pg_planning_time
            # else:
            hinted = hinter.vec_to_hint(jg, out.detach().numpy().flatten())
            cvae_times, _, planning_time = gather_exec_time(hinted, executor, statement_timeout=TIMEOUT, n_runs=5,
                                            return_all=True, return_planning_time=True)
            # TODO: add Postgres planning time as a part of inference as based on Postgres plan 
            cvae_results.append(('cVAE', 0, EXPERIMENTS[args.experiment], qid, 'test', inference_time, planning_time, cvae_times[-3]))
            cvae_results.append(('cVAE', 1, EXPERIMENTS[args.experiment], qid, 'test', inference_time, planning_time, cvae_times[-2]))
            cvae_results.append(('cVAE', 2, EXPERIMENTS[args.experiment], qid, 'test', inference_time, planning_time, cvae_times[-1]))
            # print(cvae_times)
            # plot_data.append((qid, pg_df[(pg_df.query_ident == qid) & (pg_df.run_id == 2)]['execution_time'].values[0], cvae_times[-1]))
            if args.pg_live:
                cvae_results.extend(pg_results)
            print(f'{(np.mean(pg_times[-3:]) + pg_planning_time):.2f}', f'{(np.mean(cvae_times[-3:]) + planning_time + inference_time):.2f}')

            diff_test_data.append({
                'diff': diff.item(),
                'query': qid,
                'inference_time': inference_time,
                'planning_time': planning_time,
                'execution_time': cvae_times[-1],
            })
            pd.DataFrame(diff_test_data).to_csv('./data/processed/diff_test_data.csv', index=False)

    # plot_df = pd.DataFrame(plot_data, columns=['qid', 'pg_time', 'cvae_time'])
    # plot_df['query'] = plot_df['qid'].str.replace('.sql','')
    # plot_df['base_query'] = plot_df['query'].apply(lambda x: int(x[:-1]))
    # plot_df['variant'] = plot_df['query'].apply(lambda x: x[-1])
    # plot_df.drop(columns=['query'], inplace=True)
    # plot_df = plot_df.sort_values(['base_query', 'variant']).reset_index(drop=True)
    temp = pd.DataFrame(cvae_results, columns=['method', 'run_id', 'experiment', 
                                               'query_ident', 'split', 'inference_time', 'planning_time', 'execution_time'])
    pg_df = pd.DataFrame(pg_results, columns=['method', 'run_id', 'experiment', 
                                               'query_ident', 'split', 'inference_time', 'planning_time', 'execution_time'])
    print(f'Number of decisions to keep a plan from PostgreSQL: {pg_decision_count}/{len(test_queries)}')
    print(f"Total Postgres time: {1/3 * (pg_df[pg_df.query_ident.isin(test_queries)]['execution_time'].sum() + pg_df[pg_df.query_ident.isin(test_queries)]['planning_time'].sum() + pg_df[pg_df.query_ident.isin(test_queries)]['inference_time'].sum()):.2f}ms")
    print(f"Total cVAE time: {1/3 * (temp[temp.method == 'cVAE']['execution_time'].sum() + temp[temp.method == 'cVAE']['planning_time'].sum() + temp[temp.method == 'cVAE']['inference_time'].sum()):.2f}ms")
    
    if args.dump:
        df = df[df.experiment != EXPERIMENTS[args.experiment]]
        df = pd.concat([df, temp])
        df.to_csv(f'./data/processed/{SCHEMA_FOLDER}/full_execution_time.csv', index=False)
        
    # 2024-03-02 suli: otherwise it continues hanging in a script
    sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, required=True, help='Defines the worload to be used, i.e. as of now only `join_order_benchmark` for JOB or `stack` for STACK.')
    parser.add_argument('--experiment', type=str, required=True, help='Defines which train-test split to use (see #get_train_test_split method).')
    parser.add_argument('--dump', action='store_true', default=False, help='Saving the results')
    parser.add_argument('--pg_live', action='store_true', default=False, help='If to make live PostgreSQL runs for "fairness" of results comparison')
    args = parser.parse_args()

    main(args)
    
    # 2024-03-02 suli: otherwise it continues hanging in a script
    sys.exit()

