# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: workload-specific settings, JSONL training-data reconstruction,
# encoding caches, training metrics, and checkpoint output.

import os
import sys
sys.path.append(os.getcwd())
import random
import math
import json
import logging
import numpy as np
import pandas as pd
import networkx as nx
import argparse
import itertools
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.masked import masked_tensor, as_masked_tensor
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import random_split
from scipy import stats

from src.database.postgres_executor import PostgresExecutor
from src.database.query_loader import QueryLoader, WorkloadFileLoader
from src.database.utils import PlanParser
from src.database.selectivity_extractor import PostgresSelectivityExtractor, SelectivityExtractorCannotExtractException
from src.encoding.encoder import PSTHistogramEncoder, PSTEncoder, positional_encoding, QueryEncoder
from src.encoding.join_graph import JoinGraph, SimpleJoinGraph, JoinType, RedundantJoinException, JoinNotPossibleException
from src.database.utils import ParsingNotPossible
from src.models.cvae import CVAE_G
from src.utils.query_helper import parse_sql_query
from src.utils.path_helper import rl_model_folder_path, WORKLOADS
from src.utils.train_test_splits import get_train_test_split
from src.encoding.database_graph import DatabaseGraph
from src.utils.logging_config import setup_logging
from src.utils.path_helper import DB_CONFIG
from tqdm.auto import tqdm
import wandb

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='error', category=RuntimeWarning)

# NUM_POSSIBLE_JOINS = 53  JOB: 53, STACK: 32
NUM_JOIN_TYPES = 3
QUERY_ENCODING_SIZE = 3

DEMO_BASE_PATH = f'data/demonstrations/'
PROCESSED_BASE_PATH = f'data/processed/'
DEMO_TYPE = 'cvae'
LOGS_FILENAME = 'logs'

device = torch.device("cuda")
softmax = nn.Softmax(dim=1)

NEW_WORKLOADS = ('job_light', 'stats_ceb', 'ergastf1')


def get_num_possible_joins(workload_name):
    """Read join_conditions count from predicates.json."""
    schema_folder = WORKLOADS[workload_name]
    pred_path = f'workloads/{schema_folder}/predicates.json'
    with open(pred_path, 'r') as f:
        pred = json.load(f)
    return len(pred['join_conditions'])


def reconstruct_query_enc(alias_selectivities, join_graph):
    """Reconstruct QueryEncoder.encode() output from stored alias_selectivities."""
    join_graph.alias_selectivities = alias_selectivities
    return QueryEncoder.encode(join_graph)


def reconstruct_plan_enc(flat_idx_list, join_graph, norm=100):
    """
    Reconstruct PlanEncoder.encode() output from compact flat_idx list.

    flat_idx_list: sorted by rank ascending. Each element = state_edge_idx * 3 + join_type.
    Length K = number of participating undirected edges in this query.
    """
    NUM_JOIN_TYPES = 3
    num_edges = join_graph.num_undirected_edges
    enc = np.zeros((num_edges, NUM_JOIN_TYPES))

    available_edges = join_graph.get_available_joins(remove_duplicates=False)
    participating = sorted(available_edges.state_edge_idx.unique())
    K = len(participating)
    assert len(flat_idx_list) == K, \
        f"flat_idx length {len(flat_idx_list)} != participating edges {K}"

    available_joins = {e: [0, 1, 2] for e in participating}
    ranks = list(range(1, K * 3 + 1))

    # Fill priority picks from flat_idx list (position = rank - 1)
    for pos, fidx in enumerate(flat_idx_list):
        e = fidx // NUM_JOIN_TYPES
        jt = fidx % NUM_JOIN_TYPES
        available_joins[e].remove(jt)
        ranks.remove(pos + 1)
        enc[e, jt] = positional_encoding(pos + 1, norm=norm)

    # Step 3: remaining 2K positions get random ranks (fixed seed for reproducibility)
    np.random.seed(1234)
    for e, remaining in available_joins.items():
        for jt in remaining:
            idx = np.random.randint(0, len(ranks))
            enc[e, jt] = positional_encoding(ranks.pop(idx), norm=norm)

    return torch.from_numpy(enc.flatten()).to(dtype=torch.float32)


def kld_gaussians(mu1, std1, mu2, std2):
    """
    KL divergence between 2 Gaussians: N(mu1, std1**2) and N(mu2, std2**2)
    
    Source: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    return torch.mean(torch.log(std2 / std1) + (std1**2 + (mu1 - mu2)**2) / (2*std2**2) - 1/2)

def loss_function(loss_fn, pred, out, mu, logvar):
    """
    Batch-mean reconstruction loss, KL divergence, and target entropy.
    
    See Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    mask = (out > 0).detach()
    target = softmax(out.masked_fill(~mask, float('-inf')))
    ce = loss_fn(pred, target)
    target_entropy = -(target * target.clamp_min(torch.finfo(target.dtype).tiny).log()) \
        .sum(dim=1).mean()
    std = torch.exp(0.5*logvar)
    KLD_latent = kld_gaussians(mu, std, 0, 1)
    return ce, KLD_latent, target_entropy

def train(epoch, train_loader, model, optimizerG, loss_G):
    model.train()
    totals = {
        'loss': 0.0,
        'ce': 0.0,
        'kl': 0.0,
        'target_entropy': 0.0,
    }
    num_samples = 0
    for batch_idx, (i_enc, q_enc, o_enc, p_dist, o_dist, c_dist) in enumerate(train_loader):
        i_enc, q_enc, o_enc, p_dist, o_dist, c_dist = i_enc.to(device), q_enc.to(device), o_enc.to(device), p_dist.to(device), o_dist.to(device), c_dist.to(device)
        ## Updating generator ##
        optimizerG.zero_grad()
        o_pred, mu, logvar, z = model(i_enc, q_enc, c_dist, o_dist)
        ce, kl, target_entropy = loss_function(loss_G, o_pred, o_enc, mu, logvar)
        errG = ce + kl
        errG.backward()
        optimizerG.step()

        batch_size = i_enc.size(0)
        num_samples += batch_size
        totals['loss'] += errG.item() * batch_size
        totals['ce'] += ce.item() * batch_size
        totals['kl'] += kl.item() * batch_size
        totals['target_entropy'] += target_entropy.item() * batch_size

    metrics = {name: value / num_samples for name, value in totals.items()}
    metrics['ce_excess'] = metrics['ce'] - metrics['target_entropy']
    metrics['learning_rate'] = optimizerG.param_groups[0]['lr']
    logging.info(
        f"Epoch:{epoch} Loss:{metrics['loss']:.8f} CE:{metrics['ce']:.8f} "
        f"KL:{metrics['kl']:.10f} TargetEntropy:{metrics['target_entropy']:.8f} "
        f"CE-Entropy:{metrics['ce_excess']:.8f} LR:{metrics['learning_rate']:.8g}"
    )
    return metrics

def main(args):

    torch.manual_seed(1234)
    np.random.seed(1234)

    WORKLOAD_NAME = args.workload
    SCHEMA_FOLDER = WORKLOADS[WORKLOAD_NAME]
    IS_NEW_WORKLOAD = WORKLOAD_NAME in NEW_WORKLOADS

    setup_logging('training', log_dir='logs')
    logging.info(f"Training start: workload={WORKLOAD_NAME}, experiment={args.experiment}, run_id={args.run_id}")

    pg_df = pd.read_csv(os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, 'postgres_execution_time.csv'))

    config = {'join_order_benchmark': {
                    'NUM_POSSIBLE_JOINS': 53,
                    'batch_size': 128,
                    'max_batch_size': 2**12,
                    'lr': 1e-3,
                    'epochs': 150,
                    'latent_size': 100,
                    'layers_size': [275, 225],
                    'step_size': 25
                    },
              'stack': {
                    'NUM_POSSIBLE_JOINS': 32,
                    'batch_size': 128,
                    'max_batch_size': 2**12,
                    'lr': 1e-3,
                    'epochs': 150,
                    'latent_size': 50,
                    'layers_size': [75, 60],
                    'step_size': 25
              },
              'job_light': {
                    'NUM_POSSIBLE_JOINS': get_num_possible_joins('job_light'),
                    'batch_size': 128,
                    'max_batch_size': 2**12,
                    'lr': 1e-3,
                    'epochs': 150,
                    'latent_size': 100,
                    'layers_size': [275, 225],
                    'step_size': 25,
                    'norm': 23,
              },
              'stats_ceb': {
                    'NUM_POSSIBLE_JOINS': get_num_possible_joins('stats_ceb'),
                    'batch_size': 128,
                    'max_batch_size': 2**12,
                    'lr': 1e-3,
                    'epochs': 150,
                    'latent_size': 100,
                    'layers_size': [275, 225],
                    'step_size': 25,
                    'norm': 24,
              },
              'ergastf1': {
                    'NUM_POSSIBLE_JOINS': get_num_possible_joins('ergastf1'),
                    'batch_size': 64,
                    'max_batch_size': 2**11,
                    'lr': 1e-3,
                    'epochs': 150,
                    'latent_size': 100,
                    'layers_size': [275, 225],
                    'step_size': 25,
                    'norm': 28,
              }
    }

    all_query_enc = []
    all_input_enc = []
    all_out_enc = []
    all_pg_dist = []
    all_out_dist = []
    all_conf_dist = []

    train_queries, _ = get_train_test_split(args.experiment, workload_name=WORKLOAD_NAME)
    logging.info(f"Experiment: {args.experiment}, train queries: {len(train_queries)}")

    cache_dir = os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, args.experiment)
    cache_filenames = (
        'query_enc.pt', 'input_enc.pt', 'out_enc.pt',
        'pg_dist.pt', 'out_dist.pt', 'conf_dist.pt',
    )
    cache_is_valid = os.path.isdir(cache_dir) and all(
        os.path.isfile(os.path.join(cache_dir, filename))
        for filename in cache_filenames
    )
    norm = config[WORKLOAD_NAME].get('norm')
    cache_config_path = os.path.join(cache_dir, 'encoding_config.json')
    if cache_is_valid and IS_NEW_WORKLOAD:
        try:
            with open(cache_config_path, 'r') as f:
                cache_config = json.load(f)
            cache_is_valid = cache_config.get('norm') == norm
        except (FileNotFoundError, json.JSONDecodeError):
            cache_is_valid = False

        if not cache_is_valid:
            logging.info(f"Encoding cache does not match norm={norm}; rebuilding.")

    if cache_is_valid:
        query_enc = torch.load(os.path.join(cache_dir, 'query_enc.pt'))
        input_enc = torch.load(os.path.join(cache_dir, 'input_enc.pt'))
        out_enc = torch.load(os.path.join(cache_dir, 'out_enc.pt'))
        pg_dist = torch.load(os.path.join(cache_dir, 'pg_dist.pt'))
        out_dist = torch.load(os.path.join(cache_dir, 'out_dist.pt'))
        conf_dist = torch.load(os.path.join(cache_dir, 'conf_dist.pt'))
    elif IS_NEW_WORKLOAD:
        # Load from JSON LINE compact format
        os.makedirs(cache_dir, exist_ok=True)
        jsonl_path = os.path.join(DEMO_BASE_PATH, SCHEMA_FOLDER, 'train_data.jsonl')

        # Build SQL loader for per-query SQL access
        sql_loader = WorkloadFileLoader(WORKLOAD_NAME,
                                        os.path.join('workloads', SCHEMA_FOLDER),
                                        file_pattern='train_workload.sql')
        pg_time_map = (
            pg_df.assign(query_ident=pg_df['query_ident'].astype(str))
            .groupby('query_ident', sort=False)['execution_time']
            .apply(lambda times: times.to_numpy(dtype=np.float64))
            .to_dict()
        )
        
        train_qids_set = set(train_queries)
        
        # Construct relevant classes
        executor = PostgresExecutor.create_postgres_engine(
            local_port=DB_CONFIG[WORKLOAD_NAME]['DB_PORT'],
            db_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
            statement_timeout=1*60*1000
        )
        selectivity_extractor = PostgresSelectivityExtractor(workload_name=WORKLOAD_NAME,
                                                         database_name=DB_CONFIG[WORKLOAD_NAME]['DB_NAME'],
                                                         executor=executor)
        database_graph = DatabaseGraph(selectivity_extractor=selectivity_extractor,
                                    workload_name=WORKLOAD_NAME)
        
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc='Loading training data...'):
                record = json.loads(line.strip())
                qid = record['qid']
                qid_str = str(qid)

                if qid_str not in train_qids_set:
                    continue

                alias_sels = record['alias_selectivities']
                plans = record['plans']

                if len(plans) < 2:
                    continue

                # Rebuild join_graph from SQL to get encoding structure
                query = sql_loader.get_query(qid)
                jg = SimpleJoinGraph(query, workload_name=WORKLOAD_NAME,
                                        database_graph=database_graph,
                                        selectivity_extractor=selectivity_extractor)
                
                query_enc = reconstruct_query_enc(alias_sels, jg)

                all_exec_times = torch.tensor([p['exec_times'] for p in plans], dtype=torch.float32)
                exec_time = torch.mean(all_exec_times, axis=1)
                exec_time, indices = torch.sort(exec_time, descending=True)

                pg_times_list = pg_time_map.get(qid_str)
                if pg_times_list is None:
                    continue

                for in_pos, out_pos in itertools.combinations(range(len(exec_time)), 2):
                    if exec_time[in_pos] <= exec_time[out_pos]:
                        continue
                    try:
                        _, od = stats.ttest_ind(
                            all_exec_times[indices[in_pos]].numpy(),
                            all_exec_times[indices[out_pos]].numpy(),
                            alternative='greater')
                    except RuntimeWarning:
                        od = 1.0
                    if math.isnan(od):
                        od = 1.0
                    try:
                        _, pd_ = stats.ttest_ind(
                            all_exec_times[indices[in_pos]].numpy(),
                            pg_times_list,
                            alternative='greater')
                    except RuntimeWarning:
                        pd_ = 1.0
                    if math.isnan(pd_):
                        pd_ = 1.0
                    try:
                        _, cd = stats.ttest_ind(
                            pg_times_list,
                            all_exec_times[indices[out_pos]].numpy(),
                            alternative='greater')
                    except RuntimeWarning:
                        cd = 1.0
                    if math.isnan(cd):
                        cd = 1.0

                    if od < 0.025:
                        all_query_enc.append(query_enc.view(1, -1))
                        all_input_enc.append(
                            reconstruct_plan_enc(plans[indices[in_pos].item()]['flat_idx'], jg, norm=norm).view(1, -1))
                        all_out_enc.append(
                            reconstruct_plan_enc(plans[indices[out_pos].item()]['flat_idx'], jg, norm=norm).view(1, -1))
                        all_pg_dist.append(torch.tensor([pd_]).float().view(1, -1))
                        all_out_dist.append(torch.tensor([od]).float().view(1, -1))
                        all_conf_dist.append(torch.tensor([cd]).float().view(1, -1))

        if len(all_query_enc) == 0:
            logging.error("No training pairs generated. Check JSONL data and PG baseline.")
            return

        query_enc = torch.cat(all_query_enc)
        input_enc = torch.cat(all_input_enc)
        out_enc = torch.cat(all_out_enc)
        pg_dist = torch.cat(all_pg_dist)
        out_dist = torch.cat(all_out_dist)
        conf_dist = torch.cat(all_conf_dist)

        torch.save(query_enc, os.path.join(cache_dir, 'query_enc.pt'))
        torch.save(input_enc, os.path.join(cache_dir, 'input_enc.pt'))
        torch.save(out_enc, os.path.join(cache_dir, 'out_enc.pt'))
        torch.save(pg_dist, os.path.join(cache_dir, 'pg_dist.pt'))
        torch.save(out_dist, os.path.join(cache_dir, 'out_dist.pt'))
        torch.save(conf_dist, os.path.join(cache_dir, 'conf_dist.pt'))
        with open(cache_config_path, 'w') as f:
            json.dump({'norm': norm}, f)
    else:
        os.makedirs(os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, args.experiment))
        for train_query in tqdm(train_queries, desc="Processing training queries ...", total=len(train_queries)):
            # counter = 0
            query_enc = torch.load(os.path.join(DEMO_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, train_query, 'query_enc.pt'))
            plan_enc = torch.load(os.path.join(DEMO_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, train_query, 'plan_enc.pt'))
            exec_times = torch.load(os.path.join(DEMO_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, train_query, 'exec_times.pt'))
            exec_time = torch.mean(exec_times, axis=1)
            exec_time, indices = torch.sort(exec_time, descending=True)
            plan_enc = plan_enc[indices]

            pg_times = pg_df[pg_df.query_ident == f'{train_query}'].execution_time.values

            for in_id, out_id in itertools.combinations(np.arange(0, len(exec_time) - 1), 2):
                # Problem with calculating p-values when data is close to identical
                try:
                    _, out_dist = stats.ttest_ind(exec_times[in_id].numpy(), exec_times[out_id].numpy(),
                                                alternative='greater')
                except RuntimeWarning:
                    out_dist = 1.0
                if math.isnan(out_dist):
                    out_dist = 1.0
                try:
                    _, pg_dist = stats.ttest_ind(exec_times[in_id].numpy(), pg_times,
                                                alternative='greater')
                except RuntimeWarning:
                    pg_dist = 1.0
                if math.isnan(pg_dist):
                    pg_dist = 1.0
                try:
                    _, conf_dist = stats.ttest_ind(pg_times, exec_times[out_id].numpy(),
                                                alternative='greater')
                except RuntimeWarning:
                    conf_dist = 1.0
                if math.isnan(conf_dist):
                    conf_dist = 1.0

                if out_dist < 0.025:
                    all_query_enc.append(query_enc.view(1, -1))
                    all_input_enc.append(plan_enc[in_id].view(1, -1))
                    all_out_enc.append(plan_enc[out_id].view(1, -1))
                    all_pg_dist.append(torch.tensor([pg_dist]).float().view(1, -1))
                    all_out_dist.append(torch.tensor([out_dist]).float().view(1, -1))
                    all_conf_dist.append(torch.tensor([conf_dist]).float().view(1, -1))

        query_enc = torch.cat(all_query_enc)
        input_enc = torch.cat(all_input_enc)
        out_enc = torch.cat(all_out_enc)
        pg_dist = torch.cat(all_pg_dist)
        out_dist = torch.cat(all_out_dist)
        conf_dist = torch.cat(all_conf_dist)

        torch.save(query_enc, os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, args.experiment, 'query_enc.pt'))
        torch.save(input_enc, os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, args.experiment, 'input_enc.pt'))
        torch.save(out_enc, os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, args.experiment, 'out_enc.pt'))
        torch.save(pg_dist, os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, args.experiment, 'pg_dist.pt'))
        torch.save(out_dist, os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, args.experiment, 'out_dist.pt'))
        torch.save(conf_dist, os.path.join(PROCESSED_BASE_PATH, SCHEMA_FOLDER, DEMO_TYPE, args.experiment, 'conf_dist.pt'))
        with open(cache_config_path, 'w') as f:
            json.dump({'norm': norm}, f)


    logging.info(f"Train data size: {len(query_enc)}")

    model_name = f'cvae_{args.experiment}.pt'

    out_dist = (torch.exp(out_dist) - 1) / (np.exp(0.025) - 1)
    pg_dist = (torch.exp(pg_dist) - 1) / (np.exp(1) - 1)
    conf_dist = (torch.exp(conf_dist) - 1) / (np.exp(1) - 1)

    train_data = torch.utils.data.TensorDataset(input_enc, query_enc, out_enc, pg_dist, out_dist, conf_dist)
    if IS_NEW_WORKLOAD:
        # Use all training data, no split
        pass
    elif WORKLOAD_NAME == 'join_order_benchmark':
        train_ratio = 0.3
    elif WORKLOAD_NAME == 'stack':
        train_ratio = 0.45
    if not IS_NEW_WORKLOAD:
        train_len = int(len(train_data)*train_ratio)
        train_data, _ = random_split(train_data, [train_len, len(train_data) - train_len])
    logging.info(f"Train data size after filtering: {len(train_data)}")
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config[WORKLOAD_NAME]['batch_size'],
                                                   shuffle=True, drop_last=True)

    model = CVAE_G(config[WORKLOAD_NAME]['NUM_POSSIBLE_JOINS'] * NUM_JOIN_TYPES,
                 config[WORKLOAD_NAME]['latent_size'],
                 config[WORKLOAD_NAME]['layers_size'],
                 config[WORKLOAD_NAME]['NUM_POSSIBLE_JOINS'] * QUERY_ENCODING_SIZE).to(device)

    optimizerG = torch.optim.RMSprop(model.parameters(), lr=config[WORKLOAD_NAME]['lr'])
    schedulerG = StepLR(optimizerG, step_size=config[WORKLOAD_NAME]['step_size'], gamma=0.75)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, config[WORKLOAD_NAME]['epochs'] + 1):
        train(epoch, train_dataloader, model, optimizerG, ce_loss)
        schedulerG.step()
        if epoch % 25 == 0 or epoch == config[WORKLOAD_NAME]['epochs']:
            model_dir = os.path.join('models', SCHEMA_FOLDER, f'{DEMO_TYPE}', f'run{args.run_id}')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model, os.path.join(model_dir, model_name))

    logging.info(f"Training done. Model saved: {os.path.join('models', SCHEMA_FOLDER, f'{DEMO_TYPE}', f'run{args.run_id}', model_name)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, required=True, help='Defines the worload to be used, i.e. as of now only `join_order_benchmark` for JOB or `stack` for STACK.')
    parser.add_argument('--experiment', type=str, default="default", required=False, help='Defines which train-test split to use (see #get_train_test_split method).')
    parser.add_argument('--threshold', type=float, default=None, help='Defines filter threshold on significance (p-value) between 2 execution times.')
    parser.add_argument('--run_id', type=int, default=0, help='The run_id that is to be used for testing of N models for the purposes of stability.')
    args = parser.parse_args()

    main(args)
    
# python scripts/training/train_cvae.py --workload job_light/stats_ceb/ergastf1
