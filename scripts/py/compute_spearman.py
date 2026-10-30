import os
import sys
import json
import logging
import argparse
from scipy import stats
import time
import numpy as np
from multiprocessing import Lock, Pool
import polars as pl

from pilotscope import PilotTransData
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from pilotscope.Factory.DBControllerFectory import DBControllerFactory
from pilotscope.DBController.BaseDBController import BaseDBController
from pilotscope.Common.CardMetricCalc import p_error_calc
from pilotscope.Common.Util import get_pg_hints

from typing import List, Tuple, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

from common.constants import DB_CONNECTION_ARGS
from common.sql_utils import UnionFind
from schema.schemas import get_schema
from schema.sql_schema import SQLSchema, SQLTable
from query.sql_query import SQLQuery
from common.hint import Hint


sys.path.append(os.path.join(parent_dir, "cpp", "build"))
from enumerate_plan import ScanMethod, enumerate_plan


_interactor_init_lock = None


def _init_worker(interactor_init_lock):
    """Give every pool worker the same lock for PilotScope initialization."""
    global _interactor_init_lock
    _interactor_init_lock = interactor_init_lock


def setup_logger(log_dir, query_set, estimator):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{query_set}_{estimator}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True,
    )


def get_data_interactor(db):
    user, password, host, port = DB_CONNECTION_ARGS["user"], DB_CONNECTION_ARGS["password"], DB_CONNECTION_ARGS["host"], DB_CONNECTION_ARGS["port"]
    config = PostgreSQLConfig(db_host=host, db_port=port, db_user=user, db_user_pwd=password, db=db)
    data_interactor = PilotDataInteractor(config)
    return data_interactor


def _process_plan_chunk(args: Tuple) -> List[Tuple]:
    """Process a chunk of plans. Each worker process owns one PilotDataInteractor."""
    chunk_data, db, sql, subquery_2_true_card, subquery_2_est_card, aliases = args

    logger = logging.getLogger(__name__)
    # PilotScope first probes a free HTTP port and then binds it in a separate
    # step. Serialize construction so concurrent workers cannot select the same
    # port during that race window.
    if _interactor_init_lock is None:
        data_interactor = get_data_interactor(db)
    else:
        with _interactor_init_lock:
            data_interactor = get_data_interactor(db)
    results = []

    for plan_idx, hint_init_pg, hint_init_leading, hint_init_access_method in chunk_data:
        hint_init = Hint(hint_init_leading, hint_init_access_method)

        data_interactor.push_card(subquery_2_true_card)
        data_interactor.push_pg_hint_comment(hint_init_pg)
        data_interactor.pull_physical_plan()
        data_interactor.pull_estimated_cost()
        result = data_interactor.execute(sql, is_reset=False)
        data_interactor._anchor_to_handlers.clear()
        true_physical_plan, true_cost = result.physical_plan, result.estimated_cost

        hint_new = Hint.from_pg_physical_plan_json(true_physical_plan)
        pg_hint_comment = get_pg_hints(true_physical_plan)

        data_interactor.push_card(subquery_2_est_card)
        data_interactor.push_pg_hint_comment(pg_hint_comment)
        data_interactor.pull_physical_plan()
        data_interactor.pull_estimated_cost()
        result = data_interactor.execute(sql, is_reset=False)
        data_interactor._anchor_to_handlers.clear()
        estimated_cost = result.estimated_cost

        hint_rejected = hint_init != hint_new
        if hint_rejected:
            pass
            # logger.warning(f"PostgreSQL reject the initial hint:\n{hint_init}\nIt adopts the new hint:\n{pg_hint_comment}")

        results.append((plan_idx, pg_hint_comment, true_cost, estimated_cost, hint_rejected))

    return results


def get_sql_along_with_subquery_est_card_true_card(experiment_dir) -> List[Tuple[int, str, Dict[str,float], Dict[str,float]]]:
    end2end_path = os.path.join(experiment_dir, "end2end.csv")
    estimation_path = os.path.join(experiment_dir, "estimation.csv")

    df_end2end = pl.read_csv(end2end_path)
    df_estimation = pl.read_csv(estimation_path)

    result = []
    for idx in df_end2end['index'].to_list():
        main_sql = df_end2end.filter(pl.col('index') == idx)['sql'][0]

        sub_rows = df_estimation.filter(pl.col('index') == idx)
        sub_sqls = sub_rows['sql'].to_list()
        sub_est_cards = sub_rows['estimated_card'].to_list()
        sub_true_cards = sub_rows['true_card'].to_list()

        subquery_2_est_card = dict(zip(sub_sqls, sub_est_cards))
        subquery_2_true_card = dict(zip(sub_sqls, sub_true_cards))

        result.append((idx, main_sql, subquery_2_est_card, subquery_2_true_card))

    return result



def get_query_relevent_info(query: SQLQuery):
    query_nodes = sorted(query.nodes(), key=lambda x: x.table().alias())
    aliases = [node.table().alias() for node in query_nodes]
    alias_to_idx = {alias: i for i, alias in enumerate(aliases)}
    name_to_alias = {node.table().name(): node.table().alias() for node in query_nodes}
    alias_to_relevant_cols = {alias: set() for alias in aliases}
    alias_to_filter_cols = {alias: set() for alias in aliases}
    join_keys = set()

    for node in query_nodes:
        alias = node.table().alias()
        for predicate in node.predicates():
            col = predicate.column.name()
            alias_to_relevant_cols[alias].add(col)
            alias_to_filter_cols[alias].add(col)

    for edge in query.edges():
        join_cond = str(edge)
        for key in join_cond.split(" = "):
            tb, col = key.strip().split('.')
            alias = name_to_alias[tb]
            join_keys.add((alias, col))
            alias_to_relevant_cols[alias].add(col)

    join_keys = list(join_keys)
    join_key_to_idx = {key: i for i, key in enumerate(join_keys)}

    uf = UnionFind(len(join_keys))
    for edge in query.edges():
        join_cond = str(edge)
        keys = [tuple(key.strip().split('.')) for key in join_cond.split(" = ")]
        idxs = [join_key_to_idx[(name_to_alias[tb], col)] for tb, col in keys]
        uf.union(idxs[0], idxs[1])
    equivalent_groups = [[join_keys[g] for g in group] for group in uf.equivalent_groups()]

    adjacency_list = [[] for _ in range(len(aliases))]

    for group in equivalent_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                alias_i, _ = group[i]
                alias_j, _ = group[j]
                idx_i = alias_to_idx[alias_i]
                idx_j = alias_to_idx[alias_j]
                adjacency_list[idx_i].append(idx_j)
                adjacency_list[idx_j].append(idx_i)

    supported_scan_methods = []
    support_bitmap_as_nlj_right = []
    for i, alias in enumerate(aliases):
        table: SQLTable = query_nodes[i].table()
        scan_methods = []
        scan_methods.append(ScanMethod.SeqScan)
        indexes = table.get_indexes()

        if indexes is not None and len(indexes) > 0:
            scan_methods.append(ScanMethod.IndexScan)

            can_index_only_scan = False
            can_bitmap_scan = False
            for index_columns in indexes.values():
                index_columns = set(index_columns)
                if all(col in index_columns for col in alias_to_relevant_cols[alias]):
                    can_index_only_scan = True
                    break
                if any(col in index_columns for col in alias_to_filter_cols[alias]):
                    can_bitmap_scan = True

            if can_index_only_scan:
                scan_methods.append(ScanMethod.IndexOnlyScan)
            if can_bitmap_scan:
                scan_methods.append(ScanMethod.BitmapScan)

            support_bitmap_as_nlj_right.append(not can_bitmap_scan)
        else:
            support_bitmap_as_nlj_right.append(False)

        supported_scan_methods.append(scan_methods)

    return adjacency_list, aliases, supported_scan_methods, support_bitmap_as_nlj_right


def get_all_plans_with_est_cost_and_true_cost(schema: SQLSchema, db: str, idx: int, sql: str,
                                             subquery_2_est_card: Dict[str, float],
                                             subquery_2_true_card: Dict[str, float], limit_plan_num: int = 100000,
                                             workers: int = 32) -> Tuple[List[str], List[float], List[float]]:
    logger = logging.getLogger(__name__)
    query = SQLQuery.parse(schema, sql)
    adjacency_list, aliases, supported_scan_methods, support_bitmap_as_nlj_right = get_query_relevent_info(query)
    plans = enumerate_plan(adjacency_list, limit_plan_num, aliases, supported_scan_methods, support_bitmap_as_nlj_right)
    logger.info(f"[{idx}] Enumerated {len(plans)} plans for query:\n{sql}")

    # Pre-compute hints
    plan_data = [(hint.to_pg_hint(), hint.leading_hint, hint.access_method)
                 for hint in (Hint.from_custom_plan(p, aliases) for p in plans)]

    n = len(plans)
    actual_workers = min(workers, n)
    chunk_size = (n + actual_workers - 1) // actual_workers
    chunks = []
    for w in range(actual_workers):
        start = w * chunk_size
        end = min(start + chunk_size, n)
        if start >= end:
            break
        chunk_data = [(i,) + plan_data[i] for i in range(start, end)]
        chunks.append((chunk_data, db, sql, subquery_2_true_card, subquery_2_est_card, aliases))

    start_time = time.perf_counter()
    interactor_init_lock = Lock()
    with Pool(processes=actual_workers, initializer=_init_worker,
              initargs=(interactor_init_lock,)) as pool:
        chunk_results = pool.map(_process_plan_chunk, chunks)

    results: List[Tuple] = [None] * n
    for chunk_result in chunk_results:
        for plan_idx, pg_hint_comment, true_cost, estimated_cost, hint_rejected in chunk_result:
            results[plan_idx] = (pg_hint_comment, true_cost, estimated_cost, hint_rejected)
    assert all(r is not None for r in results)

    latency_sec = time.perf_counter() - start_time

    pg_hint_comments = [r[0] for r in results]
    true_costs = [r[1] for r in results]
    est_costs = [r[2] for r in results]
    reject_count = sum(1 for r in results if r[3])

    logger.info(f"Query Plan Enumeration Complete. Total plans: {n}, Rejected plans: {reject_count}")
    logger.info(f"Total time: {latency_sec:.2f} seconds. Avg Time: {latency_sec / n:.4f} seconds per plan.")

    return pg_hint_comments, true_costs, est_costs


def main():
    parser = argparse.ArgumentParser(description='Get spearman correlation of plan ranking.')

    parser.add_argument('--query_set', type=str, choices=['job_light', 'stats_ceb', 'ergastf1', 'job_m_ours', 'job_large_ours'],
                        help='Test workload name', required=True)
    parser.add_argument('--estimator',
                        type=str,
                        choices=['postgresql', 'postgresql_ext', 'true', 'mscn', 'robustmscn', 'jgmp', 'alece', 'price_zs',
                                 'price_ft', 'deepdb', 'neurocard', 'factorjoin', 'bayescard', 'asm', 'lpbound'],
                        help='Cardinality estimator', required=True)
    parser.add_argument('--type', type=str, choices=['overall', 'training_data_efficiency', 'generalization', 'scalability',],
                        help='Experiment type', default='overall')
    parser.add_argument('--exp_name', type=str, default='default', help='Experiment name, will be used as the folder name')
    parser.add_argument('--log_dir', type=str, default='logs/spearman',
                        help='Directory for log output (default: logs/spearman)')
    parser.add_argument('--limit_plan_num', type=int, default=100000,
                        help='Limit the number of plans to enumerate (default: 100000)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel worker processes for plan evaluation (default: 32)')

    args = parser.parse_args()

    setup_logger(args.log_dir, args.query_set, args.estimator)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting calculation of spearman correlation for query set: {args.query_set}, estimator: {args.estimator}")
    logger.info(f"Limit plan number: {args.limit_plan_num}")

    experiment_dir = os.path.join(f"experiments/{args.type}", args.query_set, args.estimator, args.exp_name)

    schema_name = {
        'job_light': 'imdb',
        'job_m_ours': 'imdb_m',
        'job_large_ours': 'imdb_large',
        'stats_ceb': 'stats',
        'ergastf1': 'ergastf1'
    }[args.query_set]

    db = schema_name.split('_')[0]

    schema = get_schema(schema_name, DB_CONNECTION_ARGS['user'], DB_CONNECTION_ARGS['password'],
                        DB_CONNECTION_ARGS['host'], DB_CONNECTION_ARGS['port'])

    sql_and_subquery_card = get_sql_along_with_subquery_est_card_true_card(experiment_dir)

    rows = []
    for idx, sql, subquery_2_est_card, subquery_2_true_card in sql_and_subquery_card:
        # subquery spearman: true vs estimated cardinalities
        sub_sqls = list(subquery_2_true_card.keys())
        query_spearman = stats.spearmanr(
            [subquery_2_true_card[s] for s in sub_sqls],
            [subquery_2_est_card[s] for s in sub_sqls],
        ).correlation

        # plan spearman: true-cost vs estimated-cost plan rankings
        _, true_cost, est_cost = get_all_plans_with_est_cost_and_true_cost(
            schema, db, idx, sql, subquery_2_est_card, subquery_2_true_card, args.limit_plan_num, args.workers)
        plan_spearman = stats.spearmanr(true_cost, est_cost).correlation

        logger.info(f"SQL: {sql}\nSubquery spearman: {query_spearman}\nPlan spearman: {plan_spearman}\n")
        rows.append((idx, query_spearman, plan_spearman, sql))

    out_path = os.path.join(experiment_dir, "spearman.csv")
    pl.DataFrame(rows, schema=["index", "query_spearman", "plan_spearman", "sql"]).write_csv(out_path)
    logger.info(f"Saved to {out_path}")

    q_vals = np.array([r[1] for r in rows])
    p_vals = np.array([r[2] for r in rows])
    for name, vals in [("Subquery", q_vals), ("Plan", p_vals)]:
        logger.info(f"{name} Spearman: "
                    f"MIN={np.min(vals):.4f} "
                    f"5%={np.percentile(vals, 5):.4f} "
                    f"25%={np.percentile(vals, 25):.4f} "
                    f"50%={np.percentile(vals, 50):.4f} "
                    f"75%={np.percentile(vals, 75):.4f} "
                    f"95%={np.percentile(vals, 95):.4f} "
                    f"MAX={np.max(vals):.4f}")

if __name__ == '__main__':
    main()
