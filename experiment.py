import os
import copy
import time
import json
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from typing import List, Optional, Dict
from query.sql_query import SQLQuery
from query.workload import load_workload
from schema.schemas import imdb_schema, stats_schema, ergastf1_schema, imdb_m_schema, imdb_large_schema
from schema.sql_schema import SQLSchema
from common.logger import Logger
from common.card_utils import q_error_value
from common.sql_utils import get_tables_from_comment, single_table_estimation


db_connection_args = {
    "user": "pilotscope",
    "password": "pilotscope",
    "host": os.environ.get("PGHOST", "localhost"),
    "port": int(os.environ.get("PGPORT", "5432")),
}


def load_config(estimator: str):
    estimator = 'price' if estimator.startswith('price') else estimator
    
    config_path = "./config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    estimator_config = {}
    for k, v in config.items():
        if k != estimator:
            # add global config
            if not isinstance(v, dict):
                estimator_config[k] = v

    # config for this estimator
    if estimator in config:
        estimator_config.update(config[estimator])

    if "sample_bitmap" not in estimator_config or "sample_number" not in estimator_config:
        estimator_config["sample_bitmap"] = False
    if "build_histogram" not in estimator_config:
        estimator_config["build_histogram"] = False

    return estimator_config


def load_schema(db: str, query_set: str, sample_number: Optional[int] = None):
    user, password, host, port = db_connection_args["user"], db_connection_args["password"], db_connection_args["host"], db_connection_args["port"]
   
    if db == "imdb":
        if query_set == "job_light":
            schema = imdb_schema(username=user, password=password, database_name=db, host=host, port=port, sample_size=sample_number)
        elif query_set == "job_m_ours":
            schema = imdb_m_schema(username=user, password=password, database_name=db, host=host, port=port, sample_size=sample_number)
        elif query_set == "job_large_ours":
            schema = imdb_large_schema(username=user, password=password, database_name=db, host=host, port=port, sample_size=sample_number)
        else:
            raise ValueError(f"Invalid query set {query_set}")
    elif db.startswith("stats"):
        schema = stats_schema(username=user, password=password, database_name=db, host=host, port=port, sample_size=sample_number)
    elif db == "ergastf1":
        schema = ergastf1_schema(username=user, password=password, database_name=db, host=host, port=port, sample_size=sample_number)
    return schema

def train_estimator(estimator_name: str, estimator_config: dict, query_set: str, exp_type: str, exp_name: str, schema: SQLSchema, dir: str, statistics: Dict = dict()):
    is_query_driven = estimator_name in ["mscn", "mscn_hybrid", "robustmscn", "jgmp", "alece", "price_ft"]

    if is_query_driven:
        scale_card = None
        if exp_type == 'scalability' and query_set == 'stats_ceb':
            assert(exp_name in ['sf10', 'sf100'])
            scale_card = int(exp_name[2:])

        if exp_type in ['overall', 'training_data_efficiency', 'scalability',]:
            train_workload = load_workload(query_set, "train")
            valid_workload = load_workload(query_set, "valid")
        elif exp_type == 'generalization':
            assert(exp_name in ['template', 'small'])
            train_workload = load_workload(query_set, "train_" + exp_name)
            valid_workload = load_workload(query_set, "valid_" + exp_name)
        
        alias_dict = {table.alias(): table.name() for table in schema.tables()}

        scale = scale_card if scale_card is not None else 1

        def prepare(records):
            queries = []
            true_cards = []
            postgres_cards = []
            for record in records:
                subquery_sqls = [subquery.sql for subquery in record.subqueries]
                subquery_cards = [
                    subquery.postgres_cardinality * scale
                    for subquery in record.subqueries
                ]
                query = SQLQuery.parse(schema, record.sql).canonicalize()
                query.add_single_table_estimation(
                    single_table_estimation(
                        alias_dict, subquery_sqls, subquery_cards
                    )
                )
                queries.append(query)
                true_cards.append(record.true_cardinality * scale)
                postgres_cards.append(record.postgres_cardinality * scale)
            return queries, true_cards, postgres_cards

        train_queries, train_cardinality, train_pg_est = prepare(train_workload)
        valid_queries, valid_cardinality, valid_pg_est = prepare(valid_workload)
        
        if exp_type == 'training_data_efficiency':
            query_num = int(exp_name)
            assert(query_num > 0 and query_num < len(train_workload))
            print(f"Sample {query_num} training queries.")
            
            random.seed(8888)
            indices = random.sample(range(len(train_workload)), query_num)
            train_queries = [train_queries[i] for i in indices]
            train_cardinality = [train_cardinality[i] for i in indices]
            train_pg_est = [train_pg_est[i] for i in indices]

    if estimator_name == "mscn":
        from estimators.mscn.api import MSCNEstimator
        estimator = MSCNEstimator(schema, estimator_config, dir)
        estimator._statistics.update(statistics)
        estimator.train(train_queries, train_cardinality, valid_queries, valid_cardinality)
        estimator.save()
    elif estimator_name == "mscn_hybrid":
        from estimators.mscn_hybrid.api import MSCNHybridEstimator
        estimator = MSCNHybridEstimator(schema, estimator_config, dir)
        estimator._statistics.update(statistics)
        estimator.train(train_queries, train_cardinality, valid_queries, valid_cardinality)
        estimator.save()
    elif estimator_name == "robustmscn":
        from estimators.robustmscn.api import RobustMSCNEstimator
        estimator = RobustMSCNEstimator(schema, estimator_config, dir)
        estimator._statistics.update(statistics)
        estimator.train(train_queries, train_pg_est, train_cardinality, valid_queries, valid_pg_est, valid_cardinality)
        estimator.save()
    elif estimator_name == "jgmp":
        from estimators.jgmp.api import JGMPEstimator
        estimator = JGMPEstimator(schema, estimator_config, dir)
        estimator._statistics.update(statistics)
        estimator.train(train_queries, train_cardinality, valid_queries, valid_cardinality)
        estimator.save()
    elif estimator_name == "alece":
        from estimators.alece.api import ALECEEstimator
        estimator = ALECEEstimator(schema, estimator_config, dir)
        estimator._statistics.update(statistics)
        histogram = np.expand_dims(schema.histogram_feature(), 0)
        estimator.train(train_queries, histogram, np.array([0]), train_cardinality,
                        valid_queries, histogram, np.array([0]), valid_cardinality)
        estimator.save()
    elif estimator_name == "price_zs":
        from estimators.price.api import PRICEEstimator
        estimator = PRICEEstimator(schema, estimator_config, dir)
        estimator.save()
    elif estimator_name == "price_ft":
        from estimators.price.api import PRICEEstimator
        estimator = PRICEEstimator(schema, estimator_config, dir)
        estimator.train(train_queries, train_pg_est, train_cardinality, valid_queries, valid_pg_est, valid_cardinality)
        estimator.save()
    elif estimator_name == "deepdb":
        from estimators.deepdb.api import DeepDBEstimator
        estimator = DeepDBEstimator(schema, estimator_config, dir)
        estimator.train()   # The deepdb will automatically be saved after training.
        estimator.save()
    elif estimator_name == "neurocard":
        from estimators.neurocard.api import NeuroCardEstimator
        estimator = NeuroCardEstimator(schema, estimator_config, dir)
        estimator.train()
        estimator.save()
    elif estimator_name == "factorjoin":
        from estimators.factorjoin.api import FactorJoinEstimator
        estimator = FactorJoinEstimator(schema, estimator_config, dir)
        estimator.train()
        estimator.save()
    elif estimator_name == "bayescard":
        from estimators.bayescard.api import BayesCardEstimator
        estimator = BayesCardEstimator(schema, estimator_config, dir)
        estimator.train()
        estimator.save()
    elif estimator_name == "asm":
        from estimators.asm.api import ASMEstimator
        estimator = ASMEstimator(schema, estimator_config, dir)
        estimator.train()
        estimator.save()
    elif estimator_name == "asm_hybrid":
        # No training; loads model from sibling asm/ directory
        from estimators.asm_hybrid.api import ASMHybridEstimator
        # Check sibling asm dir exists
        parent = os.path.dirname(dir)
        exp_name = os.path.basename(dir)
        asm_dir = os.path.join(os.path.dirname(parent), "asm", exp_name)
        if not os.path.exists(asm_dir):
            raise FileNotFoundError(f"ASM model not found at {asm_dir}. Run 'asm' experiment first.")
        estimator = ASMHybridEstimator(schema, estimator_config, dir)
        estimator.save()
    elif estimator_name == "flat":
        from estimators.flat.api import FlatEstimator
        estimator = FlatEstimator(schema, estimator_config, dir)
        estimator.train()
        estimator.save()
    else:
        raise NotImplementedError


def estimator_predict(
        estimator_name: str,
        query_set: str,
        schema: SQLSchema,
        dir: str,
        device: str = None,
        price_statistics_folder: str = None
    ):
    if estimator_name == "postgresql" or estimator_name == "true":
        pass
    elif estimator_name == "mscn":
        from estimators.mscn.api import MSCNEstimator
        estimator = MSCNEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "mscn_hybrid":
        from estimators.mscn_hybrid.api import MSCNHybridEstimator
        estimator = MSCNHybridEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "robustmscn":
        from estimators.robustmscn.api import RobustMSCNEstimator
        estimator = RobustMSCNEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "jgmp":
        from estimators.jgmp.api import JGMPEstimator
        estimator = JGMPEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "alece":
        from estimators.alece.api import ALECEEstimator
        estimator = ALECEEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "price_zs" or estimator_name == "price_ft":
        from estimators.price.api import PRICEEstimator
        estimator = PRICEEstimator.load_from_dir(
            schema,
            dir,
            device=device,
            price_statistics_folder=price_statistics_folder
        )
    elif estimator_name == "deepdb":
        from estimators.deepdb.api import DeepDBEstimator
        estimator = DeepDBEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "neurocard":
        from estimators.neurocard.api import NeuroCardEstimator
        estimator = NeuroCardEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "factorjoin":
        from estimators.factorjoin.api import FactorJoinEstimator
        estimator = FactorJoinEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "bayescard":
        from estimators.bayescard.api import BayesCardEstimator
        estimator = BayesCardEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "asm":
        from estimators.asm.api import ASMEstimator
        estimator = ASMEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "asm_hybrid":
        from estimators.asm_hybrid.api import ASMHybridEstimator
        estimator = ASMHybridEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "flat":
        from estimators.flat.api import FlatEstimator
        estimator = FlatEstimator.load_from_dir(schema, dir, device=device)
    else:
        raise NotImplementedError
    
    scale_card = None
    if query_set == 'stats_ceb' and schema.name().startswith('stats_sf'):
        scale_card = int(schema.name()[len('stats_sf'):])
    
    workload = load_workload(query_set, "test")
    scale = scale_card if scale_card is not None else 1
    queries_id = [query.query_id for query in workload]
    subqueries, subqueries_pg_est = [], []
    subqueries_true_card, subqueries_is_outer_most, subqueries_sql = [], [], []

    need_canonicalize = estimator_name not in ["asm", "asm_hybrid", "factorjoin"]
    
    alias_dict = {table.alias(): table.name() for table in schema.tables()}

    for workload_query in workload:
        subquery_sqls = [subquery.sql for subquery in workload_query.subqueries]
        pg_estimates = [
            subquery.postgres_cardinality * scale
            for subquery in workload_query.subqueries
        ]
        true_cards = [
            subquery.true_cardinality * scale
            for subquery in workload_query.subqueries
        ]
        single_table_est = single_table_estimation(alias_dict, subquery_sqls, pg_estimates)
        if need_canonicalize:
            subqueries.append([SQLQuery.parse(schema, sql).canonicalize().add_single_table_estimation(single_table_est) for sql in subquery_sqls])
        else:
            subqueries.append([SQLQuery.parse(schema, sql).add_single_table_estimation(single_table_est) for sql in subquery_sqls])
        outer_size = max([len(get_tables_from_comment(sql)) for sql in subquery_sqls])
        subqueries_is_outer_most.append([outer_size == len(get_tables_from_comment(sql)) for sql in subquery_sqls])
        subqueries_pg_est.append(pg_estimates)
        subqueries_true_card.append(true_cards)
        subqueries_sql.append(subquery_sqls)

    # Now, estimate the cardinality for each subquery
    subqueries_est_card = []
    subqueries_est_time = []

    if estimator_name == "postgresql":
        subqueries_est_card = subqueries_pg_est
        subqueries_est_time = [[0.0] * len(subquery) for subquery in subqueries]
    elif estimator_name == "true":
        subqueries_est_card = subqueries_true_card
        subqueries_est_time = [[0.0] * len(subquery) for subquery in subqueries]
    elif estimator_name in ["mscn", "jgmp"]:
        for cur_subqueries in subqueries:
            est_card, est_time = estimator.predict(cur_subqueries)
            subqueries_est_card.append(est_card)
            subqueries_est_time.append(est_time)
    elif estimator_name in ["robustmscn", "price_zs", "price_ft", "mscn_hybrid"]:
        for cur_subqueries, cur_subqueries_pg_est in zip(subqueries, subqueries_pg_est):
            est_card, est_time = estimator.predict(cur_subqueries, cur_subqueries_pg_est)
            subqueries_est_card.append(est_card)
            subqueries_est_time.append(est_time)
    elif estimator_name == "alece":
        for cur_subqueries in subqueries:
            est_card, est_time = estimator.predict(cur_subqueries, schema.histogram_feature())
            subqueries_est_card.append(est_card)
            subqueries_est_time.append(est_time)
    elif estimator_name == "asm_hybrid":
        subqueries_est_card, subqueries_est_time = estimator.predict(subqueries, subqueries_pg_est)
    elif estimator_name in ["neurocard", "factorjoin", "bayescard", "asm", "flat", "deepdb"]:
        subqueries_est_card, subqueries_est_time = estimator.predict(subqueries)
    else:
        raise NotImplementedError

    cardinality_result = {
        "index": [],
        "is_outer_most": [],
        "estimated_card": [],
        "true_card": [],
        "q_error": [],
        "estimation_time": [],
        "query_id": [],
        "sql": []
    }

    q_error_subqueries = []
    q_error_queries = []
    total_time = 0
    query_num = len(workload)

    for idx, (query_id, cur_subqueries_is_outer_most, cur_subqueries_true_card, cur_subqueries_est_card, cur_subqueries_est_time, cur_subqueries_sql) \
        in enumerate(zip(queries_id, subqueries_is_outer_most, subqueries_true_card, subqueries_est_card, subqueries_est_time, subqueries_sql)):
        for pred_card, true_card, time, is_outer_most, sql in zip(cur_subqueries_est_card, cur_subqueries_true_card, \
                                                                  cur_subqueries_est_time, cur_subqueries_is_outer_most, cur_subqueries_sql):
            q_error = q_error_value(pred_card, true_card)
            cardinality_result["index"].append(idx)
            cardinality_result["is_outer_most"].append(is_outer_most)
            cardinality_result["estimated_card"].append(pred_card)
            cardinality_result["true_card"].append(true_card)
            cardinality_result["q_error"].append(q_error)
            cardinality_result["estimation_time"].append(time)
            cardinality_result["query_id"].append(query_id)
            cardinality_result["sql"].append(sql)
            if is_outer_most:
                q_error_queries.append(q_error)
            q_error_subqueries.append(q_error)
            total_time += time

    df = pd.DataFrame(cardinality_result)
    df.to_csv(os.path.join(dir, "estimation.csv"), index=False)

    q_error_subqueries = np.array(q_error_subqueries)
    q_error_queries = np.array(q_error_queries)

    log = Logger(os.path.join(dir, "predict.log"), mode='a')
    log.write("\n====================================== Cardinality Estimation Result ======================================\n")
    log.write("estimator: %s, query_set: %s, query_num: %d, avg_time: %f\n" % (estimator_name, query_set, query_num, total_time / query_num))
    
    log.write("Subqueries Q-error Percentile\n50%% %.3f\n80%% %.3f\n90%% %.3f\n95%% %.3f\n99%% %.3f\nmax %.3f\n"
              %(
                  np.percentile(q_error_subqueries, 50),
                  np.percentile(q_error_subqueries, 80),
                  np.percentile(q_error_subqueries, 90),
                  np.percentile(q_error_subqueries, 95),
                  np.percentile(q_error_subqueries, 99),
                  q_error_subqueries.max(),
              ))

    log.write("Queries Q-error Percentile\n50%% %.3f\n80%% %.3f\n90%% %.3f\n95%% %.3f\n99%% %.3f\nmax %.3f\n"
              %(
                  np.percentile(q_error_queries, 50),
                  np.percentile(q_error_queries, 80),
                  np.percentile(q_error_queries, 90),
                  np.percentile(q_error_queries, 95),
                  np.percentile(q_error_queries, 99),
                  q_error_queries.max(),
              ))

    log.close()


def construct_statistics_over_dynamic_data(estimator_name: str, schema: SQLSchema, estimator_config: Dict, init_data_perc: int):
    def construct_price_statistics(schema: SQLSchema, perc: int):
        from synopsis.price_statistics import PRICEStatistics
        
        db = schema.name()
        assert db in ('stats', 'imdb')
        
        table_csv_path = f"datasets/{db}/snapshots/{perc}"

        statistics_dir = os.path.join(table_csv_path, "statistics/price")
        
        os.makedirs(statistics_dir, exist_ok=True)
        
        filenames = [f for f in os.listdir(statistics_dir) if f.endswith('.pkl')]
        
        if not all(f in filenames for f in ['abbrev_col_type.pkl', 'histogram40.pkl', 'fanout40.pkl', 'size.pkl', 'summary40.pkl']):
        
            statistics = PRICEStatistics.construct_statistics(schema, table_csv_path, 40)
        
            statistics.save(statistics_dir)
        
        else:
            statistics = PRICEStatistics.load(statistics_dir)
        
        return statistics
    
    def construct_robustmscn_join_samples(schema: SQLSchema, perc: int, estimator_config: Dict):
        user, password, host, port = db_connection_args["user"], db_connection_args["password"], db_connection_args["host"], db_connection_args["port"]
        db = schema.name()
        assert db in ('stats', 'imdb')
        
        join_samples_path = f"datasets/{db}/snapshots/{perc}/statistics/robustmscn/join_samples.pkl"
        
        os.makedirs(os.path.dirname(join_samples_path), exist_ok=True)
        
        if os.path.exists(join_samples_path):
            with open(join_samples_path, "rb") as f:
                join_samples = pickle.load(f)
            return join_samples

        if db == 'imdb':
            schema_perc = imdb_schema(username=user, password=password, database_name=f"{db}_p{perc}", host=host, port=port)
        else:
            schema_perc = stats_schema(username=user, password=password, database_name=f"{db}_p{perc}", host=host, port=port)
        
        from synopsis.join_bitmap import JoinBitmap
        
        join_samples = JoinBitmap(schema_perc, estimator_config["sample_number"], estimator_config["foreign_table_ratio"])
        join_samples.update(schema_perc)
        
        with open(join_samples_path, "wb") as f:
            pickle.dump(join_samples, f)
        
        return join_samples          
        
    
    def construct_alece_histograms(schema: SQLSchema, perc: int):
        db = schema.name()
        assert db in ('stats', 'imdb')
        
        table_csv_dir = f"datasets/{db}/snapshots/{perc}"
        histograms_dir = os.path.join(table_csv_dir, "statistics/alece")
        
        os.makedirs(histograms_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(histograms_dir, "histograms.pkl")):
            with open(os.path.join(histograms_dir, "histograms.pkl"), "rb") as f:
                histograms = pickle.load(f)
            return histograms
        
        histograms = {}
        
        for table in schema.tables():
            if table.histogram() is None:
                continue
            
            histogram = copy.deepcopy(table.histogram())
            
            df = pd.read_csv(os.path.join(table_csv_dir, f"{table.name()}.csv"), sep='|', na_values=['', 'NULL'])
            
            histogram._n_rows = df.shape[0]
            
            pos_attr_dict = {i: attr for attr, i in histogram.attr_pos_map().items()}
            
            initial_data = np.zeros(shape=(histogram._n_attrs, len(df)), dtype=np.float32)
            boundary_points = []
            
            for i in range(histogram._n_attrs):
                initial_data[i] = np.sort(df[pos_attr_dict[i]].values)
            
                arange_idxes = np.reshape(np.arange(1, histogram._n_bins + 1, dtype=np.float32), [1, -1])
                boundary_points.append(arange_idxes)
                
            boundary_points = np.concatenate(boundary_points, axis=0)
            boundary_points *= np.reshape(histogram._bin_sizes, [-1, 1])
            boundary_points += np.reshape(histogram._min_vals, [-1, 1])
            boundary_points[:, -1] += 1e6

            new_histogram = []
            
            for i in range(histogram._n_attrs):
                
                sorted_idxes = np.searchsorted(initial_data[i], boundary_points[i], side='left')
                
                assert sorted_idxes[-1] == initial_data[i].shape[0] or np.isnan(initial_data[i,sorted_idxes[-1]])
            
                sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
                sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
                sorted_idxes_left_translation[0] = 0
                diff = sorted_idxes - sorted_idxes_left_translation
                new_histogram.append(np.reshape(diff, [1, -1]))
                
            new_histogram = np.concatenate(new_histogram, axis=0, dtype=np.float32)
            histogram._histogram = new_histogram
            
            histograms[table.name()] = histogram
                
        with open(os.path.join(histograms_dir, "histograms.pkl"), "wb") as f:
            pickle.dump(histograms, f)
            
        return histograms
    
    if estimator_name == 'alece':
        for i in range(1, 10 + 1):
            construct_alece_histograms(schema, i * init_data_perc // 10)
    elif estimator_name == 'robustmscn':
        for i in range(1, 10 + 1):
            construct_robustmscn_join_samples(schema, i * init_data_perc // 10, estimator_config)
    elif estimator_name.startswith('price'):
        for i in range(1, 10 + 1):
            construct_price_statistics(schema, i * init_data_perc // 10)
    else:
        raise ValueError("Unknown estimator name: %s" % estimator_name)
    

def read_training_workload_over_dynamic_data(queryset: str, init_data_perc: int):
    def read_training_workload_from_file(path):
        sqls, true_cards, estimated_cards, subquery_est_cards = [], [], [], []
        with open(path, "r") as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                j = i + 1
                while j < len(lines):
                    if not lines[j].startswith("/*"):
                        break
                    j += 1
            
                tokens = lines[i].strip().split("||")
                sql, true_card, est_card = tokens[0], float(tokens[1]), float(tokens[2])
                
                subquery_est_dict = {}
                for k in range(i + 1, j):
                    tokens = lines[k].strip().split("||")
                    subquery_est_dict[tokens[0]] = float(tokens[1])
                
                sqls.append(sql)
                true_cards.append(true_card)
                estimated_cards.append(est_card)
                subquery_est_cards.append(subquery_est_dict)
                
                i = j
        
        return sqls, true_cards, estimated_cards, subquery_est_cards
    
    result_dict = {}
    
    for i in range(1, 10 + 1):
        perc = i * init_data_perc // 10
        path = f"workloads/{queryset}/dynamic/{perc}/train_workload.sql"
        sqls, true_cards, estimated_cards, subquery_est_cards = read_training_workload_from_file(path)
        result_dict[perc] = (sqls, true_cards, estimated_cards, subquery_est_cards)
    
    return result_dict


def read_test_workload_postgres_est_over_dynamic_data(queryset: str, init_data_perc: int):
    with open(f"workloads/{queryset}/dynamic/{init_data_perc}/{queryset}_all.sql", "r") as f:
        lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
        subquery_pg_estimates = []
        
        i = 0
        while i < len(lines):
            j = i + 1
            while j < len(lines):
                if not lines[j].startswith("/*"):
                    break
                j += 1
            
            sub_sqls, pg_estimates = [], []
            for k in range(i, j):
                tokens = lines[k].strip().split("||")
                sub_sqls.append(tokens[0])
                pg_estimates.append(float(tokens[1]))
            
            est_dict = {sub_sql: est for sub_sql, est in zip(sub_sqls, pg_estimates)}
            subquery_pg_estimates.append(est_dict)
            i = j

    return subquery_pg_estimates
    
def train_and_update_estimator(estimator_name: str, estimator_config: dict, schema: SQLSchema, dir: str, exp_name: str, statistics: Dict = dict()):
    perc = int(exp_name.split("_")[0])
    option = None if "_" not in exp_name else exp_name.split("_")[1]
    estimator_config["update_strategy"] = option
    estimator_config['perc'] = perc
    
    if estimator_name in ('robustmscn', 'alece', 'price_ft'):
        construct_statistics_over_dynamic_data(estimator_name, schema, estimator_config, perc)
        queryset = 'job_light' if schema.name() == 'imdb' else 'stats_ceb'
        workload_each_perc = read_training_workload_over_dynamic_data(queryset, perc)
    
    if estimator_name == "deepdb":
        from estimators.deepdb.api import DeepDBEstimator
        estimator = DeepDBEstimator(schema, estimator_config, dir)
        estimator.train_update()
    elif estimator_name == "neurocard":
        from estimators.neurocard.api import NeuroCardEstimator
        estimator = NeuroCardEstimator(schema, estimator_config, dir)
        estimator.train_update()
    elif estimator_name == "bayescard":
        from estimators.bayescard.api import BayesCardEstimator
        estimator = BayesCardEstimator(schema, estimator_config, dir)
        estimator.train_update()
    elif estimator_name == "factorjoin":
        from estimators.factorjoin.api import FactorJoinEstimator
        estimator = FactorJoinEstimator(schema, estimator_config, dir)
        estimator.train_update()
    elif estimator_name == "asm":
        from estimators.asm.api import ASMEstimator
        estimator = ASMEstimator(schema, estimator_config, dir)
        estimator.train_update()
    elif estimator_name == "robustmscn":
        from estimators.robustmscn.api import RobustMSCNEstimator
        estimator = RobustMSCNEstimator(schema, estimator_config, dir)
        estimator.train_update(workload_each_perc)
        estimator.save()
    elif estimator_name == "alece":
        from estimators.alece.api import ALECEEstimator
        estimator = ALECEEstimator(schema, estimator_config, dir)
        estimator._statistics.update(statistics)
        estimator.train_update(workload_each_perc)
        estimator.save()
    elif estimator_name == 'price_ft':
        from estimators.price.api import PRICEEstimator
        estimator = PRICEEstimator(schema, estimator_config, dir)
        estimator.train_update(workload_each_perc)
        estimator.save()
    else:
        raise NotImplementedError
    

    
def updated_estimator_predict(estimator_name: str, query_set: str, schema: SQLSchema, dir: str, exp_name: str, device: str = None):
    assert estimator_name in ["deepdb", "neurocard", "bayescard", "factorjoin", "asm", "robustmscn", "alece", "price_zs", "price_ft"]
    perc = int(exp_name.split("_")[0])
    assert perc in [20, 40, 60, 80]

    if estimator_name == "deepdb":
        from estimators.deepdb.api import DeepDBEstimator
        estimator = DeepDBEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "neurocard":
        from estimators.neurocard.api import NeuroCardEstimator
        estimator = NeuroCardEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "bayescard":
        from estimators.bayescard.api import BayesCardEstimator
        estimator = BayesCardEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "factorjoin":
        from estimators.factorjoin.api import FactorJoinEstimator
        estimator = FactorJoinEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "asm":
        from estimators.asm.api import ASMEstimator
        estimator = ASMEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "robustmscn":
        from estimators.robustmscn.api import RobustMSCNEstimator
        estimator = RobustMSCNEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name == "alece":
        from estimators.alece.api import ALECEEstimator
        estimator = ALECEEstimator.load_from_dir(schema, dir, device=device)
    elif estimator_name in ('price_zs', 'price_ft'):
        from estimators.price.api import PRICEEstimator
        estimator = PRICEEstimator.load_from_dir(schema, dir, device=device)
    else:
        raise NotImplementedError
    
    
    workload = load_workload(query_set, "test")
    queries_id = [query.query_id for query in workload]
    subqueries, subqueries_pg_est, subqueries_pg_est_stale = [], [], []
    subqueries_true_card, subqueries_is_outer_most, subqueries_sql = [], [], []
    single_table_est_all, single_table_est_stale_all = [], []
    
    # Read the PostgreSQL estimates over the stale snapshot at `perc`.
    subqueries_pg_est_dict_stale = read_test_workload_postgres_est_over_dynamic_data(query_set, perc)
    
    need_canonicalize = estimator_name not in ["asm", "asm_hybrid", "factorjoin"]

    alias_dict = {table.alias(): table.name() for table in schema.tables()}
    
    for workload_query, pg_estimates_dict_stale in zip(workload, subqueries_pg_est_dict_stale):
        subquery_sqls = [subquery.sql for subquery in workload_query.subqueries]
        pg_estimates = [subquery.postgres_cardinality for subquery in workload_query.subqueries]
        pg_estimates_stale = [pg_estimates_dict_stale[sql] for sql in subquery_sqls]
        single_table_est = single_table_estimation(alias_dict, subquery_sqls, pg_estimates)
        single_table_est_stale = single_table_estimation(alias_dict, subquery_sqls, pg_estimates_stale)
        if need_canonicalize:
            subqueries.append([SQLQuery.parse(schema, sql).canonicalize().add_single_table_estimation(single_table_est) for sql in subquery_sqls])
        else:
            subqueries.append([SQLQuery.parse(schema, sql).add_single_table_estimation(single_table_est) for sql in subquery_sqls])
        outer_size = max([len(get_tables_from_comment(sql)) for sql in subquery_sqls])
        subqueries_is_outer_most.append([outer_size == len(get_tables_from_comment(sql)) for sql in subquery_sqls])
        subqueries_pg_est.append(pg_estimates)
        subqueries_pg_est_stale.append(pg_estimates_stale)
        single_table_est_all.append(single_table_est)
        single_table_est_stale_all.append(single_table_est_stale)
        subqueries_true_card.append([
            subquery.true_cardinality for subquery in workload_query.subqueries
        ])
        subqueries_sql.append(subquery_sqls)
    
    # Now, estimate the cardinality for each subquery
    if estimator_name in ["deepdb", "neurocard", "bayescard", "factorjoin", "asm", "alece"]:
        subqueries_est_card_stale, subqueries_est_time_stale, subqueries_est_card_updated, \
                subqueries_est_time_updated = estimator.predict_update(subqueries)
    elif estimator_name == "robustmscn":
        subqueries_est_card_stale, subqueries_est_time_stale, subqueries_est_card_updated, \
                subqueries_est_time_updated = estimator.predict_update(subqueries, subqueries_pg_est,
                                                                       subqueries_pg_est_stale,
                                                                       single_table_est_all,
                                                                       single_table_est_stale_all)
    elif estimator_name == "price_ft":
        subqueries_est_card_stale, subqueries_est_time_stale, subqueries_est_card_updated, \
                subqueries_est_time_updated = estimator.predict_update(subqueries, subqueries_pg_est,
                                                                       subqueries_pg_est_stale)
    else:
        raise NotImplementedError

    for i, (subqueries_est_card, subqueries_est_time) in enumerate([(subqueries_est_card_stale, subqueries_est_time_stale),
                                                      (subqueries_est_card_updated, subqueries_est_time_updated)]):
        cardinality_result = {
            "index": [],
            "is_outer_most": [],
            "estimated_card": [],
            "true_card": [],
            "q_error": [],
            "estimation_time": [],
            "query_id": [],
            "sql": []
        }

        q_error_subqueries = []
        q_error_queries = []
        total_time = 0
        query_num = len(workload)
        

        for idx, (query_id, cur_subqueries_is_outer_most, cur_subqueries_true_card, cur_subqueries_est_card, cur_subqueries_est_time, cur_subqueries_sql) \
            in enumerate(zip(queries_id, subqueries_is_outer_most, subqueries_true_card, subqueries_est_card, subqueries_est_time, subqueries_sql)):
            
            for pred_card, true_card, time, is_outer_most, sql in zip(cur_subqueries_est_card, cur_subqueries_true_card, \
                                                                      cur_subqueries_est_time, cur_subqueries_is_outer_most, cur_subqueries_sql):
                q_error = q_error_value(pred_card, true_card)
                cardinality_result["index"].append(idx)
                cardinality_result["is_outer_most"].append(is_outer_most)
                cardinality_result["estimated_card"].append(pred_card)
                cardinality_result["true_card"].append(true_card)
                cardinality_result["q_error"].append(q_error)
                cardinality_result["estimation_time"].append(time)
                cardinality_result["query_id"].append(query_id)
                cardinality_result["sql"].append(sql)
                if is_outer_most:
                    q_error_queries.append(q_error)
                q_error_subqueries.append(q_error)
                total_time += time

        df = pd.DataFrame(cardinality_result)
        if i == 0:
            df.to_csv(os.path.join(dir, "estimation_stale.csv"), index=False)
        else:
            df.to_csv(os.path.join(dir, "estimation_updated.csv"), index=False)

        q_error_subqueries = np.array(q_error_subqueries)
        q_error_queries = np.array(q_error_queries)

        log = Logger(os.path.join(dir, "predict.log"), mode='a')
        if i == 0:
            log.write("\n====================================== Cardinality Estimation Result (Stale) ======================================\n")
        else:
            log.write("\n====================================== Cardinality Estimation Result (Updated) ======================================\n")
        log.write("estimator: %s, query_set: %s, query_num: %d, avg_time: %f\n" % (estimator_name, query_set, query_num, total_time / query_num))
        
        log.write("Subqueries Q-error Percentile\n50%% %.3f\n80%% %.3f\n90%% %.3f\n95%% %.3f\n99%% %.3f\nmax %.3f\n"
                %(
                    np.percentile(q_error_subqueries, 50),
                    np.percentile(q_error_subqueries, 80),
                    np.percentile(q_error_subqueries, 90),
                    np.percentile(q_error_subqueries, 95),
                    np.percentile(q_error_subqueries, 99),
                    q_error_subqueries.max(),
                ))

        log.write("Queries Q-error Percentile\n50%% %.3f\n80%% %.3f\n90%% %.3f\n95%% %.3f\n99%% %.3f\nmax %.3f\n"
                %(
                    np.percentile(q_error_queries, 50),
                    np.percentile(q_error_queries, 80),
                    np.percentile(q_error_queries, 90),
                    np.percentile(q_error_queries, 95),
                    np.percentile(q_error_queries, 99),
                    q_error_queries.max(),
                ))

    log.close()


def run_cardinality_estimation_experiment(estimator: str, db: str, query_set: str, exp_type: str, exp_name: str, load_dir = None, device: str = None):
    if load_dir is not None:
        print("Cardinality Estimation: Load trained model from %s" % load_dir)
    elif estimator not in ["postgresql", "true"]:
        print("Cardinality Estimation: Train new model %s" % estimator)
    
    config = load_config(estimator)
    if estimator not in ["postgresql", "true"] and device is None:
        raise ValueError("--device is required for learned cardinality estimators")
    if device is not None:
        config["device"] = device

    schema = load_schema(db, query_set, config["sample_number"] if config["sample_bitmap"] else None)
    if config["build_histogram"]:
        start_histogram = time.perf_counter()
        schema.build_histogram(config["n_bins"])
        schema._statistics['build_histogram_time'] = time.perf_counter() - start_histogram
    
    if estimator.startswith('price'):
        from synopsis.price_statistics import PRICEStatistics
        
        statistics_folder = f"datasets/imdb/price_statistics/{query_set}" if db.startswith('imdb') else f"datasets/{db}/price_statistics"
        print(f"statistics_folder: {statistics_folder}")
        os.makedirs(statistics_folder, exist_ok=True)
        
        filenames = [f for f in os.listdir(statistics_folder) if f.endswith('.pkl')]
        if not all(f in filenames for f in ['abbrev_col_type.pkl', 'histogram40.pkl', 'fanout40.pkl', 'size.pkl', 'summary40.pkl']):
        
            statistics = PRICEStatistics.construct_statistics(schema, f"datasets/{db}", 40)
        
            statistics.save(statistics_folder)
        
        config['price_statistics_folder'] = statistics_folder
        
    if load_dir is not None:
        if exp_type == 'dynamic':
            updated_estimator_predict(estimator, query_set, schema, load_dir, exp_name, device=device)
        else:
            estimator_predict(
                estimator,
                query_set,
                schema,
                load_dir,
                device=device,
                price_statistics_folder=config.get("price_statistics_folder")
            )
    else:
        if exp_name is None:
            # Dead code, impossible branch, will not used now
            datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            experiment_dir = f"experiments/{exp_type}/{query_set}/{estimator}/{datetime_str}"
        else:
            experiment_dir = f"experiments/{exp_type}/{query_set}/{estimator}/{exp_name}"

        if os.path.exists(experiment_dir):
            shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"Experiment result save to {experiment_dir}")
        if exp_type != 'dynamic':
            if not estimator in ["postgresql", "true",]:
                train_estimator(estimator, config, query_set, exp_type, exp_name, schema, experiment_dir, statistics=schema._statistics)
            estimator_predict(
                estimator,
                query_set,
                schema,
                experiment_dir,
                device=device,
                price_statistics_folder=config.get("price_statistics_folder")
            )
        else:
            train_and_update_estimator(estimator, config, schema, experiment_dir, exp_name, statistics=schema._statistics)
            updated_estimator_predict(estimator, query_set, schema, experiment_dir, exp_name, device=device)
    
    schema.close()


def warmup(db: str, query_set: str):
    import psycopg2
    user, password, host, port = db_connection_args["user"], db_connection_args["password"], db_connection_args["host"], db_connection_args["port"]
    warm_up_path = os.path.join("./workloads", query_set, "warmup.sql")
    conn = psycopg2.connect(host=host, database=db, user=user, password=password, port=port)
    conn.autocommit = True
    cur = conn.cursor()
    with open(warm_up_path, "r") as f:
        cur.execute(f.read())
    conn.close()


def run_end_to_end_experiment(estimator: str, db: str, query_set: str, exp_type: str, exp_name: str, 
                              query_timeout_seconds: int, workload_timeout_seconds: int,
                              load_dir: str = None, stale: bool = False):
    if load_dir is None:
        if exp_name is not None:
            load_dir = f"experiments/{exp_type}/{query_set}/{estimator}/{exp_name}"
        else:
            estimator_dir = f"experiments/{exp_type}/{query_set}/{estimator}"
            if os.path.exists(estimator_dir) and os.path.isdir(estimator_dir) and len(os.listdir(estimator_dir)) > 0:
                # load the latest folder
                load_dir = max([os.path.join(estimator_dir, f) for f in os.listdir(estimator_dir)], key=os.path.getmtime)
    estimation_csv_files = []
    if os.path.isdir(load_dir):
        if stale:
            estimation_csv_files = [f for f in os.listdir(load_dir) if f == 'estimation_stale.csv']
        else:
            estimation_csv_files = [f for f in os.listdir(load_dir) if f == 'estimation_updated.csv' or f == 'estimation.csv']
    if load_dir is None or not os.path.exists(load_dir) or not os.path.isdir(load_dir) or len(estimation_csv_files) == 0:
        if estimator in ["postgresql", "true"]:
            os.makedirs(load_dir, exist_ok=True)
            schema = load_schema(db, query_set, sample_number=None)
            estimator_predict(estimator, query_set, schema, load_dir)
            estimation_csv_files = ['estimation.csv']
        else:
            print("ERROR! You should run cardinality estimation experiment for %s on %s before end-to-end experiment." %(estimator, query_set))
            return
    
    assert len(estimation_csv_files) == 1
    estimation_csv_file = estimation_csv_files[0]
    log_file = estimation_csv_file.replace('estimation', 'execute').replace('.csv', '.log')
    e2e_csv_file = estimation_csv_file.replace('estimation', 'end2end')
    log = Logger(os.path.join(load_dir, log_file))
    
    from pilotscope import PilotTransData
    from pilotscope.PilotConfig import PostgreSQLConfig
    from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
    from pilotscope.Factory.DBControllerFectory import DBControllerFactory
    from pilotscope.DBController.BaseDBController import BaseDBController
    from pilotscope.Common.CardMetricCalc import p_error_calc

    user, password, host, port = db_connection_args["user"], db_connection_args["password"], db_connection_args["host"], db_connection_args["port"]
    
    # Enable Deep Control, restart postgresql, and warmup
    log.write("[Tips] Shutdown postgres, clear system cache, restart postgres before e2e experiments. This script will only warmup pages.\n")
    pilot_config = PostgreSQLConfig(db_host=host, db_port=port, db_user=user, db_user_pwd=password, db=db, sql_execution_timeout=query_timeout_seconds)

    warmup(db, query_set)
    log.write("Warmup finished.\n")
    
    queries = [
        (query.query_id, query.sql)
        for query in load_workload(query_set, "test")
    ]
    
    def get_subquery_estimation(estimation_csv_file):
        subquery_2_est_card_each_query = {}
        subquery_2_true_card_each_query = {}
        planning_time_each_query = {}
        df = pd.read_csv(estimation_csv_file)
        for i, row in df.iterrows():
            if row["index"] not in subquery_2_est_card_each_query:
                subquery_2_est_card_each_query[row["index"]] = {}
                subquery_2_true_card_each_query[row["index"]] = {}
                planning_time_each_query[row["index"]] = 0.0
            planning_time_each_query[row["index"]] += float(row["estimation_time"])
            subquery_2_est_card_each_query[row["index"]][row["sql"]] = float(row["estimated_card"])
            subquery_2_true_card_each_query[row["index"]][row["sql"]] = float(row["true_card"])
        return subquery_2_est_card_each_query, subquery_2_true_card_each_query, planning_time_each_query
    
    subquery_2_est_card_each_query, subquery_2_true_card_each_query, planning_time_each_query = \
        get_subquery_estimation(os.path.join(load_dir, estimation_csv_file))

    
    e2e_result = {
        "index": [],
        "planning_time": [],
        "execution_time": [],
        "e2e_time": [],
        "p_error": [],
        "query_id": [],
        "sql": []
    }

    data_interactor = PilotDataInteractor(pilot_config)

    total_plan_time, total_exec_time, total_e2e_time = 0.0, 0.0, 0.0
    stop_exec_threshold = workload_timeout_seconds

    for i, (query_id, sql) in enumerate(queries):
        planning_time = planning_time_each_query[i]
        subquery_2_est_card = subquery_2_est_card_each_query[i]
        subquery_2_true_card = subquery_2_true_card_each_query[i]

        if stop_exec_threshold > 0:
            data_interactor.push_card(subquery_2_est_card)
            data_interactor.pull_execution_time()

            data: PilotTransData = data_interactor.execute(sql)
            
            if data is None:
                execution_time = query_timeout_seconds
            else:
                execution_time = float(data.execution_time)
            
            stop_exec_threshold -= execution_time
        
            e2e_time = planning_time + execution_time
        else:
            execution_time = 0.0
            e2e_time = planning_time + execution_time
        
        p_error = p_error_calc(pilot_config, sql, subquery_2_est_card, subquery_2_true_card)
        
        e2e_result["index"].append(i)
        e2e_result["planning_time"].append(planning_time)
        e2e_result["execution_time"].append(execution_time)
        e2e_result["e2e_time"].append(e2e_time)
        e2e_result["p_error"].append(p_error)
        e2e_result["query_id"].append(query_id)
        e2e_result["sql"].append(sql)

        log.write("[%d] %fs || %s\n" % (i + 1, e2e_time, sql))
        log.flush()

        total_plan_time += planning_time
        total_exec_time += execution_time
        total_e2e_time += e2e_time

    pd.DataFrame(e2e_result).to_csv(os.path.join(load_dir, e2e_csv_file), index=False)

    log.write("\n====================================== End-to-End Execution Result ======================================\n")
    log.write("%s on %s total inference time: %f second!\n" % (estimator, query_set, total_plan_time))
    log.write("%s on %s total plan execution time: %f second!\n" % (estimator, query_set, total_exec_time))
    log.write("%s on %s total e2e execution time: %f second!\n" % (estimator, query_set, total_e2e_time))
    log.write("End-to-end experiment done.\n")

    log.close()


def main():
    parser = argparse.ArgumentParser(description='Run cardinality estimation & end-to-end exection experiment.')

    parser.add_argument('card_or_e2e', type=str, choices=['card', 'e2e'], help='Experiment type: cardinality estimation or end-to-end execution')
    parser.add_argument('--db', type=str, choices=['imdb', 'stats', 'ergastf1', 'stats_sf10', 'stats_sf100'], help='Database name', required=True)
    parser.add_argument('--query_set', type=str, choices=['job_light', 'stats_ceb', 'ergastf1', 'job_m_ours', 'job_large_ours'], 
                        help='Test workload name', required=True)
    parser.add_argument('--estimator', 
                        type=str, 
                        choices=['postgresql', 'postgresql_ext', 'true', 'mscn', 'robustmscn', 'jgmp', 'alece', 'price_zs', 'price_ft',
                                 'deepdb', 'neurocard', 'factorjoin', 'bayescard', 'asm', 'asm_hybrid', 'mscn_hybrid', 'flat', 'lpbound'],
                        help='Cardinality estimator', required=True)
    parser.add_argument('--query_timeout_seconds', type=int, default=3600 * 5, help='Time limit (s) per query')
    parser.add_argument('--workload_timeout_seconds', type=int, default=3600 * 5, help='Time limit (s) for the whole workload execution, only for end-to-end experiment')
    parser.add_argument('--load_dir', type=str, default=None, help='Load directory for cardinality estimation by trained estimator or end-to-end experiment')
    parser.add_argument('--type', type=str, choices=['overall', 'training_data_efficiency', 'generalization', 'scalability', 'dynamic'], \
                        help='Store the experiment result to the corresponding folder', default='overall')
    parser.add_argument('--exp_name', type=str, default='default', help='Experiment name, will be used as the folder name')
    parser.add_argument('--stale', action='store_true', help='Whether to use stale cardinality estimation for end-to-end experiment')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the experiment')

    args = parser.parse_args()

    if args.card_or_e2e == 'card':
        run_cardinality_estimation_experiment(args.estimator, args.db, args.query_set, args.type, args.exp_name, args.load_dir, args.device)
    else:
        run_end_to_end_experiment(args.estimator, args.db, args.query_set, args.type, args.exp_name, args.query_timeout_seconds,
                                  args.workload_timeout_seconds, args.load_dir, args.stale)


if __name__ == "__main__":
    main()
