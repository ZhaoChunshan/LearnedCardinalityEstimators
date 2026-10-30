import pickle5 as pickle
import sys
import time
import os
import numpy as np
import torch
import pandas as pd
import multiprocessing
from Join_scheme.bound import Bound_ensemble

sys.path.append('AR')

from experiments import EXPERIMENT_CONFIGS
from neurocard import NeuroCard

def load_neurocard(dataset, table, ar_path, config_path, evaluate_stale,
                   init_data_ratio, asm_data_path=None):

    config = dict(EXPERIMENT_CONFIGS[config_path.format(dataset, table)])
    if asm_data_path is not None:
        table_data_path = os.path.join(asm_data_path, table)
        config['data_dir'] = table_data_path + os.path.sep
        config['PK_tuples_np_loc'] = os.path.join(table_data_path,
                                                  'tuples_np.pkl')
    config['__run'] = config['workload_name'] = config_path.format(dataset, table)
    config['evaluate_stale'] = evaluate_stale
    if evaluate_stale:
        config['init_data_perc'] = int(init_data_ratio * 100)
    else:
        config['init_data_perc'] = 100
    print("Try loading model from", ar_path.format(dataset, table))
    if os.path.exists(ar_path.format(dataset, table)):
        print('table', table, 'updated')
        config['checkpoint_to_load'] = ar_path.format(dataset, table)
    elif dataset == 'imdb-light' and os.path.exists(ar_path.format('imdb', table)):
        print('table', table, 'updated')
        config['checkpoint_to_load'] = ar_path.format('imdb', table)
    else:
        print('table', table, 'not updated')
        config['checkpoint_to_load'] = f"AR_models/{dataset}-single-{table}.tar"
    config['queries_csv'] = None
    config['__gpu'] = 1
    config['__cpu'] = 1
    config['external'] = False
    nc = NeuroCard(config)

    min_count = pickle.load(open(f"{config['data_dir']}/min_count.pkl","rb"))
    nc.min_count = dict()
    for col in min_count:
        assert np.all(min_count[col] >= 1.), f'{np.where(min_count[col]) < 1.}'
        nc.min_count[col] = torch.tensor(min_count[col], device=nc.get_device())

    return nc

def test(model_path,
         query_file,
         query_sub_plan_file,
         query_name,
         save_res,
         ar_path,
         config_path,
         sample_size,
         query_predicate_location,
         dataset,
         output_predicate_location,
         evaluate_stale,
         init_data_ratio,
         exact_support=False,
         asm_data_path=None
         ):

    bound_ensemble = None

    os.makedirs(save_res, exist_ok=True)

    with open(model_path, "rb") as f:
        bound_ensemble_nc = pickle.load(f)

    ncs = dict()
    for table_obj in bound_ensemble_nc.schema.tables:
        table = table_obj.table_name

        nc = load_neurocard(dataset, table, ar_path, config_path,
                            evaluate_stale, init_data_ratio, asm_data_path)
        use_raw_table = table_obj.table_size < 1000 and dataset != 'ergastf1' # For ergastf1 dataset, there is a bug
        nc.ready_for_evaluate(use_raw_table)

        ncs[table] = nc

    bound_ensemble_nc.ncs = ncs
    bound_ensemble_nc.enable_exact_support = exact_support

    bound_ensemble_nc.use_ar = True
    if sample_size is None:
        bound_ensemble_nc.sample_size = 2048
    else:
        bound_ensemble_nc.sample_size = sample_size

    with open(query_file, "rb") as f:
        all_queries = pickle.load(f)    # Dict[QueryName, QueryStr]
    with open(query_sub_plan_file, "rb") as f:
        all_sub_plan_queries_str = pickle.load(f) # Dict[QueryName, List[Tuple(QueryTableAlias)]]

    # total initialize model and sample time
    total_q_time = 0.
    total_m_time = 0.
    total_s_time = 0.

    res = dict()
    q_times = dict()
    m_times = dict()
    s_times = dict()
    single_q_times = dict()
    exact_failure_masks = dict()

    t = time.time()

    for i, q_name in enumerate(all_queries):
        if query_name is not None:
            if q_name != query_name:
                continue

        print(f'query: {q_name}')
        print(f"SQL: {all_queries[q_name]}")
        if os.path.exists(query_predicate_location + '/' + q_name + '.pkl'):
            with open(query_predicate_location + '/' + q_name + '.pkl', "rb") as f:
                p = pickle.load(f)  # Dict[TableAlias, Tuple[TableName, ConjunctivePredicateStr, Set[JoinKeys]]]
                bound_ensemble_nc.query_predicate = p
        else:
            continue

        if True:

            torch.manual_seed(0)

            # Multi-table CardEst
            q_time_start = time.time()
            ests, c_time, m_time, s_time = bound_ensemble_nc.get_cardinality_bound_all(
                all_queries[q_name],
                all_sub_plan_queries_str[q_name]
            )

            q_time = time.time() - q_time_start - c_time + m_time
            assert c_time >= m_time
            total_q_time += q_time
            total_m_time += m_time
            total_s_time += s_time
            q_times[q_name] = q_time
            m_times[q_name] = m_time
            s_times[q_name] = s_time

            if exact_support:
                exact_failure_masks[q_name] = {
                    alias: mask.copy()
                    for alias, mask in
                    bound_ensemble_nc.last_exact_failure_masks.items()
                }

            res[q_name] = ests

            # Single Table CardEst
            single_q_time = 0.0
            single_card_est = {}
            for alias, (tb_name, predicate, jk_set) in bound_ensemble_nc.query_predicate.items():
                nc = bound_ensemble_nc.ncs[tb_name]
                if len(predicate) == 0:
                    single_card_est[alias] = nc.table.cardinality
                else:
                    cols, ops, vals = [], [], []
                    for pred in predicate.split("AND"):
                        first_space_idx, second_space_idx = 0, 0
                        while pred[first_space_idx] == ' ':
                            first_space_idx += 1
                        while pred[first_space_idx] != " ":
                            first_space_idx += 1
                        second_space_idx = first_space_idx + 1
                        while pred[second_space_idx] == ' ':
                            second_space_idx += 1
                        while pred[second_space_idx] != " ":
                            second_space_idx += 1
                        
                        col = pred[:first_space_idx].strip().split(".")[1]
                        op = pred[first_space_idx + 1 : second_space_idx].strip()
                        val = pred[second_space_idx + 1:].strip()
                        if val[0] == "'" and val[-1] == "'":
                            val = str(val[1:-1])
                        else:
                            val = float(val)
                        cols.append(nc.table.columns[nc.table.name_to_index[col]])
                        ops.append(op)
                        vals.append(val)
                    est, time_sec = nc.evaluate_one((cols, ops, vals))
                    single_q_time += time_sec
                    single_card_est[alias] = est

            single_q_times[q_name] = single_q_time

            if save_res:
                f_query = open(os.path.join(save_res, q_name), "w")
                for alias in sorted(bound_ensemble_nc.query_predicate.keys()):
                    f_query.write(f"{single_card_est[alias]}\n")
                for est in ests:
                    f_query.write(str(est) + "\n")
                f_query.write(f"Time: {q_time + single_q_time}\n")
                f_query.close()

    df = pd.DataFrame(columns=['name', 'single_q_time', 'q_time', 'm_time', 's_time'])
    for q_name in res:
        df = df.append({'name': q_name, 'single_q_time': single_q_times[q_name],
                        'q_time': q_times[q_name], 'm_time': m_times[q_name], 's_time': s_times[q_name]},
                        ignore_index=True
                        )
    df.to_csv(os.path.join(save_res, 'time.csv'), index=False)
    if exact_support:
        with open(os.path.join(save_res, 'exact_failure_masks.pkl'), 'wb') as f:
            pickle.dump(exact_failure_masks, f)

    print("total estimation latency is: ", total_q_time)
    print("total initialize model latency is: ", total_m_time)
    print("total initialize sample latency is: ", total_s_time)
