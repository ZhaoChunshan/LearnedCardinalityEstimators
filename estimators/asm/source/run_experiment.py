import argparse
import logging
import os
import time
import shutil
import numpy as np
import pandas as pd
import cProfile, pstats

from Evaluation.training import train
from Evaluation.testing import test

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imdb')

    parser.add_argument('--generate_models', action='store_true')
    parser.add_argument('--data_path', default='../datasets/imdb/{}.csv')
    parser.add_argument('--ar_path', default='')
    parser.add_argument('--config_path', default='')
    parser.add_argument('--model_path', default='meta_models/')
    parser.add_argument('--asm_data_path', default=None,
                        help='Directory containing per-table ASM inference data')

    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--exact_support', action='store_true')

    parser.add_argument('--query_name', type=str, default=None)

    parser.add_argument('--evaluate', help='Evaluates models to compute cardinality bound', action='store_true')
    parser.add_argument('--query_file', default='job_queries/all_queries.pkl')
    parser.add_argument('--query_sub_plan_file', default='job_queries/all_sub_plan_queries_str.pkl')
    parser.add_argument('--query_predicate_location', type=str, default=None)
    parser.add_argument('--output_predicate_location', type=str, default=None)
    parser.add_argument('--save_folder', default='job_CE')

    parser.add_argument('--log_level', type=int, default=logging.DEBUG)
    parser.add_argument('--init_data_ratio', type=float, default=1.0)
    parser.add_argument('--evaluate_stale', type=bool, default=False)

    args = parser.parse_args()

    os.makedirs('logs/asm', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/asm/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    if args.dataset in ['imdb-light', 'imdb_m', 'imdb_large', 'imdb', 'stack', 'stats', 'stats_sf10', 'stats_sf100', 'stack_ss100', 'ergastf1']:
        if args.generate_models:
            bound_ensemble = train(args.data_path, args.model_path, args.dataset,
                                   args.init_data_ratio, args.asm_data_path)
        elif args.evaluate:
            dataset = args.dataset.split('_')[0] if args.dataset.startswith('imdb_') else args.dataset
            test(args.model_path,
                 args.query_file,
                 args.query_sub_plan_file,
                 args.query_name,
                 args.save_folder,
                 args.ar_path,
                 args.config_path,
                 args.sample_size,
                 args.query_predicate_location,
                 dataset,
                 args.output_predicate_location,
                 args.evaluate_stale,
                 args.init_data_ratio,
                 args.exact_support,
                 args.asm_data_path)
        else:
            assert False
    else:
        assert False
