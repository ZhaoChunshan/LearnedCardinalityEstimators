import argparse
import logging
import os
import time
import pandas as pd
from typing import List
from utils import ROOT_DIR, set_seed
from plan_generator.plan_generator import PlanGenerator, NextBestStepNotFoundError
from plan_generator.query_hint import Hint
from plan_generator.sql_query import Query
from plan_generator.connection_args import DB_CONNECTION_ARGS
from plan_generator.cost_model import CostModel
from plan_search.search import PlanSearchEngine
from pilotscope import PilotTransData
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor


def setup_logger(log_dir, exp_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}_{time.strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True,
    )

def get_workload_paths(workload):
    if workload == "job_light":
        return os.path.join(ROOT_DIR, "workloads/job_light/job_light.sql")
    elif workload == "stats_ceb":
        return os.path.join(ROOT_DIR, "workloads/stats_ceb/stats_ceb.sql")
    elif workload == "ergastf1":
        return os.path.join(ROOT_DIR, "workloads/ergastf1/ergastf1_queries.sql")
    else:
        raise ValueError(f"Unknown workload: {workload}")

def read_workload_queries(queryset: str) -> List[str]:
    path = get_workload_paths(queryset)
    with open(path, "r") as f:
        queries = f.readlines()
    return [query.strip() for query in queries if query.strip()]

def warmup(db: str, query_set: str):
    import psycopg2
    user, password, host, port = DB_CONNECTION_ARGS["user"], DB_CONNECTION_ARGS["password"], DB_CONNECTION_ARGS["host"], DB_CONNECTION_ARGS["port"]
    warm_up_path = os.path.join(ROOT_DIR, "workloads", query_set, "warmup.sql")
    conn = psycopg2.connect(host=host, database=db, user=user, password=password, port=port)
    conn.autocommit = True
    cur = conn.cursor()
    with open(warm_up_path, "r") as f:
        cur.execute(f.read())
    conn.close()


def generate_plan(queryset: str, strategy: str = "greedy", num_workers: int = None,
                  limit_num: int = 500000, result_timeout: float = 120.0):
    logger = logging.getLogger(__name__)
    sqls = read_workload_queries(queryset)
    logger.info(f"Generating execution plan hints for {len(sqls)} queries in the {queryset} workload (strategy={strategy}).")

    db = {
        'job_light': 'imdb',
        'stats_ceb': 'stats',
        'ergastf1': 'ergastf1'
    }[queryset]
    cost_model = CostModel() if strategy == "greedy" else None
    engine = None
    hints, infer_times, indexes = [], [], []
    failed_counter = 0
    for i, sql in enumerate(sqls):
        logger.info(f"Query #{i+1}: {sql}")
        query = Query.parse(sql)
        if strategy == "greedy":
            plan_generator = PlanGenerator(db, query, cost_model)
            hint_str, latency = plan_generator.search_best_plan_hint()
        elif strategy == "enumerate":
            if engine is None:
                engine = PlanSearchEngine(
                    db=db, num_workers=num_workers, limit_num=limit_num,
                    result_timeout=result_timeout,
                )
            hint_str, latency = engine.search(query)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        if len(hint_str) == 0:
            logger.warning(f"Fail to generate plan hint for query #{i+1}.")
            failed_counter += 1
        hints.append(hint_str)
        infer_times.append(latency)
        indexes.append(i)
        logger.info(f"Query #{i+1} Generated Hint: {hint_str}\n")
    if engine is not None:
        engine.close()
    result_dir = os.path.join(ROOT_DIR, "workloads", queryset, strategy)
    os.makedirs(result_dir, exist_ok=True)
    result_csv_path = os.path.join(result_dir, f"{queryset}_plan_hints.csv")
    pd.DataFrame({"index": indexes, "query": sqls, "hint": hints, "infer_time": infer_times}).to_csv(result_csv_path, index=False)
    logger.info(f"Generated plan hints saved to {result_csv_path}")
    logger.info(f"Number of failed queries: {failed_counter}")

def execute_plan(queryset: str, strategy: str, timeout_seconds: int = 3600 * 5):
    logger = logging.getLogger(__name__)
    sqls = read_workload_queries(queryset)
    result_dir = os.path.join(ROOT_DIR, "workloads", queryset, strategy)
    plan_csv = pd.read_csv(os.path.join(result_dir, f"{queryset}_plan_hints.csv"), dtype={"index": int, "query": str, "hint": str, "infer_time": float})
    logger.info(f"Executing {len(sqls)} queries in the {queryset} workload with generated plan hints.")

    user, password, host, port = DB_CONNECTION_ARGS["user"], DB_CONNECTION_ARGS["password"], DB_CONNECTION_ARGS["host"], DB_CONNECTION_ARGS["port"]
    db = {
        'job_light': 'imdb',
        'stats_ceb': 'stats',
        'ergastf1': 'ergastf1'
    }[queryset]
    config = PostgreSQLConfig(db_host=host, db_port=port, db_user=user, db_user_pwd=password, db=db,  sql_execution_timeout=timeout_seconds)
    data_interactor = PilotDataInteractor(config)
    infer_times, execution_times, e2e_times = [], [], []

    warmup(db, queryset)

    for i, (sql, hint, infer_time) in enumerate(zip(sqls, plan_csv["hint"], plan_csv["infer_time"])):
        logger.info(f"Exec Query #{i+1}: {sql}\nUsing Hint: {hint}\n")
        if not pd.isna(hint) and hint.strip() != "":
            data_interactor.push_pg_hint_comment(hint)
        data_interactor.pull_execution_time()

        data: PilotTransData = data_interactor.execute(sql)
        assert data is not None, f"Execution failed for query: {sql}"

        execution_time = float(data.execution_time)
        infer_times.append(float(infer_time))
        execution_times.append(execution_time)
        e2e_times.append(float(infer_time) + execution_time)
        logger.info(f"Query #{i+1} Execution Time: {execution_time:.4f}s, Inference Time: {infer_time:.4f}s, E2E Time: {float(infer_time) + execution_time:.4f}s\n")
    result_csv_path = os.path.join(result_dir, f"{queryset}_execution_results.csv")
    df = pd.DataFrame({
        "index": list(range(len(sqls))),
        "e2e_time": e2e_times,
        "execution_time": execution_times,
        "infer_time": infer_times,
        "query": sqls,
        "hint": plan_csv["hint"],
    })
    df.to_csv(result_csv_path, index=False)
    total_time = sum(e2e_times)
    logger.info(f"Total E2E Time for {len(sqls)} queries: {total_time:.4f}s")
    logger.info(f"Average E2E Time per query: {total_time / len(sqls):.4f}s")
    logger.info(f"Total Execution Time for {len(sqls)} queries: {sum(execution_times):.4f}s")
    logger.info(f"Average Execution Time per query: {sum(execution_times) / len(sqls):.4f}s")

    logger.info(f"Execution results saved to {result_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning", action="store_true", help="Generate execution plan hint for each query")
    parser.add_argument("--execution", action="store_true", help="Execute the queries based on the generated plan")
    parser.add_argument(
        "--query_set", type=str, required=True, choices=["job_light", "stats_ceb", "ergastf1"], help="The query set to evaluate"
    )
    parser.add_argument(
        "--strategy", type=str, default="enumerate", choices=["greedy", "enumerate"],
        help="Plan search strategy: greedy (incremental) or enumerate (exhaustive join orders)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of DB worker processes for the enumerate strategy (default: min(64, cpu_count))"
    )
    parser.add_argument(
        "--limit_num", type=int, default=500000,
        help="Max join orders to consider per query for the enumerate strategy (sampling beyond this)"
    )
    parser.add_argument(
        "--result_timeout", type=float, default=120.0,
        help="Seconds to wait for a worker result before checking for dead workers"
    )
    parser.add_argument('--query_timeout_seconds', type=int, default=3600 * 5, help='Time limit (s) per query')
    parser.add_argument("--random_seed", type=int, default=123, help="random seed")
    args = parser.parse_args()

    # transform args to configs
    configs = vars(args)

    # set random seed
    set_seed(configs["random_seed"])

    if configs["planning"]:
        setup_logger(log_dir=os.path.join(ROOT_DIR, "logs"), exp_name=f"{configs['query_set']}_planning")
        generate_plan(
            configs['query_set'],
            strategy=configs['strategy'],
            num_workers=configs['num_workers'],
            limit_num=configs['limit_num'],
            result_timeout=configs['result_timeout'],
        )
    if configs["execution"]:
        setup_logger(log_dir=os.path.join(ROOT_DIR, "logs"), exp_name=f"{configs['query_set']}_execution")
        execute_plan(configs['query_set'], configs['strategy'], timeout_seconds=configs['query_timeout_seconds'])
