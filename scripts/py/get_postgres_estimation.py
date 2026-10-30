import argparse
import os
import sys
import re
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

import pilotscope
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor

from common.constants import DB_CONNECTION_ARGS
from common.logger import Logger
from common.card_utils import q_error_value
from common.sql_utils import get_tables_from_comment


def get_estimation(db, queryset, query_file, result_folder):
    os.makedirs(result_folder, exist_ok=True)

    with open(query_file, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines() if l.strip() != ""]

    query_groups = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        tokens = line.split("||")
        outer_sql = tokens[0].strip()
        outer_true_card = float(tokens[1].strip())

        j = i + 1
        subqueries = []
        while j < len(lines):
            next_line = lines[j].strip()
            if not next_line:
                j += 1
                continue
            if not next_line.startswith("/*"):
                break
            tokens = next_line.split("||")
            sub_sql = tokens[0].strip()
            sub_true_card = float(tokens[1].strip())
            subqueries.append((sub_sql, sub_true_card))
            j += 1

        query_groups.append((outer_sql, outer_true_card, subqueries))
        i = j

    n_outer = len(query_groups)

    user, password, host, port = (
        DB_CONNECTION_ARGS["user"],
        DB_CONNECTION_ARGS["password"],
        DB_CONNECTION_ARGS["host"],
        DB_CONNECTION_ARGS["port"],
    )
    pilot_config = PostgreSQLConfig(
        db_host=host, db_port=port, db_user=user, db_user_pwd=password, db=db
    )
    data_interactor = PilotDataInteractor(pilot_config)

    result = {
        "index": [],
        "is_outer_most": [],
        "estimated_card": [],
        "true_card": [],
        "q_error": [],
        "estimation_time": [],
        "query_id": [],
        "sql": [],
    }

    q_error_subqueries = []
    q_error_queries = []

    for idx, (outer_sql, outer_true_card, file_subqueries) in enumerate(query_groups):
        data_interactor.pull_subquery_card()
        data = data_interactor.execute(outer_sql)

        pg_est_dict = {}
        if data is not None and data.subquery_2_card is not None:
            for sub_sql, est in data.subquery_2_card.items():
                tables = get_tables_from_comment(sub_sql)
                pg_est_dict[tuple(sorted(tables))] = est

        if file_subqueries:
            table_counts = [
                len(get_tables_from_comment(sub_sql))
                for sub_sql, _ in file_subqueries
            ]
            max_table_count = max(table_counts)
        else:
            max_table_count = 0

        for sub_sql, sub_true_card in file_subqueries:
            tables = get_tables_from_comment(sub_sql)
            pg_est = pg_est_dict[tuple(sorted(tables))]

            table_count = len(get_tables_from_comment(sub_sql))
            is_outer_most = table_count == max_table_count

            q_err = q_error_value(pg_est, sub_true_card)

            result["index"].append(idx)
            result["is_outer_most"].append(is_outer_most)
            result["estimated_card"].append(pg_est)
            result["true_card"].append(sub_true_card)
            result["q_error"].append(q_err)
            result["estimation_time"].append(0.0)
            result["query_id"].append(idx + 1)
            result["sql"].append(sub_sql)

            if is_outer_most:
                q_error_queries.append(q_err)
            q_error_subqueries.append(q_err)

    df = pd.DataFrame(result)
    csv_path = os.path.join(result_folder, "estimation.csv")
    df.to_csv(csv_path, index=False)
    print(f"Estimation results saved to {csv_path}")

    q_error_subqueries = np.array(q_error_subqueries)
    q_error_queries = np.array(q_error_queries)
    query_num = n_outer

    log = Logger(os.path.join(result_folder, "predict.log"), mode="w")
    log.write(
        "\n====================================== Cardinality Estimation Result ======================================\n"
    )
    log.write(
        "estimator: %s, query_set: %s, query_num: %d, avg_time: %f\n"
        % ("postgresql", queryset, query_num, 0.0)
    )

    log.write(
        "Subqueries Q-error Percentile\n50%% %.3f\n80%% %.3f\n90%% %.3f\n95%% %.3f\n99%% %.3f\nmax %.3f\n"
        % (
            np.percentile(q_error_subqueries, 50),
            np.percentile(q_error_subqueries, 80),
            np.percentile(q_error_subqueries, 90),
            np.percentile(q_error_subqueries, 95),
            np.percentile(q_error_subqueries, 99),
            q_error_subqueries.max(),
        )
    )

    log.write(
        "Queries Q-error Percentile\n50%% %.3f\n80%% %.3f\n90%% %.3f\n95%% %.3f\n99%% %.3f\nmax %.3f\n"
        % (
            np.percentile(q_error_queries, 50),
            np.percentile(q_error_queries, 80),
            np.percentile(q_error_queries, 90),
            np.percentile(q_error_queries, 95),
            np.percentile(q_error_queries, 99),
            q_error_queries.max(),
        )
    )

    log.close()
    print(f"Prediction log saved to {os.path.join(result_folder, 'predict.log')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queryset", required=True)
    args = parser.parse_args()

    query_file_dict = {
        "job_light": "workloads/job_light/job_light_all.sql",
        "stats_ceb": "workloads/stats_ceb/stats_ceb_all.sql",
        "ergastf1": "workloads/ergastf1/ergastf1_queries_all.sql",
        "job_m_ours": "workloads/job_m_ours/job_m_ours_all.sql",
        "job_large_ours": "workloads/job_large_ours/job_large_ours_all.sql",
    }

    result_folder_dict = {
        "job_light": "experiments/overall/job_light/postgresql_ext/default",
        "stats_ceb": "experiments/overall/stats_ceb/postgresql_ext/default",
        "ergastf1": "experiments/overall/ergastf1/postgresql_ext/default",
        "job_m_ours": "experiments/scalability/job_m_ours/postgresql_ext/default",
        "job_large_ours": "experiments/scalability/job_large_ours/postgresql_ext/default",
    }

    db_dict = {
        "job_light": "imdb",
        "stats_ceb": "stats",
        "ergastf1": "ergastf1",
        "job_m_ours": "imdb",
        "job_large_ours": "imdb",
    }

    workload = args.queryset
    print(f"Processing workload: {workload}")
    print(f"Query file: {query_file_dict[workload]}")
    print(f"Result folder: {result_folder_dict[workload]}")

    db = db_dict[workload]
    query_file = query_file_dict[workload]
    result_folder = result_folder_dict[workload]
    get_estimation(db, workload, query_file, result_folder)


if __name__ == "__main__":
    main()
