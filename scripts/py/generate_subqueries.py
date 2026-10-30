import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

import re

from typing import List, Tuple


QUERY_SET_CONFIGS = {
    "job_light": {
        "input_file": "workloads/job_light/job_light.sql",
        "output_file": "workloads/job_light/job_light_all.sql",
        "schema_name": "imdb",
    },
    "stats_ceb": {
        "input_file": "workloads/stats_ceb/stats_ceb.sql",
        "output_file": "workloads/stats_ceb/stats_ceb_all.sql",
        "schema_name": "stats",
    },
    "ergastf1": {
        "input_file": "workloads/ergastf1/ergastf1_queries.sql",
        "output_file": "workloads/ergastf1/ergastf1_queries_all.sql",
        "schema_name": "ergastf1",
    },
    "job_m_ours": {
        "input_file": "workloads/job_m_ours/job_m_ours.sql",
        "output_file": "workloads/job_m_ours/job_m_ours_all.sql",
        "schema_name": "imdb_m",
    },
    "job_large_ours": {
        "input_file": "workloads/job_large_ours/job_large_ours.sql",
        "output_file": "workloads/job_large_ours/job_large_ours_all.sql",
        "schema_name": "imdb_large",
    },
}


def generate_subqueries(input_file: str, output_file: str, schema_name: str, user: str, password: str, host: str = "localhost", port: int = 5432):
    from query.cardinality_calculator import CardinalityCalculator
    from schema.schemas import get_schema
    from pilotscope import PilotTransData
    from pilotscope.PilotConfig import PostgreSQLConfig
    from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor

    db = "imdb" if schema_name.startswith("imdb") else schema_name
    table_csv_path = "./datasets/imdb" if db == "imdb" else None
    schema = get_schema(schema_name, user, password, host, port)
    cc = CardinalityCalculator(schema, host, db, user, password, port, table_csv_path)
    pilot_config = PostgreSQLConfig(db_host=host, db_port=port, db_user=user, db_user_pwd=password, db=db, sql_execution_timeout=3600 * 24 * 7)
    data_interactor = PilotDataInteractor(pilot_config)
    
    queries = []

    with open(input_file, "r") as f:
        for line in f:
            query = line.strip().split("||")[0].strip()
            if query:
                queries.append(query)

    with open(output_file, "w") as f:
        for query in queries:
            query_with_subqueries: List[Tuple[str, int, float]] = []  # subquery, true_card, est_card

            data_interactor.pull_subquery_card()
            data: PilotTransData = data_interactor.execute(query)
            
            est_card, query_size, true_card = 0.0, 0, 0
            query_with_subqueries.append((query, true_card, est_card))

            for subquery, subquery_est_card in data.subquery_2_card.items():
                subquery_true_card = cc.calculate_cardinality_sql(subquery)
                query_with_subqueries.append((subquery, subquery_true_card, subquery_est_card))
                
                comment_pattern = r"\/\*(.*)\*\/"
                comment_match = re.search(comment_pattern, subquery)
                if comment_match:
                    comment = comment_match.group(1)
                    comment = comment.replace("(", "").replace(")", "").strip()
                    cur_size = len(comment.split(","))
                    
                    # The largest subquery is the query itself
                    if cur_size > query_size:
                        query_size = cur_size
                        est_card = subquery_est_card
                        true_card = subquery_true_card
                else:
                    assert(False)
            
            query_with_subqueries[0] = (query, true_card, est_card)

            for subquery, subquery_true_card, subquery_est_card in query_with_subqueries:
                f.write("%s||%d||%f\n" % (subquery, subquery_true_card, subquery_est_card))
            f.flush()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate true and PostgreSQL-estimated cardinalities for all "
            "subqueries in a workload."
        )
    )
    parser.add_argument(
        "--query_set",
        required=True,
        choices=sorted(QUERY_SET_CONFIGS),
        help="workload to process",
    )
    parser.add_argument(
        "--input_file",
        help="override the query set's default input file",
    )
    parser.add_argument(
        "--output_file",
        help="override the query set's default output file",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", default="pilotscope")
    parser.add_argument("--password", default="pilotscope")
    args = parser.parse_args()

    config = QUERY_SET_CONFIGS[args.query_set]
    input_file = args.input_file or config["input_file"]
    output_file = args.output_file or config["output_file"]

    if not os.path.isfile(input_file):
        parser.error(f"input file does not exist: {input_file}")

    generate_subqueries(
        input_file=input_file,
        output_file=output_file,
        schema_name=config["schema_name"],
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
    )

if __name__ == "__main__":
    main()
