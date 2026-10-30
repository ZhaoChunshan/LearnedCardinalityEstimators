import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

import re
import multiprocessing

from typing import List, Tuple
from collections import OrderedDict
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope import PilotTransData
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from common.sql_utils import get_tables_from_comment

def generate_subqueries(input_file: str, output_file: str, schema_name: str, perc: int, user: str = 'pilotscope', password: str = '', host: str = "localhost", port: int = 5432):
    assert perc in [20, 40, 60, 80], "Percentage must be one of [20, 40, 60, 80]"

    db = "imdb" if schema_name.startswith("imdb") else schema_name
    assert db in ["imdb", "stats"]

    pilot_config = PostgreSQLConfig(db_host=host, db_port=port, db_user=user, db_user_pwd=password, db=f"{db}_p{perc}", sql_execution_timeout=3600 * 24 * 7)
    data_interactor = PilotDataInteractor(pilot_config)
    
    queries = []
    subqueries = []

    with open(input_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        i = 0
        while i < len(lines):
            j = i + 1
            while j < len(lines) and lines[j].startswith("/*"):
                j += 1
            
            query = lines[i].split("||")[0].strip()
            queries.append(query)
            
            subquery = OrderedDict()
            for line in lines[i+1:j]:
                subquery[tuple(sorted(get_tables_from_comment(line.split("||")[0].strip())))] = line.split("||")[0].strip()
            subqueries.append(subquery)
            
            i = j

    with open(output_file, "w") as f:
        for query, subquery_dict in zip(queries, subqueries):
            query_with_subqueries: List[Tuple[str, float]] = []  # subquery, est_card

            data_interactor.pull_subquery_card()
            data: PilotTransData = data_interactor.execute(query)
            
            est_card, query_size = 0.0, 0
            query_with_subqueries.append((query, est_card))

            subquery_est_dict = {}
            for subquery, subquery_est_card in data.subquery_2_card.items():
                tables = get_tables_from_comment(subquery)
                subquery = subquery_dict[tuple(sorted(tables))]
                
                subquery_est_dict[tuple(sorted(tables))] = (subquery, subquery_est_card)

                cur_size = len(tables)
                
                # The largest subquery is the query itself
                if cur_size > query_size:
                    query_size = cur_size
                    est_card = subquery_est_card
        
            query_with_subqueries[0] = (query, est_card)
            
            for k in subquery_dict.keys():
                assert k in subquery_est_dict, f"Subquery with tables {k} not found in estimation results."
                subquery, subquery_est_card = subquery_est_dict[k]
                query_with_subqueries.append((subquery, subquery_est_card))

            for subquery, subquery_est_card in query_with_subqueries:
                f.write("%s||%f\n" % (subquery, subquery_est_card))
            f.flush()


def main():
    process_args = [
        ("workloads/job_light/job_light_all.sql", "workloads/job_light/dynamic/20/job_light_all.sql", "imdb", 20),
        ("workloads/job_light/job_light_all.sql", "workloads/job_light/dynamic/40/job_light_all.sql", "imdb", 40),
        ("workloads/job_light/job_light_all.sql", "workloads/job_light/dynamic/60/job_light_all.sql", "imdb", 60),
        ("workloads/job_light/job_light_all.sql", "workloads/job_light/dynamic/80/job_light_all.sql", "imdb", 80),
        ("workloads/stats_ceb/stats_ceb_all.sql", "workloads/stats_ceb/dynamic/20/stats_ceb_all.sql", "stats", 20),
        ("workloads/stats_ceb/stats_ceb_all.sql", "workloads/stats_ceb/dynamic/40/stats_ceb_all.sql", "stats", 40),
        ("workloads/stats_ceb/stats_ceb_all.sql", "workloads/stats_ceb/dynamic/60/stats_ceb_all.sql", "stats", 60),
        ("workloads/stats_ceb/stats_ceb_all.sql", "workloads/stats_ceb/dynamic/80/stats_ceb_all.sql", "stats", 80),
    ]

    for args in process_args:
        print("Processing:", args)
        generate_subqueries(*args)

if __name__ == "__main__":
    main()
