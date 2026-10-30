import re
from tqdm import tqdm
from typing import List, Tuple
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope import PilotTransData
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor
from multiprocessing import Pool, cpu_count


class SubqueryCollector:
    def __init__(self, db_host, db_port, db_user, db_password, db):
        self.pilot_config = PostgreSQLConfig(db_host=db_host, db_port=db_port, db_user=db_user, db_user_pwd=db_password, db=db, sql_execution_timeout=60*20)
        self.data_interactor = PilotDataInteractor(self.pilot_config)

    def collect_subquery(self, sql: str, only_keep_single_table_est: bool = True) -> List[Tuple[str, float]]:
        sql_est, query_size = None, -1
        result = []
        result.append((sql, sql_est))

        self.data_interactor.pull_subquery_card()
        data: PilotTransData = self.data_interactor.execute(sql)

        if data is None: 
            return result

        for subquery, subquery_est_card in data.subquery_2_card.items():
            subquery_size = self.get_tables_num_from_comment(subquery)
            if only_keep_single_table_est:
                if subquery_size == 1:
                    result.append((subquery, subquery_est_card))
            else:
                result.append((subquery, subquery_est_card))

            if subquery_size > query_size:
                query_size = subquery_size
                sql_est = subquery_est_card

        result[0] = (sql, sql_est)
        return result

    @staticmethod
    def get_tables_num_from_comment(sql: str):
        return len(re.search(r"\/\*(.*)\*\/", sql).group(1).replace("(","").replace(")","").strip().split(","))

# Global variable to store the collector instance in each process
collector_instance = None

def init_worker(db_info):
    """
    Initializer function for each worker process.
    It's called once per process and creates a single SubqueryCollector instance.
    """
    global collector_instance
    collector_instance = SubqueryCollector(
        db_info['host'], db_info['port'], db_info['user'],
        db_info['password'], db_info['db']
    )

def process_sql(sql):
    """
    Worker function to process a single SQL query.
    It reuses the collector instance created by the initializer.
    """
    global collector_instance
    # The collector_instance is already created and available in this process
    return collector_instance.collect_subquery(sql)

if __name__ == "__main__":
    db_info = {
        "host": "localhost",
        "port": 5432,
        "user": "pilotscope",
        "password": "pilotscope",
        "db": "imdb"
    }

    with open("workloads/job_large_ours/train_workload.sql", "r") as f:
        sqls = [line.split("||")[0].strip() for line in f.readlines() if line.strip() != ""]

    out_path = "workloads/job_large_ours/train_workload_with_sub_est.sql"
    out_path_none = "workloads/job_large_ours/train_workload_with_sub_est_unsolved.sql"
    result_none = []
    user_input = input(f"Subqueries with cardinality estimation will be wrote to `{out_path}`. Press Enter to continue.")
    
    num_processes = 24
    print(f"Using {num_processes} processes for parallel processing.")
    
    # Use Pool with the initializer argument
    with Pool(num_processes, initializer=init_worker, initargs=(db_info,)) as p:
        # Use imap_unordered for efficient, out-of-order results processing
        # The tqdm wrapper provides a progress bar
        results_iterator = tqdm(p.imap_unordered(process_sql, sqls), total=len(sqls))

        with open(out_path, "w") as f:
            for result in results_iterator:
                if result[0][1] is None:
                    result_none.append(result[0][0])
                else:
                    for subquery, subquery_est_card in result:
                        f.write(f"{subquery}||{subquery_est_card}\n")
    
    with open(out_path_none, "w") as f:
        for sql in result_none:
            f.write(f"{sql}\n")
    
    print("Parallel processing complete.")
    