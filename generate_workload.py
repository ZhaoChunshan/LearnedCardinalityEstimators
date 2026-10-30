import os
import signal
import time
import argparse
import psycopg2


from psycopg2 import OperationalError, ProgrammingError
from typing import Tuple, List, Set, Optional
from multiprocessing import Process, Queue, Event
from query.query_generator import QueryGenerator, TemplateBasedQueryGenerator
from query.sql_query import SQLQuery, SQLTemplate
from query.cardinality_calculator import RobustCardinalityCalculator
from schema.sql_schema import SQLSchema
from schema.schemas import imdb_schema, stats_schema, ergastf1_schema, get_schema
from common.conn_utils import RobustConnection


def get_query_generator(db: str, seed: int, user: str, password: str, host: str = "localhost", port: int = 5432, small_query: bool = False) -> QueryGenerator: 
    schema = get_schema(db, user, password, host, port)
    
    query_gen_args = {
        "imdb": {
            "default_min_nodes": 1,
            "default_max_nodes": 6,
            "default_predicate_factor": 0.4,
            "default_max_predicate_columns": 8,
            "db": "imdb", 
            "user": "pilotscope", 
            "password": "", 
            "host": "localhost",
            "port": 5432,
            "pk_fk_only": True,
            "seed": seed
        },
        "imdb_m": {
            "default_min_nodes": 1,
            "default_max_nodes": 16,
            "default_predicate_factor": 0.4,
            "default_max_predicate_columns": 8,
            "db": "imdb", 
            "user": "pilotscope", 
            "password": "", 
            "host": "localhost",
            "port": 5432,
            "pk_fk_only": True,
            "seed": seed
        },
        "imdb_large": {
            "default_min_nodes": 1,
            "default_max_nodes": 21,
            "default_predicate_factor": 0.3,
            "default_max_predicate_columns": 16,
            "db": "imdb", 
            "user": "pilotscope", 
            "password": "", 
            "host": "localhost",
            "port": 5432,
            "pk_fk_only": True,
            "seed": seed
        },
        "stats": {
            "default_min_nodes": 1,
            "default_max_nodes": 8,
            "default_predicate_factor": 0.3,
            "default_max_predicate_columns": 16,
            "db": "stats", 
            "user": "pilotscope", 
            "password": "", 
            "host": "localhost",
            "port": 5432,
            "pk_fk_only": True,
            "seed": seed
        },
        "ergastf1": {
            "default_min_nodes": 1,
            "default_max_nodes": 14,
            "default_predicate_factor": 0.3,
            "default_max_predicate_columns": 16,
            "db": "ergastf1", 
            "user": "pilotscope", 
            "password": "", 
            "host": "localhost",
            "port": 5432,
            "pk_fk_only": True,
            "seed": seed
        },
    }
    
    # only generate queries with 0~2 joins
    if small_query:
        query_gen_args[db]["default_max_nodes"] = 3

    query_generator = QueryGenerator(schema=schema, **query_gen_args[db])

    return query_generator


def generate_worker(db: str, small_query: bool, queue: Queue, event, timeout_seconds: int, seed: int, user: str, password: str, host: str = "localhost", port: int = 5432):
    query_generator = get_query_generator(db, seed, user, password, host, port, small_query)
    db_name = "imdb" if db.startswith("imdb") else db
    generated_queries: Set[Tuple[str]] = set()

    with RobustConnection(db_name, user, password, host, port, retry=False) as conn:
        while not event.is_set():
            query: SQLQuery = query_generator.generate_query()
            sql = query.sql()
            sorted_tuple_repr = sorted_tuple_representation(sql)
            if sorted_tuple_repr in generated_queries:
                continue
            generated_queries.add(sorted_tuple_repr)

            results, exec_time = conn.run(sql, timeout_seconds)
            if results is None:
                continue
            card = results[0][0]
            if card != 0:
                queue.put((sql, card, exec_time))
            
    query_generator._schema.close()


def write_worker(requied_query_num: int, queue: Queue, event, output_file: str):
    cur_query_num = 0
    exsiting_queries: Set[Tuple[str]] = set()

    with open(output_file, "a") as f:
        while cur_query_num < requied_query_num:
            time.sleep(5)
            while not queue.empty():
                item = queue.get()
                sql, card, exec_time = item
                sorted_tuple_repr = sorted_tuple_representation(sql)
                if sorted_tuple_repr not in exsiting_queries:
                    exsiting_queries.add(sorted_tuple_repr)
                    cur_query_num += 1
                    f.write("%s||%d||%f\n" % (sql, card, exec_time))
                    f.flush()

                    if cur_query_num >= requied_query_num:
                        event.set()
                        break


def get_template_based_query_generator(db: str, seed: int, user: str, password: str, host: str = "localhost", port: int = 5432) -> TemplateBasedQueryGenerator:
    schema = get_schema(db, user, password, host, port)    
    query_generator = TemplateBasedQueryGenerator(schema=schema, seed=seed)
    return query_generator


def generate_worker_template_based(db: str, template_queue: Queue, result_queue: Queue, event, timeout_seconds: int, seed: int, user: str, password: str, host: str = "localhost", port: int = 5432):
    query_generator = get_template_based_query_generator(db, seed, user, password, host, port)
    generated_queries: Set[Tuple[str]] = set()
    rcc = RobustCardinalityCalculator(query_generator._schema, host, db, user, password, port)
    
    def run_query(query: SQLQuery, sql: str, sorted_tuple_repr: Tuple[str]) -> Tuple[int, float]:
        nonlocal rcc
        start_run = time.perf_counter()
        card = rcc.run(sql)
        exec_time = time.perf_counter() - start_run
        
        if card != 0:
            generated_queries.add(sorted_tuple_repr)
        
        return card, exec_time
    
    while not event.is_set():
        # A Circular Queue
        template_str = template_queue.get()
        template = SQLTemplate.parse(query_generator.get_schema(), template_str)
        template_queue.put(template_str)
        
        card = 0
        
        # Get a valid query
        retry_count, max_try_limit = 1, 100
        while card == 0 and retry_count < max_try_limit:
            query: SQLQuery = query_generator.generate_query(template)
            sql = query.sql()
            sorted_tuple_repr = sorted_tuple_representation(sql)
            
            if sorted_tuple_repr in generated_queries:
                retry_count += 1
                continue
            
            card, exec_time = run_query(query, sql, sorted_tuple_repr)
        
        if card == 0:
            continue
        
        result_queue.put((sql, card, exec_time))
        
        if event.is_set():
            break
        
        # Run all subqueries
        subqueries = query.all_subqueries(contain_itself=False)
        for subquery in subqueries:
            sql = subquery.sql()
            sorted_tuple_repr = sorted_tuple_representation(sql)
            if sorted_tuple_repr not in generated_queries:
                card, exec_time = run_query(subquery, sql, sorted_tuple_repr)
                if card != 0:
                    result_queue.put((sql, card, exec_time))

    
def generate_workload(db: str, query_num: int, process_num: int, output_file: str, log_file: str, timeout_seconds: int, seed: int,\
    user: str, password: str, host: str = "localhost", port: int = 5432, template_file: Optional[str] = None, small_query: bool = False):
    
    args = locals()

    has_template = template_file is not None
    if has_template:
        template_queue = Queue()
        with open(template_file, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                template_queue.put(line)
    result_queue = Queue()

    start_time = time.time()
    
    event = Event()

    processes = []
    proc_id = []

    for i in range(process_num):
        # different seed for each process
        if has_template:
            p = Process(target=generate_worker_template_based, args=(db, template_queue, result_queue, event, timeout_seconds, seed + i, user, password, host, port))
        else:
            p = Process(target=generate_worker, args=(db, small_query, result_queue, event, timeout_seconds, seed + i, user, password, host, port))
        p.start()
        processes.append(p)
        proc_id.append(p.pid)
    
    writer = Process(target=write_worker, args=(query_num, result_queue, event, output_file))
    writer.start()
    proc_id.append(writer.pid)
    with open(log_file, "a") as f:
        f.write("Now: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        f.write("Database: " + db + "\n")
        f.write("Arguments: " + str(args) + "\n")
        f.write("Subprocesses: " + " ".join([str(pid) for pid in proc_id[:-1]]) + "\n")
        f.write("Writer: %d\n" % proc_id[-1])
        
    writer.join()
    
    for pid in proc_id:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    for p in processes:
        p.join()

    time_used = time.time() - start_time
    with open(log_file, "a") as f:
        f.write("Time used: %f hours.\n" % (time_used / 3600))
        f.write("-----------------Finished-----------------\n\n")


def sorted_tuple_representation(sql: str) -> Tuple[str]:
    from_and_where = sql.split(" FROM ")[1].strip().replace(";", "").split(" WHERE ")
    
    tables: List[str] = []
    predicates: List[str] = []

    table_str, predicate_str = from_and_where[0], ""
    if len(from_and_where) == 2:
        predicate_str = from_and_where[1]
    
    for table_alias in table_str.strip().split(','):
        tables.append(table_alias.strip().split()[-1])
    
    if predicate_str != "":
        for predicate in predicate_str.strip().split(' AND '):
            predicates.append(predicate.replace("(", "").replace(")", "").strip())
    
    return tuple(sorted(tables) + sorted(predicates))

def main():
    parser = argparse.ArgumentParser(description='Generate training workload queries for a given database')

    parser.add_argument('--db', type=str, choices=['imdb', 'stats', 'ergastf1', 'imdb_m', 'imdb_large'], help='Database name', required=True)
    parser.add_argument('--query_num', type=int, default=1024, help='Number of queries', required=True)
    parser.add_argument('--process_num', type=int, default=16, help='Number of parallel processes')
    parser.add_argument('--output_file', type=str, help='Output file name', required=True)
    parser.add_argument('--template_file', type=str, default=None, help='Template file path')
    parser.add_argument('--small_query', action='store_true', help="Generate small queries with 0~2 joins")
    parser.add_argument('--log_file', type=str, default="generate_workload.log", help='Log file')
    parser.add_argument('--timeout_seconds', type=int, default=300, help='Timeout in seconds')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')
    parser.add_argument('--user', type=str, default='pilotscope', help='Database user')
    parser.add_argument('--password', type=str, default='pilotscope', help='Database password')
    parser.add_argument('--host', type=str, default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')

    generate_workload(**vars(parser.parse_args()))

if __name__ == "__main__":
    main()
