#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import re
import time
import argparse
import polars as pl
import multiprocessing as mp


from typing import Tuple, List, Set

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

from query.cardinality_calculator import CardinalityCalculator
from query.query_generator import QueryGenerator
from schema.schemas import imdb_schema, stats_schema, ergastf1_schema
from pilotscope import PilotTransData
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor


parser = argparse.ArgumentParser(description="Generate SQL query & Collect true / estimated cardinality")

parser.add_argument("--output", required=True, help="output file path")
parser.add_argument("--query_set", default="imdb")
parser.add_argument("--snapshot", type=int, default=None, help="inital data percentage of snapshot",
                    choices=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 
                             32, 36, 40, 42, 48, 54, 56, 60, 64, 72, 80],)
parser.add_argument("--query_num", type=int, default=10000)
parser.add_argument("--num_workers", type=int, default=16, help="number of parallel workers")
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=5432)
parser.add_argument("--user", default="pilotscope")
parser.add_argument("--password", default="")


args = parser.parse_args()


def _init_worker(user: str, password: str, host: str, port: int, queryset: str, p: int):
    assert queryset in ['job_light', 'stats_ceb', 'ergastf1']
    
    db = {"job_light": "imdb", "stats_ceb": "stats", "ergastf1": "ergastf1"}[queryset]
    db_name = db + f"_p{p}"
    
    table_csv_path = f"datasets/{db}/snapshots/{p}"
    
    if queryset == 'job_light':
        schema = imdb_schema(user, password, db_name, host, port)
    elif queryset == 'stats_ceb':
        schema = stats_schema(user, password, db_name, host, port)
    elif queryset == 'ergastf1':
        schema = ergastf1_schema(user, password, db_name, host, port)
    
    tbls_cols_types = load_tbls_cols_types(f"datasets/{db}/snapshots/{p}/postgres_create_{db_name}.sql")
    
    card_calculator = CardinalityCalculator(schema, host, db_name, user, password, port, table_csv_path, tbls_cols_types)
    
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
            "seed": p   # Simply use snapshot number as seed
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
            "seed": p   # Simply use snapshot number as seed
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
            "seed": p   # Simply use snapshot number as seed
        },
    }
    
    query_generator = QueryGenerator(schema=schema, **query_gen_args[db])

    pilot_config = PostgreSQLConfig(db_host=host, db_port=port, db_user=user, db_user_pwd=password, db=db_name, sql_execution_timeout=3600 * 24 * 7)
    data_interactor = PilotDataInteractor(pilot_config)

    return card_calculator, query_generator, data_interactor


def generate_worker_on_snapshot(user: str, password: str, host: str, port: int, queryset: str, p: int, queue: mp.Queue, event):
    
    card_calculator, query_generator, data_interactor = _init_worker(user, password, host, port, queryset, p)
    
    generated_queries: Set[Tuple[str]] = set()

    while not event.is_set():
        
        query = query_generator.generate_query()
        sql = query.sql()
        
        sorted_tuple_repr = sorted_tuple_representation(sql)
        if sorted_tuple_repr in generated_queries:
            continue
        
        generated_queries.add(sorted_tuple_repr)

        true_card = card_calculator.calculate_cardinality(query)
        
        if true_card == 0:
            continue
        
        est_card, query_size = 0.0, 0
        subqueries_with_est = []
        
        data_interactor.pull_subquery_card()
        data: PilotTransData = data_interactor.execute(sql)
        
        for subquery_sql, subquery_est_card in data.subquery_2_card.items():
            
            comment_pattern = r"\/\*(.*)\*\/"
            comment_match = re.search(comment_pattern, subquery_sql)
            
            if comment_match:
                comment = comment_match.group(1)
                comment = comment.replace("(", "").replace(")", "").strip()
                cur_size = len(comment.split(","))
                
                # The largest subquery is the query itself
                if cur_size > query_size:
                    query_size = cur_size
                    est_card = subquery_est_card

                # Only add estimated card for subqueries with size 1
                if cur_size == 1:
                    subqueries_with_est.append((subquery_sql, subquery_est_card))

            else:
                assert(False)

        queue.put((sql, true_card, est_card, subqueries_with_est))


def write_worker(requied_query_num: int, queue: mp.Queue, event, output_file):
    cur_query_num = 0
    exsiting_queries: Set[Tuple[str]] = set()

    with open(output_file, "w") as f:
        while cur_query_num < requied_query_num:
            try:
                item = queue.get(block=True, timeout=1) 
            except mp.queues.Empty:
                continue
            
            sql, true_card, est_card, subqueries_with_est = item
            
            sorted_tuple_repr = sorted_tuple_representation(sql)
            if sorted_tuple_repr in exsiting_queries:
                continue

            exsiting_queries.add(sorted_tuple_repr)
            cur_query_num += 1

            f.write("%s||%d||%f\n" % (sql, true_card, est_card))
            
            for subquery_sql, est_card in subqueries_with_est:
                f.write("%s||%f\n" % (subquery_sql, est_card))
            
            f.flush()

            if cur_query_num >= requied_query_num:
                event.set()
                break


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


def load_tbls_cols_types(create_db_sql_path):
    assert os.path.exists(create_db_sql_path)

    tbls_cols_types = {}
    
    with open(create_db_sql_path, 'r') as file:
        for line in file:
            if "create table" in line:
                tbl = line.split("create table")[1].split()[0].replace('"', '')
                tbls_cols_types[tbl] = {}
            elif "integer" in line or "bigint" in line or "smallint" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Int64
            elif "character" in line or "varchar(" in line or "char(" in line: 
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.String
            elif "decimal(" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Float64
            elif "double precision" in line or "float" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Float64
            elif "date" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Date
            elif "interval" in line:
                col = line.strip().split(" ")[0].replace('"', '')
                tbls_cols_types[tbl][col] = pl.Duration
            else:
                pass

    return tbls_cols_types


def main():

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
        
    start_time = time.time()

    queue = mp.Queue() 
    event = mp.Event()

    processes = []
    proc_id = []

    # generate_worker_on_snapshot(args.user, args.password, args.host, args.port, args.query_set, args.snapshot, queue, event)
    for i in range(args.num_workers):
        p = mp.Process(target=generate_worker_on_snapshot, args=(args.user, args.password, args.host, args.port, args.query_set, args.snapshot, queue, event))
        p.start()
        processes.append(p)
        proc_id.append(p.pid)
    
    writer = mp.Process(target=write_worker, args=(args.query_num, queue, event, args.output))
    writer.start()
    proc_id.append(writer.pid)
   
    print("Now: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
    print("Arguments: " + str(args) + "\n")
    print("Subprocesses: " + " ".join([str(pid) for pid in proc_id[:-1]]) + "\n")
    print("Writer: %d\n" % proc_id[-1])
    
    writer.join()
    
    for p in processes:
        p.join()

    time_used = time.time() - start_time
    
    print("Time used: %f hours.\n" % (time_used / 3600))
    print("-----------------Finished-----------------\n\n")


if __name__ == '__main__':
    main()
