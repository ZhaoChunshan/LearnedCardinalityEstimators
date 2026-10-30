# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: select databases explicitly and make SSH tunneling optional.

import os
import csv

from sqlalchemy import create_engine, text
from sshtunnel import SSHTunnelForwarder
import pandas.io.sql as psql
import time
import numpy as np
from decouple import config
from psycopg2.errors import QueryCanceled
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool


QUERY_LOG_FILE_PATH = 'data/queries/query_log_file.csv'


class PostgresExecutor:
    """
    Create executor for connection to PostgreSQL and running queries
    """
    def __init__(self, engine):
        self.engine = engine
        self.conn = self.engine.connect()
        self.statement_timeout = None
    
    @staticmethod
    def create_postgres_engine(local_port=config('DB_PORT', cast=int), db_name=None, statement_timeout=None, suppress_print=False):
        """
        In case you want to create a Postgres engine using the remote server DB, specify the variable `SSH_PRIVATE_KEY` in `.env` file in home
        directory via setting it as     e.g. `SSH_PRIVATE_KEY='/home/user/.ssh/id_rsa'`
        
        Parameters:
            - statement_timeout: Timeout before aborting the transaction (in milliseconds!)
        """
        port = local_port
        # Try SSH tunnel only if private key is configured
        ssh_key = config('SSH_PRIVATE_KEY', default='')
        ssh_user = config('SSH_USER', default='')
        if ssh_key and ssh_user:
            try:
                server = SSHTunnelForwarder(
                (config('DB_HOST'), 22),
                ssh_private_key=ssh_key,
                ssh_username=ssh_user,
                remote_bind_address=('localhost', port))
                server.start()
                port = str(server.local_bind_port)
                if not suppress_print:
                    print("Connected via SSH")
            except Exception as e:
                if not suppress_print:
                    print(f"SSH failed: {e}, connecting locally")
        elif not suppress_print:
            print("SSH skipped (no key configured), connecting locally")
            
        pg_cfg = f"postgresql://{config('DB_USER')}:{config('DB_PASSWORD')}@localhost:{port}/{db_name}"
        if port == local_port:
            print("Connected locally")

        if statement_timeout is None:
            statement_timeout = 0
        else:
            if not suppress_print:
                print(f"Adding timeout option to all queries, with a statement_timeout of {statement_timeout}")

        executor = None
        while executor is None:
            try:
                executor = PostgresExecutor(create_engine(pg_cfg, connect_args={'options': f'-c statement_timeout={statement_timeout}'}, poolclass=NullPool))
            except Exception as e:
                print(f"Trying to connect to Postgres, retrying in 10sec. Error occurred: {e}")
                time.sleep(10)
                
        if statement_timeout:
            executor.statement_timeout = statement_timeout

        return executor

    def execute(self, sql_query, num_retries=0):
        for _ in range(1+num_retries):
            try:
                result = self.conn.execute(text(sql_query))

            except Exception as e:
                print('pgexec.execute', str(e)[:150])
                self.conn.execute(text('ROLLBACK;'))
                time.sleep(8)
                result = None

            if result is not None:
                break
        
        return result
    
    def read_to_pandas(self, sql_query):
        df = psql.read_sql(text(sql_query), self.conn)
        return df
    
    def get_exec_time(self, raw_query, n_runs=3, return_all=False, log_execution=False, return_planning_time=False):
        """
        Returns aggregated execution time
        
        As of now returning the last one as it occurs to be the most stable after all the caching done.
        Default `n_runs` is set to 3 as exeprimentally first two are not yet converged (additionally, considering
        the 3rd element to be approximately the same as the median of 5 runs).
        """
        # print(f"PostgresExecutor - get_exec_time")
        exec_times = []
        planning_times = []

        if not 'EXPLAIN' in raw_query:
            sql_query = "EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) " + raw_query
        else:
            sql_query = raw_query

        is_timeout = False
        for _ in range(n_runs):
            try:
                result = self.conn.execute(text(sql_query)).fetchall()[0][0][0]
                exec_times.append(result['Execution Time'])
                planning_times.append(result['Planning Time'])
            
            except QueryCanceled as e:
                print('pgexec.get_exec_time', str(e)[:150])
                self.conn.execute(text('ROLLBACK;'))
                is_timeout = True
                break
                
            except OperationalError as e:
                print('pgexec.get_exec_time', str(e)[:150])
                self.conn.execute(text('ROLLBACK;'))
                is_timeout = True
                break

            except Exception as e:
                print('pgexec.get_exec_time', str(e)[:150])
                self.conn.execute(text('ROLLBACK;'))
                time.sleep(8)

        # Log Execution if query was running but not failed
        if log_execution:
            is_error = (len(exec_times) == 0) and (not is_timeout)
            if not is_error:
                final_execution_time = -1 if is_timeout else exec_times[-1]
                self.log_execution(sql_query, final_execution_time, is_timeout)

        # In case all N executions resulted in errors, there will be no
        # measured execution times -> return None
        #
        if return_planning_time:
            if len(exec_times) == 0:
                if return_all:
                    return None, [], None
                return None, None
            if return_all:
                return exec_times[-1], exec_times, planning_times[-1]
            return exec_times[-1], planning_times[-1]
        else:
            if len(exec_times) == 0:
                if return_all:
                    return None, []
                return None
            if return_all:
                return exec_times[-1], exec_times
            return exec_times[-1]

        
    
    def get_exec_time_w_error(self, raw_query: str, n_runs: int = 3, return_all: bool = False, 
                                                                     log_execution: bool = False):
        """
        Returns aggregated execution time
        
        As of now returning the last one as it occurs to be the most stable after all the caching done.
        Default `n_runs` is set to 3 as exeprimentally first two are not yet converged (additionally, considering
        the 3rd element to be approximately the same as the median of 5 runs).
        
        Returns:
            exec_time (float): execution time after `n_runs`
            error_type (str, default=None): the type of error if happened. Helps to catch timeouts etc.
        """
        print(f"PostgresExecutor - get_exec_time")
        exec_times = []

        if not 'EXPLAIN' in raw_query:
            sql_query = "EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) " + raw_query
        else:
            sql_query = raw_query

        error_type = None
        for _ in range(n_runs):
            try:
                exec_times.append(self.conn.execute(text(sql_query)).fetchall()[0][0][0]['Execution Time'])
            except QueryCanceled as e:
                print('pgexec.get_exec_time', str(e)[:150])
                self.conn.execute(text('ROLLBACK;'))
                error_type = 'timeout'
                break
            except OperationalError as e:
                print('pgexec.get_exec_time', str(e)[:150])
                self.conn.execute(text('ROLLBACK;'))
                # 2023-05-20 suli: both errors of timeout and "server closed the connection unexpectedly" are
                # `OperationalError`, so need to distinguish beteween them
                if 'timeout' in str(e):
                    error_type = 'timeout'
                else:
                    error_type = 'operational'
                break
            except Exception as e:
                print('pgexec.get_exec_time', str(e)[:150])
                self.conn.execute(text('ROLLBACK;'))
                time.sleep(8)

        # Log Execution if query was running but not failed
        if log_execution:
            is_error = (len(exec_times) == 0) and (not error_type)
            if not is_error:
                final_execution_time = -1 if error_type else exec_times[-1]
                self.log_execution(sql_query, final_execution_time, error_type)

        # In case all N executions resulted in errors, there will be no
        # measured execution times -> return None
        #
        if len(exec_times) == 0:
            if return_all:
                return None, [], error_type
            return None, error_type

        if return_all:
            return exec_times[-1], exec_times, error_type
        return exec_times[-1], error_type

    def close_connection(self):
        self.conn.close()

    def log_execution(self, sql_query: str, execution_time: float, is_timeout: bool):
        file_path = QUERY_LOG_FILE_PATH
        is_new_file = not os.path.isfile(file_path)
        open_mode = 'w' if is_new_file else 'a'

        with open(file_path, open_mode) as f:
            writer = csv.writer(f)

            if is_new_file:
                writer.writerow(('sql_query', 'execution_time', 'is_timeout'))
            writer.writerow((str(sql_query), str(execution_time), str(is_timeout)))    
