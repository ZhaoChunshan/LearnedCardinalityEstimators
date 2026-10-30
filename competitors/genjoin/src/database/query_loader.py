# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: load the study workloads and their SQL/metadata formats, with
# filtering of unsupported training queries.

import os
import sys
import numpy as np
import pandas as pd

from src.utils.path_helper import queries_path


WORKLOADS = {
    'join_order_benchmark': 'JOB/JOB',
    'join_order_benchmark_extended': 'JOB/JOB-EXTENDED',
    'join_order_benchmark_light': 'JOB/JOB-LIGHT',
    'stats': 'STATS-CEB',
    'stack': 'STACK'
}


class QueryLoader:
    def __init__(self, workload, debug=False):
        self.workload = workload

        # New workloads live under workloads/, not data/queries/
        if workload in ('job_light', 'stats_ceb', 'ergastf1'):
            workload_path = os.path.join('workloads', workload)
        else:
            workload_path = os.path.join(queries_path(), WORKLOADS[self.workload])
        assert os.path.exists(workload_path) and os.path.isdir(workload_path), f"Should choose a workload that exists. Are you sure that the directory {workload_path} exists?"
        self.workload_path = workload_path
        self._init_loader()

        if debug:
            print(f"There are {self.num_queries()} queries in the workload '{self.workload}' located at '{self.workload_path}'.")
    
    def _init_loader(self):
        if self.workload == 'stats':
            self.loader = SingleFileLoader(self.workload, self.workload_path)
        elif self.workload in ('job_light', 'stats_ceb', 'ergastf1'):
            self.loader = WorkloadFileLoader(
                self.workload, self.workload_path,
                file_pattern='train_workload.sql'
            )
        else:
            self.loader = MultipleFileLoader(self.workload, self.workload_path)

    @staticmethod
    def available_workloads():
        return WORKLOADS.keys()

    def sample(self):
        qid = np.random.choice(self.queries())
        return self.get_query(qid)

    def queries(self):
        try:
            self.query_files
        except AttributeError:
            self.query_files = self.loader.get_query_files()
        
        return self.query_files

    def get_query(self, query_ident):
        return self.loader.get_query(query_ident)

    def num_queries(self):
        return len(self.queries())


class FileLoader:
    def __init__(self, workload, workload_path):
        self.workload = workload
        self.workload_path = workload_path

class MultipleFileLoader(FileLoader):
    def get_query_files(self):
        return [f for f in os.listdir(self.workload_path) if f.endswith('.sql')]

    def get_query(self, query_ident):
        if not query_ident.endswith('.sql'):
            query_ident += '.sql'

        if query_ident.startswith('0'):
            query_ident = query_ident.strip('0')

        query_file_path = os.path.join(self.workload_path, query_ident)
        if not os.path.exists(query_file_path):
            raise FileNotFoundError(f"The file {query_file_path} does not exist.")

        with open(query_file_path, 'r') as f:
            return ''.join(f.readlines())
        
class SingleFileLoader(FileLoader):
    def __init__(self, workload, workload_path):
        super().__init__(workload, workload_path)

        if self.workload == 'stats':
            self.df = pd.read_csv(os.path.join(workload_path, 'stats_CEB.sql'), sep='|', header=None)
            self.df.columns = ['cardinality', '1', 'query']
        else:
            raise ValueError(f"Workload {self.workload} should be added to SingleFileLoader.")

    def get_query_files(self):
        return list(range(self.df.shape[0]))

    def get_query(self, query_ident):
        return self.df.loc[int(query_ident), 'query']


class WorkloadFileLoader(FileLoader):
    """
    Loads queries from a single file. Handles two formats:
      1. Plain SQL (one per line) — for test queries.
      2. Pipe-separated: SQL||TrueCard||ExecTime — extracts SQL only.
    """
    def __init__(self, workload, workload_path, file_pattern):
        super().__init__(workload, workload_path)
        self.file_pattern = file_pattern
        self._load_queries()

    def _load_queries(self):
        file_path = os.path.join(self.workload_path, self.file_pattern)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Workload file not found: {file_path}")
        self.queries = []
        skipped = 0
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '||' in line:
                    sql = line.split('||')[0].strip()
                else:
                    sql = line.rstrip(';').strip()
                if not sql:
                    continue
                # For STATS-CEB, skip training queries containing 'lasteditoruserid' because the test workload does not use this column.
                if self.workload == 'stats_ceb' and 'lasteditoruserid' in sql:
                    skipped += 1
                    continue
                # Skip single-table queries (no joins to learn from)
                tables = sql.split(" FROM ")[1].split(" WHERE ")[0].split(",")
                if len(tables) < 2:
                    skipped += 1
                    continue
                self.queries.append(sql)
        if skipped:
            print(f"Skipped {skipped} single-table queries from {file_path}")

    def get_query_files(self):
        return list(range(len(self.queries)))

    def get_query(self, query_ident):
        return self.queries[int(query_ident)]
