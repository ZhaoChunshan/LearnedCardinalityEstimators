# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: select the workload database and handle queries without filters.

import os
import time
import jsonpickle
from abc import ABC
import pandas as pd
import numpy as np
from collections import defaultdict

from mo_sql_parsing import parse
from src.database.postgres_executor import PostgresExecutor
# from src.encoding.database_graph import SchemaExtractor
from src.utils.path_helper import processed_path, WORKLOADS
from src.utils.query_helper import parse_sql_query
from tqdm.auto import tqdm


from src.utils.debug import timeit_decorator


class SelectivityExtractorCannotExtractException(Exception):
    """
    Raised when there is an error in trying to extract selectivities
    from the database.
    """
    pass


class SelectivityExtractor(ABC):
    def __init__(self, workload_name='JOB', database_name='imdbload', database_schema='public'):
        self.database_name = database_name
        self.database_schema = database_schema
        self.workload_name = workload_name
        self.workload_path =  WORKLOADS[workload_name]
        
    def _get_alias_dict(self, query: str) -> dict:
        parsed = parse(query)
        if isinstance(parsed['from'], list):  
            reverse_aliases_dict = {p['name']: p['value'] for p in parsed['from']}
        else:
            reverse_aliases_dict = {parsed['from']['name']: parsed['from']['value']}
        return reverse_aliases_dict
        
    def get_full_cardinalities(self) -> None:
        raise NotImplementedError('#get_full_cardinalities should be implemented on subclass.')


    def get_true_selectivity(self, query: str) -> dict:
        raise NotImplementedError('#get_true_selectivity should be implemented on subclass.')


    def get_estimated_selectivity(self, query: str) -> dict:
        raise NotImplementedError('#get_estimated_selectivity should be implemented on subclass.')


class PostgresSelectivityExtractor(SelectivityExtractor):
    # @timeit_decorator
    def __init__(self, workload_name='join_order_benchmark', database_name='imdbload', database_schema='public', executor=None):
        self.database_name = database_name
        # if workload_name == 'join_order_benchmark':
        #     self.database_name = 'imdbload'
        # elif workload_name == 'stack':
        #     self.database_name = 'so'
        self.database_schema = database_schema
        self.workload_name = workload_name
        self.workload_path = WORKLOADS[workload_name]
        if executor:
            self.executor = executor
        else:
            self.executor = PostgresExecutor.create_postgres_engine(db_name=self.database_name)
        self.get_full_cardinalities()  # required for parsing the selectivities!!!


    # @timeit_decorator
    def get_full_cardinalities(self):
        folder_path = os.path.join(processed_path(), self.workload_path)
        file_path = os.path.join(folder_path, 'full_table_cardinalities.csv')

        if os.path.exists(file_path):
            print("Loading full cardinalities from file ... ")
            self.full_cardinalities = pd.read_csv(file_path)
            
        else:
            print("There're no full cardinalities created yet, calculating from scratch ...")
            full_cardinalities = dict()
            res = self.executor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{self.database_name}' AND table_schema = '{self.database_schema}'", num_retries=1)
            if res is None:
                raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
            tables = res.fetchall()
            tables = [t[0] for t in tables]

            for table_name in tables:
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                res = self.executor.execute(count_query, num_retries=1)
                if res is None:
                    raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
                full_cardinalities[table_name] = res.fetchall()[0][0]

            self.full_cardinalities = pd.DataFrame(full_cardinalities, index=['total_rows']).transpose().reset_index()
            self.full_cardinalities.columns = ['table_name', 'total_rows']

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            self.full_cardinalities.to_csv(file_path, index=False)

        # Turn full cardinalities into a dictionary for faster access
        self.full_cardinalities = {row['table_name']: row['total_rows'] for _, row in self.full_cardinalities.iterrows()}


    # @timeit_decorator
    def _extract_cardinalities(self, node: dict, cardinalities: dict, num_rows_key = 'Plan Rows', do_estimate=False) -> dict:
        "Iterates through plan and returns the cardinalities"
        if 'Node Type' in node.keys():
            if 'scan' in str(node['Node Type']).lower():
                # The "Bitmap Index Scan" leaf node did not contain an Alias, but is parent
                # "Bitmap Heap Scan" did, we assume that for every alias, there will be at least
                # one node containing its cardinality
                if 'Alias' in node.keys():
                    # 2024-02-27 suli FIXME: hotfix for STACK with 'interval'
                    # 2024-02-28 suli TODO: it occurs that `Filter` on scan could be like `site_id = u2.site_id`, i.e. Postgres is making some additional filtering on other attribute - with '::' forcing we meanwhile ignore such filters and only allow numerical and text ones
                    if 'Filter' in node.keys() and 'interval' not in node['Filter'] and '::' in node['Filter']:
                        count_query = f"SELECT COUNT(*) FROM {node['Relation Name']} AS {node['Alias']} WHERE {node['Filter']}"
                        res = self.executor.execute(count_query, num_retries=0)
                        if res is None:
                            raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
                        cardinalities[node['Alias']] = res.fetchall()[0][0]
                    else:
                        # setting "full" cardinality if no filters were applied
                        cardinalities[node['Alias']] = self.full_cardinalities[self.full_cardinalities['table_name'] == node['Relation Name']]['total_rows'].item()
                        
        if 'Plans' in node.keys():
            for child in node['Plans']:
                cardinalities = self._extract_cardinalities(child, cardinalities, num_rows_key=num_rows_key, do_estimate=do_estimate)
        return cardinalities


    # This function parses the plan and extracts the filters, storing them in a dictionary
    # where the alias is the key and as value we store the table name and the filter (or None)
    # @timeit_decorator
    def _parse_plan_for_filters(self, node: dict, filters: dict) -> dict:
        if 'Node Type' in node.keys():
            if 'scan' in str(node['Node Type']).lower():
                # The "Bitmap Index Scan" leaf node did not contain an Alias, but is parent
                # "Bitmap Heap Scan" did, we assume that for every alias, there will be at least
                # one node containing its cardinality
                if 'Alias' in node.keys():
                    # 2024-02-27 suli FIXME: hotfix for STACK with 'interval'
                    # 2024-02-28 suli TODO: it occurs that `Filter` on scan could be like `site_id = u2.site_id`, i.e. Postgres is making some additional filtering on other attribute - with '::' forcing we meanwhile ignore such filters and only allow numerical and text ones
                    if 'Filter' in node.keys() and 'interval' not in node['Filter'] and '::' in node['Filter']:
                        filters[node['Alias']] = {
                            'filter': node['Filter'],
                            'table': node['Relation Name'],
                            'planned_rows': node['Plan Rows'] if 'Plan Rows' in node.keys() else None
                        }
                    else:
                        filters[node['Alias']] = {
                            'filter': None,
                            'table': node['Relation Name'],
                            'planned_rows': None
                        }
        if 'Plans' in node.keys():
            for child in node['Plans']:
                filters = self._parse_plan_for_filters(child, filters)
        return filters


    # new implementation using EXPLAIN estimates
    # @timeit_decorator
    def _extract_cardinalities__new(self, node: dict, cardinalities: dict, num_rows_key = 'Plan Rows', do_estimate=False) -> dict:
        filters_dict = self._parse_plan_for_filters(node, dict())
        for alias, filter_dict in filters_dict.items():
            if filter_dict['filter'] is not None:
                if do_estimate:
                    # Uses PostgreSQL EXPLAIN estimates
                    explain_query = f"EXPLAIN (FORMAT JSON) SELECT * FROM {filter_dict['table']} AS {alias} WHERE {filter_dict['filter']}"
                    res = self.executor.execute(explain_query, num_retries=0)
                    if res is None:
                        raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
                    cardinalities[alias] = res.fetchall()[0][0][0]['Plan']['Plan Rows']
                else:
                    # Extracts the true number of rows
                    count_query = f"SELECT COUNT(*) FROM {filter_dict['table']} AS {alias} WHERE {filter_dict['filter']}"
                    res = self.executor.execute(count_query, num_retries=0)
                    if res is None:
                        raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
                    cardinalities[alias] = res.fetchall()[0][0]
            else:
                cardinalities[alias] = self.full_cardinalities[filter_dict['table']]

        return cardinalities


    # new implementation using just the single EXPLAIN that is passed as plan node
    # from outside (see get_estimated_selecitvity and its call of this function)
    #
    # Remove extraction of planned rows in _parse_plan_for_filters in case this
    # method is not used further!
    #
    # @timeit_decorator
    def _extract_cardinalities__new_single_explain(self, node: dict, cardinalities: dict, num_rows_key = 'Plan Rows', do_estimate=False) -> dict:
        filters_dict = self._parse_plan_for_filters(node, dict())
        for alias, filter_dict in filters_dict.items():
            if filter_dict['filter'] is not None:
                if do_estimate:
                    # Uses estimated planned rows from outer single EXPLAIN
                    if filter_dict['planned_rows'] is not None:
                        cardinalities[alias] = filter_dict['planned_rows']
                    else:
                        cardinalities[alias] = self.full_cardinalities[filter_dict['table']]

                else:
                    # Extracts the true number of rows
                    count_query = f"SELECT COUNT(*) FROM {filter_dict['table']} AS {alias} WHERE {filter_dict['filter']}"
                    res = self.executor.execute(count_query, num_retries=0)
                    if res is None:
                        raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
                    cardinalities[alias] = res.fetchall()[0][0]
            else:
                cardinalities[alias] = self.full_cardinalities[filter_dict['table']]

        return cardinalities
    

    # @timeit_decorator
    def get_estimated_selectivity(self, query: str, alias_dict: dict=None, filter_conditions: list = None, use_logarithm: bool = True) -> dict:
        # print(f"SelectivityExtractor - get_estimated_selectivity")

        if alias_dict is None:
            explain_query = f"EXPLAIN (FORMAT JSON) {query}"
            alias_to_table = self._get_alias_dict(query)

        else:
            alias_to_table = alias_dict
            if not filter_conditions:
                return {alias: 1.0 for alias in alias_to_table}

            # go for SeqScan on all aliases, such that we can extract the individual result set size estimates
            filter_union = ' UNION '.join([f"SELECT COUNT(*) FROM {alias_to_table[condition['names'][0]]} AS {condition['names'][0]} WHERE {condition['condition']}" for condition in filter_conditions])
            # aliased_seq_scan_hints = ' '.join([f"SeqScan({x})" for x in alias_to_table.keys()])
            # explain_query = f"/*+ {aliased_seq_scan_hints} */ EXPLAIN (FORMAT JSON) {query}"
            explain_query = f"EXPLAIN (FORMAT JSON) {filter_union}"

        
        # 2023-04-28: Adding multiple tries to get the selecitivites, but raise an exception
        # to allow skipping this query during training
        exception_counter = 0
        result = None
        while result is None:
            result = self.executor.execute(explain_query, num_retries=0)
            if result is not None:
                result = result.fetchall()
            else:
                time.sleep(10)
                exception_counter += 1
                print(f"SelectivityExtractor - get_estimated_selectivity: Timeout #{exception_counter}")
                
                if exception_counter >= 5:
                    raise SelectivityExtractorCannotExtractException('SelectivityExtractor could not do his job.')
            
        plan = result[0][0][0]['Plan']
        selectivities = dict()

        # selectivities = self._extract_cardinalities(plan, selectivities, num_rows_key='Plan Rows', do_estimate=True)
        # selectivities = self._extract_cardinalities__new(plan, selectivities, num_rows_key='Plan Rows', do_estimate=True)
        selectivities = self._extract_cardinalities__new_single_explain(plan, selectivities, num_rows_key='Plan Rows', do_estimate=True)
        
        for alias in alias_to_table.keys():
            total_table_rows = self.full_cardinalities[alias_to_table[alias]]
            if alias in selectivities.keys():
                filtered_selectivity = selectivities[alias]
            else:
                filtered_selectivity = total_table_rows

            if use_logarithm:
                filtered_selectivity = np.log(filtered_selectivity + 1)
                total_table_rows = np.log(total_table_rows + 1)

            selectivities[alias] = filtered_selectivity / total_table_rows
        return selectivities


    def get_true_selectivity(self, query: str, use_logarithm: bool = True) -> dict:
        explain_query = f"EXPLAIN (ANALYZE, FORMAT JSON) {query}"
        alias_to_table = self._get_alias_dict(query)
        res = self.executor.execute(explain_query, num_retries=1)
        if res is None:
            raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
        plan = res.fetchall()[0][0][0]['Plan']

        selectivities = dict()
        selectivities = self._extract_cardinalities(plan, selectivities, num_rows_key='Actual Rows', do_estimate=False)
        
        for alias in selectivities.keys():
            total_table_rows = self.full_cardinalities[alias_to_table[alias]]         
            filtered_selectivity = selectivities[alias]

            if use_logarithm:
                filtered_selectivity = np.log(filtered_selectivity + 1)
                total_table_rows = np.log(total_table_rows + 1)

            selectivities[alias] = filtered_selectivity / total_table_rows
        return selectivities


class PostgresHistogramExtractor(SelectivityExtractor):
    """
    Extracting the selectivities in form of histograms instead of single values
    """
    def __init__(self, workload_name='join_order_benchmark', database_name='imdbload', database_schema='public'):
        self.database_name = database_name
        self.database_schema = database_schema
        self.workload_name = workload_name
        self.workload_path = WORKLOADS[workload_name]
        self.executor = PostgresExecutor.create_postgres_engine(db_name=self.database_name)
        # self.se = SchemaExtractor(workload_name)
        # self.target_columns, _ = self.se.enrich_from_queries()
        self.target_columns = []  # FIXME: due to a circular import meanwhile disabled
        self.get_full_cardinalities()  # required for parsing the selectivities!!!:
    
    def get_full_cardinalities(self):
        """
        Creating holder of original histograms for all PK/FK columns
        """
        folder_path = os.path.join(processed_path(), self.workload_path)
        file_path = os.path.join(folder_path, 'full_histograms.json')

        if os.path.exists(file_path):
            print(f"Loading full cardinalities from file {file_path} ")
            with open(file_path, 'r') as file:
                self.full_cardinalities = jsonpickle.decode(file.read())

        else:
            print(f"Need to generate full cardinalities for {self.workload_name} yet, calculating from scratch ...")
            self.full_cardinalities = dict()
            for table, columns in tqdm(self.target_columns.items(), total=len(self.target_columns.items()), desc='Iterating over tables...'):
                self.full_cardinalities[table] = defaultdict(dict)

                for target_column in tqdm(columns, total=len(columns), desc='Querying individual columns...', leave=False):
                    res = self.executor.execute(f"""
                    SELECT {target_column} FROM {table} WHERE {target_column} IS NOT NULL""", num_retries=1)
                    
                    if res is None:
                        raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
                    
                    rec_pg = res.fetchall()
                    result = [r[0] for r in rec_pg if r[0] is not None]
                    # self.full_cardinalities[table][target_column]['min'] = min(result)
                    # self.full_cardinalities[table][target_column]['max'] = max(result)
                    bins = np.histogram_bin_edges(result, bins='scott')
                    hist, _ = np.histogram(result, bins=bins)
                    self.full_cardinalities[table][target_column]['bins'] = bins
                    self.full_cardinalities[table][target_column]['hist'] = hist

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            with open(file_path, 'w') as file:
                file.write(jsonpickle.encode(self.full_cardinalities))

        return self.full_cardinalities
                
    def get_true_selectivity(self, query: str, use_alias: bool = True) -> dict:
        query_tables, aliases_dict, query_conditions = parse_sql_query(query)
        table_to_alias = {value: key for key, value in aliases_dict.items()}
        filters = defaultdict(list)

        for condition in query_conditions:
            if not condition['is_join'] and aliases_dict[condition['names'][0]] in self.target_columns.keys():
                filters[condition['names'][0]].append(condition['condition'])

        selectivities = dict()
        for alias, table in aliases_dict.items():
            for target_column in self.target_columns[table]:
                filter_condition = ' AND '.join(filters[alias]) if filters[alias] else '1=1'
                # print(filters[table], filter_condition)
                query = f"SELECT {target_column} FROM {table} AS {alias} WHERE {filter_condition}"
                res = self.executor.execute(query, num_retries=1)
                if res is None:
                    raise SelectivityExtractorCannotExtractException('An error occured querying Postgres.')
                rec_pg = res.fetchall()
                result = [r[0] for r in rec_pg if r[0] is not None]

                init_hist = self.full_cardinalities[table][target_column]['hist']
                hist, _ = np.histogram(result, bins=self.full_cardinalities[table][target_column]['bins'])
                hist = np.divide(hist, init_hist, out=np.zeros_like(hist, dtype='float64'), where=init_hist!=0)
                if use_alias:  # using alias by default for consistency with `JoinGraph`
                    selectivities[f'{alias}.{target_column}'] = hist
                else:
                    selectivities[f'{table}.{target_column}'] = hist
        return selectivities
