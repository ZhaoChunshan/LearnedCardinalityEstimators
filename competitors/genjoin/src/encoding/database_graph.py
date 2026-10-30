# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: construct workload join graphs from predicates.json and remove
# automatic connection cleanup in the destructor.

import os
import json

import networkx as nx
import pandas as pd
from src.database.postgres_executor import PostgresExecutor
from src.database.selectivity_extractor import PostgresSelectivityExtractor, PostgresHistogramExtractor
from src.utils.query_helper import parse_sql_query
from src.utils.path_helper import WORKLOADS, DB_CONFIG
from src.database.query_loader import QueryLoader


class SchemaExtractor:
    """
    Helps to extract database schema as a graph, on which arbitrary queries
    then could be generated
    """
    
    NUMERIC_TYPES = ['integer', 'smallint', 'timestamp']
    STRING_TYPES = ['text', 'character']
    
    def __init__(self, workload_name):
        self.workload_name = workload_name
        self.schema_name = WORKLOADS[workload_name].split('/')[0]
        self.name_to_alias = dict()
        self.alias_to_name = dict()
        self.schema_graph = nx.MultiGraph()

    def enrich_from_queries(self):
        """
        Initial schema graph creation using existing workload for edges building
        """
        ql = QueryLoader(self.workload_name)

        filters = defaultdict(set)
        target_columns = defaultdict(set)

        conditions_processed = set()  # monitoring if join condition has been added to the edge
        # Possible join columns
        for qid in ql.queries():
            query = ql.get_query(qid)
            if self.workload_name == 'stats':
                # query = query.rstrip().split('||')[1]
                query = query.replace('::timestamp', '')
            elif self.workload_name == 'stack':
                query = query.replace('::interval', '')
            query_tables, aliases_dict, query_conditions = parse_sql_query(query)
            self.alias_to_name.update(aliases_dict)
            # adding only "base" aliases into `name_to_alias`
            for alias, name in aliases_dict.items():
                if not bool(set(string.digits).intersection(set(alias))):
                    self.name_to_alias.update({name: alias})
            for condition in query_conditions:
                if condition['is_join']:
                    # creating connection between 2 nodes based on join condition
                    # sorting is done in order not to create multiple edges when order of conditions is different in different queries 
                    columns = tuple(sorted(map(lambda x: x.lower(), condition['columns'])))
                    col_name1, col_name2 = map(lambda x: x.split('.')[1], columns)
                    # not to confuse the columns and tables after sorting, extracting tables from `columns` but not from `condition['names']`
                    table1, table2 = map(lambda x: aliases_dict[x.split('.')[0]], columns)
                    # need to check that aliases are "base", i.e. have no numbers
                    base_left_alias = bool(set(string.digits).intersection(set(columns[0])))
                    base_right_alias = bool(set(string.digits).intersection(set(columns[1])))
                    target_columns[table1].update({col_name1})
                    target_columns[table2].update({col_name2})
                    if columns not in conditions_processed and not (base_left_alias + base_right_alias):
                        # As a hotfix adding tables from join condition:
                        # it does not drop edges if nodes and connection between them exists
                        self.schema_graph.add_node(table1)
                        self.schema_graph.add_node(table2)
                        self.schema_graph.add_edge(table1, table2, join_condition={*columns})
                        conditions_processed.add(columns)
                else:
                    full_name, *_ = condition['condition'].split()
                    table = aliases_dict[condition['names'][0]]
                    # FIXME: hotfix for not processing `title.title`
                    if not (table == 'title' and 'title' in full_name):
                        filters[table].add(full_name.split('.')[1].lower())
                    
        return target_columns, filters

    def enrich_from_schema(self, target_columns, filters):
        """
        Additional schema graph enrichment with information about columns, i.e.
        ones used for joins (target columns) and for filtering (of different types)
        """
        with open(os.path.join(db_content_path(), self.schema_name, 'schema.sql'), "r") as schema:
            table_processing = False
            for line in schema:
                if line.startswith("CREATE TABLE"):
                    table_processing = True
                    table_name = line.lstrip("CREATE TABLE ").rstrip(" (\n")
                    if table_name in target_columns.keys():  # checking that table participates in query workload
                        self.schema_graph.add_node(table_name, alias=self.name_to_alias[table_name],
                                                   numeric_columns=list(), string_columns=list(), 
                                                   target_columns=list())
                elif table_processing and ";" not in line:
                    column_name, column_type, *_ = line.lstrip().replace(',', '').split()
                    if column_type in self.NUMERIC_TYPES:
                        if column_name in filters[table_name]:
                            self.schema_graph.nodes[table_name]['numeric_columns'].append(column_name)
                        elif column_name in target_columns[table_name]:
                            self.schema_graph.nodes[table_name]['target_columns'].append(column_name)
                    elif column_type in self.STRING_TYPES and column_name in filters[table_name]:
                        self.schema_graph.nodes[table_name]['string_columns'].append(column_name)
                elif table_processing and ";" in line:
                    table_processing = False
                    
    def add_filter_processors(self):
        """
        Add FilterProcessors as an attribute of schema graph nodes
        """
        for node in self.schema_graph.nodes():
            fp = FiltersProcessor(node, self.schema_graph.nodes()[node]['alias'],
                                  self.schema_graph.nodes()[node]['target_columns'],
                                  string_columns=self.schema_graph.nodes()[node]['string_columns'],
                                  numeric_columns=self.schema_graph.nodes()[node]['numeric_columns'])
            self.schema_graph.nodes()[node]['fp'] = fp


class DatabaseGraph:
    def __init__(self, selectivity_extractor, workload_name: str = 'join_order_benchmark'):
        self.workload_name = workload_name

        self.ql = QueryLoader(self.workload_name)
        self.exec = PostgresExecutor.create_postgres_engine(local_port=DB_CONFIG[self.workload_name]['DB_PORT'], db_name=DB_CONFIG[self.workload_name]['DB_NAME'])
        self.selectivity_extractor = selectivity_extractor

        # make sure graph is calculated or loaded on creation
        self.build_edgelist()

    def load_primary_keys(self):
        q_pk = """
        SELECT ccu.table_name, ccu.column_name, ccu.constraint_name
            FROM information_schema.constraint_column_usage ccu
            JOIN information_schema.table_constraints tc
                ON tc.constraint_name = ccu.constraint_name AND constraint_type = 'PRIMARY KEY'
        """

        df_pk = pd.DataFrame(self.exec.execute(q_pk).fetchall())
        df_pk['combined'] = df_pk['table_name'] + '.' + df_pk['column_name']
        df_pk = df_pk.sort_values(['table_name', 'column_name']).reset_index(drop=True)
        return df_pk

    def load_foreign_keys(self):
        q_fk = """
        SELECT kcu.constraint_name, kcu.table_name AS t1_name, kcu.column_name AS t1_col, ccu.table_name AS t2_name, ccu.column_name AS t2_col
        FROM information_schema.key_column_usage kcu
            JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = kcu.constraint_name
            JOIN information_schema.table_constraints tc ON kcu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        """

        df_fk = pd.DataFrame(self.exec.execute(q_fk).fetchall())
        df_fk = df_fk.sort_values(['t1_name', 't1_col', 't2_name', 't2_col']).reset_index(drop=True)
        return df_fk

    def load_tables(self):
        q_t = "SELECT table_name FROM information_schema.tables WHERE table_schema ='public'"

        df_tables = pd.DataFrame(self.exec.execute(q_t).fetchall())
        df_tables = df_tables.sort_values(['table_name']).reset_index(drop=True)
        return df_tables

    def evaluate_join_conditions(self, df_pk):
        primary_keys = df_pk['combined'].tolist()
        all_joins = list()
        aliases = dict()

        for q_ident in self.ql.queries():
            q = self.ql.get_query(q_ident)
            # print(q_ident)

            try:
                query_tables, aliases_dict, query_conditions = parse_sql_query(q)
            except:
                print(q_ident)
            else:
                aliases = dict(aliases, **aliases_dict)
                all_joins.extend([x['columns'] for x in query_conditions if x['is_join']])

        all_join_conditions = list()

        for names in all_joins:
            l_cond, r_cond = sorted(names)

            l_alias, l_col = l_cond.split('.')
            r_alias, r_col = r_cond.split('.')

            left = f"{aliases[l_alias]}.{l_col}"
            right = f"{aliases[r_alias]}.{r_col}"

            if left in primary_keys:
                if right in primary_keys:
                    type = 'pk_pk'
                else:
                    type = 'left_pk'
            elif right in primary_keys:
                type = 'right_pk'
            else:
                type = 'fk_fk'

            new_cond = [left, right, type]

            if new_cond not in all_join_conditions:
                all_join_conditions.append(new_cond)

        print(f"Unique JOIN conditions: {len(all_join_conditions)}")

        return all_join_conditions

    def build_conditions(self, all_join_conditions):
        df_cond = pd.DataFrame(all_join_conditions, columns=['left', 'right', 'join_type'])

        df_cond['left_table'] = [x[0] for x in df_cond['left'].str.split('.')]
        df_cond['left_column'] = [x[1] for x in df_cond['left'].str.split('.')]
        df_cond['right_table'] = [x[0] for x in df_cond['right'].str.split('.')]
        df_cond['right_column'] = [x[1] for x in df_cond['right'].str.split('.')]

        return df_cond

    def build_edgelist(self):
        # 2023-11-18 suli: this graph needs to be re-created when switching from cardinalities and historgrams
        file_path = f"data/processed/{WORKLOADS[self.workload_name]}/possible_joins__directed.edgelist"

        if os.path.exists(file_path):
            G = nx.read_edgelist(path=file_path, create_using=nx.MultiDiGraph())
            print(f"Database graph edge list already exists here: {file_path}")
            self.edge_list = nx.to_pandas_edgelist(G)
            self.graph = G

        elif self.workload_name in ('job_light', 'stats_ceb', 'ergastf1'):
            # New workloads: build from predicates.json instead of DB schema
            pred_path = f"workloads/{WORKLOADS[self.workload_name]}/predicates.json"
            self.build_edgelist_from_predicates(pred_path)

        else:
            df_pk = self.load_primary_keys()
            df_fk = self.load_foreign_keys()
            df_tables = self.load_tables()

            all_join_conditions = self.evaluate_join_conditions(df_pk)
            df_cond = self.build_conditions(all_join_conditions)

            alias_selectivities = self.selectivity_extractor.get_full_cardinalities()

            G = nx.MultiDiGraph()
            for i, row in df_tables.iterrows():
                G.add_node(row['table_name'])

            state_edge_idx = 0
            action_edge_idx = 0
            offset = 0

            for i, row in df_cond.iterrows():
                left = min(row['left'], row['right'])
                right = max(row['left'], row['right'])
                # 2023-11-18 suli: using it only in case of histograms
                if isinstance(self.selectivity_extractor, PostgresHistogramExtractor):
                    left_hist_size = len(alias_selectivities[row['left_table']][row['left_column']]['hist'])
                    right_hist_size = len(alias_selectivities[row['right_table']][row['right_column']]['hist'])
                else:
                    left_hist_size = 1
                    right_hist_size = 1
                join_type = 'FK-FK' if row['join_type'] == 'fk_fk' else 'PK-FK'
                G.add_edge(left.split('.')[0], right.split('.')[0],
                           join_condition=(left, right),
                           join_type=join_type,
                           state_edge_idx=state_edge_idx,
                           action_edge_idx=action_edge_idx,
                           left_hist_size=left_hist_size,
                           right_hist_size=right_hist_size,
                           offset=offset,
                           join_directionality=0
                )
                action_edge_idx += 1
                offset += (1 + 3 + left_hist_size + right_hist_size)

                G.add_edge(right.split('.')[0], left.split('.')[0],
                           join_condition=(right, left),
                           join_type=join_type,
                           state_edge_idx=state_edge_idx,
                           action_edge_idx=action_edge_idx,
                           left_hist_size=right_hist_size,
                           right_hist_size=left_hist_size,
                           offset=offset,
                           join_directionality=1
                )
                action_edge_idx += 1
                state_edge_idx += 1
                offset += (1 + 3 + left_hist_size + right_hist_size)

            nx.write_edgelist(G, path=file_path)
            print(f"Saved database graph edge list to {file_path}")
            self.edge_list = nx.to_pandas_edgelist(G)
            self.graph = G

        self.num_undirected_edges = self.edge_list['state_edge_idx'].nunique()

    def build_edgelist_from_predicates(self, predicates_path: str):
        """
        Build possible join edge list from predicates.json.
        Format: {"tables": [...], "join_conditions": ["t1.col=t2.col", ...], "predicates": {...}}
        """
        with open(predicates_path, 'r') as f:
            predicates = json.load(f)

        join_conditions = predicates['join_conditions']
        edges = []
        for i, jc in enumerate(join_conditions):
            left, right = jc.split('=')
            left_table, left_col = left.strip().split('.')
            right_table, right_col = right.strip().split('.')

            state_edge_idx = i
            edges.append({
                'source': left_table,
                'target': right_table,
                'join_condition': (f"{left_table}.{left_col}", f"{right_table}.{right_col}"),
                'state_edge_idx': state_edge_idx,
                'action_edge_idx': state_edge_idx * 2,
                'join_directionality': 0,
            })
            edges.append({
                'source': right_table,
                'target': left_table,
                'join_condition': (f"{right_table}.{right_col}", f"{left_table}.{left_col}"),
                'state_edge_idx': state_edge_idx,
                'action_edge_idx': state_edge_idx * 2 + 1,
                'join_directionality': 1,
            })

        self.edge_list = pd.DataFrame(edges)
        self.num_undirected_edges = len(join_conditions)

        # Persist as edgelist file for downstream compatibility
        G = nx.from_pandas_edgelist(self.edge_list, edge_attr=True, create_using=nx.MultiDiGraph())
        edgelist_path = f"data/processed/{WORKLOADS[self.workload_name]}/possible_joins__directed.edgelist"
        os.makedirs(os.path.dirname(edgelist_path), exist_ok=True)
        nx.write_edgelist(G, edgelist_path)
        self.graph = G
        print(f"Saved database graph edge list to {edgelist_path}")
        return self.edge_list
