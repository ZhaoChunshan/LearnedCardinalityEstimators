# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: disable an unused FiltersProcessor import.

import os
import re
import string
import functools
import itertools
import networkx as nx
import numpy as np
import pandas as pd

from src.database.query_loader import QueryLoader
# from src.encoding.filters import FiltersProcessor  # dead code: only used by SchemaExtractor
from src.utils.path_helper import db_content_path, WORKLOADS
from collections import defaultdict
from src.utils.query_helper import parse_sql_query
from src.encoding.join_graph import JoinGraph, JoinType, RedundantJoinException, JoinNotPossibleException

from src.utils.debug import timeit_decorator


class ParsingNotPossible(Exception):
    "Raised when parsing fails, i.e. most likely CROSS join has been found."
    pass

def get_table_columns(table, executor):
    columns = executor.execute(f"SELECT * FROM information_schema.columns WHERE table_schema = 'public'   AND table_name = '{table}';").fetchall()
    columns = [col[0] for col in columns]
    return columns

def process_workload(path_to_workload):
    """
    Returns the tables, corresponding filter/join columns that are specific for particular workload
    """
    pass


class PlanParser:
    """
    Basic parser of Postgres EXPLAIN query plan. Requires the query to be executed with a prefix `EXPLAIN (VERBOSE, FORMAT JSON)`,
    and later on extracted as `executor.execute(query).fetchall()[0][0][0]['Plan']`
    
    Currently is used in order to extract information about join order/type, and the estimated number of rows (the last one can be easily
    changed/extended to real cardinality or execution time).
    """
    def __init__(self, query, plan):
        self._reset_join_path()
        self.query = query
        self.plan = plan
        self._extract_join_conditions()
        
    def _reset_join_path(self):
        self.join_path = {'children': []}
        
    def _extract_join_conditions(self):
        query_tables, alias_dict, query_conditions = parse_sql_query(self.query)
        self.join_conditions = [set(cond['columns']) for cond in query_conditions if cond['is_join']]

    @staticmethod
    def process_condition(cond):
        """
        Splits condition and returns aliases
        
        Sometimes Nested Loop Join is possible to be done on several conditions,
        i.e. multiple edges exist, e.g. `ml.movie_id = t1.id AND kt1.id = t1.kind_id`.
        
        This happens most likely because while still doing nested for-loop, Postgres can filter out
        more data using several conditions. It's important to notice that anyway all conditions will 
        contain target table to be joined (e.g. `t1` in the case described above)
        
        As of now we are simply taking the first condition after splitting on `AND` without processing of
        multi-edge cases separately - under the assumption that first condition is the main one, and rest
        added by Postgres are redundant and made for efficiency of filtering out the data.
        """
        cond = cond.replace('(', '').replace(')', '')
        cond = cond.split('AND')[0]
        cond = cond.replace('(', '').replace(')', '')
        # 2024-02-28 suli: this should solve the issues with e.g. `(c1.date > (c2.date + '6 mons'::interval))`
        conditions = []
        for string in cond.split(' '):
            if re.search(r'\.', string):
                conditions.append(string)
        left_condition, right_condition = conditions
        # Sometimes there could be a space in condition
        left_condition = left_condition.strip()
        right_condition = right_condition.strip()
        
        left_alias = left_condition.split(".")[0]
        right_alias = right_condition.split(".")[0]
            
        return left_alias, right_alias, (left_condition, right_condition)

    def traverse(self, current, parent_join, for_postgres=False):
        if 'Plans' in current:
            # print(current['Node Type'])
            for plan in current['Plans']:
                # print(plan['Node Type'])
                parent_join = self.traverse(plan, parent_join, for_postgres=for_postgres)

        node_type = current['Node Type']
        condition = None
        if node_type == "Hash Join" or node_type == "Merge Join" or node_type == "Nested Loop":
            # Inspired by: https://github.com/postgres/pgadmin4/blob/c1ba645dceed5c9551a5f408e37a14d1041ee598/web/pgadmin/misc/static/explain/js/explain.js#L617
            if 'Hash Cond' in current:
                condition = current['Hash Cond']
                join_type = 'HashJoin'
            elif 'Merge Cond' in current:
                condition = current['Merge Cond']
                join_type = 'MergeJoin'
            elif node_type == 'Nested Loop' and 'Join Filter' in current.keys():
                condition = current['Join Filter']
                join_type = 'NestLoop'
            elif node_type == 'Nested Loop' and 'Join Filter' not in current.keys():
                left_subplan, right_subplan = current['Plans']
                left_output = left_subplan['Output']
                right_output = right_subplan['Output']
                for cand_cond in itertools.product(left_output, right_output):
                    if set(cand_cond) in self.join_conditions:
                        condition = ' = '.join(cand_cond)
                        join_type = 'NestLoop'
                        break
            
            if condition:
                left_alias, right_alias, condition = self.process_condition(condition) 
                join = {'left': left_alias, 'right': right_alias, 'est_rows': current['Plan Rows'], 'condition': condition, 'join_type': join_type, 'children': []}
                parent_join['children'].append(join)
                return join
            else:
                if not for_postgres:
                    raise ParsingNotPossible("It's impossible to parse the plan. Most likely, CROSS join takes place in the plan.")
                else:
                    pass
                
        return parent_join
    
    def traverse_all(self, for_postgres=False):
        self._reset_join_path()
        return self.traverse(self.plan, self.join_path, for_postgres=for_postgres)
    
    
def gather_join_order(query, executor=None, for_postgres=False, n_runs=5, return_planning_time=False):
    query_tables, alias_dict, query_conditions = parse_sql_query(query)
    n_joins = len(query_tables) - 1
    explain_query = "EXPLAIN (VERBOSE, FORMAT JSON) " + query
    result = executor.execute(explain_query)
    if result is None:
        raise JoinNotPossibleException(f"Executor returned no result.")
    plan = result.fetchall()[0][0][0]
    
    is_timeout = False
    try:
        execution_time, execution_times, planning_time = executor.get_exec_time(query, log_execution=False, return_all=True, n_runs=n_runs, return_planning_time=True)
    except Exception as e:
        raise JoinNotPossibleException(f"Extracting execution time failed:\t{e}")
    else:
        if execution_time is None:
            is_timeout = True
            execution_times = [executor.statement_timeout] * 5
            planning_time = 0

    try:
        parser = PlanParser(query, plan['Plan'])
        parser.traverse_all(for_postgres=for_postgres)
    except Exception as e:
        raise JoinNotPossibleException(f"Plan Parser had an issue:\t{e}")

    ordered_joins = []
    while parser.join_path['children']:
        parser.join_path = parser.join_path['children'][0]
        ordered_joins.append((parser.join_path['condition'], parser.join_path['join_type']))
    
    if len(ordered_joins) != n_joins and not for_postgres:
        # When extracting plan for Postgres to e.g. for a vector with hint vector, bad NestLoop
        # parsing does not matter, as long as we still include all possible joins in ranking
        raise JoinNotPossibleException(f"Plan Parser extracted NestedLoop join incorrectly. Skipping query.")
    
    if return_planning_time:
        return ordered_joins, execution_times, planning_time
    return ordered_joins, execution_times
    
    
class PlanHinter:
    """
    Adding hints for `pg_hint_plan` on JoinGraph
    """ 
    @staticmethod
    def create_join_cond(join_type, *aliases):
        """
        Creates forced join condition on 2 aliases of tables for the `pg_hint_plan`
        """
        return f"{join_type}({' '.join(aliases)})"
    
    @staticmethod
    def form_sql(hint, query):
        """
        Creating the SQL query with hint
        """
        return f"/*+ {hint} */ EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {query}"
        
    def hint_from_history(self, join_graph: JoinGraph):
        """
        Returns inital query with the hint formed based on order and types
        of joins accrording to join history
        """
        hint = ''
        # hint = f"/*+ "
        for join_perf in join_graph.join_history:
            involved_aliases = join_perf.contracted_node_name.replace('(', '').replace(')', '').split(' ')
            hint += self.create_join_cond(join_perf.join_type.name, *involved_aliases)
            hint += '\n'

        # We take the last join_perf from the loop and turn it into a Leading hint
        # (for Postgres Policy idea: change this to include all history up to current state)
        #
        hint += f"Leading({join_perf.contracted_node_name})"
        # hint += "\n*/ \n"
        
        return self.form_sql(hint, join_graph.query)
    
    # @timeit_decorator
    def vec_to_hint(self, join_graph: JoinGraph, enc_vector: np.array) -> np.array:
        """
        Forming the ordered set of hints from ranked list for inputing to `pg_hint_plan`
        """
        hints = []
        available_edges = join_graph.get_available_joins(remove_duplicates=True)

        def generate_hints(hints, edge_tuple):
            for i in range(len(JoinType)):
                hint = f"{JoinType(i).name}({edge_tuple[2]} {edge_tuple[3]})"
                index = len(JoinType) * edge_tuple[1] + i
                hints.append((enc_vector[index], hint))
            return hints

        available_edges = sorted(map(
            lambda row: (
                row[1]['action_edge_idx'], row[1]['state_edge_idx'],
                row[1]['source'], row[1]['target']
            ),
            available_edges.iterrows()
        ), key=lambda x: x[0])
        hints = functools.reduce(generate_hints, available_edges, hints)

        hints = [x[1] for x in sorted(hints, key=lambda x: x[0])]
        return self.form_sql(' '.join(hints), join_graph.query)
