from ntpath import join
import re
import logging
from collections import defaultdict, OrderedDict
from copy import deepcopy
from pathlib import Path
import psycopg2
from mo_sql_parsing import parse
from src.encoding.plan import Plan

from src.utils.debug import timeit_decorator

LOG = logging.getLogger(__name__)


def explain_analyze(sql_query, conn):
    cur = conn.cursor()
    try:
        cur.execute(f"""EXPLAIN ANALYZE {sql_query}""")
        rows = cur.fetchall()
    except Exception as e:
        cur.close()
        raise e
    cur.close()
    for row in rows:
        print(row[0])


def _parse_single_query_condition(qc):
    simple_ops = {
        'eq': '=',
        'gt': '>',
        'gte': '>=',
        'lt': '<',
        'lte': '<=',
        'neq': '!='
    }
    like_ops = {
        'not_like': 'NOT LIKE',
        'like': 'LIKE',
        'ilike': 'ILIKE'
    }
    exists_ops = {
        'missing': 'IS NULL',
        'exists': 'IS NOT NULL',
    }

    k, v = list(qc.items())[0]
    is_join = False
    if k in simple_ops.keys():
        names = []
        names.append(v[0].split('.')[0])

        second_statement = v[1]
        if isinstance(second_statement, dict):
            if 'literal' in second_statement.keys():
                second_statement = f"'{second_statement['literal']}'"
            # Otherwise it should be a join condition like `b2.date > b1.date + '4 months'::interval`
            else:
                names.append(second_statement['add'][0].split('.')[0])
                is_join = True
        elif isinstance(second_statement, str):
            names.append(v[1].split('.')[0])
            is_join = True
        elif isinstance(second_statement, (int, float)):
            second_statement = str(second_statement)
        else:
            return None

        cond = f"{v[0]} {simple_ops[k]} {second_statement}"
    elif k == 'in':
        names = [v[0].split('.')[0]]
        if isinstance(v[1]['literal'], str):
            in_part = f"'{v[1]['literal']}'"
        else:
            in_part = ', '.join([f"'{state}'" for state in v[1]['literal']])
        cond = f"{v[0]} in ({in_part})"
    elif k == 'nin':
        names = [v[0].split('.')[0]]
        if isinstance(v[1]['literal'], str):
            in_part = f"'{v[1]['literal']}'"
        else:
            in_part = ', '.join([f"'{state}'" for state in v[1]['literal']])
        cond = f"{v[0]} not in ({in_part})"
    elif k in like_ops.keys():
        names = [v[0].split('.')[0]]
        cond = f"{v[0]} {like_ops[k]} '{v[1]['literal']}'"
    elif k in exists_ops.keys():
        v = [v]
        names = [v[0].split('.')[0]]
        cond = f"{v[0]} {exists_ops[k]}"
    elif k in ['or', 'and']:
        names = []
        cond = []
        for _v in v:
            condition = _parse_single_query_condition(_v)
            names += condition['names']
            cond.append(condition['condition'])
        cond = '(' + f' {k} '.join(cond) + ')'
        names = list(set(names))
    elif k == 'between':
        names = [v[0].split('.')[0]]
        left, right = v[1], v[2]

        if isinstance(left, dict):
            left = v[1]['literal']
        if isinstance(right, dict):
            right = v[2]['literal']

        if isinstance(left, str):
            left = f"'{left}'"
        if isinstance(right, str):
            right = f"'{right}'"

        cond = f"{v[0]} BETWEEN {left} AND {right}"
    else:
        return None

    return {
        'names': names,
        'condition': cond,
        'is_join': is_join, # whether it is a join or a filter
        'columns': _get_involved_columns(names, cond)
    }

def _get_involved_columns(names, cond):
    columns = []
    for name in names:
        pattern = f"('|^|\s+){name}\.\w+"
        res = re.search(pattern, cond)
        res = res.string[res.span()[0]:res.span()[1]]
        columns.append(res.strip().replace("'", ""))
    return columns

# @timeit_decorator
def _get_query_condition(qc):
    conditions = []

    def _get_query_condition_internal(qc, conditions):
        if isinstance(qc, list):
            for _qc in qc:
                _get_query_condition_internal(_qc, conditions)

        elif isinstance(qc, dict):
            single_cond = _parse_single_query_condition(qc)
            conditions.append(single_cond)
        else:
            raise Exception(
                f"Not supported type {type(qc)}, must be list or dict")

    _get_query_condition_internal(qc, conditions)
    return conditions


# @timeit_decorator
def parse_sql_query(sql_query):
    parsed = parse(sql_query)
    if isinstance(parsed['from'], list):  
        query_tables = [p['value'] for p in parsed['from']]
        reverse_aliases_dict = {p['name']: p['value'] for p in parsed['from']}
    else:
        query_tables = [parsed['from']['value']]
        reverse_aliases_dict = {parsed['from']['name']: parsed['from']['value']}
    query_conditions = _get_query_condition(
        parsed['where'].get('and', parsed['where']))
    return query_tables, reverse_aliases_dict, query_conditions


def parse_explain_json_to_list_of_nodes(
        plan,
        index_cond_names=['Index Cond', 'Recheck Cond'],
        join_cond_names=['Join Filter', 'Hash Cond', 'Merge Cond']):
    # possible_node_types = [
    #     'Hash',
    #     'Materialize',
    #     'Gather',
    #     'Sort',
    #     'Aggregate',
    #     'Partial Aggregate',
    #     'Vector Aggregate',
    #     'Vector Streaming (type: GATHER)',
    #     'Vector Streaming(type: REDISTRIBUTE)',
    #     'Vector Streaming(type: BROADCAST)',
    #     'Vector Streaming(type: PART REDISTRIBUTE PART ROUNDROBIN)',
    #     'Vector Streaming(type: PART LOCAL PART BROADCAST)',
    #     'Vector Materialize',
    #     'Vector Sort',
    # ] + join_types

    join_types = {
        'Hash Join': "HashJoin",
        'Merge Join': "MergeJoin",
        'Nested Loop': "NestLoop",
        'Vector Sonic Hash Join': "HashJoin",
        'Vector Nest Loop': "NestLoop",
        'Vector Merge Join': "MergeJoin",
    }
    same_filter_types = ['Filter']

    dict_of_nodes = {}
    list_of_joins = []
    conds = []
    alias_to_table = {}
    tables = []

    def _extract_index_conditions(_plan, tab_name):
        for cond_name in index_cond_names + same_filter_types:
            if cond_name in _plan:
                cond = _plan[cond_name]
                conds.append({'names': [tab_name], 'condition': cond})

    def _extract_filter_conditions(_plan):
        # find conditions with 2 tables
        for cond_name in join_cond_names + same_filter_types:
            if cond_name in _plan:
                cond = _plan[cond_name]
                results = re.findall(
                    r'([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)( [\=\>\<]+ )+([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)',
                    cond
                )
                for tab1, col1, op, tab2, col2 in results:
                    condition = f'{tab1}.{col1}{op}{tab2}.{col2}'
                    _plan[cond_name] = condition
                    query_condition = {
                        'names': [tab1, tab2],
                        'condition': f'{tab1}.{col1}{op}{tab2}.{col2}',
                    }
                    if query_condition not in conds:
                        conds.append(query_condition)

    def _extract_table_name(_plan):
        if 'Relation Name' in _plan:
            tab_name = _plan['Relation Name']
            if tab_name not in tables:
                tables.append(tab_name)
            if 'Alias' in _plan:
                alias_to_table[_plan['Alias']] = tab_name
                tab_name = _plan['Alias']
            return tab_name

    def _extract_order_and_join_info(plan):
        tabs = []
        if 'Plans' in plan:
            join_nodes = []
            for _plan in plan['Plans']:
                _tabs = _extract_order_and_join_info(_plan)
                join_nodes.append(_tabs)
                tabs.extend(_tabs)
            if len(plan['Plans']) >= 2:
                list_of_joins.append((tuple(tabs), join_nodes))
        if 'Relation Name' in plan:
            alias = _extract_table_name(plan)
            tabs.append(alias)
        tabs = tuple(tabs)
        _extract_filter_conditions(plan)
        if len(tabs) == 1:
            _extract_index_conditions(plan, tabs[0])
        if plan['Node Type'] in join_types:
            tmp_plan = deepcopy(plan)
            del tmp_plan['Plans']
            tmp_plan["tab_entries"] = set(tabs)
            tmp_plan["Node Type"] = join_types[tmp_plan["Node Type"]]
            dict_of_nodes[tabs] = tmp_plan
        return tabs

    _extract_order_and_join_info(plan)
    alias_to_table = dict(sorted(alias_to_table.items()))
    labels = {(n,): i for i, n in enumerate(alias_to_table.keys())}
    relabeled_join_list = []
    for p, (c1, c2) in list_of_joins:
        labels[p] = len(labels)
        dict_of_nodes[labels[p]] = dict_of_nodes.pop(p)
        relabeled_join_list.append((labels[p], (labels[c1], labels[c2])))
    return relabeled_join_list, dict_of_nodes, conds, alias_to_table, tables


def explain_plan_parser(plan, query, conn, exec_time=False):
    """
    Parameters
    ----------
    plan: plan.Plan
    query: str
        sql query
    conn: psycopg2.extensions.connection
    exec_time: bool, optional
        estimate time if True

    Returns
    -------
    plan.Plan
    float
        Actual total cost
    float
        Actual total execution time if exec_time=True

    WARNING: can change left/right join order.
    """
    cur = conn.cursor()
    try:
        cur.execute(
            f"EXPLAIN (FORMAT JSON{', ANALYZE' if exec_time else ''}) {query}")
        data = cur.fetchall()
    except Exception as e:
        cur.close()
        raise e
    cur.close()
    index_cond_names = ['Index Cond', 'Recheck Cond']
    join_cond_names = ['Join Filter', 'Hash Cond', 'Merge Cond']
    query_plan = data[0][0][0]['Plan']
    list_of_joins, joins_info, _, alias_to_table, _ = parse_explain_json_to_list_of_nodes(
        query_plan, index_cond_names, join_cond_names
    )
    p = Plan(plan.query_tables, plan.alias_to_table,
             plan.query_join_conditions, plan.initial_query)
    for node, (c1, c2) in list_of_joins:
        join_info = joins_info[node]
        info_dict = {
            'cost': join_info['Total Cost'],
            'time': join_info.get('Actual Total Time') or join_info.get('Actual Max Total Time'),
            'rows': join_info.get('Actual Rows'),
            'join_type': join_info.get('Node Type')
        }
        p.join(c1, c2, **info_dict)
    if exec_time:
        return p, query_plan.get('Actual Total Time')
    else:
        return p, query_plan.get('Total Cost')


def get_column_dtype(conn, table_name, column_name):
    cur = conn.cursor()
    cur.execute(
        f"SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = '{table_name}' and COLUMN_NAME = '{column_name}'")
    data = cur.fetchone()
    cur.close()
    return data[0]


def get_min_max_of_column(conn, table_name, column_name):
    cur = conn.cursor()
    cur.execute(
        f"SELECT MIN({column_name}), MAX({column_name}) FROM {table_name}")
    data = cur.fetchone()
    cur.close()
    return data
