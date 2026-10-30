import copy
import sqlglot

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set
from sqlglot import exp


class Query:
    """Represents query"""

    def __init__(self):
        self.original_sql: Optional[str] = None
        self.table_ocurrances = set()   # use table aliases to handle multiple ocurrances of same table in query
        self.table_alias_to_name_dict = {} # table alias to table name mapping
        self.join_condition_set = set() # each is alias1.attr1 = alias2.attr2
        self.table_where_condition_dict = {} # each is table alias to list of conditions (exp.Expression) on that table
        self.table_column_where_condition_dict = {} # table alias -> column name -> list of conditions on that column

    def copy_query(self):
        query = Query()
        query.table_ocurrances = copy.copy(self.table_ocurrances)
        query.table_alias_to_name_dict = copy.copy(self.table_alias_to_name_dict)
        query.join_condition_set = copy.copy(self.join_condition_set)
        query.table_where_condition_dict = copy.copy(self.table_where_condition_dict)
        query.table_column_where_condition_dict = copy.copy(self.table_column_where_condition_dict)
        return query

    @staticmethod
    def parse(sql_query) -> 'Query':
        """Parses SQL query into Query object"""
        ast = sqlglot.parse_one(sql_query)
        query = Query()
        query.original_sql = sql_query

        # Extract table occurrences and their aliases
        for table in ast.find_all(exp.Table):
            alias = table.alias_or_name
            name = table.name
            query.table_ocurrances.add(alias)
            query.table_alias_to_name_dict[alias] = name

        where = ast.args.get('where')
        if where is None:
            return query

        where_expr: exp.Expression = where.this
        and_conditions = Query.flatten_and(where_expr)

        filter_candidates = []
        for condition in and_conditions:
            if isinstance(condition, exp.EQ) and isinstance(condition.left, exp.Column) and isinstance(condition.right, exp.Column):
                # This is a join condition
                left_table_alias = condition.left.table
                left_column = condition.left.name
                right_table_alias = condition.right.table
                right_column = condition.right.name
                if (left_table_alias, left_column) > (right_table_alias, right_column):
                    left_table_alias, left_column, right_table_alias, right_column = right_table_alias, right_column, left_table_alias, left_column
                if left_table_alias is None or right_table_alias is None or left_column is None or right_column is None:
                    raise ValueError("Join condition must have both table aliases and column names. The predicate is: " + condition.sql())
                elif left_table_alias == right_table_alias:
                    raise ValueError("Join condition must be between two different tables. The predicate is: " + condition.sql())
                else:
                    query.join_condition_set.add(f"{left_table_alias}.{left_column} = {right_table_alias}.{right_column}")
                    continue

            filter_candidates.append(condition)

        for condition in filter_candidates:
            aliases = Query.get_table_aliases_from_expression(condition)
            assert len(aliases) == 1, "Only support filtering predicates on a single table. The predicate is: " + condition.sql()

            alias = next(iter(aliases))

            if alias not in query.table_where_condition_dict:
                query.table_where_condition_dict[alias] = []
            query.table_where_condition_dict[alias].append(condition)

        # Build column-level index: for unary predicates (involving exactly one column),
        # record which column each condition touches.
        non_unary_expressions = []
        for condition in filter_candidates:
            aliases = Query.get_table_aliases_from_expression(condition)
            alias = next(iter(aliases))
            columns_in_condition = [col for col in condition.find_all(exp.Column) if col.table == alias]
            if len(columns_in_condition) == 1:
                col_name = columns_in_condition[0].name
                if alias not in query.table_column_where_condition_dict:
                    query.table_column_where_condition_dict[alias] = {}
                if col_name not in query.table_column_where_condition_dict[alias]:
                    query.table_column_where_condition_dict[alias][col_name] = []
                query.table_column_where_condition_dict[alias][col_name].append(condition)
            else:
                non_unary_expressions.append(condition)
        assert len(non_unary_expressions) == 0, \
            f"Only unary filter predicates (single column) are supported. Found: {[e.sql() for e in non_unary_expressions]}"

        return query

    @staticmethod
    def flatten_and(expr: exp.Expression) -> List[exp.Expression]:
        """Flattens an expression tree of ANDs into a list of expressions. For example, (A AND B) AND C will be flattened to [A, B, C]."""
        if isinstance(expr, exp.And):
            return Query.flatten_and(expr.left) + Query.flatten_and(expr.right)
        elif isinstance(expr, exp.Paren):
            return Query.flatten_and(expr.this)
        else:
            return [expr]

    @staticmethod
    def get_table_aliases_from_expression(expr: exp.Expression) -> Set[str]:
        """Get all table aliases involved in the expression."""
        table_aliases = {col.table for col in expr.find_all(exp.Column) if col.table is not None}
        return table_aliases
