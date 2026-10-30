"""Join-order enumeration: wrap the C++ enumerator and map Query objects onto
the join graph and Leading-only pg_hint_plan hints.

A "join order" is a binary tree whose leaves are relation indices and whose
internal nodes are joins. The C++ module (csrc, built via pybind11) produces
these trees; this module converts a Query into the adjacency list the C++
expects, and converts the enumerated trees back into Leading-only hint strings.

Deliberately no torch import here: this module is imported by worker processes
too, and they must stay lightweight.
"""

import os
import sys
from typing import List, Optional, Tuple

from plan_generator.query_hint import Hint
from plan_generator.sql_query import Query

# Make the compiled C++ extension importable. Build it with:
#   cmake -B csrc/build csrc && cmake --build csrc/build
_CSRC_BUILD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc", "build"
)
if _CSRC_BUILD_DIR not in sys.path:
    sys.path.insert(0, _CSRC_BUILD_DIR)
try:
    from enumerate_plan import PlanNode, Scan, Join, enumerate_plan as _cpp_enumerate_plan
except ImportError as e:
    raise ImportError(
        "C++ join-order enumerator is not built. Run "
        "`cmake -B csrc/build csrc && cmake --build csrc/build` first."
    ) from e


def build_adjacency_list(query: Query) -> Tuple[List[str], List[List[int]]]:
    """Map a Query onto (sorted aliases, adjacency list over those aliases).

    Two relations are adjacent iff they share a join predicate. The adjacency
    list is indexed by position in the sorted alias list, which is exactly the
    relation-index space the C++ enumerator works in.
    """
    aliases = sorted(query.table_ocurrances)
    index_of = {a: i for i, a in enumerate(aliases)}
    adj = [set() for _ in aliases]
    for jc in query.join_condition_set:
        left, right = jc.split(" = ")
        a = left.split(".")[0].strip()
        b = right.split(".")[0].strip()
        if a in index_of and b in index_of:
            adj[index_of[a]].add(index_of[b])
            adj[index_of[b]].add(index_of[a])
    return aliases, [sorted(s) for s in adj]


def enumerate_join_orders(query: Query, limit_num: int) -> Tuple[List[str], List[PlanNode]]:
    """Enumerate join orders for a query. Returns (aliases, list of plan trees)."""
    aliases, adj = build_adjacency_list(query)
    if len(aliases) <= 1:
        return aliases, []
    trees = _cpp_enumerate_plan(adj, limit_num)
    return aliases, trees


def leading_from_tree(node: PlanNode, aliases: List[str]) -> str:
    """Render a join-order tree as a pg_hint_plan Leading fragment.

    Format matches Hint.from_pg_physical_plan_json: a leaf is the alias, an
    internal node is "(<left> <right>)".
    """
    if isinstance(node, Scan):
        return aliases[node.relation]
    if isinstance(node, Join):
        left = leading_from_tree(node.left, aliases)
        right = leading_from_tree(node.right, aliases)
        return f"({left} {right})"
    raise TypeError(f"Unknown plan node: {type(node)}")


def hint_from_leading(leading: str) -> str:
    """Wrap a Leading fragment into a full pg_hint_plan comment (Leading only)."""
    return Hint(leading, set()).to_pg_hint()


def build_subquery_sql(query: Query) -> str:
    """Build the EXPLAIN-able SQL for the full join of all query relations.

    Mirrors PlanGenerator.subplan_tree_to_subquery_sql for the whole tree:
        SELECT COUNT(*) FROM real_table alias, ... WHERE <all join conds AND all filters>;
    This is constant across every candidate join order (all relations appear
    in every candidate), so the workers can reuse the same SQL.
    """
    aliases = sorted(query.table_ocurrances)
    from_parts = [f"{query.table_alias_to_name_dict[a]} {a}" for a in aliases]
    from_clause = ", ".join(from_parts)

    conditions = list(sorted(query.join_condition_set))
    for a in sorted(query.table_where_condition_dict):
        for cond_expr in query.table_where_condition_dict[a]:
            cond_sql = cond_expr.sql()
            if cond_sql not in conditions:
                conditions.append(cond_sql)

    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
    return f"SELECT COUNT(*) FROM {from_clause}{where_clause};"
