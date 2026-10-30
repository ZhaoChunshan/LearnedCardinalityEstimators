"""Exhaustive join-order plan search.

Enumerates join orders (C++/pybind11 in csrc/), fetches PostgreSQL physical
plans for each in a multiprocessed worker pool, and picks the plan with the
lowest DACE-estimated cost.
"""

from plan_search.search import PlanSearchEngine, WorkerPool, WorkerPoolError
from plan_search.enumerator import enumerate_join_orders, leading_from_tree, hint_from_leading

__all__ = [
    "PlanSearchEngine",
    "WorkerPool",
    "WorkerPoolError",
    "enumerate_join_orders",
    "leading_from_tree",
    "hint_from_leading",
]
