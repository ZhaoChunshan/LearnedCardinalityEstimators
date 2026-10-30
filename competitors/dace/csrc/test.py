import sys
import os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import enumerate_plan
from enumerate_plan import (
    PlanNode,
    Scan,
    Join,
    enumerate_plan,
)


def make_clique(size: int) -> List[List[int]]:
    return [[j for j in range(size) if j != i] for i in range(size)]


def plan_to_str(node: PlanNode) -> str:
    """Render a join-order tree as 'Join(...)' / 'Scan(relation)'."""
    if isinstance(node, Join):
        return f"Join({plan_to_str(node.left)}, {plan_to_str(node.right)})"
    elif isinstance(node, Scan):
        return f"Scan({node.relation})"
    return "?"


def collect_relations(node: PlanNode, out: set) -> None:
    if isinstance(node, Scan):
        out.add(node.relation)
    elif isinstance(node, Join):
        collect_relations(node.left, out)
        collect_relations(node.right, out)


def check_valid(plans: List[PlanNode], size: int) -> bool:
    """Every plan must be a tree covering all relations exactly once."""
    for plan in plans:
        rels: set = set()
        collect_relations(plan, rels)
        if rels != set(range(size)):
            return False
    return True


# Test 1: clique of size 2 -> 2 join orders
clique_size: int = 2
adj_list: List[List[int]] = make_clique(clique_size)
print("adj_list:", adj_list)
plans: List[PlanNode] = enumerate_plan(adj_list, 100)
print(f"Total join orders (clique 2): {len(plans)}")
for plan in plans:
    print(f"  {plan_to_str(plan)}")
assert len(plans) == 2
assert check_valid(plans, clique_size)

# Test 2: clique of size 3 -> 12 join orders
plans = enumerate_plan(make_clique(3), 100)
print(f"Total join orders (clique 3): {len(plans)}")
for plan in plans:
    print(f"  {plan_to_str(plan)}")
assert len(plans) == 12
assert check_valid(plans, 3)

# Test 3: star graph of size 4 (center 0) -> 48 join orders
adj_list = [[1, 2, 3], [0], [0], [0]]
plans = enumerate_plan(adj_list, 100)
print(f"Total join orders (star 4): {len(plans)}")
assert len(plans) == 48
assert check_valid(plans, 4)

print("All python tests passed.")
