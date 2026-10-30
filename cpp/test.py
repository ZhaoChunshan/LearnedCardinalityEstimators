import sys
import os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import enumerate_plan
from enumerate_plan import (
    PlanNode,
    ScanNode,
    JoinNode,
    ScanMethod,
    JoinMethod,
    enumerate_plan,
)

# Test with small clique (size 2 → 96 plans with swap+methods)
clique_size: int = 2
adj_list: List[List[int]] = [
    [j for j in range(clique_size) if j != i] for i in range(clique_size)
]
relation_names: List[str] = [f"t{i}" for i in range(clique_size)]
all_scan: List[ScanMethod] = [
    ScanMethod.SeqScan,
    ScanMethod.IndexScan,
    ScanMethod.IndexOnlyScan,
    ScanMethod.BitmapScan,
]

print("adj_list:", adj_list)
plans: List[PlanNode] = enumerate_plan(
    adj_list, 100, relation_names, [all_scan] * clique_size, [False] * clique_size
)

SCAN_NAMES: List[str] = ["SeqScan", "IndexScan", "IndexOnlyScan", "BitmapScan"]
JOIN_NAMES: List[str] = ["NestedLoopJoin", "HashJoin", "MergeJoin"]


def print_plan_node(node: PlanNode) -> str:
    if isinstance(node, JoinNode):
        return f"{JOIN_NAMES[node.join_method.value]}({print_plan_node(node.left)}, {print_plan_node(node.right)})"
    elif isinstance(node, ScanNode):
        return f"{SCAN_NAMES[node.scan_method.value]}({node.relation})"
    return "?"


print(f"Total plans: {len(plans)}")
for plan in plans:
    print(f"  {print_plan_node(plan)}")
