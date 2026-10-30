from typing import List, Set, Tuple, Union
from plan_generator.plan_tree import JoinNode, ScanNode, ScanMethod, JoinMethod

# Mapping from PostgreSQL JSON EXPLAIN node type names to pg_hint_plan keywords.
PG_SCAN_METHOD = {
    "Seq Scan": ScanMethod.SeqScan.value,         # "SeqScan"
    "Index Scan": ScanMethod.IndexScan.value,       # "IndexScan"
    "Index Only Scan": ScanMethod.IndexOnlyScan.value,  # "IndexOnlyScan"
    "Bitmap Heap Scan": ScanMethod.BitmapScan.value,    # "BitmapScan"
}

PG_JOIN_METHOD = {
    "Nested Loop": JoinMethod.NestedLoopJoin.value,  # "NestLoop"
    "Hash Join": JoinMethod.HashJoin.value,          # "HashJoin"
    "Merge Join": JoinMethod.MergeJoin.value,        # "MergeJoin"
}


class Hint:
    def __init__(self, leading_hint: str, access_method: Set[str]):
        self.leading_hint = leading_hint
        self.access_method = access_method

    def __eq__(self, other):
        if not isinstance(other, Hint):
            return False
        return (self.leading_hint == other.leading_hint and
                self.access_method == other.access_method)

    def __repr__(self):
        return f"Hint(leading_hint={self.leading_hint!r}, access_method={self.access_method!r})"

    def __str__(self):
        methods = ", ".join(sorted(self.access_method))
        return f"leading_hint: {self.leading_hint}\naccess_method: [{methods}]"


    @staticmethod
    def from_custom_plan(plan: Union[JoinNode, ScanNode]) -> 'Hint':
        access_method: Set[str] = set()

        def walk(node: Union[JoinNode, ScanNode]) -> Tuple[List[str], str]:
            """Return relation names in the subtree and the leading-hint fragment."""
            if isinstance(node, JoinNode):
                left_tables, left_leading = walk(node.left_child)
                right_tables, right_leading = walk(node.right_child)
                tables = left_tables + right_tables
                access_method.add(
                    f"{node.join_method.value}({' '.join(tables)})"
                )
                return tables, f"({left_leading} {right_leading})"
            access_method.add(
                f"{node.scan_method.value}({node.alias})"
            )
            return [node.alias], node.alias

        _, leading_hint = walk(plan)
        return Hint(leading_hint, access_method)


    @staticmethod
    def from_pg_physical_plan_json(json_plan) -> 'Hint':
        # Unwrap to the root plan node: tolerate the full EXPLAIN envelope
        # ([{"Plan": {...}}]), a bare {"Plan": {...}}, or the node itself.
        node = json_plan
        if isinstance(node, list) and node:
            node = node[0]
        if isinstance(node, dict) and "Plan" in node:
            node = node["Plan"]

        access_method: Set[str] = set()

        def relation_name(plan_node: dict) -> str:
            return plan_node.get("Alias") or plan_node.get("Relation Name")

        def walk(plan_node: dict) -> Tuple[List[str], str]:
            """Return relation names under node and its leading-hint fragment."""
            node_type = plan_node.get("Node Type")

            # Leaf: a relation scan.
            if node_type in PG_SCAN_METHOD:
                rel = relation_name(plan_node)
                scan_kw = PG_SCAN_METHOD[node_type]
                access_method.add(f"{scan_kw}({rel})")
                return [rel], rel

            # Binary join: recurse into each child, then combine.
            if node_type in PG_JOIN_METHOD:
                tables: List[str] = []
                leadings: List[str] = []
                for child in plan_node.get("Plans", []):
                    child_tables, child_leading = walk(child)
                    if not child_tables and not child_leading:
                        continue
                    tables += child_tables
                    leadings.append(child_leading)
                join_kw = PG_JOIN_METHOD[node_type]
                access_method.add(
                    f"{join_kw}({' '.join(tables)})"
                )
                return tables, "(" + " ".join(leadings) + ")"

            # Passthrough node (Hash, Sort, Materialize, Aggregate, Limit, ...):
            # neither a relation nor a join, so delegate to its single subtree.
            children = plan_node.get("Plans", [])
            if children:
                return walk(children[0])
            return [], ""

        _, leading_hint = walk(node)
        return Hint(leading_hint, access_method)


    def to_pg_hint(self) -> str:
        # Emit a pg_hint_plan block comment, one hint per line for readability
        # (matches the layout pilotscope.Common.Util.get_pg_hints produces):
        #     /*+
        #     SeqScan(t0)
        #     IndexScan(t1)
        #     HashJoin(t0 t1 t2)
        #     Leading((t0 (t1 t2)))
        #     */
        scan_keywords = {str(ScanMethod.SeqScan), str(ScanMethod.IndexScan), str(ScanMethod.IndexOnlyScan), str(ScanMethod.BitmapScan)}

        scan_hints: List[str] = []
        join_hints: List[str] = []
        for hint in self.access_method:
            keyword = hint.split("(", 1)[0]
            if keyword in scan_keywords:
                scan_hints.append(hint)
            else:
                join_hints.append(hint)

        lines = ["/*+"]
        lines.extend(sorted(scan_hints))
        lines.extend(sorted(join_hints))
        if self.leading_hint:
            lines.append(f"Leading({self.leading_hint})")
        lines.append("*/")
        return "\n".join(lines) + "\n"
