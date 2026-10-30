import time
import logging
from plan_generator.sql_query import Query
from plan_generator.plan_tree import ScanNode, JoinNode, ScanMethod, JoinMethod
from plan_generator.query_hint import Hint
from plan_generator.index_info import get_db_indexes_info
from plan_generator.cost_model import CostModel
from plan_generator.connection_args import DB_CONNECTION_ARGS
from plan_generator.union_find import UnionFind
from typing import Dict, List, Union, Optional, Tuple, Set
from pilotscope.PilotConfig import PostgreSQLConfig
from pilotscope.DBInteractor.PilotDataInteractor import PilotDataInteractor

class NextBestStepNotFoundError(Exception):
    """Raised when no valid next join step can be found."""
    pass





class PlanGenerator:
    def __init__(self, db: str, query: Query, cost_model: CostModel):
        self.db = db
        self.query = query
        self.cost_model = cost_model
        self.index_info = get_db_indexes_info(db)
        self._build_alias_columns()
        self.equivalent_key_groups = self._build_equivalent_key_groups()
        self.equivalent_table_groups = [{key.split(".")[0] for key in group} for group in self.equivalent_key_groups]
        self.data_interactor = self._get_pilotscope_data_interactor()

    def _get_pilotscope_data_interactor(self):
        user, password, host, port = DB_CONNECTION_ARGS["user"], DB_CONNECTION_ARGS["password"], DB_CONNECTION_ARGS["host"], DB_CONNECTION_ARGS["port"]

        config = PostgreSQLConfig(db_host=host, db_port=port, db_user=user, db_user_pwd=password, db=self.db)
        data_interactor = PilotDataInteractor(config)

        return data_interactor

    def _two_tables_joinable(self, table1: str, table2: str) -> bool:
        """Check if two tables can be joined."""
        for g in self.equivalent_table_groups:
            if table1 in g and table2 in g:
                return True
        return False


    def _two_subtree_joinable(self, left: Union[ScanNode, JoinNode], right: Union[ScanNode, JoinNode]) -> bool:
        """Check if two subtrees can be joined."""
        left_aliases = self._collect_aliases(left)
        right_aliases = self._collect_aliases(right)
        for l in left_aliases:
            for r in right_aliases:
                if self._two_tables_joinable(l, r):
                    return True
        return False


    @staticmethod
    def merge_subtrees_with_join(left: Union[ScanNode, JoinNode], right: Union[ScanNode, JoinNode], join_method: JoinMethod, step_cost: Optional[float]=None, total_cost: Optional[float]=None) -> JoinNode:
        """Merge two subtrees with a join node."""
        join_node = JoinNode(
            left_child=left,
            right_child=right,
            join_method=join_method,
            step_cost=step_cost if step_cost is not None else 0.0,
            total_cost=total_cost if total_cost is not None else 0.0
        )
        return join_node


    def plan_node_to_physical_plan(self, node: Union[ScanNode, JoinNode]) -> Optional[dict]:
        """Convert a plan node to a postgresql physical plan."""
        subquery_sql = self.subplan_tree_to_subquery_sql(node)
        hint = Hint.from_custom_plan(node)
        hint_str = hint.to_pg_hint()

        self.data_interactor.push_pg_hint_comment(hint_str)
        self.data_interactor.pull_physical_plan()
        self.data_interactor.pull_estimated_cost()
        result = self.data_interactor.execute(subquery_sql)
        pg_physical_plan = result.physical_plan

        hint_new = Hint.from_pg_physical_plan_json(pg_physical_plan)

        if hint_new != hint:
            logging.warning(f"PostgreSQL rejects the generated hint: {hint.to_pg_hint()} and adopts {hint_new.to_pg_hint()} for SQL (sub)query {subquery_sql}. The full SQL Query is {self.query.original_sql}.")
            return None

        return pg_physical_plan


    @staticmethod
    def _collect_aliases(node: Union[ScanNode, JoinNode]) -> Set[str]:
        """Collect all table aliases in the plan tree."""
        if isinstance(node, ScanNode):
            return {node.alias}
        else:
            left_aliases = PlanGenerator._collect_aliases(node.left_child)
            right_aliases = PlanGenerator._collect_aliases(node.right_child)
            return left_aliases | right_aliases


    def subplan_tree_to_subquery_sql(self, node: Union[ScanNode, JoinNode]) -> str:
        """Convert a subplan tree to a SQL subquery string.

        Generates:
            SELECT COUNT(*) FROM real_table_1 alias_1, real_table_2 alias_2
            WHERE join_cond AND filter_cond;
        """
        aliases = PlanGenerator._collect_aliases(node)

        # FROM clause: alias -> real table name for each table in subtree
        from_parts = []
        for alias in sorted(aliases):
            table_name = self.query.table_alias_to_name_dict.get(alias, alias)
            from_parts.append(f"{table_name} {alias}")
        from_clause = ", ".join(from_parts)

        # WHERE clause: join conditions + filter conditions
        conditions = []

        # Join conditions involving only tables in this subtree
        for group in self.equivalent_key_groups:
            contained_keys = [key for key in group if key.split(".")[0] in aliases]
            if len(contained_keys) >= 2:
                for i in range(1, len(contained_keys)):
                    conditions.append(f"{contained_keys[i]} = {contained_keys[0]}")

        # Filter (where) conditions for each table in this subtree
        for alias in sorted(aliases):
            if alias in self.query.table_where_condition_dict:
                for cond_expr in self.query.table_where_condition_dict[alias]:
                    cond_sql = cond_expr.sql()
                    if cond_sql not in conditions:
                        conditions.append(cond_sql)

        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        return f"SELECT COUNT(*) FROM {from_clause}{where_clause};"


    def get_cost_of_plan_tree(self, node: Union[ScanNode, JoinNode]) -> Tuple[float, float]:
        """Estimate the cost of a plan tree using the cost model. Return a tuple of (plan_cost, infer_time)."""
        physical_plan = self.plan_node_to_physical_plan(node)
        if physical_plan is None:
            return float('inf'), 0.0  # If PostgreSQL rejects the hint, return infinite cost

        start_time = time.perf_counter()
        plan_cost = self.cost_model.estimate_cost(physical_plan)
        infer_time = time.perf_counter() - start_time
        return plan_cost, infer_time


    def _build_alias_columns(self) -> None:
        """Precompute filter columns and join columns per alias."""
        # Dict[alias, Set[filter_column_name]]
        self._alias_filter_cols: Dict[str, Set[str]] = {}
        # Dict[alias, Set[join_column_name]]
        self._alias_join_cols: Dict[str, Set[str]] = {}

        for alias in self.query.table_ocurrances:
            self._alias_filter_cols[alias] = set(
                self.query.table_column_where_condition_dict.get(alias, {}).keys()
            )

        for jc in self.query.join_condition_set:
            left, right = jc.split(" = ")
            for part in (left, right):
                tbl, col = part.strip().split(".")
                if tbl not in self._alias_join_cols:
                    self._alias_join_cols[tbl] = set()
                self._alias_join_cols[tbl].add(col)


    def _build_equivalent_key_groups(self) -> List[Set[str]]:
        join_key_to_idx = dict()
        for jc in self.query.join_condition_set:
            left, right = jc.split(" = ")
            for key in (left.strip(), right.strip()):
                if key not in join_key_to_idx:
                    join_key_to_idx[key] = len(join_key_to_idx)
        idx_to_key = {v: k for k, v in join_key_to_idx.items()}
        uf = UnionFind(len(join_key_to_idx))
        for jc in self.query.join_condition_set:
            left, right = jc.split(" = ")
            uf.union(join_key_to_idx[left.strip()], join_key_to_idx[right.strip()])
        groups = uf.equivalent_groups()
        return [{idx_to_key[idx] for idx in group} for group in groups]


    def get_access_methods(self) -> Tuple[List[ScanNode], float]:
        """For each table alias, generate candidate access methods, evaluate cost,
        and return the best ScanNode per alias."""
        best_scans: List[ScanNode] = []
        infer_time = 0.0

        for alias in sorted(self.query.table_ocurrances):
            table_name = self.query.table_alias_to_name_dict[alias]
            indexes = self.index_info.get(table_name, {})
            filter_cols = self._alias_filter_cols[alias]
            join_cols = self._alias_join_cols.get(alias, set())
            all_cols = filter_cols | join_cols

            # --- Build candidate scan methods ---
            candidates = [ScanMethod.SeqScan]

            # IndexScan: any filter column matches any index
            if filter_cols:
                for idx_cols in indexes.values():
                    if filter_cols & set(idx_cols):
                        candidates.append(ScanMethod.IndexScan)
                        break

            # IndexOnlyScan: ALL filter + join columns covered by a single index
            if all_cols:
                for idx_cols in indexes.values():
                    if all_cols.issubset(set(idx_cols)):
                        candidates.append(ScanMethod.IndexOnlyScan)
                        break

            # BitmapScan: any filter column matches any index
            if filter_cols:
                for idx_cols in indexes.values():
                    if filter_cols & set(idx_cols):
                        candidates.append(ScanMethod.BitmapScan)
                        break

            # --- Evaluate each candidate, pick the cheapest ---
            best_method = candidates[0]
            best_cost = float('inf')
            for method in candidates:
                scan_node = ScanNode(alias=alias, scan_method=method)
                plan_cost, latency = self.get_cost_of_plan_tree(scan_node)
                infer_time += latency
                if plan_cost < best_cost:
                    best_cost = plan_cost
                    best_method = method

            best_scans.append(ScanNode(alias=alias, scan_method=best_method, step_cost=best_cost, total_cost=best_cost))

        return best_scans, infer_time


    def next_best_join_step(self, forests) ->  Tuple[int, int, JoinMethod, float, float]:
        """Get the next step to join, return left_child_idx, right_child_idx, join_approach, step_cost, plan_cost, infer_time"""
        subtree_idxs = sorted(list(forests.keys()))
        best_step_cost = float('inf')
        best_total_cost, best_left_idx, best_right_idx, best_join_method = None, None, None, None
        total_infer_time = 0.0

        for i in range(len(subtree_idxs)):
            for j in range(i + 1, len(subtree_idxs)):
                left_idx = subtree_idxs[i]
                right_idx = subtree_idxs[j]
                left_subtree = forests[left_idx]
                right_subtree = forests[right_idx]

                if not self._two_subtree_joinable(left_subtree, right_subtree):
                    continue

                # For each join method, estimate the cost
                for join_method in JoinMethod:
                    new_join_tree = PlanGenerator.merge_subtrees_with_join(left_subtree, right_subtree, join_method)
                    plan_cost, infer_time = self.get_cost_of_plan_tree(new_join_tree)
                    total_infer_time += infer_time

                    if plan_cost - (left_subtree.total_cost + right_subtree.total_cost) < best_step_cost:
                        best_step_cost = plan_cost - (left_subtree.total_cost + right_subtree.total_cost)
                        best_total_cost = plan_cost
                        best_left_idx = left_idx
                        best_right_idx = right_idx
                        best_join_method = join_method

                    new_join_tree = PlanGenerator.merge_subtrees_with_join(right_subtree, left_subtree, join_method)
                    plan_cost, infer_time = self.get_cost_of_plan_tree(new_join_tree)
                    total_infer_time += infer_time

                    if plan_cost - (right_subtree.total_cost + left_subtree.total_cost) < best_step_cost:
                        best_step_cost = plan_cost - (right_subtree.total_cost + left_subtree.total_cost)
                        best_total_cost = plan_cost
                        best_left_idx = right_idx
                        best_right_idx = left_idx
                        best_join_method = join_method

        return best_left_idx, best_right_idx, best_join_method, best_step_cost, best_total_cost, total_infer_time


    def search_best_plan_hint(self) -> Tuple[str, float]:
        """Search for the best plan hint based on the cost model."""
        # Get access methods of base tables
        latency = 0.0
        scans, infer_time = self.get_access_methods()
        latency += infer_time
        forests = {i: scan for i, scan in enumerate(scans)}
        next_idx = len(forests)
        while len(forests) > 1:
            left_child_idx, right_child_idx, join_method, step_cost, plan_cost, infer_time = self.next_best_join_step(forests)
            latency += infer_time

            if left_child_idx is None or right_child_idx is None or join_method is None:
                return "", latency  # No valid join step found, return empty hint

            new_join_tree = PlanGenerator.merge_subtrees_with_join(
                forests[left_child_idx],
                forests[right_child_idx],
                join_method,
                step_cost=step_cost,
                total_cost=plan_cost
            )
            forests.pop(left_child_idx)
            forests.pop(right_child_idx)
            forests[next_idx] = new_join_tree
            next_idx += 1

        assert len(forests) == 1, "There should be exactly one plan tree left."
        best_plan_tree = next(iter(forests.values()))
        best_hint_str = Hint.from_custom_plan(best_plan_tree).to_pg_hint()

        return best_hint_str, latency
