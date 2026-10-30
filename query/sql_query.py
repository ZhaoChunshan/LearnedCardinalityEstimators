import re
import copy
import datetime

from enum import Enum
from typing import List, Set, Dict, Union, Tuple, Any
from schema.sql_schema import SQLColumn, SQLTable, SQLForeignKey, SQLSchema
from schema.comparison_operator import ColumnType, OperatorType, OPERATORS
from common.sql_utils import UnionFind
from common.value_wrapper import value_wrapper, parse_value, date_to_int, AnyLiteral, AnyColumn, AnyOperator


class JoinType(Enum):
    PK_FK = 1     # PK-FK Join 
    FK_FK = 2     # FK-FK Join


class Predicate:
    def __init__(self, table: SQLTable, column: SQLColumn, operator: str, value: Any) -> None:
        self.table = table
        self.column = column
        self.operator = operator
        self.value = value

    def float_value(self) -> float:
        data_type, column_type = self.column.data_type(), self.column.column_type()
        if data_type.name() in ["smallint", "integer", "bigint", "double precision"]:
            return float(self.value)
        elif data_type.name() == "character varying":
            return float(self.column.value_2_id(self.value) / self.column.length_of_value_2_id())
        elif data_type.name() == "date":
            return float(date_to_int(self.value))
        elif data_type.name() == "interval":
            return float(int(self.value.total_seconds()))

    def __str__(self) -> str:
        return "%s.%s %s %s" % (self.table.try_alias(),
                                self.column.name(),
                                self.operator,
                                value_wrapper(self.value)
                                )
    
    def str_with_alias(self, alias, encode_comparable_string=False) -> str:
        return "%s.%s %s %s" % (alias,
                                self.column.name(),
                                self.operator,
                                value_wrapper(self.column.value_2_id(self.value)) if encode_comparable_string and self.column.column_type() == ColumnType.STRING_WITH_CMP else value_wrapper(self.value)
                                )
        
    def copy_without_value(self) -> "Predicate":
        return Predicate(self.table, self.column, self.operator, AnyLiteral())
    
    def copy_without_operator(self) -> "Predicate":
        return Predicate(self.table, self.column, AnyOperator(), AnyLiteral())
    
    def copy_without_column(self) -> "Predicate":
        return Predicate(self.table, AnyColumn(), AnyOperator(), AnyLiteral())


class QueryNode:
    def __init__(self, table: SQLTable, predicates: List[Predicate], is_virtual: bool = False) -> None:
        self._table = table
        self._predicates = predicates
        self._pg_estimated_card = None
        self._is_virtual = is_virtual

    def table(self) -> SQLTable:
        return self._table
    
    def is_virtual(self) -> bool:
        return self._is_virtual
    
    def predicates(self) -> List[Predicate]:
        return self._predicates

    def set_pg_estimated_card(self, pg_estimated_card: float) -> None:
        self._pg_estimated_card = pg_estimated_card

    def pg_estimated_card(self) -> float:
        return self._pg_estimated_card

class QueryEdge:
    def __init__(self, foreign_key: Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]) -> None:
        self._foreign_key = foreign_key

    def foreign_key(self) -> Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]:
        return self._foreign_key

    def join_type(self) -> JoinType:
        if isinstance(self._foreign_key, tuple):
            return JoinType.FK_FK
        else:
            return JoinType.PK_FK

    def is_pk_fk(self) -> bool:
        return isinstance(self._foreign_key, SQLForeignKey)

    def is_fk_fk(self) -> bool:
        return isinstance(self._foreign_key, tuple)
    
    def __str__(self) -> str:
        if self.is_pk_fk():
            return "%s.%s = %s.%s" % (self.foreign_key().table.name(), self.foreign_key().column.name(), 
                                            self.foreign_key().referenced_table.name(), self.foreign_key().referenced_column.name())
        else:
            return "%s.%s = %s.%s" % (self.foreign_key()[0].table.name(), self.foreign_key()[0].column.name(), 
                                            self.foreign_key()[1].table.name(), self.foreign_key()[1].column.name())


class SQLQuery:
    def __init__(self, nodes: List[QueryNode], edges: List[QueryEdge], join_predicate_2_fk: Dict[str, SQLForeignKey] = None) -> None:
        self._nodes = nodes
        self._edges = edges
        self._join_predicate_2_fk = join_predicate_2_fk
        self._adj_list: Dict[QueryNode, List[QueryEdge]] = {node: [] for node in self._nodes}

        table_dict = {node.table(): node for node in nodes}
        for edge in edges:
            if edge.is_pk_fk():
                fk_table = edge.foreign_key().table
                pk_table = edge.foreign_key().referenced_table
                self._adj_list[table_dict[pk_table]].append(edge)
                self._adj_list[table_dict[fk_table]].append(edge)
            else:
                table_1 = edge.foreign_key()[0].table
                table_2 = edge.foreign_key()[1].table
                self._adj_list[table_dict[table_1]].append(edge)
                self._adj_list[table_dict[table_2]].append(edge)

    def nodes(self) -> List[QueryNode]:
        return self._nodes

    def edges(self) -> List[QueryEdge]:
        return self._edges
    
    def has_filters(self) -> bool:
        return any([len(node.predicates()) != 0 for node in self._nodes])
    
    def adj_list(self, node: QueryNode) -> List[QueryEdge]:
        return self._adj_list[node]

    def is_acyclic(self) -> bool:
        """
        Check if the query is acyclic or not. Current implementation assumes that all queries are tree-structured.

        :Note: call this function after `canonicalize_equal_join_keys`

        :return: True if the query is acyclic, False otherwise
        """
        return len(self._nodes) == len(self._edges) + 1
    
    def contains_fk_fk_joins(self) -> bool:
        return any([edge.is_fk_fk() for edge in self._edges])

    def all_subqueries(self, contain_itself: bool) -> List["SQLQuery"]:
        """
        Enumerate all possible subqueries of this query.
        
        [Ref.] Guido Moerkotte, Thomas Neumann. "Analysis of Two Existing and One New Dynamic Programming Algorithm \ 
        for the Generation of Optimal Bushy Join Trees without Cross Products." VLDB 2006.
        
        Args:
        - contain_itself: If true, then the subquery contains itself.

        Returns:
        A list of all possible subqueries of this query.
        """
        subqueries: List["SQLQuery"] = []
        
        table_dict = {node.table(): node for node in self._nodes}
        
        nodes_all = self._nodes
        edges_all = self._edges
        edges_all_set = {str(edge) for edge in edges_all}
        
        # Add all possible pk-fk joins to current edges_all
        for edge in edges_all:
            if edge.is_pk_fk():
                continue
            for fk in edge.foreign_key():
                if fk.table not in table_dict or fk.referenced_table not in table_dict:
                    continue
                new_edge = QueryEdge(fk)
                new_edge_str = str(new_edge)
                if new_edge_str not in edges_all_set:
                    edges_all.append(new_edge)
                    edges_all_set.add(new_edge_str)
        
        # Add all possible fk-fk joins to current edges_all
        for node in nodes_all:
            node_table = node.table()
            adj_edges = self.adj_list(node)
            for edge_1 in adj_edges:
                if not edge_1.is_pk_fk():
                    continue
                fk1 = edge_1.foreign_key()
                if fk1.referenced_table != node_table:
                    continue
                
                for edge_2 in adj_edges:
                    if not edge_2.is_pk_fk() or edge_2 == edge_1:
                        continue
                    fk2 = edge_2.foreign_key()
                    if fk2.referenced_table != node_table or \
                        fk2.referenced_column != fk1.referenced_column:
                        continue
                    
                    if fk1.name < fk2.name:
                        new_edge = QueryEdge((fk1, fk2))
                    else:
                        new_edge = QueryEdge((fk2, fk1))
                    
                    new_edge_str = str(new_edge)
                    if new_edge_str not in edges_all_set:
                        edges_all.append(new_edge)
                        edges_all_set.add(new_edge_str)
        
        def possible_combinations(l):
            result = []
            def rec(partial_result, i):
                if i == len(l):
                    if len(partial_result) > 0:
                        result.append(partial_result)
                else:
                    rec(partial_result, i + 1)
                    rec(partial_result + [l[i]], i + 1)
            rec([], 0)
            return result
            
        def enumerate_subqueries_rec(nodes: Set[QueryNode], edges: List[QueryEdge], visited: Set[QueryNode]) -> None:
            neighbors: List[Tuple[QueryNode, QueryEdge]] = []
                        
            for edge in edges_all:
                if edge in edges:
                    continue
                if edge.is_pk_fk():
                    t1 = edge.foreign_key().table
                    t2 = edge.foreign_key().referenced_table
                else:
                    t1 = edge.foreign_key()[0].table
                    t2 = edge.foreign_key()[1].table
                
                n1 = table_dict[t1]
                n2 = table_dict[t2]
                
                if n1 in visited and n2 in visited:
                    continue
                
                if not n1 in nodes and not n2 in nodes:
                    continue
                elif n1 in nodes and not n2 in nodes:
                    pass
                elif not n1 in nodes and n2 in nodes:
                    n1, n2 = n2, n1
                else:
                    assert(False)
                
                neighbors.append((n2, edge))
            
            if len(neighbors) == 0:
                return
            
            neighbors_comb = possible_combinations(neighbors)
            for subset_neighbors in neighbors_comb:
                subquery_nodes = list(nodes)
                subquery_edges = edges.copy()
                for n, e in subset_neighbors:
                    subquery_nodes.append(n)
                    subquery_edges.append(e)
                
                if not contain_itself and len(subquery_nodes) == len(nodes_all):
                    continue
                
                subquery = SQLQuery(subquery_nodes, subquery_edges)
                subqueries.append(subquery)
            
            new_visited = set(visited)
            for n, _ in neighbors:
                new_visited.add(n)
            
            for subset_neighbors in neighbors_comb:
                subquery_nodes = nodes.copy()
                subquery_edges = edges.copy()
                for n, e in subset_neighbors:
                    subquery_nodes.add(n)
                    subquery_edges.append(e)
                enumerate_subqueries_rec(subquery_nodes, subquery_edges, new_visited)
        
        for i in range(len(nodes_all) -1, -1, -1):
            visited = {nodes_all[j] for j in range(i + 1)}
            
            if contain_itself or len(nodes_all) != 1:
                subqueries.append(SQLQuery([nodes_all[i]], []))
                        
            enumerate_subqueries_rec({nodes_all[i]}, [], visited)
        
        return subqueries

    def sql(self, use_as: bool = True, alias_dict = None, encode_comparable_string = False) -> str:
        def alias(table: SQLTable):
            if alias_dict is not None and alias_dict[table.name()] is not None:
                return alias_dict[table.name()]
            else:
                return table.alias() or table.name()
        
        table_alias = []
        for node in self._nodes:
            if use_as:
                table_alias.append("%s AS %s" % (node.table().name(), alias(node.table())))
            else:
                table_alias.append("%s %s" % (node.table().name(), alias(node.table())))
            
        alias_str = ", ".join(table_alias)

        joins_and_predicates = []
        for edge in self._edges:
            if edge.is_pk_fk():
                joins_and_predicates.append("%s.%s = %s.%s" % (alias(edge.foreign_key().table), edge.foreign_key().column.name(), 
                                                alias(edge.foreign_key().referenced_table), edge.foreign_key().referenced_column.name()))
            else:
                joins_and_predicates.append("%s.%s = %s.%s" % (alias(edge.foreign_key()[0].table), edge.foreign_key()[0].column.name(), 
                                                alias(edge.foreign_key()[1].table), edge.foreign_key()[1].column.name()))

        for node in self._nodes:
            joins_and_predicates.extend([predicate.str_with_alias(alias(node.table()), encode_comparable_string) for predicate in node.predicates()])

        where_str = " AND ".join(joins_and_predicates)

        if where_str == "":
            return "SELECT COUNT(*) FROM %s;" % alias_str
        
        return "SELECT COUNT(*) FROM %s WHERE %s;" % (alias_str, where_str)

    def canonicalize(self) -> "SQLQuery":
        """
            In our testing workloads, there are no table multi occurances. Also, we are sure that
            the missed pk table doesn't occur in the query.
            See the assertion in line 350. Also, run `test/test_canonicalize.py` to check the workloads.
        """
        new_nodes: List[QueryNode] = copy.copy(self.nodes())
        new_edges: List[QueryEdge] = []
        tables = [node.table() for node in new_nodes]
        occurances_as_pk = {edge.foreign_key().referenced_table for edge in self.edges() if edge.is_pk_fk()}
        foreign_keys = []

        for edge in self.edges():
            if edge.join_type() == JoinType.PK_FK:
                fk = edge.foreign_key()
                if fk not in foreign_keys:
                    new_edges.append(edge)
                    foreign_keys.append(fk)
            else:
                fk1, fk2 = edge.foreign_key()
                if fk1.referenced_table not in tables:
                    new_nodes.append(QueryNode(fk1.referenced_table, [], True))
                    tables.append(fk1.referenced_table)
                    occurances_as_pk.add(fk1.referenced_table)
                else:
                    assert(fk1.referenced_table in occurances_as_pk)

                if fk1 not in foreign_keys:
                    new_edges.append(QueryEdge(fk1))
                    foreign_keys.append(fk1)

                if fk2 not in foreign_keys:
                    new_edges.append(QueryEdge(fk2))
                    foreign_keys.append(fk2)
        
        return SQLQuery(new_nodes, new_edges)

    def canonicalize_equal_join_keys(self) -> "SQLQuery":
        new_nodes: List[QueryNode] = copy.copy(self.nodes())
        new_edges: List[QueryEdge] = []
        
        join_key_dict: Dict[Tuple[str, str], Tuple[int, bool]] = {}
        pk_dict: Dict[Tuple[str, str], Tuple[str, str]] = {}
        join_keys: List[Tuple[str, str]] = []
        
        for edge in self._edges:
            if edge.join_type() == JoinType.PK_FK:
                fk = edge.foreign_key()
                p = (fk.referenced_table.name(), fk.referenced_column.name())
                f = (fk.table.name(), fk.column.name())
                if p not in join_key_dict:
                    join_key_dict[p] = (len(join_key_dict), False)
                    join_keys.append(p)
                if f not in join_key_dict:
                    join_key_dict[f] = (len(join_key_dict), True)
                    join_keys.append(f)
                pk_dict[f] = p
                pk_dict[p] = p
            else:
                fk1, fk2 = edge.foreign_key()
                p = (fk1.referenced_table.name(), fk1.referenced_column.name())
                f1 = (fk1.table.name(), fk1.column.name())
                f2 = (fk2.table.name(), fk2.column.name())
                if f1 not in join_key_dict:
                    join_key_dict[f1] = (len(join_key_dict), True)
                    join_keys.append(f1)
                if f2 not in join_key_dict:
                    join_key_dict[f2] = (len(join_key_dict), True)
                    join_keys.append(f2)
                pk_dict[f1] = p
                pk_dict[f2] = p
                pk_dict[p] = p
        
        uf = UnionFind(len(join_key_dict))
        for edge in self._edges:
            if edge.join_type() == JoinType.PK_FK:
                fk = edge.foreign_key()
                p = (fk.referenced_table.name(), fk.referenced_column.name())
                f = (fk.table.name(), fk.column.name())
                uf.union(join_key_dict[p][0], join_key_dict[f][0])
            else:
                fk1, fk2 = edge.foreign_key()
                f1 = (fk1.table.name(), fk1.column.name())
                f2 = (fk2.table.name(), fk2.column.name())
                uf.union(join_key_dict[f1][0], join_key_dict[f2][0])
        
        equivalent_groups = uf.equivalent_groups()
        equivalent_groups = [[join_keys[i] for i in g] for g in equivalent_groups]
        for i, g in enumerate(equivalent_groups):
            sorted_keys = sorted(g, key=lambda x: (join_key_dict[x][1], x[0], x[1]))
            equivalent_groups[i] = sorted_keys
            first_key = sorted_keys[0]
            pk_of_equivalent_group = pk_dict[first_key]
            is_pk = pk_of_equivalent_group == first_key
            for j in range(1, len(sorted_keys)):
                key_j = sorted_keys[j]
                if is_pk:
                    join_repr = f"{key_j[0]}.{key_j[1]} = {pk_of_equivalent_group[0]}.{pk_of_equivalent_group[1]}"
                    edge = QueryEdge(self._join_predicate_2_fk[join_repr])
                else:
                    join_repr_1 = f"{key_j[0]}.{key_j[1]} = {pk_of_equivalent_group[0]}.{pk_of_equivalent_group[1]}"
                    join_repr_2 = f"{first_key[0]}.{first_key[1]} = {pk_of_equivalent_group[0]}.{pk_of_equivalent_group[1]}"
                    fk1 = self._join_predicate_2_fk[join_repr_1]
                    fk2 = self._join_predicate_2_fk[join_repr_2]
                    if fk1.name < fk2.name:        
                        edge = QueryEdge((fk1, fk2))
                    else:
                        edge = QueryEdge((fk2, fk1))
                new_edges.append(edge)
        
        return SQLQuery(new_nodes, new_edges, self._join_predicate_2_fk)
    
    def canonicalize_join_graph_rep(self, query: "SQLQueryJGMPRep" = None) -> "SQLQueryJGMPRep":
        edges: Set[Tuple[QueryNode, SQLForeignKey, QueryNode]] = set()

        if query is None:
            nodes = [QueryNode(node.table(), node.predicates(), False) for node in self.nodes()]
        else:
            nodes = [query.real_node(node.table()) for node in self.nodes()]
        
        real_node_dict = {node.table() : node for node in nodes}        
        occurances_as_pk = {}
        
        for edge in self.edges():
            if edge.is_pk_fk():
                fk = edge.foreign_key()
                occurances_as_pk[fk.referenced_table] = real_node_dict[fk.referenced_table]
        
        for edge in self.edges():
            if edge.is_pk_fk():
                fk = edge.foreign_key()
                from_node, to_node = real_node_dict[fk.table], real_node_dict[fk.referenced_table]
                edges.add((from_node, fk, to_node))
            else:
                fk1, fk2 = edge.foreign_key()
                from_node_1, from_node_2 = real_node_dict[fk1.table], real_node_dict[fk2.table]
                pk_table = fk1.referenced_table
                if pk_table in occurances_as_pk:
                    to_node = occurances_as_pk[pk_table]
                    edges.add((from_node_1, fk1, to_node))
                    edges.add((from_node_2, fk2, to_node))
                else:
                    if query is None:
                        virtual_node = QueryNode(pk_table, [], True)
                    else:
                        virtual_node = query.virtual_node(pk_table)
                    nodes.append(virtual_node)
                    occurances_as_pk[pk_table] = virtual_node
                    edges.add((from_node_1, fk1, virtual_node))
                    edges.add((from_node_2, fk2, virtual_node))
        
        return SQLQueryJGMPRep(nodes, list(edges))
    
        def add_single_table_estimation(self, single_table_est: Dict[str, float]) -> "SQLQuery":
            for node in self.nodes():
                if node.is_virtual():
                    node.set_pg_estimated_card(node.table().cardinality())
                elif node.table().name() in single_table_est:
                    node.set_pg_estimated_card(single_table_est[node.table().name()])
            return self

    @staticmethod
    def parse(schema: SQLSchema, sql: str) -> "SQLQuery":
        nodes: List[QueryNode] = []
        edges: List[QueryEdge] = []
        
        table_dict: Dict[SQLTable, QueryNode] = {}
        alias_dict: Dict[str, str] = {} # alias -> table name

        comment_pattern = r"\/\*.*?\*\/"
        sql = re.sub(comment_pattern, "", sql)

        from_index = sql.lower().find(" from ")
        where_index = sql.lower().find(" where ")
        if where_index == -1:
            from_str = sql[from_index + 6:].strip()
            from_str = from_str[:-1].strip() if from_str[-1] == ";" else from_str
            where_str = ""
        else:
            from_str = sql[from_index + 6: where_index].strip()
            where_str = sql[where_index + 7:].strip()
            where_str = where_str[:-1].strip() if where_str[-1] == ";" else where_str

        for alias_str in from_str.strip().split(','):
            table_alias =  alias_str.strip().split()

            if len(table_alias) == 1:
                table = schema.table(table_alias[0].strip().lower())
                alias = table_alias[0].strip().lower()
            elif len(table_alias) == 2:
                table = schema.table(table_alias[0].strip().lower())
                alias = table_alias[1].strip().lower()
            else:
                table = schema.table(table_alias[0].strip().lower())
                alias = table_alias[2].strip().lower()
            
            nodes.append(QueryNode(table, []))
            table_dict[table] = nodes[-1]
            alias_dict[alias] = table.name()

        if where_str == "":
            return SQLQuery(nodes, edges)
        
        predicate_list = SQLQuery._split_predicate(where_str)
        for predicate_str in predicate_list:
            predicate_str = predicate_str.strip()
            if predicate_str[0] == '(' and predicate_str[-1] == ')':
                predicate_str = predicate_str[1:-1]
            
            first_dot_index = predicate_str.find('.')
            
            operator_index = -1
            for i, c in enumerate(predicate_str):
                if c in "<=>":
                    operator_index = i
                    break
            left_table_alias = predicate_str[:first_dot_index].strip().lower()
            left_table_name = alias_dict[left_table_alias]
            left_column_name = predicate_str[first_dot_index+1:operator_index].strip().lower()

            if predicate_str[operator_index + 1] in "<=>":
                operator = predicate_str[operator_index:operator_index + 2]
                right_operand_index = operator_index + 2
            else:
                operator = predicate_str[operator_index]
                right_operand_index = operator_index + 1

            right_str = predicate_str[right_operand_index:].strip()
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9_]*$", right_str):
                right_table_alias, right_column_name = right_str.split('.')
                right_table_name = alias_dict[right_table_alias]
                right_column_name = right_column_name.lower()
                
                left_table = schema.table(left_table_name)
                right_table = schema.table(right_table_name)
                left_column = left_table.attribute(left_column_name)
                right_column = right_table.attribute(right_column_name)

                edges.append(SQLQuery.get_query_edge(schema, left_table, left_column, right_table, right_column))
            else:
                value = parse_value(right_str)

                left_table = schema.table(left_table_name)
                left_column = left_table.attribute(left_column_name)

                table_dict[left_table]._predicates.append(Predicate(left_table, left_column, operator, value))

        return SQLQuery(nodes, edges, schema.get_join_predicate_2_fk())

    @staticmethod
    def _split_predicate(predicate_str: str) -> List[str]:
        n = len(predicate_str)
        result = []
        in_quote = False    
        i, j = 0, 0
        while j < n:
            if predicate_str[j] == "'":
                if in_quote:
                    if (j + 1 < n and predicate_str[j+1] == "'"):
                        j += 2
                    else:
                        in_quote = False
                        j += 1
                else:
                    in_quote = True
                    j += 1
            elif in_quote:
                j += 1
            else:
                if predicate_str[j: j + 5].lower() == " and ":    
                    result.append(predicate_str[i:j].strip())
                    i = j + 5
                    j = i
                else:
                    j += 1
        result.append(predicate_str[i:].strip())      
        return result      

    @staticmethod
    def get_query_edge(schema: SQLSchema, table_1: SQLTable, column_1: SQLColumn, table_2: SQLTable, column_2: SQLColumn) -> QueryEdge:
        assert(table_1 != table_2)

        fk_fk_edge = None

        for fk, tb in schema.out_neighbors(table_1):
            if fk.column != column_1:
                continue

            if fk.referenced_table == table_2 and fk.referenced_column == column_2:
                return QueryEdge(fk)
            
            for fk2, _ in schema.in_neighbors(tb):
                if fk2.table == table_2 and fk2.column == column_2 and fk2.referenced_table == fk.referenced_table and fk2.referenced_column == fk.referenced_column:
                    if fk.name < fk2.name:
                        fk_fk_edge = QueryEdge((fk, fk2))
                    else:
                        fk_fk_edge = QueryEdge((fk2, fk))
        
        for fk, _ in schema.in_neighbors(table_1):
            if fk.referenced_column != column_1:
                continue

            if fk.table == table_2 and fk.column == column_2:
                return QueryEdge(fk)
        
        # prior pk-fk edge, finally return fk-fk edge
        if fk_fk_edge is not None:
            return fk_fk_edge

        raise Exception("No edge found")


class SQLQueryJGMPRep:
    def __init__(self, nodes: List[QueryNode], edges: List[Tuple[QueryNode, SQLForeignKey, QueryNode]]) -> None:
        self._nodes = nodes
        self._edges = edges
        self._table_to_node_real = {}
        self._table_to_node_virtual = {}
        
        for node in nodes:
            if node.is_virtual():
                self._table_to_node_virtual[node.table()] = node
            else:
                self._table_to_node_real[node.table()] = node
    
    def nodes(self) -> List[QueryNode]:
        return self._nodes

    def real_node(self, table: SQLTable):
        return self._table_to_node_real.get(table, None)
    
    def virtual_node(self, table: SQLTable):
        return self._table_to_node_virtual.get(table, None)
    
    def duplicated(self) -> bool:
        for t in self._table_to_node_virtual:
            if t in self._table_to_node_real:
                return True
        return False
    
    def __str__(self) -> str:
        node_dict = {n: i for i, n in enumerate(self._nodes)}
        node_str = [ f"Node {i}: table{n.table().name()}" for i, n in enumerate(self._nodes)]
        edge_str = [ f"Edge ({node_dict[n1]},{node_dict[n2]}) {fk.join_predicate_repr()}" for n1, fk, n2 in self._edges]
        return "\n".join(node_str + edge_str)

class SQLTemplate(SQLQuery):
    def __init__(self, nodes: List[QueryNode], edges: List[QueryEdge], join_predicate_2_fk: Dict[str, SQLForeignKey] = None) -> None:
        super().__init__(nodes, edges, join_predicate_2_fk)
        
    @staticmethod
    def from_query(query:SQLQuery, keep_filter_table: bool = False, keep_column: bool = False, keep_operator: bool = False) -> "SQLTemplate":
        cmp_operator_dict = {
            '>=': 0,
            '>': 1,
            '=':2,
            '<': 3,
            '<=': 4
        }
        template_nodes = []
        
        # We make sure node, predicate, edge in sorted order
        for node in query.nodes():
            if not keep_filter_table:
                # Only keep join schema
                template_nodes.append(QueryNode(node.table(), []))
            elif not keep_column:
                # keep filtered tables
                predicate_list = [p.copy_without_column() for p in node.predicates()]
                predicate_list = sorted(predicate_list, key=lambda p: p.table.name())
                unique_predicate_list = []
                for i, predicate in enumerate(predicate_list):
                    if i == 0 or predicate.table.name() != predicate_list[i-1].table.name():
                        unique_predicate_list.append(predicate)
                template_nodes.append(QueryNode(node.table(), unique_predicate_list))
            elif keep_operator:
                predicate_list = [p.copy_without_value() for p in node.predicates()]
                predicate_list = sorted(predicate_list, key=lambda p: (p.table.name(), p.column.name(), cmp_operator_dict[p.operator]))
                template_nodes.append(QueryNode(node.table(), predicate_list))
            else:
                predicate_list = sorted([p.copy_without_operator() for p in node.predicates()], key=lambda p: (p.table.name(), p.column.name()))
                unique_predicate_list = []
                for i, predicate in enumerate(predicate_list):
                    if i == 0 or predicate.table.name() != predicate_list[i-1].table.name() \
                        or predicate.column.name() != predicate_list[i-1].column.name():
                        unique_predicate_list.append(predicate)
                template_nodes.append(QueryNode(node.table(), unique_predicate_list))
        
        template_nodes = sorted(template_nodes, key=lambda n: n.table().name())
        template_edges = sorted(query.edges(), key=lambda e: str(e))
        return SQLTemplate(template_nodes, template_edges)
    
    @staticmethod
    def parse(schema: SQLSchema, sql: str) -> "SQLTemplate":
        query = SQLQuery.parse(schema, sql)
        nodes = query.nodes()
        edges = query.edges()
        return SQLTemplate(nodes, edges, schema.get_join_predicate_2_fk())
    