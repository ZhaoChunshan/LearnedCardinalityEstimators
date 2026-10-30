from random import Random
from schema.sql_schema import SQLColumn, SQLTable, SQLForeignKey, SQLSchema
from schema.comparison_operator import OperatorType, ColumnType, OPERATORS, COLUMN_OPERATOR_TYPE, OPERATOR_TYPE_PROB
from query.sql_query import QueryNode, QueryEdge, Predicate, SQLQuery, SQLTemplate
from common.conn_utils import RobustConnection
from typing import List, Dict, Tuple, Set, Union, Any


class ValuePicker:
    def __init__(self, schema: SQLSchema, random: Random, db: str, user: str, password: str, host: str = "localhost", port: int = 5432, buffer_size: int = 1000) -> None:
        self._connection = RobustConnection(db, user, password, host, port)
        self._schema = schema
        self._random = random
        self._value_buffer: Dict[Tuple[SQLTable, SQLColumn], List[Any]] = { (table, column): [] for table in self._schema.tables() for column in table.columns() }
        self._buffer_size = buffer_size

    def pick(self, table: SQLTable, column: SQLColumn) -> Any:
        if len(self._value_buffer[(table, column)]) == 0:
            self._fill_buffer(table, column)

        return self._value_buffer[(table, column)].pop()
    
    def _fill_buffer(self, table: SQLTable, column: SQLColumn) -> None:
        sample_ratio = min(table.cardinality(), self._buffer_size / (1 - column.null_ratio())) / table.cardinality()
        sample_query = """SELECT %s
                          FROM %s
                          TABLESAMPLE SYSTEM(%f) REPEATABLE (%d)
                          WHERE NOT %s IS NULL;""" % (column.name(),
                                                      table.name(),
                                                      sample_ratio * 100,
                                                      self._random.randint(0, 1000000),
                                                      column.name())
        results, _ = self._connection.run(sample_query)

        self._random.shuffle(results)

        self._value_buffer[(table, column)] = [row[0] for row in results]
        
        if len(self._value_buffer[(table, column)]) == 0:
            self._fill_buffer(table, column)

class QueryGenerator:
    def __init__(self, schema: SQLSchema, default_min_nodes: int, default_max_nodes: int, default_predicate_factor: float,
                 default_max_predicate_columns: int,  db: str, user: str, password: str, host: str = "localhost", 
                 port: int = 5432, pk_fk_only: bool = False, seed: int = 42) -> None:
        self._schema = schema
        self._default_min_nodes = max(default_min_nodes, 1)
        self._default_max_nodes = min(default_max_nodes, len(schema.tables()))
        self._default_predicate_factor = default_predicate_factor
        self._default_max_predicate_columns = max(0, default_max_predicate_columns)
        self._random = Random(seed)
        self._value_picker = ValuePicker(schema, self._random, db, user, password, host, port)
        self._pk_fk_only = pk_fk_only

    def generate_query(self) -> SQLQuery:
        num_nodes = self._random.randint(self._default_min_nodes, self._default_max_nodes)
        num_predicates = QueryGenerator.geometric_distribution(self._random, self._default_predicate_factor)
        nodes, edges = self._generate_nodes_and_edges(num_nodes, num_predicates)
        return SQLQuery(nodes, edges)

    def get_schema(self) -> SQLSchema:
        return self._schema

    def _generate_nodes_and_edges(self, num_nodes: int, num_predicates: int) -> Tuple[List[QueryNode], List[QueryEdge]]:
        tables: List[SQLTable] = []
        join_conditions: List[Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]] = []
        possible_joins: List[Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]] = []
        
        start_table = self._random.choice(self._schema.tables())
        tables.append(start_table)
        possible_joins.extend(self._possible_joins(self._schema, start_table))

        for _ in range(num_nodes - 1):
            while True:
                assert(len(possible_joins) > 0)

                join_condition = self._random.choice(possible_joins)
                possible_joins.remove(join_condition)

                if isinstance(join_condition, SQLForeignKey):
                    if join_condition.table in tables and join_condition.referenced_table in tables:
                        continue

                    if join_condition.table in tables:
                        table = join_condition.referenced_table
                    else:
                        table = join_condition.table
                    
                    tables.append(table)
                    possible_joins.extend(self._possible_joins(self._schema, table))
                    join_conditions.append(join_condition)
                    break
                
                else:
                    fk1, fk2 = join_condition
                    if fk1.table in tables and fk2.table in tables:
                        continue

                    if fk1.table in tables:
                        table = fk2.table
                    else:
                        table = fk1.table

                    tables.append(table)
                    possible_joins.extend(self._possible_joins(self._schema, table))
                    join_conditions.append(join_condition)
                    break
        
        predicable_columns = [(table, column) for table in tables for column in table.columns() if column.predicable()]
        predicates: Dict[SQLTable, List[Predicate]] = {table: [] for table in tables}

        if len(predicable_columns) <= num_predicates:
            num_predicates = len(predicable_columns)
        
        if num_predicates != 0:
            predicable_columns = self._random.sample(predicable_columns, num_predicates)
        
        if len(predicable_columns) > self._default_max_predicate_columns:
            predicable_columns = self._random.sample(predicable_columns, self._default_max_predicate_columns)

        for table, column in predicable_columns:
            predicates[table].extend(self._generate_predicate(table, column))

        nodes = [QueryNode(table, predicates[table]) for table in tables]
        edges = [QueryEdge(join_condition) for join_condition in join_conditions]

        return nodes, edges
    
    def _possible_joins(self, schema: SQLSchema, table: SQLTable) -> List[Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]]:
        joins = []

        for fk, tb in schema.out_neighbors(table):
            joins.append(fk)
            if not self._pk_fk_only:
                for fk2, _ in schema.in_neighbors(tb):
                    if fk != fk2:
                        if fk.name < fk2.name:
                            joins.append((fk, fk2))
                        else:
                            joins.append((fk2, fk))

        for fk, _ in schema.in_neighbors(table):
            joins.append(fk)
        
        return joins


    def _generate_predicate(self, table: SQLTable, column: SQLColumn) -> List[Predicate]:
        assert(column.predicable())
        column_type = column.column_type()
        operator_types = COLUMN_OPERATOR_TYPE[column_type]
        operator_prob = [OPERATOR_TYPE_PROB[operator_type] for operator_type in operator_types]

        operator_type = self._random.choices(operator_types, weights=operator_prob)[0]

        if operator_type == OperatorType.EQU:
            return self._generate_equ_predicate(table, column)
        else:
            return self._generate_cmp_predicate(table, column)

    def _generate_equ_predicate(self, table: SQLTable, column: SQLColumn) -> List[Predicate]:
        value = self._value_picker.pick(table, column)
        op = OPERATORS[OperatorType.EQU][0]
        return [Predicate(table, column, op, value)]
    
    def _generate_cmp_predicate(self, table: SQLTable, column: SQLColumn) -> List[Predicate]:
        op = self._random.choice(OPERATORS[OperatorType.CMP])
        predicate: List[Predicate] = []

        if op in [">", "<", ">=", "<="]:
            value = self._value_picker.pick(table, column)
            predicate.append(Predicate(table, column, op, value))

        else:
            value1 = self._value_picker.pick(table, column)
            value2 = self._value_picker.pick(table, column)
            predicate.append(Predicate(table, column, op[0], min(value1, value2)))
            predicate.append(Predicate(table, column, op[1], max(value1, value2)))

        return predicate
    

    @staticmethod
    def geometric_distribution(random: Random, p: float) -> int:
        i = 0
        while random.random() > p:
            i += 1
        return i
    
    
class TemplateBasedQueryGenerator:
    def __init__(self, schema: SQLSchema, seed: int = 42) -> None:
        self._schema = schema
        self._random = Random(seed)
        self._value_picker = ValuePicker(schema, self._random)
        
    def generate_query(self, template: SQLTemplate) -> SQLQuery:
        query_nodes = []
        for node in template.nodes():
            predicates = []
            predicates_template = node.predicates()
            i = 0
            while i < len(predicates_template):
                j = i + 1
                while j < len(predicates_template) and \
                    predicates_template[j].column.name() == predicates_template[i].column.name():
                    j += 1
                ops = [predicates_template[k].operator for k in range(i, j)]
                predicates.extend(self._generate_predicate(node.table(), predicates_template[i].column, ops))
                i = j
            query_nodes.append(QueryNode(node.table(), predicates))
        return SQLQuery(query_nodes, template.edges())
    
    def get_schema(self) -> SQLSchema:
        return self._schema
        
    def _generate_predicate(self, table: SQLTable, column: SQLColumn, operators: List[str]) -> List[Predicate]:
        assert(len(operators) == 1 or len(operators) == 2)
        
        if len(operators) == 1:
            op = operators[0]
            value = self._value_picker.pick(table, column)
            return [Predicate(table, column, op, value)]
        
        else:
            op1, op2 = operators
            value1 = self._value_picker.pick(table, column)
            value2 = self._value_picker.pick(table, column)
            
            retry_count = 1
            while value2 == value1 and retry_count < 5:
                value2 = self._value_picker.pick(table, column)
                retry_count += 1
            
            if op1 == '<' or op1 == '<=':
                assert(op2 == '>' or op2 == '>=')
                return [Predicate(table, column, op1, max(value1, value2)), Predicate(table, column, op2, min(value1, value2))]
            elif op1 == '>' or op1 == '>=':
                assert(op2 == '<' or op2 == '<=')
                return [Predicate(table, column, op1, min(value1, value2)), Predicate(table, column, op2, max(value1, value2))]
            else:
                assert(False)
