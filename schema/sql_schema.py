import time
import psycopg2
import numpy as np

from typing import List, Set, Dict, Tuple, Any, Optional, Union
from schema.data_type import DataType, DATATYPES
from schema.comparison_operator import ColumnType
from synopsis.samples import Samples
from synopsis.histogram import TableHistogram
from common.value_wrapper import date_to_int


class SQLColumn:
    column_count = 0
    def __init__(self, name: str, type: DataType, nullable: bool, null_ratio: float, column_type: ColumnType, \
                 min: Any = None, max: Any = None, value_2_id: Dict[Any, int] = {}) -> None:
        self._name = name
        self._type = type
        self._nullable = nullable
        self._null_ratio = null_ratio
        self._column_type = column_type
        
        self._min = min    # For numerical and non-string categorical value
        self._max = max    # For numerical and non-string categorical value

        self._value_2_id = value_2_id   # For string categorical value;
                                        # However for neurocard, all value should map to id
        
        self._id = SQLColumn.column_count
        SQLColumn.column_count += 1

    def name(self) -> str:
        return self._name

    def id(self) -> int:
        return self._id
    
    def data_type(self) -> DataType:
        return self._type

    def nullable(self) -> bool:
        return self._nullable

    def null_ratio(self) -> float:
        return self._null_ratio
    
    def column_type(self) -> ColumnType:
        return self._column_type
    
    def predicable(self) -> bool:
        return self._column_type == ColumnType.NUMERICAL or self._column_type == ColumnType.CATEGORICAL or self._column_type == ColumnType.STRING_WITH_CMP

    def min(self) -> Any:
        return self._min
    
    def min_float(self) -> float:
        if self._type.name() in ["smallint", "integer", "bigint", "double precision", "character varying"]:
            return float(self._min)
        elif self._type.name() == "date":
            return float(date_to_int(self._min))
        elif self._type.name() == 'interval':
            return float(self._min.interval())

    def max(self) -> Any:
        return self._max
    
    def max_float(self) -> float:
        if self._type.name() in ["smallint", "integer", "bigint", "double precision", "character varying"]:
            return float(self._max)
        elif self._type.name() == "date":
            return float(date_to_int(self._max))
        elif self._type.name() == 'interval':
            return float(self._max.interval())

    def value_2_id(self, value: Any) -> int:
        return self._value_2_id[value]
    
    def length_of_value_2_id(self) -> int:
        return len(self._value_2_id)

    def __str__(self) -> str:
        return "[SQLColumn] %s: %s %s" % (self._name, self._type.name(), str(self._column_type))

    @staticmethod
    def from_connection(connection: psycopg2.extensions.connection, table_name: str, column_name: str, column_type: ColumnType) -> "SQLColumn":
        cursor = connection.cursor()
        
        sql = "SELECT data_type FROM information_schema.columns WHERE table_name = '%s' AND column_name = '%s';" % (table_name, column_name)
        cursor.execute(sql)
        row = cursor.fetchone()
        type = DATATYPES[row[0]]

        sql = "SELECT COUNT(*) FROM %s WHERE %s IS NULL;" % (table_name, column_name)
        cursor.execute(sql)
        row = cursor.fetchone()
        null_count = row[0]
        nullable = null_count != 0

        sql = "SELECT COUNT(*) FROM %s;" % (table_name)
        cursor.execute(sql)
        row = cursor.fetchone()
        cardinality = row[0]
        null_ratio = null_count / cardinality

        min_value, max_value, value_2_id = None, None, {}

        if column_type == ColumnType.NUMERICAL or column_type == ColumnType.CATEGORICAL:
            if type.python_type() == str:
                if column_type == ColumnType.CATEGORICAL:
                    value_count = 0
                    sql = "SELECT %s FROM %s;" % (column_name, table_name)
                    cursor.execute(sql)
                    for row in cursor.fetchall():
                        value = row[0]
                        if value not in value_2_id:
                            value_2_id[value] = value_count
                            value_count += 1
                    min_value, max_value = 0, value_count - 1
            else:
                sql = "SELECT MIN(%s), MAX(%s) FROM %s;" % (column_name, column_name, table_name)
                cursor.execute(sql)
                row = cursor.fetchone()
                min_value, max_value = row
        elif column_type == ColumnType.STRING_WITH_CMP:
            value_count = 0
            sql = "SELECT %s FROM %s ORDER BY %s;" % (column_name, table_name, column_name)
            cursor.execute(sql)
            for row in cursor.fetchall():
                value = row[0]
                if value not in value_2_id:
                    value_2_id[value] = value_count
                    value_count += 1
            min_value, max_value = 0, value_count - 1

        cursor.close()
        connection.commit()
        
        return SQLColumn(column_name, type, nullable, null_ratio, column_type, min_value, max_value, value_2_id)


class SQLTable:
    table_count = 0
    def __init__(self, name: str, alias: Optional[str], columns: List[SQLColumn], cardinality: int, samples: Optional[Samples], sample_time: Optional[float] = None, indexes: Optional[Dict[str, List[str]]] = None) -> None:
        self._name = name
        self._alias = alias
        self._columns = columns
        self._column_dict = {column.name(): column for column in columns}
        self._cardinality = cardinality
        self._samples = samples
        self._sample_time = sample_time
        self._indexes = indexes # index name to columns
        self._histogram = None

        self._id = SQLTable.table_count
        SQLTable.table_count += 1

    def name(self) -> str:
        return self._name
    
    def id(self) -> int:
        return self._id
    
    def alias(self) -> Optional[str]:
        return self._alias
    
    def try_alias(self) -> str:
        return self._alias if self._alias is not None else self._name

    def columns(self) -> List[SQLColumn]:
        return self._columns
    
    def attribute(self, column_name: str) -> Optional[SQLColumn]:
        return self._column_dict.get(column_name)

    def cardinality(self) -> int:
        return self._cardinality
    
    def set_cardinality(self, cardinality: int) -> None:
        self._cardinality = cardinality
    
    def samples(self) -> Samples:
        return self._samples
    
    def set_samples(self, samples: Samples):
        self._samples = samples
    
    def histogram(self) -> Optional[TableHistogram]:
        return self._histogram
    
    def set_histogram(self, histogram: TableHistogram) -> None:
        self._histogram = histogram
    
    def get_indexes(self) -> Optional[Dict[str, List[str]]]:
        return self._indexes
    
    def __str__(self) -> str:
        return "[SQLTable] %s %s\n" % (self._name, self._alias) + "\n".join([str(column) for column in self._columns])

    @staticmethod
    def from_connection(connection: psycopg2.extensions.connection, name: str, alias: Optional[str], column_types: Dict[str, ColumnType], sample_size: Optional[int] = None) -> "SQLTable":
        cursor = connection.cursor()
        get_columns_sql = "SELECT column_name FROM information_schema.columns WHERE table_name = '%s';" % name
        cursor.execute(get_columns_sql)

        columns = []
        for row in cursor.fetchall():
            column_name: str = row[0]
            columns.append(SQLColumn.from_connection(connection, name, column_name, column_types.get(column_name, ColumnType.OTHER)))

        get_indexes_sql = "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = '%s';" % name
        cursor.execute(get_indexes_sql)
        indexes = {}
        for row in cursor.fetchall():
            index_name: str = row[0]
            index_def: str = row[1]
            index_columns = index_def[index_def.index("(") + 1:index_def.index(")")].replace(" ", "").split(",")
            indexes[index_name] = index_columns
        
        cardinality_sql = "SELECT COUNT(*) FROM %s;" % name
        cursor.execute(cardinality_sql)
        cardinality = cursor.fetchone()[0]

        cursor.close()

        samples, sample_time = None, None
        if sample_size is not None:
            start_sample = time.perf_counter()
            samples = Samples.from_connection(connection, name, [column for column in columns if column.predicable()], sample_size)
            sample_time = time.perf_counter() - start_sample

        return SQLTable(name, alias, columns, cardinality, samples, sample_time, indexes)


class SQLForeignKey:
    foreign_key_count = 0
    def __init__(self, name: str, table: SQLTable, column: SQLColumn, referenced_table: SQLTable, referenced_column: SQLColumn) -> None:
        self.name = name
        self.table = table
        self.column = column
        self.referenced_table = referenced_table
        self.referenced_column = referenced_column
        self.id = SQLForeignKey.foreign_key_count
        SQLForeignKey.foreign_key_count += 1

    def __str__(self) -> str:
        return "[ForeignKey] %s: %s.%s -> %s.%s" % (self.name, self.table.name(), self.column.name(), self.referenced_table.name(), self.referenced_column.name())
    
    def join_predicate_repr(self) -> str:
        return "%s.%s = %s.%s" % (self.table.name(), self.column.name(), self.referenced_table.name(), self.referenced_column.name())


class SQLSchema:
    def __init__(self, connection: psycopg2.extensions.connection, name: str, tables: List[SQLTable], foreign_keys: List[SQLForeignKey], possible_joins) -> None:
        self._connection = connection
        self._name = name
        self._tables = tables
        self._foreign_keys = foreign_keys
        self._possible_joins = possible_joins
        self._join_predicate_2_fk = {fk.join_predicate_repr(): fk for fk in self._foreign_keys}
        self._edges: Dict[SQLTable, List[Tuple[SQLForeignKey, SQLTable]]] = {table: [] for table in self._tables}
        self._reverse_edges: Dict[SQLTable, List[Tuple[SQLForeignKey, SQLTable]]] = {table: [] for table in self._tables}
        self._table_dict: Dict[str, SQLTable] = {table.name(): table for table in self._tables}
        self._table_id_dict: Dict[int, SQLTable] = {table.id(): table for table in self._tables}
        self._statistics: Dict[str, Any] = {}
        for fk in self._foreign_keys:
            self._edges[fk.table].append((fk, fk.referenced_table))
            self._reverse_edges[fk.referenced_table].append((fk, fk.table))

    def connection(self) -> psycopg2.extensions.connection:
        return self._connection

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    def name(self) -> str:
        return self._name

    def table(self, name: str) -> Optional[SQLTable]:
        return self._table_dict.get(name)

    def table_by_id(self, id: int) -> Optional[SQLTable]:
        return self._table_id_dict.get(id)

    def tables(self) -> List[SQLTable]:
        return self._tables
    
    def foreign_keys(self) -> List[SQLForeignKey]:
        return self._foreign_keys
    
    def possible_joins(self) -> List[Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]]:
        return self._possible_joins
    
    def get_join_predicate_2_fk(self) -> Dict[str, SQLForeignKey]:
        return self._join_predicate_2_fk
    
    def table_num(self) -> int:
        return len(self._tables)
    
    def foreign_key_num(self) -> int:
        return len(self._foreign_keys)
    
    def column_num(self) -> int:
        return sum(len(table.columns()) for table in self._tables)
    
    def predicable_column_num(self) -> int:
        return sum(len([column for column in table.columns() if column.predicable()]) for table in self._tables)

    def out_neighbors(self, table: SQLTable) -> List[Tuple[SQLForeignKey, SQLTable]]:
        return self._edges.get(table, [])
    
    def in_neighbors(self, table: SQLTable) -> List[Tuple[SQLForeignKey, SQLTable]]:
        return self._reverse_edges.get(table, [])

    def build_histogram(self, n_bins: int) -> None:
        for table in self._tables:
            table.set_histogram(TableHistogram.from_connection(self._connection, table, n_bins))
            self._connection.commit()

    def histogram_feature(self) -> np.ndarray:
        tables: List[SQLTable] = sorted([table for table in self._tables if table.histogram() is not None], key=lambda table: table.id())
        return np.concatenate([table.histogram().histogram() for table in tables], axis=0)

    def collect_table_samples(self) -> Dict[str, Samples]:
        return {table.name(): table.samples() for table in self._tables}
    
    def set_table_samples(self, table_samples: Dict[str, Samples]) -> None:
        for table in self._tables:
            table.set_samples(table_samples.get(table.name()))

    @staticmethod
    def from_connection(name: str, 
                        connection: psycopg2.extensions.connection,
                        table_column_types: Dict[Tuple[str, str], ColumnType] = {},
                        mask: Set[str] = None,
                        additional_possible_joins: List[Tuple[str, str, str, str, str]] = [],
                        alias: Dict[str,str] = {},
                        sample_size: Optional[int] = None
                        ) -> "SQLSchema":
        tables = SQLSchema._get_tables(connection, table_column_types, mask, alias, sample_size)
        foreign_keys, possible_joins = SQLSchema._get_foreign_keys(connection, tables, additional_possible_joins)

        # Reset column_count, table_count, foreign_key_count for next schema
        SQLTable.table_count = SQLColumn.column_count = SQLForeignKey.foreign_key_count = 0

        schema = SQLSchema(connection, name, tables, foreign_keys, possible_joins)
        if sample_size is not None:
            schema._statistics['sample_time'] = 0.0
            for table in schema._tables:
                schema._statistics['sample_time'] += table._sample_time

        connection.commit()
        return schema

    @staticmethod
    def _get_tables(connection: psycopg2.extensions.connection, table_column_types: Dict[str, Dict[str, ColumnType]], mask: Set[str], alias: Dict[str, str], sample_size: Optional[int] = None) -> List[SQLTable]:
        cursor = connection.cursor()

        sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';"
        cursor.execute(sql)

        tables = []
        for row in cursor.fetchall():
            table_name = row[0]
            if mask is None or table_name in mask:
                column_types: Dict[str, ColumnType] = table_column_types.get(table_name,{})

                table = SQLTable.from_connection(connection, table_name, alias.get(table_name, None), column_types, sample_size)

                tables.append(table)

        cursor.close()

        return tables
    
    def __str__(self) -> str:
        s = "[SQLSchema] %s\n" % self._name

        for table in self._tables:
            s += str(table) + "\n"

        for fk in self._foreign_keys:
            s += str(fk) + "\n"

        return s

    @staticmethod
    def _get_foreign_keys(connection: psycopg2.extensions.connection, tables: List[SQLTable], additional_possible_joins: List[Tuple[str, str, str, str, str]] = []) -> Tuple[List[SQLForeignKey], List[Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]]]:
        table_dict = {table.name(): table for table in tables}
        foreign_keys = []

        cursor = connection.cursor()
        foreign_key_query = """SELECT DISTINCT tc.constraint_name, kcu.table_name, kcu.column_name, ccu.table_name, ccu.column_name
                                FROM information_schema.table_constraints AS tc
                                JOIN information_schema.key_column_usage AS kcu ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
                                JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
                                JOIN information_schema.key_column_usage AS kcup ON kcup.table_schema = tc.table_schema AND kcup.column_name = ccu.column_name AND kcup.ordinal_position = kcu.position_in_unique_constraint
                                AND ccu.table_schema = tc.table_schema
                                WHERE tc.constraint_schema = 'public'
                                AND tc.constraint_type = 'FOREIGN KEY';"""
        cursor.execute(foreign_key_query)
        for name, ft, fc, tt, tc in cursor.fetchall():
            if ft not in table_dict or tt not in table_dict:
                continue
            
            from_table = table_dict[ft]
            from_column = from_table.attribute(fc)
            to_table = table_dict[tt]
            to_column = to_table.attribute(tc)

            foreign_keys.append(SQLForeignKey(name, from_table, from_column, to_table, to_column))

        cursor.close()
        
        possible_joins = SQLSchema._get_possible_joins(foreign_keys)

        for name, ft, fc, tt, tc in additional_possible_joins:
            if ft not in table_dict or tt not in table_dict:
                continue
            
            from_table = table_dict[ft]
            from_column = from_table.attribute(fc)
            to_table = table_dict[tt]
            to_column = to_table.attribute(tc)

            foreign_keys.append(SQLForeignKey(name, from_table, from_column, to_table, to_column))
            possible_joins.append(foreign_keys[-1])

        return foreign_keys, possible_joins


    @staticmethod
    def _get_possible_joins(foreign_keys: List[SQLForeignKey]) -> List[Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]]:
        possbile_joins: List[Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]] = []

        pk_2_fk: Dict[str, List[str]] = {}
        
        join_cond_2_fk: Dict[str, SQLForeignKey] = {}
                
        for pk_fk in foreign_keys:
            pk = f"{pk_fk.referenced_table.name()}.{pk_fk.referenced_column.name()}"
            fk = f"{pk_fk.table.name()}.{pk_fk.column.name()}"

            join_cond_2_fk[f"{fk} = {pk}"] = pk_fk
            
            if pk not in pk_2_fk:
                pk_2_fk[pk] = [fk]
            else:
                pk_2_fk[pk].append(fk)
            
            possbile_joins.append(pk_fk)
            
        for pk, fk_list in pk_2_fk.items():
            n = len(fk_list)
            for i in range(0, n - 1):
                for j in range(i + 1, n):
                    pk_fk_1 = join_cond_2_fk[f"{fk_list[i]} = {pk}"]         
                    pk_fk_2 = join_cond_2_fk[f"{fk_list[j]} = {pk}"]
                    if pk_fk_1.name < pk_fk_2.name:
                        possbile_joins.append((pk_fk_1, pk_fk_2))
                    else:
                        possbile_joins.append((pk_fk_2, pk_fk_1))
        
        return possbile_joins
    