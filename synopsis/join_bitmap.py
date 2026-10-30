import psycopg2
from schema.sql_schema import SQLSchema
from schema.comparison_operator import OPERATOR_CMP_FUNC
from query.sql_query import SQLQuery
from typing import List, Tuple, Any

class JoinBitmap:
    def __init__(self, schema: SQLSchema, sample_number: int, foreign_table_ratio: float):
        self.primary_key_sample = dict() # tb -> { sample }
        self.foreign_key_sample = dict() # tb.pk -> { tb.fk -> sample }

        self.sample_number = sample_number
        self.foreign_table_sample_number = dict() # tb -> int
        for table in schema.tables():
            self.foreign_table_sample_number[table.name()] = min(max(
                int(foreign_table_ratio * table.cardinality()), sample_number), 
                table.cardinality())
        
        self.table_columns = {table.name(): [column.name() for column in table.columns()] for table in schema.tables()}
        self._column_positions = {}
        self._foreign_key_positions = {}
        self._unfiltered_foreign_key_values = {}
        self._evaluation_cache_ready = False

    def update(self, schema: SQLSchema):
        primary_key_sample, foreign_key_sample = dict(), dict()
        primary_table_pk_cache = dict() # tb -> { pk -> List[int] }
        conn = schema.connection()
        
        for table in schema.tables():
            pk_table_name = table.name()
            pk_tb_cols = [column.name() for column in table.columns()]
            samples = JoinBitmap.sample_primary_table(conn, pk_table_name, pk_tb_cols, self.sample_number)
            primary_key_sample[pk_table_name] = samples

            in_neigh = schema.in_neighbors(table)
            if len(in_neigh) == 0:
                continue
            # This is a PK table    
            primary_table_pk_cache[pk_table_name] = dict()
            for fk, fk_tb in in_neigh:
                pk_col = fk.referenced_column.name()
                pk_col_idx = pk_tb_cols.index(pk_col)
                if pk_col not in primary_table_pk_cache[pk_table_name]:
                    primary_table_pk_cache[pk_table_name][pk_col] = [sample[pk_col_idx] for sample in samples]

                fk_col = fk.column.name()
                fk_tb_cols = [column.name() for column in fk_tb.columns()]

                samples = JoinBitmap.sample_foreign_table(conn, fk_tb.name(), fk_tb_cols, fk_col, 
                    primary_table_pk_cache[pk_table_name][pk_col], self.foreign_table_sample_number[fk_tb.name()])
                
                pk_name = f"{pk_table_name}.{pk_col}"
                fk_name = f"{fk_tb.name()}.{fk_col}"
                if pk_name not in foreign_key_sample:
                    foreign_key_sample[pk_name] = dict()
                foreign_key_sample[pk_name][fk_name] = samples

        self.primary_key_sample = primary_key_sample
        self.foreign_key_sample = foreign_key_sample
        self._rebuild_evaluation_cache()

    @staticmethod
    def sample_primary_table(connection: psycopg2.extensions.connection, table: str, columns: List[str], sample_size: int) -> List[Tuple[Any]]:
        cursor = connection.cursor()

        sql = "SELECT %s FROM %s ORDER BY random() LIMIT %d;" % (", ".join(columns), table, sample_size)
        cursor.execute(sql)

        records = cursor.fetchall()
        cursor.close()

        return records
    
    @staticmethod
    def sample_foreign_table(connection: psycopg2.extensions.connection, table: str, columns: List[str], 
                             fk_column: str, fk_values: List[int], sample_size: int) -> List[Tuple[Any]]:
        cursor = connection.cursor()

        sql = "SELECT %s FROM %s WHERE %s IN %s ORDER BY random() LIMIT %d;" % \
            (", ".join(columns), table, fk_column, str(tuple(fk_values)), sample_size)
        cursor.execute(sql)

        records = cursor.fetchall()
        cursor.close()

        return records
    
    def _rebuild_evaluation_cache(self) -> None:
        """Build immutable lookup data used by query-time bitmap evaluation."""
        self._column_positions = {
            table_name: {column_name: i for i, column_name in enumerate(columns)}
            for table_name, columns in self.table_columns.items()
        }
        self._foreign_key_positions = {}
        self._unfiltered_foreign_key_values = {}

        for pk_name, foreign_keys in self.foreign_key_sample.items():
            self._foreign_key_positions[pk_name] = {}
            self._unfiltered_foreign_key_values[pk_name] = {}

            for fk_name, rows in foreign_keys.items():
                fk_table_name, fk_column_name = fk_name.rsplit(".", 1)
                fk_column_pos = self._column_positions[fk_table_name][fk_column_name]
                self._foreign_key_positions[pk_name][fk_name] = fk_column_pos
                self._unfiltered_foreign_key_values[pk_name][fk_name] = frozenset(
                    row[fk_column_pos] for row in rows
                )
        self._evaluation_cache_ready = True

    def _ensure_evaluation_cache(self) -> None:
        # JoinBitmap objects written before these caches were introduced do
        # not contain the attributes. Rebuild them lazily after unpickling.
        if not getattr(self, "_evaluation_cache_ready", False):
            self._rebuild_evaluation_cache()

    def _compile_predicates(self, table_name, predicates):
        column_positions = self._column_positions[table_name]
        return [
            (
                column_positions[predicate.column.name()],
                OPERATOR_CMP_FUNC[predicate.operator],
                predicate.value,
            )
            for predicate in predicates
        ]

    @staticmethod
    def _row_matches(row, predicate_specs) -> bool:
        for column_pos, compare, value in predicate_specs:
            if not compare(row[column_pos], value):
                return False
        return True

    @staticmethod
    def _matching_foreign_key_values(rows, fk_column_pos, predicate_specs):
        matching_values = set()
        add_value = matching_values.add

        for row in rows:
            for column_pos, compare, value in predicate_specs:
                if not compare(row[column_pos], value):
                    break
            else:
                add_value(row[fk_column_pos])

        return matching_values

    def evaluate(self, query: SQLQuery) -> List[Tuple[Tuple[str, str], List[str], List[bool]]]:
        """
        Return one correlated bitmap for every equivalent join-key group.

        Each result is ``((pk_table, pk_column), involved_tables, bitmap)``.
        Queries must be canonicalized first, so every join is represented as a
        PK-FK edge and the referenced PK table is present (possibly virtually).
        """
        self._ensure_evaluation_cache()

        nodes_by_table = {node.table().name(): node for node in query.nodes()}
        predicate_specs_by_table = {
            table_name: self._compile_predicates(
                table_name, node.predicates()
            )
            for table_name, node in nodes_by_table.items()
        }
        join_groups = {}

        for edge in query.edges():
            if edge.is_fk_fk():
                raise ValueError("JoinBitmap requires canonicalized PK-FK joins")

            fk = edge.foreign_key()
            group_key = (fk.referenced_table.name(), fk.referenced_column.name())
            join_groups.setdefault(group_key, []).append(fk)

        group_bitmaps = []
        for pk_table_name, pk_column_name in sorted(join_groups):
            group_key = (pk_table_name, pk_column_name)
            foreign_keys = join_groups[group_key]
            pk_node = nodes_by_table[pk_table_name]
            pk_column_pos = self._column_positions[pk_table_name][pk_column_name]
            pk_samples = self.primary_key_sample[pk_table_name]

            # A PK node inserted only to canonicalize an FK-FK join anchors the
            # correlated sample, but it is not a table in the original query.
            involved_tables = {fk.table.name() for fk in foreign_keys}
            if not pk_node.is_virtual():
                involved_tables.add(pk_table_name)
            pk_name = f"{pk_table_name}.{pk_column_name}"

            # Intersect surviving join-key values as each FK relation is
            # processed. For unfiltered tables, use the precomputed value set
            # instead of scanning and copying the full FK sample per query.
            eligible_pk_values = None
            for fk in foreign_keys:
                fk_table_name = fk.table.name()
                fk_column_name = fk.column.name()
                fk_name = f"{fk_table_name}.{fk_column_name}"
                predicate_specs = predicate_specs_by_table[fk_table_name]

                if len(predicate_specs) == 0:
                    matching_values = self._unfiltered_foreign_key_values[pk_name][fk_name]
                else:
                    matching_values = self._matching_foreign_key_values(
                        self.foreign_key_sample[pk_name][fk_name],
                        self._foreign_key_positions[pk_name][fk_name],
                        predicate_specs,
                    )

                if eligible_pk_values is None:
                    eligible_pk_values = set(matching_values)
                else:
                    eligible_pk_values.intersection_update(matching_values)

                if len(eligible_pk_values) == 0:
                    break

            bitmap = [False] * self.sample_number
            pk_predicate_specs = predicate_specs_by_table[pk_table_name]

            if eligible_pk_values:
                if len(pk_predicate_specs) == 0:
                    for i, row in enumerate(pk_samples):
                        bitmap[i] = row[pk_column_pos] in eligible_pk_values
                else:
                    for i, row in enumerate(pk_samples):
                        if self._row_matches(row, pk_predicate_specs):
                            bitmap[i] = row[pk_column_pos] in eligible_pk_values

            group_bitmaps.append((group_key, sorted(involved_tables), bitmap))

        return group_bitmaps

    def display(self):
        import tabulate
        result = []
        for table_name, sample in self.primary_key_sample.items():
            result.append(f"PK sample {table_name}\n")
            headers = self.table_columns[table_name]
            data = sample
            result.append(tabulate.tabulate(data, headers, tablefmt="grid"))
        for pk, fk_and_sample in self.foreign_key_sample.items():
            for fk, sample in fk_and_sample.items():
                result.append(f"FK sample {pk} <- {fk}\n")
                headers = self.table_columns[fk.split('.')[0]]
                data = sample
                result.append(tabulate.tabulate(data, headers, tablefmt="grid"))
        return "".join(result)
