import os
import psycopg2
import polars as pl
from functools import reduce
from time import perf_counter
from typing import List, Optional
from collections import deque
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from schema.schemas import imdb_m_schema



class EvaluationTreeNode:
    def __init__(self, sql: str, parent_key: Optional[str] = None, my_key: Optional[str] = None, children: List['EvaluationTreeNode'] = None):
        self.sql = sql
        self.parent_key = parent_key
        self.my_key = my_key
        self.children = [] if children is None else children
        self.join_key_count = {} # Store (key, count/expansion_degree) in dict format
        self.join_key_count_df: Optional[pl.DataFrame] = None  # Store (key, count/expansion_degree) in DataFrame format
        self.cardinality: Optional[int] = None  # Save cardinality when at the root
    
    def add_child(self, child: 'EvaluationTreeNode'):
        self.children.append(child)

    def add_join_key_count(self, key: int, count: int):
        if key in self.join_key_count:
            self.join_key_count[key] += count
        else:
            self.join_key_count[key] = count


class CardinalityCalculator:
    def __init__(self, schema: SQLSchema, host: str, db_name: str, user: str, password: str, port: int = 5432, table_csv_path = None,
                 polars_csv_schema = None):
        self.schema = schema
        self.db_name = db_name
        self.connection = psycopg2.connect(host=host, database=db_name, user=user, password=password, port=port)
        self.table_csv_path = table_csv_path
        if self.table_csv_path is not None:
            self.df_cache = {}
            for table in self.schema.tables():
                table_name = table.name()
                csv_path = os.path.join(self.table_csv_path, table_name + ".csv")
                
                if polars_csv_schema is None:
                    self.df_cache[table_name] = pl.read_csv(csv_path, separator="|", null_values=["NULL", ""])
                else:
                    self.df_cache[table_name] = pl.read_csv(csv_path, separator="|", null_values=["NULL", ""], schema=polars_csv_schema.get(table_name, None))

    def calculate_cardinality(self, query: SQLQuery) -> int:
        """
        Calculates the cardinality of a given query. The query must be acyclic.
        """
        root = self.traverse_query_bfs(query)
        if self.table_csv_path is None:
            self.evaluate_tree_postgres(root)
        else:
            self.evaluate_tree_polars(root)
        assert(type(root.cardinality) == int)
        return root.cardinality
    
    def calculate_cardinality_sql(self, sql: str) -> int:
        query = SQLQuery.parse(self.schema, sql)
        return self.calculate_cardinality(query)
    
    def traverse_query_bfs(self, query: SQLQuery) -> EvaluationTreeNode:
        table_dict = {node.table(): node for node in query.nodes()}
        
        root_query_node = query.nodes()[0]
        visited = {root_query_node}
        dq = deque()
        root = EvaluationTreeNode(SQLQuery([root_query_node], []).sql())
        dq.append((root, root_query_node))
        
        while dq:
            parent, query_node = dq.popleft()
            for edge in query.adj_list(query_node):
                if edge.is_pk_fk():
                    fk = edge.foreign_key()
                    if fk.referenced_table == query_node.table():
                        parent_key = fk.referenced_column.name()
                        my_key = fk.column.name()
                        my_query_node = table_dict[fk.table]                
                    else:
                        parent_key = fk.column.name()
                        my_key = fk.referenced_column.name()
                        my_query_node = table_dict[fk.referenced_table]
                else:
                    fk1, fk2 = edge.foreign_key()
                    if fk1.table == query_node.table():
                        parent_key = fk1.column.name()
                        my_key = fk2.column.name()
                        my_query_node = table_dict[fk2.table]
                    else:
                        parent_key = fk2.column.name()
                        my_key = fk1.column.name()
                        my_query_node = table_dict[fk1.table]
                
                if my_query_node in visited:
                    continue
                
                tree_node = EvaluationTreeNode(SQLQuery([my_query_node], []).sql(),
                                               parent_key, my_key)
                parent.add_child(tree_node)
                visited.add(my_query_node)
                dq.append((tree_node, my_query_node))
        
        return root


    def evaluate_tree_postgres(self, tree: EvaluationTreeNode):
        for child in tree.children:
            self.evaluate_tree_postgres(child)
        
        with self.connection.cursor(name='stream_cursor') as cursor:
            batch_size = 1000
            
            if tree.parent_key is None:
                if len(tree.children) == 0:
                    cursor.execute(tree.sql)
                    row = cursor.fetchone()
                    cardinality = row[0]
                
                else:
                    cardinality = 0
                    my_key = [c.parent_key for c in tree.children]
                    sql = CardinalityCalculator.modifiy_sql(tree.sql, my_key)
                    cursor.execute(sql)
                    
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break
                    
                        for row in rows:
                            expand = 1
                            for i, child in enumerate(tree.children):
                                if type(row[i]) == int:
                                    pass
                                elif row[i] is None:
                                    expand = 0
                                    print("None detected on branch 2.")
                                    break
                                else:
                                    print(f"Unexpected type {type(row[i])} detected on {sql} with column = {my_key[i]}.")    
                                    assert(False)
                                
                                expand *= child.join_key_count.get(row[i], 0)
                                if expand == 0:
                                    break
                            cardinality += expand
                
                tree.cardinality = cardinality
                
            else:
                if len(tree.children) == 0:
                    sql = CardinalityCalculator.modifiy_sql(tree.sql, [tree.my_key])
                    cursor.execute(sql)
                    
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break

                        for row in rows:
                            if type(row[0]) == int:
                                pass
                            elif row[0] is None:
                                print("None detected on branch 3.")
                                continue
                            else:
                                print(f"Unexpected type {type(row[0])} detected on {sql} with column = {tree.my_key}.")    
                                assert(False)
                            
                            tree.add_join_key_count(row[0], 1)
                
                else:
                    my_key = [c.parent_key for c in tree.children] + [tree.my_key]
                    sql = CardinalityCalculator.modifiy_sql(tree.sql, my_key)
                    cursor.execute(sql)
                    
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break
                        
                        for row in rows:
                            expand = 1
                            for i, child in enumerate(tree.children):
                                if type(row[i]) == int:
                                    pass
                                elif row[i] is None:
                                    expand = 0
                                    print("None detected on branch 4.")
                                    break
                                else:
                                    print(f"Unexpected type {type(row[i])} detected on {sql} with column = {my_key[i]}.")    
                                    assert(False)
                                
                                expand *= child.join_key_count.get(row[i], 0)
                                if expand == 0:
                                    break
                            tree.add_join_key_count(row[-1], expand)
    
    def evaluate_tree_polars(self, tree: EvaluationTreeNode):
        # Recursively calculate child nodes
        for child in tree.children:
            self.evaluate_tree_polars(child)

        query = SQLQuery.parse(self.schema, tree.sql)
        table_name = query.nodes()[0].table().name()
        predicates = query.nodes()[0].predicates()
        df = self.df_cache[table_name]

        # Apply predicate filtering
        if len(predicates) > 0:
            df = self._apply_predicates(df, predicates)

        if tree.parent_key is None:  # root
            if len(tree.children) == 0:
                # Is root and has no children
                tree.cardinality = df.height
            else:
                df_expanded = df
                for i, child in enumerate(tree.children):
                    # Directly use child.join_key_count_df
                    child_ext = child.join_key_count_df.rename({"ext": f"ext_child{i}"})
                    df_expanded = df_expanded.join(
                        child_ext,
                        left_on=child.parent_key,   # Column in the parent table
                        right_on=child.my_key,      # Key column in the child statistics table
                        how="left"
                    ).fill_null(0)

                # Product of expansion degrees
                ext_cols = [f"ext_child{i}" for i in range(len(tree.children))]
                df_expanded = df_expanded.with_columns(
                    reduce(lambda a, b: a * b, [pl.col(c) for c in ext_cols]).alias("ext_total")
                )
                tree.cardinality = int(df_expanded["ext_total"].sum())

        else:  # Non-root
            if len(tree.children) == 0:
                # Count by my_key
                grouped = df.group_by(tree.my_key).agg(pl.count().cast(pl.Int64).alias("ext"))
                tree.join_key_count_df = grouped
            else:
                df_expanded = df
                for i, child in enumerate(tree.children):
                    child_ext = child.join_key_count_df.rename({"ext": f"ext_child{i}"})
                    df_expanded = df_expanded.join(
                        child_ext,
                        left_on=child.parent_key,   # Column in the parent table
                        right_on=child.my_key,      # Key column in the child statistics table
                        how="left"
                    ).fill_null(0)

                ext_cols = [f"ext_child{i}" for i in range(len(tree.children))]
                df_expanded = df_expanded.with_columns(
                    reduce(lambda a, b: a * b, [pl.col(c) for c in ext_cols]).alias("ext_total")
                )

                # Aggregate to my_key
                grouped = df_expanded.group_by(tree.my_key).agg(pl.sum("ext_total").alias("ext"))
                tree.join_key_count_df = grouped

    def _apply_predicates(self, df: pl.DataFrame, predicates):
        exprs = []
        for p in predicates:
            col = p.column.name()
            val = p.value
            op = p.operator

            if op == "<":
                exprs.append(pl.col(col) < val)
            elif op == ">":
                exprs.append(pl.col(col) > val)
            elif op == "<=":
                exprs.append(pl.col(col) <= val)
            elif op == ">=":
                exprs.append(pl.col(col) >= val)
            elif op == "=":
                exprs.append(pl.col(col) == val)
        return df.filter(reduce(lambda a, b: a & b, exprs))

    @staticmethod
    def modifiy_sql(sql: str, keys: List[str]):
        return sql.replace("SELECT COUNT(*)", "SELECT " + ", ".join(keys))

class RobustCardinalityCalculator(CardinalityCalculator):
    def __init__(self, schema: SQLSchema, host: str, db_name: str, user: str, password: str, port: int = 5432):
        super().__init__(schema, host, db_name, user, password, port)
        self.host = host 
        self.db_name = db_name
        self.user = user
        self.password = password
        self.port = port
        self.connection = self.get_connection()
    
    def get_connection(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(host=self.host, database=self.db_name, user=self.user, password=self.password, port=self.port)
    
    def run(self, sql: str) -> int:
        timeout_seconds = 10
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"SET statement_timeout TO {timeout_seconds * 1000};")  # Unit: milliseconds
                cursor.execute(sql)
                result = cursor.fetchone()
                return int(result[0])
        
        except psycopg2.errors.QueryCanceled as e:
            # print(f"Query failed due to: {e}")
            self.connection.rollback()
            query = SQLQuery.parse(self.schema, sql)
            return self.calculate_cardinality(query)
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            exit(0)

