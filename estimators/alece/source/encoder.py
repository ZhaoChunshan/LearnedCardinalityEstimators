import numpy as np
from typing import List
from schema.sql_schema import SQLSchema, SQLColumn
from query.sql_query import SQLQuery


class ALECEEncoder:
    def __init__(self, schema: SQLSchema) -> None:
        self._schema = schema
        self._columns: List[SQLColumn] = sorted(
            [column for table in self._schema.tables() for column in table.columns() if column.predicable()],
            key=lambda column: column.id()
        )
        self._column_2_pos = {column: i for i, column in enumerate(self._columns)}

        self._table_num = self._schema.table_num()
        self._foreign_key_num = self._schema.foreign_key_num()
        self._predicate_column_num = self._schema.predicable_column_num()

        self._epsilon_for_int = 0.5
        self._epsilon_for_float = 1e-6 

        epsilon = np.array([self._epsilon_for_float] * self._predicate_column_num, dtype=np.float32)
        epsilon[[column.data_type() in ["smallint", "integer", "bigint", "character varying"] for column in self._columns]] = self._epsilon_for_int
        self._attr_min = np.array([column.min_float() for column in self._columns], dtype=np.float32)
        self._attr_max = epsilon + np.array([column.max_float() for column in self._columns], dtype=np.float32)

        self._attr_range = self._attr_max - self._attr_min

    def feature_size(self) -> int:
        return self._table_num + self._foreign_key_num + self._predicate_column_num * 2

    def encode(self, query: SQLQuery) -> np.ndarray:
        pass

    def encode_batch(self, queries: List[SQLQuery]) -> np.ndarray:
        batch_size = len(queries)
        features = np.zeros((batch_size, self.feature_size()), dtype=np.float32)
        
        for i, query in enumerate(queries):
            relavant_tables = [node.table().id() for node in query.nodes()]
            features[i][relavant_tables] = 1.0
            
            relavant_fk = [self._table_num + edge.foreign_key().id for edge in query.edges()]
            if len(relavant_fk) > 0:
                features[i][relavant_fk] = 1.0
            
            features[i][self._table_num + self._foreign_key_num : ] = self.encode_filtering_predicate(query)
        
        return features

    def encode_filtering_predicate(self, query: SQLQuery) -> np.ndarray:
        filter_feature = np.zeros((self._predicate_column_num, 2), dtype=np.float32)
        filter_feature[:, 1] = 1.0

        for node in query.nodes():
            for predicate in node.predicates():
                epsilon = self._epsilon_for_int if predicate.column.data_type() in ["smallint", "integer", "bigint", "character varying"] \
                            else self._epsilon_for_float
                index = self._column_2_pos[predicate.column]
                value = predicate.float_value()
                
                if predicate.operator == '<':
                    filter_feature[index, 1] = (value - epsilon - self._attr_min[index]) / self._attr_range[index]
                elif predicate.operator == '>':
                    filter_feature[index, 0] = (value + epsilon - self._attr_min[index]) / self._attr_range[index]
                elif predicate.operator == '<=':
                    filter_feature[index, 1] = (value + epsilon - self._attr_min[index]) / self._attr_range[index]
                elif predicate.operator == '>=':
                    filter_feature[index, 0] = (value - epsilon - self._attr_min[index]) / self._attr_range[index]
                elif predicate.operator == '=':
                    filter_feature[index, 0] = (value - epsilon - self._attr_min[index]) / self._attr_range[index]
                    filter_feature[index, 1] = (value + epsilon - self._attr_min[index]) / self._attr_range[index]
                else:
                    assert(False)
        return filter_feature.reshape(-1) * 2.0 - 1
