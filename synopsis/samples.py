import psycopg2
import numpy as np

from random import Random
from typing import List, Tuple, Set, Dict, Any
from schema.comparison_operator import OPERATOR_CMP_FUNC
from schema.data_type import DataType
from synopsis.greedy_feature_selection import greedy_feature_selection


class Samples:
    def __init__(self, names: List[str], types: List[DataType], name_2_pos: Dict[str, int], records: List[Tuple[Any]]):
        self._names = names
        self._types = types
        self._name_2_pos = name_2_pos
        self._records = records
    

    def __len__(self):
        return len(self._records)

    def evaluate(self, predicates: List, bitmap_size: int) -> List[bool]:
        bitmap = [False] * bitmap_size
        predicate_num = len(predicates)

        predicate_columns_pos = [self._name_2_pos[predicate.column.name()] for predicate in predicates]
        predicate_cmp_func = [OPERATOR_CMP_FUNC[predicate.operator] for predicate in predicates]
        predicate_literal = [predicate.value for predicate in predicates]

        for i, record in enumerate(self._records[:bitmap_size]):
            result = True

            for j in range(predicate_num):
                if not predicate_cmp_func[j](record[predicate_columns_pos[j]], predicate_literal[j]):
                    result = False
                    break
            
            bitmap[i] = result

        return bitmap

    def evaluate_one_record(self, record_index: int, predicates: List) -> bool:
        for predicate in predicates:
            predicate_column_pos = self._name_2_pos[predicate.column.name()]
            predicate_cmp_func = OPERATOR_CMP_FUNC[predicate.operator]
            predicate_literal = predicate.value
            if not predicate_cmp_func(self._records[record_index][predicate_column_pos], predicate_literal):
                return False
        
        return True

    def update_records(self, records: List[List[Any]]):
        self._records = records

    def select_feature_samples(self, query_nodes: List, max_fs_choices: int, feature_num: int) -> None:
        a_set: Set[Tuple[bool]] = set()
        a_list: List[List[bool]] = []
        candidates: List[List[Any]] = []

        for i, record in enumerate(self._records):
            a = [self.evaluate_one_record(i, query_node.predicates()) for query_node in query_nodes]
            a_sum = sum(a)
            if a_sum == 0 or a_sum == len(query_nodes):
                continue
            a_tuple = tuple(a)
            if a_tuple not in a_set:
                a_set.add(a_tuple)
                a_list.append(a)
                candidates.append(record)
                if len(a_list) >= max_fs_choices:
                    break
        
        if len(candidates) != 0:
            A = np.array(a_list, dtype=np.float32).T
            candidate_ids = greedy_feature_selection(A, min(feature_num, len(candidates)))
            self.update_records([candidates[i] for i in candidate_ids])
        else:
            self._records = self._records[:feature_num]

    def __str__(self) -> str:
        import tabulate
        
        headers = [n for _, n in sorted([(i, n) for n, i in self._name_2_pos.items()])]
        data = self._records
        return tabulate.tabulate(data, headers, tablefmt="grid")
    
    @staticmethod
    def sample_records(connection: psycopg2.extensions.connection, table: str, columns: List[str], sample_size: int) -> List[Tuple[Any]]:
        cursor = connection.cursor()

        sql = "SELECT %s FROM %s ORDER BY random() LIMIT %d;" % (", ".join(columns), table, sample_size)
        cursor.execute(sql)

        records = cursor.fetchall()
        cursor.close()

        return records


    @staticmethod
    def from_connection(connection: psycopg2.extensions.connection, table: str, columns: List, sample_size: int) -> "Samples":
        columns_name = [column.name() for column in columns]
        records = Samples.sample_records(connection, table, columns_name, sample_size)
        return Samples(columns_name, [column.data_type() for column in columns], {column.name(): i for i, column in enumerate(columns)}, records)
