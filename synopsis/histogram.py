import copy
import psycopg2
import numpy as np

from typing import List, Tuple, Dict, Any, Optional
from schema.comparison_operator import ColumnType
from common.value_wrapper import date_to_int

class TableHistogram:
    def __init__(self, n_bins: int, n_attrs: int, n_rows: int, attr_pos_map: Dict[str, int], min_vals: np.ndarray, max_vals: np.ndarray, 
                 bin_sizes: np.ndarray, histogram: np.ndarray) -> None:
        self._n_bins = n_bins
        self._n_attrs = n_attrs
        self._n_rows = n_rows
        # Keep the feature scale fixed across database snapshots.  For a
        # histogram built from the current (static) database this is identical
        # to ``_n_rows`` and therefore preserves the existing static behavior.
        # Dynamic experiments replace ``_n_rows`` with the snapshot size while
        # retaining this threshold at the full-table cardinality.
        self._table_size_threshold = n_rows

        self._attr_pos_map = attr_pos_map
        self._min_vals = min_vals
        self._max_vals = max_vals
        self._bin_sizes = bin_sizes
        self._histogram = histogram

        arange_idxes = np.reshape(np.arange(0, self._n_attrs, dtype=np.int64), [1, -1])
        # To facilitate the update of histogram
        self._indices = np.concatenate([arange_idxes, np.zeros(shape=arange_idxes.shape, dtype=arange_idxes.dtype)], axis=0)

    def attr_pos_map(self) -> Dict[str, int]:
        return self._attr_pos_map

    def set_table_size_threshold(self, n_rows: int) -> None:
        if n_rows <= 0:
            raise ValueError("table_size_threshold must be positive")
        self._table_size_threshold = n_rows

    def histogram(self) -> np.ndarray:
        # ``getattr`` keeps histogram pickles created before the fixed-scale
        # field was introduced readable.  ALECE's dynamic loading path upgrades
        # those objects to the full-table threshold before using them.
        table_size_threshold = getattr(self, "_table_size_threshold", self._n_rows)
        return self._histogram / table_size_threshold

    @staticmethod
    def from_connection(connection: psycopg2.extensions.connection, table, n_bins: int) -> Optional["TableHistogram"]:  
        columns = [column for column in table.columns() if column.predicable()]
        
        if len(columns) == 0:
            return None

        columns = sorted(columns, key=lambda column: column.id())

        attr_pos_map = {column.name(): i for i, column in enumerate(columns)}
        n_rows = table.cardinality()
        n_attrs = len(columns)

        min_vals = np.zeros(shape=(n_attrs,), dtype=np.float32)
        max_vals = np.zeros(shape=(n_attrs,), dtype=np.float32)
        
        records = TableHistogram.get_records(connection, table.name(), [column.name() for column in columns])
        attr_values = list(zip(*records))
        initial_data = np.zeros(shape=(n_attrs, n_rows), dtype=np.float32)

        for i, column in enumerate(columns):
            data_type, column_type = column.data_type(), column.column_type()

            print("Build histogram for column %s whose datatype is %s." % (column.name(), data_type.name()))
            # for int && double value
            if data_type.name() in ["smallint", "integer", "bigint", "double precision"]:
                min_vals[i], max_vals[i] = column.min(), column.max()
                initial_data[i] = np.sort(np.array(attr_values[i], dtype=np.float32))
            # for string categorical value
            elif data_type.name() == "character varying":
                assert(column_type == ColumnType.CATEGORICAL or column_type == ColumnType.STRING_WITH_CMP)
                min_vals[i], max_vals[i] = column.min(), column.max()
                initial_data[i] = np.sort(np.array([column.value_2_id(value) for value in attr_values[i]], dtype=np.float32))
            # for date
            elif data_type.name() == "date":
                min_vals[i], max_vals[i] = map(date_to_int, [column.min(), column.max()])
                initial_data[i] = np.sort(np.array([date_to_int(value) for value in attr_values[i]], dtype=np.float32))
            # for interval
            elif data_type.name() == "interval":
                min_vals[i], max_vals[i] = int(column.min().total_seconds()), int(column.max().total_seconds())
                initial_data[i] = np.sort(np.array([int(value.total_seconds()) for value in attr_values[i]], dtype=np.float32))
            else:
                raise ValueError("Unsupported data type %s." % data_type.name())
        
        bin_sizes = (max_vals - min_vals) / n_bins
        boundary_points = []
        for _ in range(n_attrs):
            arange_idxes = np.reshape(np.arange(1, n_bins + 1, dtype=np.float32), [1, -1])
            boundary_points.append(arange_idxes)
        boundary_points = np.concatenate(boundary_points, axis=0)
        boundary_points *= np.reshape(bin_sizes, [-1, 1])
        boundary_points += np.reshape(min_vals, [-1, 1])
        boundary_points[:, -1] += 1e6

        histogram = []
        for i in range(n_attrs):
            sorted_idxes = np.searchsorted(initial_data[i], boundary_points[i], side='left')
            print("sorted_idxes[-1] = {}, initial_data[i].shape[0] = {}".format(sorted_idxes[-1], initial_data[i].shape[0]))
            assert sorted_idxes[-1] == initial_data[i].shape[0] or np.isnan(initial_data[i,sorted_idxes[-1]])
            sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
            sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
            sorted_idxes_left_translation[0] = 0
            diff = sorted_idxes - sorted_idxes_left_translation
            histogram.append(np.reshape(diff, [1, -1]))
        
        histogram = np.concatenate(histogram, axis=0, dtype=np.float32)
        return TableHistogram(n_bins, n_attrs, n_rows, attr_pos_map, min_vals, max_vals, bin_sizes, histogram)

    def update_from_data(self, df):
        self._n_rows += df.shape[0]
            
        pos_attr_dict = {i: attr for attr, i in self.attr_pos_map().items()}
            
        initial_data = np.zeros(shape=(self._n_attrs, df.shape[0]), dtype=np.float32)
        boundary_points = []
            
        for i in range(self._n_attrs):
            initial_data[i] = np.sort(df[pos_attr_dict[i]].values)
            arange_idxes = np.reshape(np.arange(1, self._n_bins + 1, dtype=np.float32), [1, -1])
            boundary_points.append(arange_idxes)
            
        boundary_points = np.concatenate(boundary_points, axis=0)
        boundary_points *= np.reshape(self._bin_sizes, [-1, 1])
        boundary_points += np.reshape(self._min_vals, [-1, 1])
        boundary_points[:, -1] += 1e6

        new_histogram = []
        
        for i in range(self._n_attrs):
            
            sorted_idxes = np.searchsorted(initial_data[i], boundary_points[i], side='left')
            
            assert sorted_idxes[-1] == initial_data[i].shape[0] or np.isnan(initial_data[i,sorted_idxes[-1]])
        
            sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
            sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
            sorted_idxes_left_translation[0] = 0
            diff = sorted_idxes - sorted_idxes_left_translation
            
            new_histogram.append(np.reshape(diff, [1, -1]))
            
        new_histogram = np.concatenate(new_histogram, axis=0, dtype=np.float32)
        
        self._histogram += new_histogram

    @staticmethod
    def get_records(connection: psycopg2.extensions.connection, table_name: str, column_names: List[str]) -> List[Tuple[Any]]:
        cursor = connection.cursor()
        sql = "SELECT %s FROM %s;" % (", ".join(column_names), table_name)
        cursor.execute(sql)
        records = cursor.fetchall()
        cursor.close()
        return records
