import os
import pandas as pd
import numpy as np
import pickle

from typing import Dict, Tuple, List, Optional, Union
from schema.sql_schema import SQLSchema, SQLForeignKey
from schema.comparison_operator import ColumnType


class PRICEStatistics:
    def __init__(self, abbrev_coltype, histogram, summary, fanout, size):
        self.abbrev_coltype = abbrev_coltype
        self.histogram = histogram
        self.summary = summary
        self.fanout = fanout
        self.size = size
    
    @staticmethod
    def gen_abbrev_col_type(schema: SQLSchema):
        join_keys = set()
        
        for fk in schema.foreign_keys():
            join_keys.add(f"{fk.table.try_alias()}.{fk.column.name()}")
            join_keys.add(f"{fk.referenced_table.try_alias()}.{fk.referenced_column.name()}")
        
        abbrev, col_type = {}, {}
        
        for table in schema.tables():
            
            alias = table.try_alias()
            abbrev[table.name()] = alias
            
            col_type[alias] = {'ctn':[], 'dsct':[]}
            
            for column in table.columns():
                
                column_type = column.column_type()
                column_name = column.name()
                
                if column_type == ColumnType.OTHER:
                    continue
                
                if f"{alias}.{column_name}" in join_keys:
                    # Always consider join keys as continuous.
                    continue

                if column_type == ColumnType.NUMERICAL:
                    col_type[alias]['ctn'].append(column_name)
                    
                elif column_type == ColumnType.CATEGORICAL:
                    col_type[alias]['dsct'].append(column_name)
                    
                elif column_type == ColumnType.STRING_WITH_CMP:
                    # The string will be mapped to integer later
                    col_type[alias]['ctn'].append(column_name)
                    
                else:
                    assert False, f"Unknown column type: {column_type}"

        for join_key in join_keys:
            alias, column = join_key.split('.')
            col_type[alias]['ctn'].append(column)
        
        return {'abbrev': abbrev, 'col_type': col_type}
    
    
    @staticmethod
    def gen_histogram(table_df_dict: Dict, col_type: Dict, bin_size: int):
        results = {}
        
        for table in table_df_dict:
            
            results[table] = {}
            
            for column in table_df_dict[table].columns:
                
                if column in col_type[table]['ctn']:
                    
                    len_column = table_df_dict[table][column].shape[0]
                    tmp_column = table_df_dict[table][column].dropna()
                    tmp_column = tmp_column.astype(float)
                    
                    if len(tmp_column) == 0:
                        hist, bin_edges, len_column, min_value, max_value = np.zeros(bin_size), np.zeros(bin_size), 0, 0, 0
                    else:
                        hist, bin_edges = np.histogram(tmp_column, bins=bin_size, density=False)
                        hist = hist.astype(float)
                        min_value, max_value = min(tmp_column), max(tmp_column)
        
                    results[table][column] = {'hist':hist, 'bin_edges':bin_edges, 'len':len_column, 'min_value':min_value, 'max_value':max_value}
        
        return results

    
    @staticmethod
    def gen_summary(table_df_dict: Dict, col_type: Dict):
        results = {}
        
        for table in table_df_dict:
            
            results[table] = {}
            
            for column in table_df_dict[table].columns:
                
                if column in col_type[table]['dsct']:
                    
                    K = 99999999
                    summary = {}
                    tmp_column = table_df_dict[table][column].dropna()
                    
                    for c in tmp_column:
                        if c in summary:
                            summary[c] += 1
                        elif len(summary) < K:
                            summary[c] = 1
                        else:
                            raise ValueError('summary length must be less than K')
                        
                    keys, values = list(summary.keys()), list(summary.values())
                    
                    assert len(keys) == len(values)
        
                    results[table][column] = {'keys': keys, 'values': values}
        
        return results
    
    
    @staticmethod
    def gen_fanout(schema: SQLSchema, table_df_dict: Dict, abbrev: Dict[str, str], histogram: Dict, bin_size: int):
        possible_joins = schema.possible_joins()
        
        def table_column_tuple_from_join(join: Union[SQLForeignKey, Tuple[SQLForeignKey, SQLForeignKey]]):
            if isinstance(join, SQLForeignKey):
                t1, c1 = abbrev[join.table.name()], join.column.name(),
                t2, c2 = abbrev[join.referenced_table.name()], join.referenced_column.name()
            else:
                fk1, fk2 = join
                t1, c1 = abbrev[fk1.table.name()], fk1.column.name()
                t2, c2 = abbrev[fk2.table.name()], fk2.column.name()
            return t1, c1, t2, c2
        
        fanout = {}
        
        for join in possible_joins:
            left_table, left_column, right_table, right_column = table_column_tuple_from_join(join)

            value_counts = table_df_dict[right_table][right_column].value_counts()
            value_counts = pd.DataFrame(value_counts)
            table_merge = pd.merge(left=table_df_dict[left_table][left_column], right=value_counts, left_on=left_column, right_on=right_column, how='left')
            bin_edges = histogram[left_table][left_column]['bin_edges']
            
            left_fanout = []
            assert len(bin_edges) == bin_size + 1
            for i in range(len(bin_edges) - 1):
                if i != bin_size - 1:
                    tmp = table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] < bin_edges[i + 1]), 'count']
                    tmp = tmp.sum()
                    if tmp == 0:
                        pass
                    else:
                        tmp = tmp / len(table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] < bin_edges[i + 1])])
                        assert len(table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] < bin_edges[i + 1])]) != 0
                else:
                    tmp = table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] <= bin_edges[i + 1]), 'count']
                    tmp = tmp.sum()
                    if tmp == 0:
                        pass
                    else:
                        tmp = tmp / len(table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] <= bin_edges[i + 1])])
                        assert len(table_merge.loc[(table_merge[left_column] >= bin_edges[i]) & (table_merge[left_column] <= bin_edges[i + 1])]) != 0
                left_fanout.append(tmp)
            
            right_fanout = []
            value_counts = table_df_dict[left_table][left_column].value_counts()
            value_counts = pd.DataFrame(value_counts)
            table_merge = pd.merge(left=table_df_dict[right_table][right_column], right=value_counts, left_on=right_column, right_on=left_column, how='left')
            bin_edges = histogram[right_table][right_column]['bin_edges']
            assert len(bin_edges) == bin_size + 1
            for i in range(len(bin_edges) - 1):
                if i != bin_size - 1:
                    tmp = table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] < bin_edges[i + 1]), 'count']
                    tmp = tmp.sum()
                    if tmp == 0:
                        pass
                    else:
                        tmp = tmp / len(table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] < bin_edges[i + 1])])
                        assert len(table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] < bin_edges[i + 1])]) != 0
                else:
                    tmp = table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] <= bin_edges[i + 1]), 'count']
                    tmp = tmp.sum()
                    if tmp == 0:
                        pass
                    else:
                        tmp = tmp / len(table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] <= bin_edges[i + 1])])
                        assert len(table_merge.loc[(table_merge[right_column] >= bin_edges[i]) & (table_merge[right_column] <= bin_edges[i + 1])]) != 0
                right_fanout.append(tmp)
            
            join_cond = (f"{left_table}.{left_column}", f"{right_table}.{right_column}")
            fanout[join_cond] = [left_fanout, right_fanout]
            join_inv = (join_cond[1], join_cond[0])
            fanout[join_inv] = [right_fanout, left_fanout]

        return fanout
    
    
    @staticmethod
    def gen_size(table_df_dict: Dict):
        results = {}
        for table in table_df_dict:
            results[table] = {}
            results[table]['size'] = table_df_dict[table].shape[0]
            results[table]['num_cols'] = table_df_dict[table].shape[1]
            results[table]['num_rows'] = table_df_dict[table].shape[0]
        return results

    
    @staticmethod
    def construct_statistics(schema: SQLSchema, csv_data_folder: str, bin_size: int) -> "PRICEStatistics":
        abbrev_coltype = PRICEStatistics.gen_abbrev_col_type(schema)

        table_df_dict = {}
        
        for table_name, abbrev in abbrev_coltype['abbrev'].items():

            df = pd.read_csv(os.path.join(csv_data_folder, f"{table_name}.csv"), sep='|', quotechar='"', escapechar='\\', na_values=['NULL'])
            table_obj = schema.table(table_name)

            for column in table_obj.columns():
                if column.column_type() == ColumnType.STRING_WITH_CMP:
                    df[column.name()] = df[column.name()].map(column._value_2_id).astype(int)
            
            table_df_dict[abbrev] = df
            
        histogram = PRICEStatistics.gen_histogram(table_df_dict, abbrev_coltype['col_type'], bin_size)
        
        summary = PRICEStatistics.gen_summary(table_df_dict, abbrev_coltype['col_type'])
        
        fanout = PRICEStatistics.gen_fanout(schema, table_df_dict, abbrev_coltype['abbrev'], histogram, bin_size)
        
        size = PRICEStatistics.gen_size(table_df_dict)
        
        return PRICEStatistics(abbrev_coltype, histogram, summary, fanout, size)

    def save(self, folder: str):
        
        os.makedirs(folder, exist_ok=True)
        
        with open(os.path.join(folder, 'abbrev_col_type.pkl'), 'wb') as f:
            pickle.dump(self.abbrev_coltype, f)
        with open(os.path.join(folder, 'histogram40.pkl'), 'wb') as f:
            pickle.dump(self.histogram, f)
        with open(os.path.join(folder, 'fanout40.pkl'), 'wb') as f:
            pickle.dump(self.fanout, f)
        with open(os.path.join(folder, 'size.pkl'), 'wb') as f:
            pickle.dump(self.size, f)
        with open(os.path.join(folder, 'summary40.pkl'), 'wb') as f:
            pickle.dump(self.summary, f)
    
    
    @staticmethod
    def load(folder: str):
        with open(os.path.join(folder, 'abbrev_col_type.pkl'), 'rb') as f:
            abbrev_coltype = pickle.load(f)
        with open(os.path.join(folder, 'histogram40.pkl'), 'rb') as f:
            histogram = pickle.load(f)
        with open(os.path.join(folder, 'fanout40.pkl'), 'rb') as f:
            fanout = pickle.load(f)
        with open(os.path.join(folder, 'size.pkl'), 'rb') as f:
            size = pickle.load(f)
        with open(os.path.join(folder, 'summary40.pkl'), 'rb') as f:
            summary = pickle.load(f)
        
        return PRICEStatistics(abbrev_coltype, histogram, summary, fanout, size)
    