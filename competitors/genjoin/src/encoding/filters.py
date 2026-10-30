# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: disable an unavailable converters import.

import copy
import random
from typing import Optional

# from src.utils.converters import dt_to_int, int_to_dt


class FiltersProcessor:
    """
    Holding data and performing manipulations with histogram on particular table
    """
    def __init__(self, table, alias, target_columns, executor, string_columns=None, numeric_columns=None):
        # TODO: add check that columns are in columns of the table
        self.table = table
        self.alias = alias
        self.numeric_columns = numeric_columns if numeric_columns else []
        self.string_columns = string_columns if string_columns else []
        self.target_columns = target_columns
        self._get_sampling_values()
        
    def _get_sampling_values(self):
        """
        Extracting distinct values, along with Most Common Values (MCVs) to sample from
        """
        self.distincts = dict()
        self.mcvs = dict()

        # Distinct values
        for filter_column in self.numeric_columns:
            distinct_query = f"""
            SELECT DISTINCT({filter_column})
            FROM {self.table}
            WHERE {filter_column} IS NOT NULL
            """

            df = executor.read_to_pandas(distinct_query)
            self.distincts[filter_column] = sorted(list(df[filter_column]))
        
        # Most common values (MCVs) - currently extracted only for string columns
        mcvs_query = f"""
        SELECT attname, array_to_string(most_common_vals, E'\n') as mcvs
        FROM pg_stats
        WHERE tablename = '{self.table}';
        """
        df = executor.read_to_pandas(mcvs_query)
        # TODO: now it's a quickfix to avoid Nones - however e.g. `name.name` would not be present
        # here due to big number of unique values. Need then to fix it properly
        
        if self.string_columns:
            # For very small tables like `role_type.role` MCVs are not created
            if len(df):
                not_none_strings = df.dropna().attname.values.tolist()
                self.string_columns = set(not_none_strings).intersection(set(self.string_columns))
                for filter_column in self.string_columns:
                    mcvs_values = df[df['attname'] == filter_column].mcvs.values[0].split('\n')
                    self.mcvs[filter_column] = mcvs_values
                self.string_columns = list(self.string_columns)
            else:
                filter_column = self.string_columns[0]
                full_query = f"""
                SELECT {filter_column}
                FROM {self.table}
                """
                df = executor.read_to_pandas(full_query)
                self.mcvs[filter_column] = df[filter_column].values.tolist()

    def get_filter_values(self, filter_column: str, lower_bound: Optional[int] = 1,
                                                    nunique_threshold: Optional[int] = 30):
        """
        Extracting sample values for particular column
        
        One can tweak the `lower_bound` of proportion of filter column value to be filtered on: 
        higher this value, higher the probability of intersection of filtered target columns 
        to be non-empty.
        """
        if filter_column in self.numeric_columns:
            lower_bound /= 100  # because for numericals it should be in fraction
            values_to_sample = copy.copy(self.distincts[filter_column])
            if 'date' in filter_column:
                values_to_sample = list(map(dt_to_int, values_to_sample))
            # to avoid rounding problems
            n_elements = int(random.uniform(lower_bound, 1) * len(values_to_sample)) + 1
            if len(values_to_sample) < nunique_threshold:
                filter_values = random.sample(values_to_sample, k=n_elements)
                filter_type = 'IN'
            else:
                filter_values = values_to_sample[: n_elements]
                filter_type = 'BETWEEN'
        elif filter_column in self.string_columns:
            values_to_sample = copy.copy(self.mcvs[filter_column])
            # the meximum number of MCVs stored in `pg_stats` is 100 by default
            n_elements = random.randint(min(lower_bound, len(values_to_sample)), len(values_to_sample))
            filter_values = random.sample(values_to_sample, k=n_elements)
            filter_type = 'NOT IN'  # using `NOT` in order not to make outputs too selective
        filter_values = list(map(str, filter_values))
        if 'date' in filter_column:
            filter_values = list(map(lambda x:  f"\'{int_to_dt(x)}\'", filter_values))
        if filter_column in self.string_columns:
            filter_values = list(map(lambda x:  f"\'{x}\'", filter_values))
        
        return filter_values, filter_type
    
    def get_where_cond(self, filter_column, filter_values, filter_type):
        """
        Forming WHERE condition for the query
        """
        if filter_type == 'IN':
            local_where_cond = f"{self.alias}.{filter_column} IN ({', '.join(filter_values)})"
        elif filter_type == 'BETWEEN':
            local_where_cond = f"{self.alias}.{filter_column} BETWEEN {filter_values[0]} AND {filter_values[-1]}"
        
        return len(filter_values), local_where_cond
    
    def apply_filters(self, lower_bound: Optional[int] = 1, exec_filters: Optional[bool] = False,
                                                            exec_targets: Optional[bool] = False):
        """
        Applies randomly generated filters to target column in a table
        
        As of now processing all the filters that were extracted from queries workload
        
        NB! It could be that `target_column` is empty after combination of filters applied.
        """
        filtered_values = dict()
        target_values = dict()
        
        target_where_cond = ['1=1']
        target_select_stmt = f"{', '.join(self.target_columns)}"
        for filter_column in (*self.numeric_columns, *self.string_columns):
            filter_values, filter_type = self.get_filter_values(filter_column, lower_bound=lower_bound)
            nvalues, local_where_cond = self.get_where_cond(filter_column, filter_values, filter_type)
            if nvalues:
                target_where_cond.append(local_where_cond)
                # Applying filter only if extraction is in general required (usually only for SP-phase)
                if exec_filters:
                    local_select_stmt = f'{self.alias}.{filter_column}'
                    local_query = f"""
                    SELECT {local_select_stmt}
                    FROM {self.table} AS {self.alias}
                    WHERE {local_where_cond}
                    """
                    df = executor.read_to_pandas(local_query)

                    if 'date' in filter_column:
                        df[filter_column] = list(map(dt_to_int, df[filter_column]))

                    filtered_values[filter_column] = df[filter_column].dropna().values
                    
        target_where_cond = ' AND '.join(target_where_cond)
        target_query = f"""
        SELECT {target_select_stmt}
        FROM {self.table} AS {self.alias}
        WHERE {target_where_cond}
        """
        if exec_targets:
            df = executor.read_to_pandas(target_query)
            for target_column in self.target_columns:
                target_values[target_column] = df[target_column].dropna().values

        return target_query, target_where_cond, filtered_values, target_values
