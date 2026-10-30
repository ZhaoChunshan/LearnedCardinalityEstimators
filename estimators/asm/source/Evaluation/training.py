import pickle
import pandas as pd
import time
import numpy as np
import os
from Join_scheme.join_graph import dfs, get_PK
from Join_scheme.data_prepare import get_imdb_schema
from Join_scheme.bound import Bound_ensemble
import psycopg2

use_given_order_PKs = True
compute_tuples_np = True

def get_table_bitmap_by_timestamp(data_path, dataset, schema, perc,
                                  asm_data_path=None):
    import polars as pl
    import operator
    from functools import reduce
    
    assert dataset in ['imdb-light', 'stats']
    perc_int = int(perc * 100)
    if asm_data_path is None:
        asm_data_path = os.path.join(os.path.dirname(data_path), 'asm_data')
    bitmap_dir = os.path.join(asm_data_path, f'bitmap_{perc_int}')
    os.makedirs(bitmap_dir, exist_ok=True)
    
    if dataset == 'imdb-light':
        percentage_map_df = pl.read_csv("../../../datasets/imdb/timestamps_split/years_by_percentage.csv")
        timestamp_row = percentage_map_df.filter(pl.col('percentage') == perc_int)
        t = timestamp_row.item(0, 'year')
        
        title_path = data_path.format('title')
        title_df = (
            pl.scan_csv(title_path, separator='|', null_values=['NULL', ''])
            .collect()
        )
        filtered_title_df = title_df.filter(pl.col('production_year') <= t)
        valid_title_ids_series = filtered_title_df.get_column('id')
        is_valid_id_series = title_df['id'].is_in(valid_title_ids_series)
        is_valid_id_series = is_valid_id_series.fill_null(False)
        np_array = is_valid_id_series.to_numpy().astype(np.bool_)
        np.save(os.path.join(bitmap_dir, "title.npy"), np_array)
        
        for table_obj in schema.tables:
            name = table_obj.table_name
            table_path = table_obj.csv_file_location
            if name == 'title':
                continue
            
            fk_filter = (
                (pl.col('movie_id').is_null()) | 
                (pl.col('movie_id').is_in(valid_title_ids_series))
            )
            
            lf = pl.scan_csv(table_path, separator='|', null_values=['NULL', ''])
            is_valid_bitmap_series = lf.select(fk_filter).collect().to_series()
            is_valid_bitmap_series = is_valid_bitmap_series.fill_null(False)
            np_array = is_valid_bitmap_series.to_numpy().astype(np.bool_)
            np.save(os.path.join(bitmap_dir, f"{name}.npy"), np_array)
        
    else:
        percentage_map_df = pl.read_csv("../../../datasets/stats/timestamps_split/timestamps_by_percentage.csv")
        timestamp_row = percentage_map_df.filter(pl.col('percentage') == perc_int)
        t = timestamp_row.item(0, 'timestamp')
        
        TIME_COLS = {
            'badges': 'date',
            'comments': 'creationdate',
            'posthistory': 'creationdate',
            'postlinks': 'creationdate',
            'posts': 'creationdate',
            'users': 'creationdate',
            'votes': 'creationdate',
        }

        users_path = data_path.format('users')
        users_df = (
            pl.scan_csv(users_path, separator='|', null_values=['NULL', ''])
            .collect()
        )
        filtered_users_df = users_df.filter(pl.col('creationdate') <= t)
        valid_user_ids_series = filtered_users_df.get_column('id')
        is_valid_id_series = users_df['id'].is_in(valid_user_ids_series)
        is_valid_id_series = is_valid_id_series.fill_null(False)
        np_array = is_valid_id_series.to_numpy().astype(np.bool_)
        np.save(os.path.join(bitmap_dir, "users.npy"), np_array)
        
        posts_path = data_path.format('posts')
        posts_df = (
            pl.scan_csv(posts_path, separator='|', null_values=['NULL', ''])
            .collect()
        )
        filtered_posts_df = posts_df.filter(pl.col('creationdate') <= t) \
            .filter(pl.col('owneruserid').is_null() | pl.col('owneruserid').is_in(valid_user_ids_series)) \
            .filter(pl.col('lasteditoruserid').is_null() | pl.col('lasteditoruserid').is_in(valid_user_ids_series))
        valid_post_ids_series = filtered_posts_df.get_column('id')
        is_valid_id_series = posts_df['id'].is_in(valid_post_ids_series)
        is_valid_id_series = is_valid_id_series.fill_null(False)
        np_array = is_valid_id_series.to_numpy().astype(np.bool_)
        np.save(os.path.join(bitmap_dir, "posts.npy"), np_array)
        
        for table_obj in schema.tables:
            name = table_obj.table_name
            if name in ('users', 'posts'):
                continue
            table_path = table_obj.csv_file_location
            lf = pl.scan_csv(table_path, separator='|', null_values=['NULL', ''])
            columns = lf.columns
            
            filters = []
            if name in TIME_COLS:
                filters.append(pl.col(TIME_COLS[name]) <= t)
            
            for col_name in columns:
                col_lower = col_name.lower()

                if 'postid' in col_lower:
                    filters.append(
                        (pl.col(col_name).is_null()) |
                        (pl.col(col_name).is_in(valid_post_ids_series))
                    )

                elif 'userid' in col_lower:
                    filters.append(
                        (pl.col(col_name).is_null()) |
                        (pl.col(col_name).is_in(valid_user_ids_series))
                    )
            
            if filters:
                combined_filter = reduce(operator.and_, filters)
            else:
                combined_filter = pl.lit(True)
            
            is_valid_bitmap_series = lf.select(combined_filter).collect().to_series()
            is_valid_bitmap_series = is_valid_bitmap_series.fill_null(False)
            np_array = is_valid_bitmap_series.to_numpy().astype(np.bool_)
            np.save(os.path.join(bitmap_dir, f"{name}.npy"), np_array)

def train(data_path, model_folder, dataset, init_data_perc=1.0,
          asm_data_path=None):

    if True:
        schema, all_keys, equivalent_keys = get_imdb_schema(data_path, dataset)

        all_df_rows = dict()

        # make PK_tuples_np files
        all_tuples_np = dict()
        for table_obj in schema.tables:
            table = table_obj.table_name
            all_tuples_np[table] = dict()

        table_size = dict()

        if "stack" in dataset:
            # reading from csv is faulty
            conn = psycopg2.connect(database=dataset, user="postgres", password="postgres", host="localhost", port=5432,)
            conn.set_client_encoding('UTF8')

        P_tuples_np_time = 0

        for PK in equivalent_keys:
            [P_table, P_col] = PK.split('.')
            table_obj = schema.table_dictionary[P_table]

            if "stack" in dataset:
                query = "select * from " + table_obj.table_name + " ;"
                df_rows = pd.read_sql(query, conn)
            else:
                if os.path.exists(table_obj.csv_file_location):
                    print("reading", table_obj.csv_file_location)
                    if dataset == 'imdb' or dataset == 'imdb-light' or dataset == 'imdb_m' or dataset == 'imdb_large':
                        df_rows = pd.read_csv(table_obj.csv_file_location,
                                            low_memory=False,
                                            keep_default_na=False,
                                            na_values=['', 'NULL'],
                                            escapechar = '\\',
                                            encoding = 'utf-8',
                                            sep = '|'
                                            )
                    elif dataset.startswith('stats') or dataset == 'ergastf1':
                        df_rows = pd.read_csv(table_obj.csv_file_location,
                                            low_memory=False,
                                            keep_default_na=False,
                                            na_values=['', 'NULL'],
                                            escapechar = '\\',
                                            encoding = 'utf-8',
                                            sep = '|'
                                      )
                else:
                    assert False, f"Table file {table_obj.csv_file_location} does not exist."

            df_rows.columns = table_obj.attributes

            table_size[PK] = len(df_rows)
            all_df_rows[P_table] = df_rows
            if compute_tuples_np:
                P_tuples_np_time_start = time.time()
                P_tuples_np = np.unique(pd.concat([pd.Series([np.nan, 0]), df_rows[P_col]]).values)
                P_tuples_np_time += time.time() - P_tuples_np_time_start
                assert np.any(pd.isnull(P_tuples_np))

                for FK in equivalent_keys[PK]:
                    [table, col] = FK.split('.')

                    all_tuples_np[table][col] = P_tuples_np
                    print(f"FK {FK} tuples_np of PK {PK} = {P_tuples_np}")

        dfs_PKs_time_start = time.time()
        # set DFS order of PKs
        if use_given_order_PKs:
            if 'stack' in dataset:
                dfs_PKs = ['site.site_id', 'question.id', 'tag.id', 'so_user.id', 'account.id']
            elif dataset == 'imdb-light':
                dfs_PKs = ['title.id']
            elif dataset == 'imdb_m':
                dfs_PKs = ['title.id', 'kind_type.id', 'comp_cast_type.id', 'info_type.id', 'link_type.id', 'keyword.id', 'company_name.id', 'company_type.id']
            elif dataset == 'imdb_large':
                dfs_PKs = ['title.id', 'comp_cast_type.id', 'info_type.id', 'name.id', 'char_name.id', 'role_type.id', 'company_name.id', 'company_type.id', 'kind_type.id', 'keyword.id', 'link_type.id']
            elif dataset == 'imdb':
                dfs_PKs = ['title.id', 'kind_type.id', 'comp_cast_type.id', 'info_type.id', 'name.id', 'char_name.id', 'role_type.id', 'link_type.id', 'keyword.id', 'company_name.id', 'company_type.id']
            elif dataset.startswith('stats'):
                dfs_PKs = ['users.id', 'posts.id']
            elif dataset == 'ergastf1':
                dfs_PKs = ['races.raceid', 'drivers.driverid', 'constructors.constructorid', 'status.statusid', 'seasons.year', 'circuits.circuitid']
            else:
                assert False
        else:
            # perform DFS
            # weird, but the result of DFS changes for each trial
            # for such a case, just use the above DFS order
            if 'stack' in dataset:
                start_PK = 'site.site_id'
            elif dataset == 'imdb' or dataset == 'imdb-light' or dataset == 'imdb_m' or dataset == 'imdb_large':
                start_PK = 'title.id'
            elif dataset.startswith('stats'):
                start_PK = 'users.id'
            elif dataset == 'ergastf1':
                start_PK = 'races.raceid'
            else:
                assert False
            dfs_PKs = dfs(start_PK, schema, equivalent_keys, None, [])

        dfs_PKs_time = time.time() - dfs_PKs_time_start
        print('dfs PKs or order of PKs:', dfs_PKs)

        # create meta model
        be = Bound_ensemble(schema)
        be.all_dfs_PKs = dfs_PKs

        from AR.experiments import EXPERIMENT_CONFIGS

        be.reordered_attributes = dict()

        reorder_time = 0

        for table_obj in schema.tables:
            table = table_obj.table_name

            loc = table_obj.csv_file_location
            attrs = table_obj.attributes
            irr_attrs = table_obj.irrelevant_attributes
            print('table', table, 'loc', loc, 'attrs', attrs, 'irr attrs', irr_attrs)

            reorder_time_start = time.time()

            order = []
            temp_orders = []
            j = 0
            for attr in attrs:
                if attr not in irr_attrs:
                    PK = get_PK(table + "." + attr, equivalent_keys)
                    if PK is not None:
                        order.append(-1)
                        temp_orders.append(dfs_PKs.index(PK))
                    else:
                        order.append(j)
                        j += 1

            global_orders = sorted(list(range(len(temp_orders))), key=lambda x: temp_orders[x])

            reverse_order = [-1] * len(global_orders)
            for i, k in enumerate(global_orders):
                reverse_order[k] = i

            # make the join order consistent with the dfs_PKs
            PK_i = 0
            for i, k in enumerate(order):
                if k == -1:
                    order[i] = j + reverse_order[PK_i]
                    print(f'order[{i}] = {j + reverse_order[PK_i]}')
                    #j += 1
                    PK_i += 1

            reverse_order = [-1] * len(order)
            for i, k in enumerate(order):
                reverse_order[k] = i

            reorder_time += time.time() - reorder_time_start

            if asm_data_path is None:
                table_dir = os.path.join(os.path.dirname(data_path.format(table)),
                                         "asm_data", f"{table}")
            else:
                table_dir = os.path.join(asm_data_path, table)
            table_path = os.path.join(table_dir, 'table0.csv')

            if True:
                os.makedirs(table_dir, exist_ok=True)

                if table in all_df_rows:
                    df_rows = all_df_rows[table]
                else:
                    if dataset == 'stack':
                        query = "select * from " + table_obj.table_name + " ;"
                        print(query)
                        df_rows = pd.read_sql(query, conn)
                    elif dataset == 'imdb' or dataset == 'imdb-light' or dataset == 'imdb_m' or dataset == 'imdb_large':
                        df_rows = pd.read_csv(table_obj.csv_file_location,
                                            low_memory=False,
                                            keep_default_na=False,
                                            na_values=['', 'NULL'],
                                            escapechar = '\\',
                                            encoding = 'utf-8',
                                            sep = '|'
                                            )
                    elif dataset.startswith('stats') or dataset == 'ergastf1':
                        df_rows = pd.read_csv(table_obj.csv_file_location,
                                            low_memory=False,
                                            keep_default_na=False,
                                            na_values=['', 'NULL'],
                                            escapechar = '\\',
                                            encoding = 'utf-8',
                                            sep = '|'
                                      )
                    
                    df_rows.columns = table_obj.attributes

                reorder_time_start = time.time()

                for attr in irr_attrs:
                    df_rows = df_rows.drop(attr, axis=1)
                df_rows = df_rows.iloc[:, reverse_order]

                be.reordered_attributes[table] = list(df_rows.columns)

                reorder_time += time.time() - reorder_time_start

            # if not os.path.exists(table_path):
            if True:    # Always update the csv
                if 'stack' in dataset:
                    df_rows[0:0].to_csv(table_path, header=True, index=False)
                else:
                    df_rows.to_csv(table_path, header=True, index=False)

            if compute_tuples_np:
                assert table in all_tuples_np, f'table {table}'
                
                tuples_np_path = os.path.join(table_dir, 'tuples_np.pkl')
                # if not os.path.exists(tuples_np_path):
                if True:    # Always update the tuples_np
                    pickle.dump(all_tuples_np[table], open(tuples_np_path, 'wb'))

        # save model
        model_path = os.path.join(model_folder, f"model_{dataset}.pkl")
        if 'stack' in dataset:
            conn.close()
        # if not os.path.exists(model_path):
        if True:    # Always update the model
            pickle.dump(be, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"models save at {model_path}")

        if init_data_perc != 1.0:
            print("Generate data bitmap for updating experiments")
            get_table_bitmap_by_timestamp(data_path, dataset, schema,
                                          init_data_perc, asm_data_path)
            print("All works done.")
        
        print('P_tuples_np_time:', P_tuples_np_time)
        print('dfs_PKs_time:', dfs_PKs_time)
        print('reorder_time:', reorder_time)
        print('total time:', P_tuples_np_time + dfs_PKs_time + reorder_time)
