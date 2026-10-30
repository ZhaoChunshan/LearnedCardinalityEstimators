import numpy as np
import pickle
import time
import os
from Schemas.stats.schema import gen_stats_light_schema
from Schemas.imdb.schema import gen_job_light_imdb_schema
from Evaluation.training import train_one_stats, test_trained_BN_on_stats
from Join_scheme.data_prepare import read_table_csv, update_stats_data
from BayesCard.Models.Bayescard_BN import Bayescard_BN


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))


def timestamp_transform_unix(time_string):
    if time_string.startswith("'") and time_string.endswith("'"):
        time_string = time_string[1:-1]
    return int(time.mktime(time.strptime(time_string, "%Y-%m-%d %H:%M:%S")))


def get_data_by_date_for_stats_light(data_path, time_date):
    # time_value = timestamp_transorform(time_date)
    time_value = timestamp_transform_unix(time_date)
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_stats_light_schema(data_path)
    
    before_data = dict()
    after_data = dict()
    
    import pandas as pd
    
    users_df = pd.read_csv(schema.table_dictionary['users'].csv_file_location, sep='|', na_values=['NULL',''])
    users_df.columns = ['users.' + attr for attr in schema.table_dictionary['users'].attributes]
    users_df = users_df.apply(pd.to_numeric, errors="ignore")
    
    idx = users_df['users.creationdate'] <= time_value
    
    for attribute in schema.table_dictionary['users'].irrelevant_attributes:
        users_df = users_df.drop('users.' + attribute, axis=1)
    
    before_data['users'] = users_df[idx]
    after_data['users'] = users_df[~idx]
    valid_user_ids = set(before_data['users']['users.id'])
    
    posts_df = pd.read_csv(schema.table_dictionary['posts'].csv_file_location, sep='|', na_values=['NULL',''])
    posts_df.columns = ['posts.' + attr for attr in schema.table_dictionary['posts'].attributes]
    posts_df = posts_df.apply(pd.to_numeric, errors="ignore")
        
    condition = (
        (posts_df['posts.creationdate'] <= time_value) &
        (posts_df['posts.owneruserid'].isna() | posts_df['posts.owneruserid'].isin(valid_user_ids)) &
        (posts_df['posts.lasteditoruserid'].isna() | posts_df['posts.lasteditoruserid'].isin(valid_user_ids))
    )
    
    for attribute in schema.table_dictionary['posts'].irrelevant_attributes:
        posts_df = posts_df.drop('posts.' + attribute, axis=1)
    
    before_data['posts'] = posts_df[condition]
    after_data['posts'] = posts_df[~condition]
    valid_post_ids = set(before_data['posts']['posts.id'])
    
    for table_obj in schema.tables:
        if table_obj.table_name in ('posts', 'users'):
            continue
        
        table_name = table_obj.table_name
        df = pd.read_csv(table_obj.csv_file_location, sep='|', na_values=['NULL',''])
        df.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]
        df = df.apply(pd.to_numeric, errors="ignore")
        
        mask = []        
        if table_obj.table_name != 'tags':
            if table_obj.table_name == 'badges':
                mask.append(df['badges.date'] <= time_value)
            else:
                mask.append(df[f'{table_name}.creationdate'] <= time_value)
        
        for col_name in df.columns:
            if 'postid' in col_name.lower():
                mask.append(df[col_name].isna() | df[col_name].isin(valid_post_ids))
            if 'userid' in col_name.lower():
                mask.append(df[col_name].isna() | df[col_name].isin(valid_user_ids))
        
        for attribute in table_obj.irrelevant_attributes:
            df = df.drop(table_obj.table_name + '.' + attribute, axis=1)
        
        if mask:
            combined_mask = mask[0]
            for m in mask[1:]:
                combined_mask &= m
            before_data[table_name] = df[combined_mask]
            after_data[table_name] = df[~combined_mask]
        else:
            before_data[table_name] = df
            after_data[table_name] = df.iloc[0:0]
        
        print(f"table name is {table_name}, before data length {before_data[table_name].shape[0]}, after data length {after_data[table_name].shape[0]}, initial data ratio is {len(before_data[table_name])/len(df)}")
    return before_data, after_data


def get_data_by_date_for_imdb_light(data_path, time_date):
    from datetime import datetime
    year = int(datetime.strptime(time_date, "%Y-%m-%d %H:%M:%S").year)
    if not data_path.endswith(".csv"):
        data_path += "/{}.csv"
    schema = gen_job_light_imdb_schema(data_path)
    
    before_data = dict()
    after_data = dict()
    
    title_obj = schema.table_dictionary['title']
    title_df = read_table_csv(title_obj, db_name="imdb-light")
    idx = title_df['title.production_year'] <= year
    before_data['title'] = title_df[idx]
    after_data['title'] = title_df[~idx]
    valid_movie_ids = set(before_data['title']['title.id'].values)
    
    print(f"table name is title, before data length {before_data['title'].shape[0]}, after data length {after_data['title'].shape[0]}, initial data ratio is {idx.sum()/len(idx)}")

    for table_obj in schema.tables:
        if table_obj.table_name in ('title'):
            continue
        
        table_name = table_obj.table_name
        df_rows = read_table_csv(table_obj, db_name="imdb-light")
        
        condition = df_rows[f'{table_name}.movie_id'].isnull() | df_rows[f'{table_name}.movie_id'].isin(valid_movie_ids)
        
        before_data[table_name] = df_rows[condition]
        after_data[table_name] = df_rows[~condition]
        
        print(f"table name is {table_name}, before data length {before_data[table_name].shape[0]}, after data length {after_data[table_name].shape[0]}, initial data ratio is {len(before_data[table_name])/len(df_rows)}")
    
    return before_data, after_data


def update_one_stats(FJmodel, buckets, table_buckets, data_path, save_model_folder, save_bucket_bins=False,
                     update_BN=True, retrain_BN=False, old_data=None, validate=False):
    """
    Incrementally update the FactorJoin model
    """
    data, table_buckets, null_values = update_stats_data(data_path, save_model_folder, buckets, table_buckets,
                                                         save_bucket_bins)
    FJmodel.table_buckets = table_buckets
    if update_BN:
        # updating the single table estimator
        if retrain_BN:
            # retrain the BN based on the new and old data
            for table in FJmodel.schema.tables:
                t_name = table.table_name
                if t_name in data and data[t_name] is not None:
                    bn = Bayescard_BN(t_name, table_buckets[t_name].id_attributes, table_buckets[t_name].bin_sizes,
                                      null_values=null_values[t_name])
                    new_data = old_data[t_name].append(data[t_name], ignore_index=True)
                    bn.build_from_data(new_data)
                    if validate:
                        test_trained_BN_on_stats(bn, t_name)
                    FJmodel.bns[t_name] = bn
        else:
            # incrementally update BN
            for table in FJmodel.schema.tables:
                t_name = table.table_name
                if t_name in data and data[t_name] is not None:
                    bn = FJmodel.bns[t_name]
                    bn.null_values = null_values[t_name]
                    bn.update_from_data(data)

    model_path = os.path.join(save_model_folder, f"update_model.pkl")
    pickle.dump(FJmodel, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"models save at {model_path}")


def update_one_imdb():
    "TODO: implement the update on IMDB, should be straight-forward as it uses the sampling for base-table"
    return

def eval_update(dataset, data_folder, model_path, n_dim_dist, bin_size, bucket_method, split_date="2014-01-01 00:00:00", seed=0):
    assert dataset in ['stats', 'imdb-light']
    
    np.random.seed(seed)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    if dataset == 'stats':
        before_data, after_data = get_data_by_date_for_stats_light(data_folder, split_date)
    else:
        before_data, after_data = get_data_by_date_for_imdb_light(data_folder, split_date)
    
    print("************************************************************")
    print(f"Training the model with data before {split_date}")
    start_time = time.perf_counter()
    train_one_stats(dataset, data_folder, model_path, n_dim_dist, bin_size, bucket_method, True, actual_data=before_data)
    print(f"training completed, took {time.perf_counter() - start_time} sec")

    # loading the trained model and buckets
    with open(os.path.join(model_path, f"{dataset}_buckets.pkl"), "rb") as f:
        buckets = pickle.load(f)
    with open(os.path.join(model_path, f"model_{dataset}_{bucket_method}_{bin_size}.pkl"), "rb") as f:
        FJmodel = pickle.load(f)
    print("************************************************************")
    print(f"Updating the model with data after {split_date}")
    start_time = time.perf_counter()
    table_buckets = FJmodel.table_buckets
    null_values = FJmodel.null_value
    data, table_buckets, null_values = update_stats_data(dataset, data_folder, model_path, buckets, table_buckets,
                                                         null_values, False, after_data)
    for table in FJmodel.schema.tables:
        t_name = table.table_name
        if t_name in data and data[t_name] is not None:
            bn = FJmodel.bns[t_name]
            bn.null_values = null_values[t_name]
            bn.update_from_data(data[t_name])
    print(f"updating completed, took {time.perf_counter() - start_time} sec")
    model_path = os.path.join(model_path, f"updated_model_{dataset}_{bucket_method}_{bin_size}.pkl")
    pickle.dump(FJmodel, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"updated models save at {model_path}")

