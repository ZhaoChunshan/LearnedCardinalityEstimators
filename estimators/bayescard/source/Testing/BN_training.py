import pandas as pd
import pickle
import time
import os
from DataPrepare.join_data_preparation import JoinDataPreparator
from Models.Bayescard_BN import Bayescard_BN, build_meta_info

def train_DMV(csv_path, model_path, algorithm, max_parents, sample_size):
    data = pd.read_csv(csv_path)
    new_cols = []
    #removing unuseful columns
    for col in data.columns:
        if col in ['VIN', 'Zip', 'City', 'Make', 'Unladen Weight', 'Maximum Gross Weight', 'Passengers',
                   'Reg Valid Date', 'Reg Expiration Date', 'Color']:
            data = data.drop(col, axis=1)
        else:
            new_cols.append(col.replace(" ", "_"))
    data.columns = new_cols
    BN = Bayescard_BN('dmv')
    BN.build_from_data(data, algorithm=algorithm, max_parents=max_parents, ignore_cols=['id'], sample_size=sample_size)
    model_path += f"/{algorithm}_{max_parents}.pkl"
    pickle.dump(BN, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"model saved at {model_path}")
    return None

def train_Census(csv_path, model_path, algorithm, max_parents, sample_size):
    df = pd.read_csv(csv_path, header=0, sep=",")
    df = df.drop("caseid", axis=1)
    df = df.dropna(axis=0)
    BN = Bayescard_BN('Census')
    BN.build_from_data(df, algorithm=algorithm, max_parents=max_parents, ignore_cols=['id'], sample_size=sample_size)
    model_path += f"/{algorithm}_{max_parents}.pkl"
    pickle.dump(BN, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"model saved at {model_path}")
    return None

def train_imdb(schema, hdf_path, model_folder, algorithm, max_parents, sample_size):
    meta_data_path = hdf_path + '/meta_data.pkl'
    prep = JoinDataPreparator(meta_data_path, schema, max_table_data=20000000)
    print(f"BN will be trained on the full outer join of following relations")
    for relationship_obj in schema.relationships:
        print(relationship_obj.identifier)

    for i, relationship_obj in enumerate(schema.relationships):
        print(f"training on {relationship_obj.identifier}")
        df_sample_size = 10000000
        relation = [relationship_obj.identifier]
        df, meta_types, null_values, full_join_est = prep.generate_n_samples(
            df_sample_size, relationship_list=relation, post_sampling_factor=10)
        columns = list(df.columns)
        assert len(columns) == len(meta_types) == len(null_values)
        meta_info = build_meta_info(df.columns, null_values)
        bn = Bayescard_BN(schema, relation, column_names=columns, full_join_size=full_join_est,
                      table_meta_data=prep.table_meta_data, meta_types=meta_types, null_values=null_values,
                      meta_info=meta_info)
        model_path = model_folder + f"/{i}_{algorithm}_{max_parents}.pkl"
        if schema.dataset != 'imdb_large':
            bn.build_from_data(df, algorithm=algorithm, max_parents=max_parents, ignore_cols=[],
                           sample_size=sample_size)
        else:
            bn.build_from_data(df, algorithm=algorithm, max_parents=max_parents, ignore_cols=['id'],
                           sample_size=sample_size)
        pickle.dump(bn, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"model saved at {model_path}")
    return None

def train_and_update(dataset, schema, hdf_path, model_folder, algorithm, max_parents, sample_size, 
                     init_hdf_path, perc, update_strategy):
    stale_dir = os.path.join(model_folder, 'stale')
    updated_dir = os.path.join(model_folder, 'updated')
    os.makedirs(stale_dir, exist_ok=True)
    os.makedirs(updated_dir, exist_ok=True)
    
    prep_init = None if init_hdf_path is None else \
        JoinDataPreparator(str(os.path.join(init_hdf_path, 'meta_data.pkl')),
                               schema, max_table_data=40000000)
    prep = JoinDataPreparator(str(os.path.join(hdf_path, 'meta_data.pkl')),
                              schema, max_table_data=40000000)
    df_sample_size = 100000000
    print(f"BN will be trained on the full outer join of following relations")
    for relationship_obj in schema.relationships:
        print(relationship_obj.identifier)
        
    def get_init_df_indices_by_perc(df, perc, dataset):
        if dataset == "imdb":
            data_split_csv = 'datasets/imdb/timestamps_split/years_by_percentage.csv'
        else:
            data_split_csv = "datasets/stats/timestamps_split/timestamps_by_percentage.csv"
        timestamp_df = pd.read_csv(data_split_csv)
        timestamp = 0
        if dataset == 'imdb':
            timestamp = timestamp_df.loc[timestamp_df['percentage'] == perc, 'year'].item()
            return df['title.production_year'] <= timestamp
        else:
            timestamp = timestamp_df.loc[timestamp_df['percentage'] == perc, 'timestamp'].item()
            TIME_COLS = ['badges.date', 'comments.creationdate', 'posthistory.creationdate', 
                         'postlinks.creationdate', 'posts.creationdate', 'users.creationdate',
                         'votes.creationdate',]
            intersection = pd.Series(True, index=df.index)
            for col in TIME_COLS:
                if col in df.columns:
                    intersection &= df[col] <= timestamp
            return intersection
        
    def save_model(bn, model_path):
        pickle.dump(bn, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"Model saved at {model_path}")
    
    model_training_time, training_data_time, model_updating_time, updating_data_time = 0.0, 0.0, 0.0, 0.0
    
    for i, relationship_obj in enumerate(schema.relationships):
        print(f"training on {relationship_obj.identifier}")
        relation = [relationship_obj.identifier]
        if update_strategy == 'structure':
            start_train_data_time = time.time()
            train_data, meta_types, null_values, full_join_est = prep_init.generate_n_samples(df_sample_size, 
                                                                                              relationship_list=relation, 
                                                                                              post_sampling_factor=10)
            training_data_time += time.time() - start_train_data_time
            
            meta_info = build_meta_info(train_data.columns, null_values)
            
            start_model_training_time = time.time()
            bn = Bayescard_BN(schema, relation, column_names=list(train_data.columns), full_join_size=full_join_est,
                          table_meta_data=prep_init.table_meta_data, meta_types=meta_types, null_values=null_values,
                          meta_info=meta_info)
            bn.build_from_data(train_data, algorithm=algorithm, max_parents=max_parents, sample_size=sample_size)
            model_training_time += time.time() - start_model_training_time
            
            save_model(bn, os.path.join(stale_dir, f"{i}_{algorithm}_{max_parents}.pkl"))
            
            start_update_data_time = time.time()
            update_data, meta_types, null_values, full_join_est = prep.generate_n_samples(df_sample_size, 
                                                                                         relationship_list=relation, 
                                                                                         post_sampling_factor=10)
            updating_data_time += time.time() - start_update_data_time
            
            start_updating = time.time()
            bn.keep_structure_fit_new_data(update_data, algorithm=algorithm)
            model_updating_time += time.time() - start_updating
            
            save_model(bn, os.path.join(updated_dir, f"{i}_{algorithm}_{max_parents}.pkl"))
             
        elif update_strategy == 'incremental':
            start_train_data_time = time.time()
            train_data, meta_types, null_values, full_join_est = prep_init.generate_n_samples(df_sample_size, 
                                                                                              relationship_list=relation, 
                                                                                              post_sampling_factor=10)
            training_data_time += time.time() - start_train_data_time
            
            meta_info = build_meta_info(train_data.columns, null_values)
            
            start_model_training_time = time.time()
            bn = Bayescard_BN(schema, relation, column_names=list(train_data.columns), full_join_size=full_join_est,
                          table_meta_data=prep_init.table_meta_data, meta_types=meta_types, null_values=null_values,
                          meta_info=meta_info)
            bn.build_from_data(train_data, algorithm=algorithm, max_parents=max_parents, sample_size=sample_size,
                               store_cpd_counts=True)
            model_training_time += time.time() - start_model_training_time
            
            save_model(bn, os.path.join(stale_dir, f"{i}_{algorithm}_{max_parents}.pkl"))
            
            start_update_data_time = time.time()
            update_data, meta_types, null_values, full_join_est = prep.generate_n_samples(df_sample_size, 
                                                                                         relationship_list=relation, 
                                                                                         post_sampling_factor=10)
            # diff_A_B = train_data.merge(update_data, how='outer', indicator=True)
            # dataset_del = diff_A_B[diff_A_B['_merge'] == 'left_only'].drop(columns=['_merge'])
            # dataset_ins = diff_A_B[diff_A_B['_merge'] == 'right_only'].drop(columns=['_merge'])
            
            train_keys = pd.util.hash_pandas_object(train_data, index=False)
            update_keys = pd.util.hash_pandas_object(update_data, index=False)

            train_key_set = set(train_keys)
            update_key_set = set(update_keys)

            dataset_del = train_data[~train_keys.isin(update_key_set)]
            dataset_ins = update_data[~update_keys.isin(train_key_set)]
            
            updating_data_time += time.time() - start_update_data_time
            
            start_updating = time.time()
            bn.update_from_insertion_and_deletion(dataset_ins, dataset_del)
            model_updating_time += time.time() - start_updating
            
            save_model(bn, os.path.join(updated_dir, f"{i}_{algorithm}_{max_parents}.pkl"))
            
        elif update_strategy == 'filter':
            start_train_data_time = time.time()
            df, meta_types, null_values, full_join_est = prep.generate_n_samples(df_sample_size, 
                                                                                 relationship_list=relation, 
                                                                                 post_sampling_factor=10)
            init_indices = get_init_df_indices_by_perc(df, perc, dataset)
            train_data = df[init_indices]
            update_data = df[~init_indices]
            training_data_time += time.time() - start_train_data_time
            
            train_data_perc = train_data.shape[0] / df.shape[0]
            
            meta_info = build_meta_info(df.columns, null_values)
            start_model_training_time = time.time()
            bn = Bayescard_BN(schema, relation, column_names=list(train_data.columns), full_join_size=full_join_est * train_data_perc,
                      table_meta_data=prep.table_meta_data, meta_types=meta_types, null_values=null_values,
                      meta_info=meta_info)
            bn.build_from_data(train_data, algorithm=algorithm, max_parents=max_parents, sample_size=sample_size,
                               store_cpd_counts=True)
            model_training_time += time.time() - start_model_training_time
            
            save_model(bn, os.path.join(stale_dir, f"{i}_{algorithm}_{max_parents}.pkl"))
            
            start_updating = time.time()
            bn.update_from_data(update_data)
            model_updating_time += time.time() - start_updating
            
            save_model(bn, os.path.join(updated_dir, f"{i}_{algorithm}_{max_parents}.pkl"))
        else:
            assert False
    
    with open(os.path.join(model_folder, 'training_time.txt'), 'w') as f:
        f.write(f"model_training_time: {model_training_time}\n")
        f.write(f"model_updating_time: {model_updating_time}\n")
        f.write(f"training_data_time: {training_data_time}\n")
        f.write(f"updating_data_time: {updating_data_time}\n")
    return
