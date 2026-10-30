import copy
import os
import time
import logging

import pandas as pd

from rspn.rspn import build_ds_context
from aqp_spn.aqp_spn import AQPSPN
from data_preparation.join_data_preparation import JoinDataPreparator
from ensemble_compilation.spn_ensemble import SPNEnsemble

logger = logging.getLogger(__name__)

RATIO_MIN_INSTANCE_SLICE = 1 / 100


def create_naive_all_split_ensemble(schema, hdf_path, sample_size, ensemble_path, dataset, bloom_filters,
                                    rdc_threshold, max_table_data, post_sampling_factor, incremental_learning_rate):
    meta_data_path = hdf_path + '/meta_data.pkl'
    prep = JoinDataPreparator(meta_data_path, schema, max_table_data=max_table_data)
    spn_ensemble = SPNEnsemble(schema)

    logger.info(f"Creating naive ensemble.")

    for table_obj in schema.tables:
        logger.info(f"Learning SPN for {table_obj.table_name}.")
        if incremental_learning_rate > 0:
            df_samples, df_inc_samples, meta_types, null_values, full_join_est = prep.generate_n_samples_with_incremental_part(
                sample_size,
                single_table=table_obj.table_name,
                post_sampling_factor=post_sampling_factor,
                incremental_learning_rate=incremental_learning_rate)
            logger.debug(f"Requested {sample_size} samples and got {len(df_samples)} + {len(df_inc_samples)} "
                         f"(for incremental learning)")
        else:
            df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(sample_size,
                                                                                         single_table=table_obj.table_name,
                                                                                         post_sampling_factor=post_sampling_factor)

        # learn spn
        aqp_spn = AQPSPN(meta_types, null_values, full_join_est, schema, None, full_sample_size=len(df_samples),
                         table_set={table_obj.table_name}, column_names=list(df_samples.columns),
                         table_meta_data=prep.table_meta_data)
        min_instance_slice = RATIO_MIN_INSTANCE_SLICE * min(sample_size, len(df_samples))
        logger.debug(f"Using min_instance_slice parameter {min_instance_slice}.")
        logger.info(f"SPN training phase with {len(df_samples)} samples")
        aqp_spn.learn(df_samples.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                      rdc_threshold=rdc_threshold)
        if incremental_learning_rate > 0:
            logger.info(f"additional incremental SPN training phase with {len(df_inc_samples)} samples "
                        f"({incremental_learning_rate}%)")
            aqp_spn.learn_incremental(df_inc_samples.values)
        spn_ensemble.add_spn(aqp_spn)

    ensemble_path += '/ensemble_single_' + dataset + '_' + str(sample_size) + '.pkl'
    logger.info(f"Saving ensemble to {ensemble_path}")
    spn_ensemble.save(ensemble_path)


def naive_every_relationship_ensemble(schema, hdf_path, sample_size, ensemble_path, dataset, bloom_filters,
                                      rdc_threshold, max_table_data, post_sampling_factor,
                                      incremental_learning_rate=0):
    meta_data_path = hdf_path + '/meta_data.pkl'
    prep = JoinDataPreparator(meta_data_path, schema, max_table_data=max_table_data)
    spn_ensemble = SPNEnsemble(schema)

    # For ErgastF1, filters may be applied to pk/fk columns
    drop_redundant_columns = (dataset != "ergastf1") and (dataset != "imdb_large")
    logger.info(f"Creating naive ensemble for every relationship.")
    for relationship_obj in schema.relationships:
        logger.info(f"Learning SPN for {relationship_obj.identifier}.")

        if incremental_learning_rate > 0:
            df_samples, df_inc_samples, meta_types, null_values, full_join_est = prep.generate_n_samples_with_incremental_part(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor,
                drop_redundant_columns=drop_redundant_columns,
                incremental_learning_rate=incremental_learning_rate)
        else:
            df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor,
                drop_redundant_columns=drop_redundant_columns)
        logger.debug(f"Requested {sample_size} samples and got {len(df_samples)}")

        # learn spn
        aqp_spn = AQPSPN(meta_types, null_values, full_join_est, schema,
                         [relationship_obj.identifier], full_sample_size=len(df_samples),
                         column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)
        min_instance_slice = RATIO_MIN_INSTANCE_SLICE * min(sample_size, len(df_samples))
        logger.debug(f"Using min_instance_slice parameter {min_instance_slice}.")
        logger.info(f"SPN training phase with {len(df_samples)} samples")
        aqp_spn.learn(df_samples.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                      rdc_threshold=rdc_threshold)
        if incremental_learning_rate > 0:
            logger.info(f"additional incremental SPN training phase with {len(df_inc_samples)} samples "
                        f"({incremental_learning_rate}%)")
            aqp_spn.learn_incremental(df_inc_samples)
        spn_ensemble.add_spn(aqp_spn)

    ensemble_path += '/ensemble_relationships_' + dataset + '_' + str(sample_size) + '.pkl'
    logger.info(f"Saving ensemble to {ensemble_path}")
    spn_ensemble.save(ensemble_path)


def train_and_update(schema, hdf_path, sample_size, ensemble_path, dataset, bloom_filters, rdc_threshold, 
                     max_table_data, post_sampling_factor, init_hdf_path, perc, update_strategy):
    assert dataset in ('imdb', 'stats')
    
    stale_ensemble_path = os.path.join(ensemble_path, 'stale')
    updated_ensemble_path = os.path.join(ensemble_path, 'updated')
    os.makedirs(stale_ensemble_path, exist_ok=True)
    os.makedirs(updated_ensemble_path, exist_ok=True)
    
    prep_init = None if init_hdf_path is None else \
        JoinDataPreparator(str(os.path.join(init_hdf_path, 'meta_data.pkl')),
                               schema, max_table_data=max_table_data)
    prep = JoinDataPreparator(str(os.path.join(hdf_path, 'meta_data.pkl')),
                              schema, max_table_data=max_table_data)
    
    spn_ensemble_stale, spn_ensemble_updated = SPNEnsemble(schema), SPNEnsemble(schema)

    logger.info(f"Creating naive ensemble for every relationship.")

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
        
    def compute_ds_context(all_data, schema, meta_types, null_values, table_meta_data):
        column_names = list(all_data.columns)
        no_compression_scopes = []
        for table_obj in schema.tables:
            for attribute in table_obj.no_compression:
                column_name = table_obj.table_name + '.' + attribute
                if column_name in column_names:
                    no_compression_scopes.append(column_names.index(column_name))
        ds_context = build_ds_context(column_names, meta_types, null_values, table_meta_data,
                                        no_compression_scopes, all_data.values)
        return ds_context
        
        
    model_training_time, training_data_time, model_updating_time, updating_data_time = 0.0, 0.0, 0.0, 0.0
    ds_context = None # Filled later
    
    for relationship_obj in schema.relationships:
        logger.info(f"Learning SPN for {relationship_obj.identifier}.")
        if update_strategy == 'structure':
            # Compute initial training data
            start_train_data_time = time.time()
            train_data, meta_types, null_values, full_join_est_init = prep_init.generate_n_samples(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor,
                drop_redundant_columns=True)
            training_data_time += time.time() - start_train_data_time
            
            # Compute update data
            start_update_data_time = time.time()
            update_data, meta_types, null_values, full_join_est_update = prep.generate_n_samples(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor,
                drop_redundant_columns=True)
            updating_data_time += time.time() - start_update_data_time
            
            # Compute ds_context
            all_data = pd.concat([train_data, update_data], axis=0, ignore_index=True)
            ds_context = compute_ds_context(all_data, schema, meta_types, null_values, prep.table_meta_data)
            
            # Training initial model
            start_model_training_time = time.time()
            aqp_spn = AQPSPN(meta_types, null_values, full_join_est_init, schema,
                            [relationship_obj.identifier], full_sample_size=len(train_data),
                            column_names=list(train_data.columns), table_meta_data=prep.table_meta_data)
            min_instance_slice = RATIO_MIN_INSTANCE_SLICE * min(sample_size, len(train_data))
            logger.debug(f"Using min_instance_slice parameter {min_instance_slice}.")
            logger.info(f"SPN training phase with {len(train_data)} samples")
            aqp_spn.learn(train_data.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                          rdc_threshold=rdc_threshold, ds_context=ds_context)
            spn_ensemble_stale.add_spn(aqp_spn)
            model_training_time += time.time() - start_model_training_time
            
            # Model update
            aqp_spn_updated = copy.deepcopy(aqp_spn)
            aqp_spn_updated.table_meta_data = prep.table_meta_data
            start_updating = time.time()
            aqp_spn_updated.keep_structure_fit_new_data(update_data.values)
            aqp_spn_updated.full_join_size = full_join_est_update
            model_updating_time += time.time() - start_updating
            spn_ensemble_updated.add_spn(aqp_spn_updated)
            
        elif update_strategy == 'incremental':
            # Compute initial training data
            start_train_data_time = time.time()
            train_data, meta_types, null_values, full_join_est_init = prep_init.generate_n_samples(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor,
                drop_redundant_columns=True)
            training_data_time += time.time() - start_train_data_time
            
            # Compute update data
            start_update_data_time = time.time()
            update_data, meta_types, null_values, full_join_est_update = prep.generate_n_samples(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor,
                drop_redundant_columns=True)
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
            
            # Compute ds_context
            all_data = pd.concat([train_data, update_data], axis=0, ignore_index=True)
            ds_context = compute_ds_context(all_data, schema, meta_types, null_values, prep.table_meta_data)
            
            # Training initial model
            start_model_training_time = time.time()
            aqp_spn = AQPSPN(meta_types, null_values, full_join_est_init, schema,
                            [relationship_obj.identifier], full_sample_size=len(train_data),
                            column_names=list(train_data.columns), table_meta_data=prep.table_meta_data)
            min_instance_slice = RATIO_MIN_INSTANCE_SLICE * min(sample_size, len(train_data))
            logger.debug(f"Using min_instance_slice parameter {min_instance_slice}.")
            logger.info(f"SPN training phase with {len(train_data)} samples")
            aqp_spn.learn(train_data.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                        rdc_threshold=rdc_threshold, ds_context=ds_context)
            spn_ensemble_stale.add_spn(aqp_spn)
            model_training_time += time.time() - start_model_training_time
            
            # Model update
            aqp_spn_updated = copy.deepcopy(aqp_spn)
            aqp_spn_updated.table_meta_data = prep.table_meta_data
            start_updating = time.time()
            aqp_spn_updated.learn_incremental_from_insertion_and_deletion(dataset_ins.values, dataset_del.values)
            aqp_spn_updated.full_join_size = full_join_est_update
            model_updating_time += time.time() - start_updating
            spn_ensemble_updated.add_spn(aqp_spn_updated)
        
        elif update_strategy == 'filter':
            # Compute initial training data
            start_train_data_time = time.time()
            
            df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor,
                drop_redundant_columns=True)
            logger.debug(f"Requested {sample_size} samples and got {len(df_samples)}")
            
            init_indices = get_init_df_indices_by_perc(df_samples, perc, dataset)
            train_data = df_samples[init_indices]
            update_data = df_samples[~init_indices]
            
            training_data_time += time.time() - start_train_data_time
            
            # Compute ds_context
            ds_context = compute_ds_context(df_samples, schema, meta_types, null_values, prep.table_meta_data)
            
            start_model_training_time = time.time()
            
            # learn spn
            train_data_perc = train_data.shape[0] / df_samples.shape[0]
            aqp_spn = AQPSPN(meta_types, null_values, train_data.shape[0], schema,
                            [relationship_obj.identifier], full_sample_size=len(train_data),
                            column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)
            min_instance_slice = RATIO_MIN_INSTANCE_SLICE * min(sample_size, len(train_data))
            logger.debug(f"Using min_instance_slice parameter {min_instance_slice}.")
            logger.info(f"SPN training phase with {len(train_data)} samples")
            aqp_spn.learn(train_data.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                        rdc_threshold=rdc_threshold)
            spn_ensemble_stale.add_spn(aqp_spn)
            model_training_time += time.time() - start_model_training_time
            logger.info(f"Previous full outer join size {aqp_spn.full_join_size}, estimated size {full_join_est}")
            aqp_spn_updated = copy.deepcopy(aqp_spn)
            start_model_update = time.time()
            aqp_spn_updated.learn_incremental_from_insertion_and_deletion(update_data.values, None)
            logger.info(f"Current full outer join size {aqp_spn_updated.full_join_size}, estimated size {full_join_est}")
            #stats_20251118-183736.log
            spn_ensemble_updated.add_spn(aqp_spn_updated)            
            model_updating_time += time.time() - start_model_update

    stale_ensemble_path = os.path.join(stale_ensemble_path, 'ensemble_relationships_' + dataset + '_' + str(sample_size) + '.pkl')
    logger.info(f"Saving stale ensemble to {stale_ensemble_path}")
    spn_ensemble_stale.save(stale_ensemble_path)
    
    updated_ensemble_path = os.path.join(updated_ensemble_path, 'ensemble_relationships_' + dataset + '_' + str(sample_size) + '.pkl')
    logger.info(f"Saving updated ensemble to {updated_ensemble_path}")    
    spn_ensemble_updated.save(updated_ensemble_path)
    
    with open(os.path.join(ensemble_path, 'training_time.txt'), 'w') as f:
        f.write(f"model_training_time: {model_training_time}\n")
        f.write(f"model_updating_time: {model_updating_time}\n")
        f.write(f"training_data_time: {training_data_time}\n")
        f.write(f"updating_data_time: {updating_data_time}\n")
