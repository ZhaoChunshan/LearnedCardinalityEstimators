import os
import time
import json
import math
import torch
import pickle
import numpy as np

from typing import List, Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from estimators.alece.source.model import ALECE
from estimators.alece.source.encoder import ALECEEncoder
from estimators.alece.source.train import ALECETrainer
from estimators.alece.source.utils import *
from common.logger import Logger
from common.sql_utils import single_table_estimation
from common.data_utils import get_update_data


class ALECEEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str):
        self._config = config
        self._schema = schema
        self._result_folder = result_folder
        self._encoder = ALECEEncoder(schema)
        self._model = ALECE(
            config["n_bins"],
            self._encoder.feature_size(),
            config["num_attn_heads"],
            config["attn_head_dim"],
            config["feed_forward_dim"],
            config["num_self_attn_layers"],
            config["num_cross_attn_layers"],
            config["mlp_num_layers"],
            config["mlp_hidden_dim"],
            config["dropout"]
        )
        self._best_model_state_dict = None
        self._scaling_ratio = config["scaling_ratio"]
        self._device = config["device"]
        self._infer_batch_size = config["infer_batch_size"]
        self._statistics = {"model_size": None, "training_time": None}
        
        if "perc" in config:
            self._perc = config["perc"]

    def _set_full_table_size_thresholds(self, histograms: Dict) -> None:
        """Use the full database cardinality as the fixed histogram scale.

        Snapshot histograms may come from pickles created before
        ``TableHistogram._table_size_threshold`` existed.  Set the threshold
        explicitly whenever dynamic histograms are loaded so both old and new
        artifacts preserve table-size changes in their feature values.
        """
        for table_name, histogram in histograms.items():
            table = self._schema.table(table_name)
            if table is None:
                raise KeyError(f"Table {table_name} from histogram state is missing in schema")
            histogram.set_table_size_threshold(table.cardinality())


    def train(
            self,
            train_queries: List[SQLQuery],
            train_histogram: np.ndarray,
            train_hist_query_idx: np.ndarray,
            train_card: List[float],
            valid_queries: List[SQLQuery],
            valid_histogram: np.ndarray,
            valid_hist_query_idx: np.ndarray,
            valid_card: List[float]
            ):
        train_log = Logger(os.path.join(self._result_folder, "train.log"))
        
        begin_encode = time.perf_counter()

        num_train, num_valid = len(train_queries), len(valid_queries) if valid_queries else 0
        _, num_attrs, n_bins = train_histogram.shape

        train_query_features = self._encoder.encode_batch(train_queries)
        if num_valid != 0:
            valid_query_features = self._encoder.encode_batch(valid_queries)

        train_hist_features = np.empty((num_train, num_attrs, n_bins), dtype=np.float32)
        if num_valid != 0:
            valid_hist_features = np.empty((num_valid, num_attrs, n_bins), dtype=np.float32)

        for i in range(train_hist_query_idx.shape[0] - 1):
            train_hist_features[train_hist_query_idx[i]: train_hist_query_idx[i+1]] = train_histogram[i]
        train_hist_features[train_hist_query_idx[-1]:] = train_histogram[-1]

        if num_valid != 0:
            for i in range(valid_hist_query_idx.shape[0] - 1):
                valid_hist_features[valid_hist_query_idx[i]: valid_hist_query_idx[i+1]] = valid_histogram[i]
            valid_hist_features[valid_hist_query_idx[-1]:] = valid_histogram[-1]

        train_card = np.array(train_card, dtype=np.float32)
        train_label = cards_to_labels(train_card, self._scaling_ratio)
        train_weight = get_weights_from_labels(train_label)

        if num_valid != 0:
            valid_card = np.array(valid_card, dtype=np.float32)
            valid_label = cards_to_labels(valid_card, self._scaling_ratio)
            valid_weight = get_weights_from_labels(valid_label)

        encode_time = time.perf_counter() - begin_encode
        train_log.write("Encode finished. Time used %f\n" % encode_time)
        train_log.flush()

        begin_training = time.perf_counter()

        trainer = ALECETrainer(self._model, self._device, train_log, self._config)
        if num_valid != 0:
            trainer.train(
                (torch.from_numpy(train_hist_features), torch.from_numpy(train_query_features)),
                torch.from_numpy(train_label),
                torch.from_numpy(train_weight),
                (torch.from_numpy(valid_hist_features), torch.from_numpy(valid_query_features)),
                torch.from_numpy(valid_label),
                torch.from_numpy(valid_weight),
            )
        else:
            trainer.train_wo_valid(
                (torch.from_numpy(train_hist_features), torch.from_numpy(train_query_features)),
                torch.from_numpy(train_label),
                torch.from_numpy(train_weight),
            )
        self._best_model_state_dict = trainer._best_model

        training_time = time.perf_counter() - begin_training
        total_time = encode_time + training_time
        train_log.write("Training finished. Time used %f\n" % training_time)
        train_log.write("[Encode Time] %f\n" % encode_time)
        train_log.write("[Training Time] %f\n" % training_time)
        train_log.write("[Total Time] %f\n" % total_time)

        self._statistics["encode_time"] = encode_time
        self._statistics["model_time"] = training_time
        self._statistics["training_time"] = total_time + self._statistics["build_histogram_time"]

        train_log.flush()
        train_log.close()

    @torch.no_grad()
    def predict(self, query: List[SQLQuery], histogram: np.ndarray):
        self._model.eval()
        self._model.to(self._device)

        begin_encode = time.perf_counter()
        query_features = torch.from_numpy(self._encoder.encode_batch(query))
        hist_features = np.empty((len(query), histogram.shape[0], histogram.shape[1]), dtype=np.float32)
        hist_features[:,:,:] = histogram
        hist_features = torch.from_numpy(hist_features)
        encode_time = time.perf_counter() - begin_encode

        preds_list = []
        transfer_time = 0.0
        infer_time = 0.0

        batch_size = self._infer_batch_size if self._infer_batch_size is not None else len(query)
        num_batches = math.ceil(len(query) / batch_size)
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(query))

            begin_transfer = time.perf_counter()
            qf = query_features[start:end].to(self._device, non_blocking=True)
            hf = hist_features[start:end].to(self._device, non_blocking=True)
            transfer_time += time.perf_counter() - begin_transfer

            begin_infer = time.perf_counter()
            batch_preds = self._model(hf, qf).cpu().numpy()
            infer_time += time.perf_counter() - begin_infer

            preds_list.append(batch_preds)

        preds = np.concatenate(preds_list, axis=0)
        estimated_card = label_preds_to_card_preds(preds, self._scaling_ratio)

        predict_time = encode_time + transfer_time + infer_time
        total_time = time.perf_counter() - begin_encode

        logger = Logger(os.path.join(self._result_folder, "predict.log"), mode="a")

        logger.write(f"Cardinality estimation for SQL query:\n")
        logger.write(f"\t{query[-1].sql()}\n")
        logger.write(f"Total time (All python statements): {total_time}\n")
        logger.write(f"Predict time (main work, encode + transfer + infer): {predict_time}\n")
        logger.write(f"Encode time: {encode_time}\n")
        logger.write(f"Transfer time: {transfer_time}\n")
        logger.write(f"Infer time: {infer_time}\n")

        logger.close()

        # Since the cardinality estimation is done in batch
        # We record the total estimation time of the batch in the last query
        estimation_time = [0.] * len(estimated_card)
        estimation_time[-1] = predict_time

        return estimated_card, estimation_time


    def save(self):
        torch.save(self._model.state_dict(), os.path.join(self._result_folder, "latest_model.pt"))    
        torch.save(self._best_model_state_dict, os.path.join(self._result_folder, "best_model.pt"))
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:  
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        model_size_mb = os.path.getsize(os.path.join(self._result_folder, "best_model.pt")) / (1024 * 1024)
        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)


    @staticmethod
    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "ALECEEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device
        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)
        alece = ALECEEstimator(schema, config, dir)
        alece._statistics = statistics
        if config["load_best_model"]:
            alece._model.load_state_dict(torch.load(os.path.join(dir, "best_model.pt")))
        else:
            alece._model.load_state_dict(torch.load(os.path.join(dir, "latest_model.pt")))
        return alece


    def train_update(self, train_workload_each_perc) -> None:
        db = self._schema.name()
        assert db in ('imdb', 'stats')
        
        query_list, true_cards_list = [], []
        histograms_list, hist_query_idx = [], []
        latest_histograms = None # For update later
        
        tables_sort_by_id = sorted([table for table in self._schema.tables() if table.histogram() is not None], key=lambda x: x.id())
        
        for perc, (sqls, true_cards, _, _) in train_workload_each_perc.items():
            queries = [SQLQuery.parse(self._schema, sql).canonicalize() for sql in sqls]
        
            with open(f"datasets/{db}/snapshots/{perc}/statistics/alece/histograms.pkl", "rb") as f:
                histograms = pickle.load(f)
            self._set_full_table_size_thresholds(histograms)
            
            histograms_np = np.concatenate([histograms[table.name()].histogram() for table in tables_sort_by_id], axis=0)
            
            hist_query_idx.append(len(query_list))
            query_list.extend(queries)
            true_cards_list.extend(true_cards)
            histograms_list.append(histograms_np)
            
            if perc == self._perc:
                latest_histograms = histograms
        
        self.train(query_list, 
                   np.stack(histograms_list, axis=0),
                   np.array(hist_query_idx),
                   true_cards_list,
                   None, None, None, None)        

        with open(os.path.join(self._result_folder, "histograms_stale.pkl"), "wb") as f:
            pickle.dump(latest_histograms, f)
        
        update_data = get_update_data(db, self._perc)
        
        start_update = time.perf_counter()
        
        for table in tables_sort_by_id:
            
            histogram = latest_histograms[table.name()]
            
            table_update_data = update_data[table.name()]
            
            histogram.update_from_data(table_update_data)
        
        updating_time = time.perf_counter() - start_update
        
        train_log = Logger(os.path.join(self._result_folder, "train.log"), "a")
        train_log.write("[Updating Time] %f\n" % updating_time)

        self._statistics["updating_time"] = updating_time
        
        train_log.flush()
        train_log.close()
        
        with open(os.path.join(self._result_folder, "histograms_updated.pkl"), "wb") as f:
            pickle.dump(latest_histograms, f)


    def predict_update(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]],
                                                                   List[List[float]], List[List[float]]]:
                                                                   # stale card, stale time, updated card, updated time
        results_all = []
        
        for is_stale in [True, False]:
            if is_stale:
                with open(f"datasets/{self._schema.name()}/snapshots/{self._perc}/statistics/alece/histograms.pkl", "rb") as f:
                    histograms = pickle.load(f)
            else:
                with open(os.path.join(self._result_folder, "histograms_updated.pkl"), "rb") as f:
                    histograms = pickle.load(f)
            self._set_full_table_size_thresholds(histograms)
            
            cardinalities, latencies = [], []
            
            for queries in query:
                histogram_np = np.concatenate([histograms[table.name()].histogram() for table in sorted(
                    [table for table in self._schema.tables() if table.histogram() is not None], key=lambda x: x.id()
                )], axis=0)
                
                est_cards, est_times = self.predict(queries, histogram_np)
                
                cardinalities.append(list(est_cards))
                latencies.append(list(est_times))
            
            results_all.extend([cardinalities, latencies])
        
        return tuple(results_all)
