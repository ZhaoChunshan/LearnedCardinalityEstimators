import os
import json
import torch
import pickle

from time import perf_counter
from typing import List, Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from schema.comparison_operator import OPERATOR_ID
from synopsis.join_bitmap import JoinBitmap
from estimators.robustmscn.source.data import RobustMSCNDataEncoder
from estimators.robustmscn.source.model import SetConv
from estimators.robustmscn.source.train import RobustMSCNTrainer
from common.logger import Logger
from common.torch_utils import unnormalize_card, normalize_card
from common.data_utils import get_update_data, get_table_size_for_snapshot
from common.sql_utils import single_table_estimation

class RobustMSCNEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._config = config
        self._schema = schema
        self._result_folder = result_folder
        self._model = SetConv(
            1 + schema.table_num(),
            schema.column_num() + len(OPERATOR_ID) + 1,
            RobustMSCNDataEncoder.join_feature_num(schema, config["sample_number"]),
            config["hidden_size"],
            1,  # per-table PostgreSQL selectivity
            schema.column_num(),
            config["sample_number"],
            config["query_masking_ratio"],
        )
        self._best_model_state_dict = None
        self._device = config["device"]
        self._join_bitmap = None
        self._statistics = {"model_size": None, "training_time": None}
        
        if 'perc' in config:
            self._perc = config['perc']


    def train(self, train_queries: List[SQLQuery], train_est_card: List[float], train_card: List[float], valid_queries: List[SQLQuery], valid_est_card: List[float], valid_card: List[float]) -> None:
        
        begin_join_bitmap = perf_counter()

        self._join_bitmap = JoinBitmap(self._schema, self._config["sample_number"], self._config["foreign_table_ratio"])
        self._join_bitmap.update(self._schema)

        join_bitmap_time = perf_counter() - begin_join_bitmap

        begin_encode = perf_counter()        

        train_input = RobustMSCNDataEncoder.encode(self._schema, train_queries, self._join_bitmap)
        train_pg_est = torch.tensor(train_est_card, dtype=torch.float32)
        train_target = torch.tensor(train_card, dtype=torch.float32)

        valid_input = RobustMSCNDataEncoder.encode(self._schema, valid_queries, self._join_bitmap)
        valid_pg_est = torch.tensor(valid_est_card, dtype=torch.float32)
        valid_target = torch.tensor(valid_card, dtype=torch.float32)

        encode_time = perf_counter() - begin_encode

        train_log = Logger(os.path.join(self._result_folder, "train.log"))
        trainer = RobustMSCNTrainer(self._model, self._device, train_log, self._config)
        
        begin_train = perf_counter()
        trainer.train((*train_input, train_pg_est), train_target, (*valid_input, valid_pg_est), valid_target)
        self._best_model_state_dict = trainer._best_model
        train_time = perf_counter() - begin_train

        train_log.write("[Join Bitmap Time] %f\n" % join_bitmap_time)
        train_log.write("[Encode Time] %f\n" % encode_time)
        train_log.write("[Forward-Backward Time] %f\n" % train_time)
        train_log.write("[Total Time] %f\n" % (train_time + encode_time))

        self._statistics["join_bitmap_time"] = join_bitmap_time
        self._statistics["encode_time"] = encode_time
        self._statistics["model_time"] = train_time
        self._statistics["training_time"] = train_time + encode_time + join_bitmap_time
        
        train_log.flush()
        train_log.close()

    @torch.no_grad()
    def predict(self, query: List[SQLQuery], est_card: List[float]) -> Tuple[List[float], List[float]]:        
        device = self._device

        self._model.to(device)
        self._model.eval()
        self._model.disable_query_masking()
        self._model.disable_bitmap_shuffling()

        encode_time, transfer_time, infer_time, predict_time, total_time = 0, 0, 0, 0, 0

        begin_encode = perf_counter()
        input = RobustMSCNDataEncoder.encode(self._schema, query, self._join_bitmap)
        encode_time = perf_counter() - begin_encode

        est_card = torch.tensor(est_card, dtype=torch.float32)
        dataset = TensorDataset(*input, est_card)
        loader = DataLoader(dataset, batch_size=self._config["batch_size"], shuffle=False)

        estimated_card = []

        with torch.no_grad():
            for samples, predicates, joins, sample_mask, predicate_mask, join_mask, postgres_est in loader:
                begin_transfer = perf_counter()
                samples, predicates, joins, sample_mask, predicate_mask, join_mask, postgres_est = \
                    samples.to(device), predicates.to(device), joins.to(device), sample_mask.to(device), \
                    predicate_mask.to(device), join_mask.to(device), postgres_est.to(device)
                transfer_time += perf_counter() - begin_transfer

                begin_infer = perf_counter()
                postgres_est = normalize_card(postgres_est)
                output = self._model(samples, predicates, joins, sample_mask, predicate_mask, join_mask, postgres_est).view(-1)
                output = unnormalize_card(output)
                infer_time += perf_counter() - begin_infer

                begin_transfer = perf_counter()
                estimated_card.extend(output.tolist())
                transfer_time += perf_counter() - begin_transfer
                
        end_predict = perf_counter()
        predict_time = encode_time + transfer_time + infer_time
        
        # Total time may be slightly larger than predict time due to other overhead.
        # But they should be very close. We treat the `predict time` as the total end-to-end prediction time.
        total_time = end_predict - begin_encode

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

    def save(self) -> None:
        torch.save(self._model.state_dict(), os.path.join(self._result_folder, "latest_model.pt"))
        torch.save(self._best_model_state_dict, os.path.join(self._result_folder, "best_model.pt"))
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        model_size_mb = os.path.getsize(os.path.join(self._result_folder, "best_model.pt")) / (1024 * 1024)
        
        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)

        if self._join_bitmap != None:
            with open(os.path.join(self._result_folder, "join_bitmap.pkl"), "wb") as f:
                pickle.dump(self._join_bitmap, f)

    @staticmethod
    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "RobustMSCNEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device

        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)
        
        robust_mscn = RobustMSCNEstimator(schema, config, dir)
        robust_mscn._statistics = statistics
        if config["load_best_model"]:
            robust_mscn._model.load_state_dict(torch.load(os.path.join(dir, "best_model.pt")))
        else:
            robust_mscn._model.load_state_dict(torch.load(os.path.join(dir, "latest_model.pt")))

        if os.path.exists(os.path.join(dir, "join_bitmap.pkl")):
            with open(os.path.join(dir, "join_bitmap.pkl"), "rb") as f:
                robust_mscn._join_bitmap = pickle.load(f)
        
        return robust_mscn


    def train_update(self, train_workload_each_perc) -> None:
        db = self._schema.name()
        assert db in ('imdb', 'stats')
        
        features_samples_all, features_predicates_all, features_joins_all = [], [], []
        train_pg_est_all, train_target_all = [], []
        
        alias_dict = {table.alias(): table.name() for table in self._schema.tables()}

        table_cardinality_full = {}

        for table in self._schema.tables():
            table_cardinality_full[table.name()] = table.cardinality()
            
        latest_join_samples = None
        
        encoding_time = 0.0
        
        for perc, (sqls, true_cards, estimated_cards, subquery_est_cards) in train_workload_each_perc.items():
            
            with open(f"datasets/{db}/snapshots/{perc}/statistics/robustmscn/join_samples.pkl", "rb") as f:
                join_samples = pickle.load(f)

                if perc == self._perc:
                    latest_join_samples = join_samples
                
            table_size_for_snapshot = get_table_size_for_snapshot(db, perc)
            
            for table in self._schema.tables():
                table.set_cardinality(table_size_for_snapshot[table.name()])
            
            queries = [SQLQuery.parse(self._schema, sql).canonicalize().add_single_table_estimation(
                    single_table_estimation(alias_dict, subquery_est.keys(), subquery_est.values())
                ) for sql, subquery_est in zip(sqls, subquery_est_cards)]
            
            begin_encode = perf_counter()
            
            features = [RobustMSCNDataEncoder.encode_no_padding(self._schema, query, join_samples) for query in queries]
            features = list(zip(*features))

            encoding_time += perf_counter() - begin_encode
            
            features_samples_all.extend(features[0])
            features_predicates_all.extend(features[1])
            features_joins_all.extend(features[2])
            
            train_pg_est_all.extend(estimated_cards)
            train_target_all.extend(true_cards)
        
        # Set to actual cardinality
        for table in self._schema.tables():
            table.set_cardinality(table_cardinality_full[table.name()])
        
        begin_padding = perf_counter()
        
        features_samples_all, features_predicates_all, features_joins_all, sample_mask_all, predicate_mask_all, join_mask_all\
            = RobustMSCNDataEncoder.feature_padding(
                features_samples_all, features_predicates_all, features_joins_all
            )
        
        encoding_time += perf_counter() - begin_padding
        
        train_pg_est_all = torch.tensor(train_pg_est_all, dtype=torch.float32)
        train_target_all = torch.tensor(train_target_all, dtype=torch.float32)
        
        train_log = Logger(os.path.join(self._result_folder, "train.log"))
        
        trainer = RobustMSCNTrainer(self._model, self._device, train_log, self._config)
        
        begin_model = perf_counter()
        trainer.train_wo_valid((features_samples_all, features_predicates_all, features_joins_all, sample_mask_all, \
            predicate_mask_all, join_mask_all, train_pg_est_all), train_target_all)
        self._best_model_state_dict = trainer._best_model
        model_time = perf_counter() - begin_model
        
        with open(os.path.join(self._result_folder, "join_samples_stale.pkl"), "wb") as f:
            pickle.dump(latest_join_samples, f)
        
        begin_update = perf_counter()
        
        latest_join_samples = JoinBitmap(self._schema, self._config["sample_number"], self._config["foreign_table_ratio"])
        latest_join_samples.update(self._schema)
        
        updating_time = perf_counter() - begin_update
        
        with open(os.path.join(self._result_folder, "join_samples_updated.pkl"), "wb") as f:
            pickle.dump(latest_join_samples, f)
        
        train_log.write("[Encode Time] %f\n" % encoding_time)
        train_log.write("[Forward-Backward Time] %f\n" % model_time)
        train_log.write("[Training Time] %f\n" % (encoding_time + model_time))
        train_log.write("[Updating Time] %f\n" % updating_time)

        self._statistics["encode_time"] = encoding_time
        self._statistics["model_time"] = model_time
        self._statistics["training_time"] = model_time + encoding_time
        self._statistics["updating_time"] = updating_time
        
        train_log.flush()
        train_log.close()


    def predict_update(self, 
                       query: List[List[SQLQuery]], 
                       pg_est_cards: List[List[float]],
                       pg_est_cards_stale: List[List[float]],
                       single_table_est: List[Dict[str, float]],
                       single_table_est_stale: List[Dict[str, float]]
                       ) -> Tuple[List[List[float]], List[List[float]],
                            List[List[float]], List[List[float]]]:
                            # stale card, stale time, updated card, updated time
        db = self._schema.name()
        assert db in ('imdb', 'stats')
        
        result_all = []
        
        table_cardinality_full = {}

        for table in self._schema.tables():
            table_cardinality_full[table.name()] = table.cardinality()
        
        for is_stale in [True, False]:
            if is_stale:
                with open(f"datasets/{self._schema.name()}/snapshots/{self._perc}/statistics/robustmscn/join_samples.pkl", "rb") as f:
                    join_samples = pickle.load(f)
                
                table_size_for_snapshot = get_table_size_for_snapshot(db, self._perc)
            
                for table in self._schema.tables():
                    table.set_cardinality(table_size_for_snapshot[table.name()])
                
            else:
                with open(os.path.join(self._result_folder, "join_samples_updated.pkl"), "rb") as f:
                    join_samples = pickle.load(f)
            
            self._join_bitmap = join_samples
            
            estimated_cards_all = []
            estimation_times_all = []
            
            if is_stale:
                for qlist, pg_est_list, single_table_estimation in zip(query, pg_est_cards_stale, single_table_est_stale):
                    qlist = [q.add_single_table_estimation(single_table_estimation) for q in qlist]
                    est_cards, est_times = self.predict(qlist, pg_est_list)
                    estimated_cards_all.append(est_cards)
                    estimation_times_all.append(est_times)
            else:
                for qlist, pg_est_list, single_table_estimation in zip(query, pg_est_cards, single_table_est):
                    qlist = [q.add_single_table_estimation(single_table_estimation) for q in qlist]
                    est_cards, est_times = self.predict(qlist, pg_est_list)
                    estimated_cards_all.append(est_cards)
                    estimation_times_all.append(est_times)
            
            self._join_bitmap = None
            
            if is_stale:
                for table in self._schema.tables():
                    table.set_cardinality(table_cardinality_full[table.name()])
            
            result_all.extend((estimated_cards_all, estimation_times_all))
        
        return result_all[0], result_all[1], result_all[2], result_all[3]
