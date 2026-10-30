import os
import time
import json
import torch
import random
import pickle

from torch import FloatTensor

from typing import List, Dict, Tuple, Any
from common.logger import Logger
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from estimators.jgmp.source.models.bulk_jgmp_cardinality_model import BulkJGMPCardinalityModel
from estimators.jgmp.source.encoder.encoder import JGMPEncoder
from estimators.jgmp.source.train_cardinality import JGMPTrainer
from estimators.jgmp.source.query_data.bulk_light_cardinality_query_data import BulkLightCardinalityQueryData
from estimators.jgmp.source.query_data.bulk_light_query_data import BulkLightQueryData

class JGMPEstimator:
    def __init__(self, schema: SQLSchema, config: Dict[str, Any], result_folder: str) -> None:
        self._schema = schema
        self._config = config
        
        self.debug = True if "debug" in config and config["debug"] else False
        if self.debug:
            self.query_count = 0
        
        self._explict_encoding = True if "explicit_encoding" in config and config["explicit_encoding"] else False
        config["explicit_encoding"] = self._explict_encoding
        
        self._encoder = JGMPEncoder(schema, config["bitmap_size"])
        self._model = BulkJGMPCardinalityModel.from_config(self._encoder, config)
        self._best_model_state_dict = None
        self._device = config["device"]
        self._result_folder = result_folder
        self._statistics = {"model_size": None, "training_time": None}

    def train(self, train_queries: List[SQLQuery], train_card: List[float], valid_queries: List[SQLQuery], valid_card: List[float]) -> None:
        train_log = Logger(os.path.join(self._result_folder, "train.log"))
        
        begin_select_feature = time.perf_counter()
        tables_num, ok_num = len(self._schema.tables()), 0
        table_id_to_nodes = [[] for _ in range(tables_num)]

        if self._config["select_feature_samples"]:
            for query in train_queries:
                if ok_num == tables_num:
                    break
                for node in query.nodes():
                    table_id = node.table().id()
                    if len(table_id_to_nodes[table_id]) < self._config["instance_number"]:
                        table_id_to_nodes[table_id].append(node)
                        if len(table_id_to_nodes[table_id]) == self._config["instance_number"]:
                            ok_num += 1

            for table in self._schema.tables():
                relavant_nodes = table_id_to_nodes[table.id()]
                train_log.write("Select feature samples for table %s. relevant nodes: %d......\n" % (table.name(), len(relavant_nodes)))
                table.samples().select_feature_samples(relavant_nodes, self._config["max_fs_choices"], self._config["bitmap_size"])
                train_log.write("Finish Selection for table %s. Feature Sample Num = %d \n" % (table.name(), len(table.samples())))
        select_feature_time = time.perf_counter() - begin_select_feature
        
        train_log.write("Feature selection finished. Time used: %f\n" % select_feature_time)
        self.display_samples(train_log)
        
        train_log.flush()

        begin_encode = time.perf_counter()

        train_data: List[BulkLightCardinalityQueryData] = [
            BulkLightCardinalityQueryData(*self._encoder.bulk_light_encode_base([query]), FloatTensor([card]))
            for query, card in zip(train_queries, train_card)
        ]

        valid_data: List[BulkLightCardinalityQueryData] = [
            BulkLightCardinalityQueryData(*self._encoder.bulk_light_encode_base([query]), FloatTensor([card]))
            for query, card in zip(valid_queries, valid_card)
        ]
        encode_time = time.perf_counter() - begin_encode
        train_log.write("Encode Finished. Time used: %f\n" % encode_time)
        train_log.flush()

        begin_model_update = time.perf_counter()
        trainer = JGMPTrainer(self._model, self._device, train_log, self._config)
        trainer.train(train_data, valid_data)
        self._best_model_state_dict = trainer._best_model
        model_update_time = time.perf_counter() - begin_model_update
        train_log.write("Model update finished. Time used: %f\n" % (model_update_time))

        train_log.write("[Encode Time] %f\n" % encode_time)
        train_log.write("[Select Feature Time] %f\n" % select_feature_time)
        train_log.write("[Model Update Time] %f\n" % model_update_time)
        train_log.write("[Total Time] %f\n" % (encode_time + select_feature_time + model_update_time))

        self._statistics["feature_selection_time"] = select_feature_time
        self._statistics["encode_time"] = encode_time
        self._statistics["model_time"] = model_update_time
        # Total time includes sample time
        self._statistics["training_time"] = encode_time + select_feature_time + model_update_time + self._statistics["sample_time"]

        train_log.flush()
        train_log.close()

    @torch.no_grad()
    def predict(self, query: List[SQLQuery]) -> Tuple[List[float], List[float]]:
        logger = Logger(os.path.join(self._result_folder, "predict.log"), mode="a")

        self._model.eval()
        self._model.to(self._device)

        encode_time, transfer_time, infer_time, predict_time, total_time = 0, 0, 0, 0, 0

        begin_encode = time.perf_counter()
        data: BulkLightQueryData = BulkLightQueryData(*self._encoder.bulk_light_encode_base(query))
        encode_time = time.perf_counter() - begin_encode

        # Check intermedia data, e.g. sample bitmap, to see sampling failure ratio
        if self.debug:
            logger.write(f"Query #: {self.query_count}\n")
            logger.write(f"shared_node_labels:\n")
            logger.write(f"{data.shared_node_labels}\n")
            logger.write(f"shared_node_cardinalities:\n")
            logger.write(f"{data.shared_node_cardinalities}\n")
            logger.write(f"shared_node_samples:\n")
            logger.write(f"{data.shared_node_samples}\n")
            
            sample_num = data.shared_node_samples.shape[1]
            ratio = torch.sum(data.shared_node_samples, dim=1) / sample_num
            min_ratio = float(torch.min(ratio))
            logger.write(f"Samples non-zero ratio: {ratio}\n")
            logger.write(f"Min non-zero ratio: {min_ratio}\n")
            logger.flush()
            self.query_count += 1

        begin_transfer = time.perf_counter()
        data = data.to(self._device)
        transfer_time = time.perf_counter() - begin_transfer

        begin_infer = time.perf_counter()
        estimated_card = self._model(data).view(-1).tolist()
        infer_time = time.perf_counter() - begin_infer

        predict_time = encode_time + transfer_time + infer_time
        total_time = time.perf_counter()- begin_encode


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
        
        with open(os.path.join(self._result_folder, "samples.pkl"), "wb") as f:
            pickle.dump(self._schema.collect_table_samples(), f)
        
        model_size_mb = os.path.getsize(os.path.join(self._result_folder, "best_model.pt")) / (1024 * 1024)
        self._statistics["model_size"] = model_size_mb
        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)
            
        

    @staticmethod
    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "JGMPEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device
        with open(os.path.join(dir, "samples.pkl"), "rb") as f:
            samples = pickle.load(f)
            schema.set_table_samples(samples)
        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)
        jgmp = JGMPEstimator(schema, config, dir)
        jgmp._statistics = statistics
        if config["load_best_model"]:
            jgmp._model.load_state_dict(torch.load(os.path.join(dir, "best_model.pt")))
        else:
            jgmp._model.load_state_dict(torch.load(os.path.join(dir, "latest_model.pt")))
        return jgmp

    def display_samples(self, logger: Logger):
        sample_dict = self._schema.collect_table_samples()
        for table_name, samples in sample_dict.items():
            logger.write(f"Samples of table #{self._schema.table(table_name).id()} {table_name}:\n")
            logger.write(f"{str(samples)}\n")
            
