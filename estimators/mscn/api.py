import os
import json
import torch


from time import perf_counter
from typing import List, Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from schema.comparison_operator import OPERATOR_ID
from estimators.mscn.source.data import MSCNDataEncoder
from estimators.mscn.source.model import SetConv
from estimators.mscn.source.train import MSCNTrainer
from common.logger import Logger
from common.torch_utils import unnormalize_card


class MSCNEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._config = config
        self._schema = schema
        self._result_folder = result_folder
        self._model = SetConv(config["sample_number"] + schema.table_num(), schema.column_num() + len(OPERATOR_ID) + 1, \
                              schema.foreign_key_num(), config["hidden_size"])
        self._best_model_state_dict = None
        self._device = config["device"]
        self._statistics = {"model_size": None, "training_time": None}


    def train(self, train_queries: List[SQLQuery], train_card: List[float], valid_queries: List[SQLQuery], valid_card: List[float]) -> None:
        begin_encode = perf_counter()
        
        train_input = MSCNDataEncoder.encode(self._schema, train_queries, self._config["sample_number"])
        train_target = torch.tensor(train_card, dtype=torch.float32)

        valid_input = MSCNDataEncoder.encode(self._schema, valid_queries, self._config["sample_number"])
        valid_target = torch.tensor(valid_card, dtype=torch.float32)

        encode_time = perf_counter() - begin_encode

        train_log = Logger(os.path.join(self._result_folder, "train.log"))
        trainer = MSCNTrainer(self._model, self._device, train_log, self._config)
        
        begin_train = perf_counter()
        trainer.train(train_input, train_target, valid_input, valid_target)
        self._best_model_state_dict = trainer._best_model
        train_time = perf_counter() - begin_train

        train_log.write("[Encode Time] %f\n" % encode_time)
        train_log.write("[Forward-Backward Time] %f\n" % train_time)
        train_log.write("[Total Time] %f\n" % (train_time + encode_time))
        
        self._statistics["encode_time"] = encode_time
        self._statistics["model_time"] = train_time
        self._statistics["training_time"] = train_time + encode_time + self._statistics["sample_time"]

        train_log.close()

    @torch.no_grad()
    def predict(self, query: List[SQLQuery]) -> Tuple[List[float], List[float]]:        
        device = self._device

        self._model.to(device)
        self._model.eval()

        encode_time, transfer_time, infer_time, predict_time, total_time = 0, 0, 0, 0, 0

        begin_encode = perf_counter()
        input = MSCNDataEncoder.encode(self._schema, query, self._config["sample_number"])
        encode_time = perf_counter() - begin_encode

        dataset = TensorDataset(*input)
        loader = DataLoader(dataset, batch_size=self._config["batch_size"], shuffle=False)

        estimated_card = []

        with torch.no_grad():
            for samples, predicates, joins, sample_mask, predicate_mask, join_mask in loader:
                begin_transfer = perf_counter()
                samples, predicates, joins, sample_mask, predicate_mask, join_mask = \
                    samples.to(device), predicates.to(device), joins.to(device), sample_mask.to(device), \
                    predicate_mask.to(device), join_mask.to(device)
                transfer_time += perf_counter() - begin_transfer

                begin_infer = perf_counter()
                output = self._model(samples, predicates, joins, sample_mask, predicate_mask, join_mask).view(-1)
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

    @staticmethod
    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "MSCNEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device

        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)

        mscn = MSCNEstimator(schema, config, dir)
        mscn._statistics = statistics
        if config["load_best_model"]:
            mscn._model.load_state_dict(torch.load(os.path.join(dir, "best_model.pt")))
        else:
            mscn._model.load_state_dict(torch.load(os.path.join(dir, "latest_model.pt")))
        return mscn
