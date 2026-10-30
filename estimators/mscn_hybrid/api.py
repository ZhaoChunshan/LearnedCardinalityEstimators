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
from estimators.mscn_hybrid.source.model import SetConvEnsemble
from estimators.mscn_hybrid.source.train import MSCNHybridTrainer
from common.logger import Logger
from common.torch_utils import unnormalize_card


class MSCNHybridEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._config = config
        self._schema = schema
        self._result_folder = result_folder

        sample_feats = config["sample_number"] + schema.table_num()
        predicate_feats = schema.column_num() + len(OPERATOR_ID) + 1
        join_feats = schema.foreign_key_num()
        hid_units = config["hidden_size"]
        ensemble_size = config["ensemble_size"]

        self._model = SetConvEnsemble(
            sample_feats, predicate_feats, join_feats,
            hid_units, ensemble_size
        )
        self._best_model_state_dict = None
        self._device = config["device"]
        self._threshold = config["var_threshold"]
        self._predict_count = 0
        self._statistics = {"model_size": None, "training_time": None}

    def train(
        self,
        train_queries: List[SQLQuery],
        train_card: List[float],
        valid_queries: List[SQLQuery],
        valid_card: List[float],
    ) -> None:
        begin_encode = perf_counter()

        train_input = MSCNDataEncoder.encode(
            self._schema, train_queries, self._config["sample_number"]
        )
        train_target = torch.tensor(train_card, dtype=torch.float32)

        valid_input = MSCNDataEncoder.encode(
            self._schema, valid_queries, self._config["sample_number"]
        )
        valid_target = torch.tensor(valid_card, dtype=torch.float32)

        encode_time = perf_counter() - begin_encode

        train_log = Logger(os.path.join(self._result_folder, "train.log"))
        trainer = MSCNHybridTrainer(
            self._model, self._device, train_log, self._config
        )

        begin_train = perf_counter()
        trainer.train(train_input, train_target, valid_input, valid_target)
        self._best_model_state_dict = trainer._best_model
        train_time = perf_counter() - begin_train

        train_log.write("[Encode Time] %f\n" % encode_time)
        train_log.write("[Forward-Backward Time] %f\n" % train_time)
        train_log.write("[Total Time] %f\n" % (train_time + encode_time))

        self._statistics["encode_time"] = encode_time
        self._statistics["model_time"] = train_time
        self._statistics["training_time"] = (
            train_time + encode_time + self._statistics.get("sample_time", 0.0)
        )

        train_log.close()

    @torch.no_grad()
    def predict(
        self,
        queries: List[SQLQuery],
        pg_est_cards: List[float],
    ) -> Tuple[List[float], List[float]]:
        """
        Predict cardinalities with uncertainty-based fallback to PostgreSQL.

        Args:
            queries: list of SQLQuery objects to estimate.
            pg_est_cards: PostgreSQL cardinality estimates for each query
                          (used as fallback when uncertainty is high).

        Returns:
            (estimated_cards, estimation_times)
        """
        device = self._device
        self._model.to(device)
        self._model.eval()

        begin_encode = perf_counter()
        encoded_input = MSCNDataEncoder.encode(
            self._schema, queries, self._config["sample_number"]
        )
        encode_time = perf_counter() - begin_encode

        dataset = TensorDataset(*encoded_input)
        loader = DataLoader(
            dataset, batch_size=self._config["batch_size"], shuffle=False
        )

        estimated_card = []
        total_variances = []

        begin_infer = perf_counter()
        with torch.no_grad():
            for samples, predicates, joins, sample_mask, predicate_mask, join_mask in loader:
                samples = samples.to(device)
                predicates = predicates.to(device)
                joins = joins.to(device)
                sample_mask = sample_mask.to(device)
                predicate_mask = predicate_mask.to(device)
                join_mask = join_mask.to(device)

                mu_all, logvar_all = self._model(
                    samples, predicates, joins,
                    sample_mask, predicate_mask, join_mask
                )
                # mu_all:     [batch, ensemble_size]
                # logvar_all: [batch, ensemble_size]

                # Ensemble prediction = mean across heads
                mu_mean = mu_all.mean(dim=1)

                # Total uncertainty decomposition
                var_aleatoric = torch.exp(logvar_all).mean(dim=1)
                var_epistemic = mu_all.var(dim=1, unbiased=False)
                var_total = var_aleatoric + var_epistemic  # [batch]

                pred_card = unnormalize_card(mu_mean)
                estimated_card.extend(pred_card.tolist())
                total_variances.extend(var_total.tolist())

        infer_time = perf_counter() - begin_infer

        # Per-subquery fallback with logging
        logger = Logger(
            os.path.join(self._result_folder, "predict.log"), mode="a"
        )
        fallback_count = 0
        for i in range(len(estimated_card)):
            is_fallback = total_variances[i] > self._threshold
            if is_fallback:
                estimated_card[i] = pg_est_cards[i]
                fallback_count += 1

            self._predict_count += 1
            logger.write(
                f"Subquery {self._predict_count}: "
                f"Uncertainty={total_variances[i]:.8f}, "
                f"MSCN_Est={estimated_card[i]:.4e}, "
                f"PG_Est={pg_est_cards[i]:.4e}"
                f"{' (FALLBACK)' if is_fallback else ''}\n"
            )

        predict_time = encode_time + infer_time

        # Summary logging
        logger.write(f"Predict time (encode + infer): {predict_time:.6f}s\n"
                     f"Encode time: {encode_time:.6f}s, "
                     f"Infer time: {infer_time:.6f}s\n"
                     f"Fallback count (var > {self._threshold}): "
                     f"{fallback_count} / {len(estimated_card)}\n"
                     f"Avg total variance: "
                     f"{sum(total_variances) / len(total_variances):.8f}\n\n")
        logger.close()

        # Estimation time: assign total to the last query (batch convention)
        estimation_time = [0.0] * len(estimated_card)
        estimation_time[-1] = predict_time

        return estimated_card, estimation_time

    def save(self) -> None:
        torch.save(
            self._model.state_dict(),
            os.path.join(self._result_folder, "best_model.pt"),
        )
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        model_size_mb = (
            os.path.getsize(os.path.join(self._result_folder, "best_model.pt"))
            / (1024 * 1024)
        )
        self._statistics["model_size"] = model_size_mb
        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)

    @staticmethod
    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "MSCNHybridEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device

        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)

        mscn_hybrid = MSCNHybridEstimator(schema, config, dir)
        mscn_hybrid._statistics = statistics
        mscn_hybrid._model.load_state_dict(
            torch.load(
                os.path.join(dir, "best_model.pt"),
                map_location=config["device"],
            )
        )
        return mscn_hybrid
