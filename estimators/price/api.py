import os
from pickle import FALSE
import time
import json
import torch
import numpy as np

from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from common.logger import Logger
from estimators.price.source.model.encoder import RegressionModel
from estimators.price.source.utils.args import get_args_finetune
from estimators.price.source.feature.features_tool import Sql2Feature
from estimators.price.source.utils.padding import features_padding
from estimators.price.source.utils.dataset import CustomDataset
from estimators.price.source.train import PRICETrainer


class PRICEEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str):
        self._config = config
        self._schema = schema
        self._result_folder = result_folder
        self._device = config["device"]
        self._args = get_args_finetune()
        self._database = 'imdb' if self._schema.name().startswith('imdb') else self._schema.name()
        self._encoder = Sql2Feature(self._database, self._args.bin_size, 
                                    statistics_dir=config['price_statistics_folder'])
        self._model = RegressionModel(config['max_n_join_col'], config['max_n_fanout'], config['max_n_table'], config['max_n_filter_col'],
                                      hist_dim=self._args.bin_size, table_dim=self._args.table_dim, filter_dim=self._args.filter_dim,
                                      query_hidden_dim=self._args.query_hidden_dim, final_hidden_dim=self._args.final_hidden_dim,
                                      output_dim=self._args.output_dim, n_embd=self._args.n_embd, n_layers=self._args.n_layers, 
                                      n_heads=self._args.n_heads, dropout_rate=self._args.dropout_rate)

        self._pretrained_path = "estimators/price/pretrain_params.pth"
        state_dict = torch.load(self._pretrained_path)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_state_dict[new_k] = v
        
        self._model.load_state_dict(new_state_dict)
        self._best_model_state_dict = new_state_dict
        self._statistics = {"model_size": None, "training_time": None, "pretraining_time": 76496}
        
        if 'perc' in self._config:
            self._perc = self._config['perc']
            

    def train(
            self,
            train_queries: List[SQLQuery],
            train_est_card: List[float],
            train_card: List[float],
            valid_queries: List[SQLQuery],
            valid_est_card: List[float],
            valid_card: List[float]
        ):
        
        logger = Logger(os.path.join(self._result_folder, "train.log"))

        alias_dict = self._encoder.information_coltype['abbrev']
        
        train_sqls = [q.sql(alias_dict=alias_dict, encode_comparable_string=True) for q in train_queries]
        valid_sqls = [q.sql(alias_dict=alias_dict, encode_comparable_string=True) for q in valid_queries]
        
        begin_encode = time.perf_counter()

        train_rets = [self._encoder.create_sql_features(sql) for sql in train_sqls] 
        train_data_features, train_n_join_cols, train_n_fanouts, train_n_tables, train_n_filter_cols = map(list, zip(*train_rets))
        
        valid_rets = [self._encoder.create_sql_features(sql) for sql in valid_sqls]
        valid_data_features, valid_n_join_cols, valid_n_fanouts, valid_n_tables, valid_n_filter_cols = map(list, zip(*valid_rets))
        
        max_n_join_col = max(max(train_n_join_cols), max(valid_n_join_cols))
        max_n_fanout = max(max(train_n_fanouts), max(valid_n_fanouts))
        max_n_table = max(max(train_n_tables), max(valid_n_tables))
        max_n_filter_col = max(max(train_n_filter_cols), max(valid_n_filter_cols))
        
        train_data, train_padding_masks = features_padding(self._args.bin_size, self._args.table_dim, self._args.filter_dim,
                                                           train_data_features, train_n_join_cols, train_n_fanouts, 
                                                           train_n_tables, train_n_filter_cols,
                                                           max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
        valid_data, valid_padding_masks = features_padding(self._args.bin_size, self._args.table_dim, self._args.filter_dim,
                                                           valid_data_features, valid_n_join_cols, valid_n_fanouts, 
                                                           valid_n_tables, valid_n_filter_cols,
                                                           max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
        
        encode_time = time.perf_counter() - begin_encode
        
        logger.write(f"[Encode Time] {encode_time:.2f} seconds\n")
        
        train_dataset = CustomDataset(list(zip(train_data, train_card, train_est_card, train_padding_masks,
                                               train_n_join_cols, train_n_fanouts, train_n_tables, train_n_filter_cols)))
        valid_dataset = CustomDataset(list(zip(valid_data, valid_card, valid_est_card, valid_padding_masks,
                                               valid_n_join_cols, valid_n_fanouts, valid_n_tables, valid_n_filter_cols)))
        
        self._model.reset_feature_num_info(max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
        
        begin_fit = time.perf_counter()
        
        trainer = PRICETrainer(model=self._model, device=self._device, log=logger, batch_size=self._config['batch_size'], 
                               epochs=self._config['epochs'], lr=self._config['lr'], weight_decay=self._args.wd,
                               step_size=self._args.step_size, gamma=self._args.gamma, accumulation_steps=self._config['accumulation_steps']
                               )
        trainer.train(train_dataset, valid_dataset)
        self._best_model_state_dict = trainer._best_model
        
        fitting_time = time.perf_counter() - begin_fit
        
        logger.write(f"[Fitting Time] {fitting_time:.2f} seconds\n")
        
        total_training_time = encode_time + fitting_time
        
        self._statistics["encode_time"] = encode_time
        self._statistics["fitting_time"] = fitting_time
        self._statistics["training_time"] = total_training_time
        
        logger.flush()
        logger.close()

    @torch.no_grad()
    def predict(self, query: List[SQLQuery], est_card: List[float]):

        logger = Logger(os.path.join(self._result_folder, "predict.log"), "a")
        
        self._model.eval()
        self._model.to(self._device)

        alias_dict = self._encoder.information_coltype['abbrev']
        
        sqls = [q.sql(alias_dict=alias_dict, encode_comparable_string=True) for q in query]
        
        encode_time, transfer_time, infer_time = 0.0, 0.0, 0.0
        
        begin_encode = time.perf_counter()

        rets = [self._encoder.create_sql_features(sql) for sql in sqls] 
        data_features, n_join_cols, n_fanouts, n_tables, n_filter_cols = map(list, zip(*rets))
        
        max_n_join_col = max(n_join_cols)
        max_n_fanout = max(n_fanouts)
        max_n_table = max(n_tables)
        max_n_filter_col = max(max(n_filter_cols), 1)
        
        data, padding_masks = features_padding(self._args.bin_size, self._args.table_dim, self._args.filter_dim,
                                               data_features, n_join_cols, n_fanouts,n_tables, n_filter_cols,
                                               max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
        
        encode_time = time.perf_counter() - begin_encode
        
        self._model.reset_feature_num_info(max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
        
        dataset = CustomDataset(list(zip(data, est_card, padding_masks, n_join_cols, n_fanouts, n_tables, n_filter_cols)))
        loader = DataLoader(dataset, batch_size=self._config["batch_size"], shuffle=False)
        
        estimated_card = []
        
        for feature, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col in loader:
            
            begin_transfer = time.perf_counter()
            
            feature = feature.to(torch.float).to(self._device)
            padding_mask = padding_mask.to(torch.float).to(self._device)
            n_join_col = n_join_col.to(torch.float).to(self._device).view(-1, 1)
            n_fanout = n_fanout.to(torch.float).to(self._device).view(-1, 1)
            n_table = n_table.to(torch.float).to(self._device).view(-1, 1)
            n_filter_col = n_filter_col.to(torch.float).to(self._device).view(-1, 1)
            pg_est_card = pg_est_card.to(torch.float).to(self._device).view(-1, 1)
            pg_est_card = torch.log(pg_est_card + 1) + 1
            
            transfer_time += time.perf_counter() - begin_transfer
            
            begin_infer = time.perf_counter()
            
            output = self._model(feature, pg_est_card, padding_mask, n_join_col, n_fanout, n_table, n_filter_col).view(-1)
            cards = torch.exp(output -1) - 1
            estimated_card.extend(cards.tolist())
            
            infer_time += time.perf_counter() - begin_infer
            
        total_predict_time = encode_time + transfer_time + infer_time
        
        logger.write(f"Cardinality estimation for SQL query:\n")
        logger.write(f"\t{query[-1].sql()}\n")
        logger.write(f"Total time (All python statements): {total_predict_time}\n")
        logger.write(f"Predict time (main work, encode + transfer + infer): {total_predict_time}\n")
        logger.write(f"Encode time: {encode_time}\n")
        logger.write(f"Transfer time: {transfer_time}\n")
        logger.write(f"Infer time: {infer_time}\n")

        logger.close()

        # Since the cardinality estimation is done in batch
        # We record the total estimation time of the batch in the last query
        estimation_time = [0.] * len(estimated_card)
        estimation_time[-1] = total_predict_time

        return estimated_card, estimation_time


    def save(self):
        torch.save(self._model.state_dict(), os.path.join(self._result_folder, "latest_model.pt"))    
        torch.save(self._best_model_state_dict, os.path.join(self._result_folder, "best_model.pt"))
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:  
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        model_size_mb = os.path.getsize(os.path.join(self._result_folder, "latest_model.pt")) / (1024 * 1024)
        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)


    @staticmethod
    def load_from_dir(
            schema: SQLSchema,
            dir: str,
            device: str = None,
            price_statistics_folder: str = None
        ) -> "PRICEEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device
        if price_statistics_folder is not None:
            config["price_statistics_folder"] = price_statistics_folder
        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)
        price = PRICEEstimator(schema, config, dir)
        price._statistics = statistics
        if config["load_best_model"]:
            price._model.load_state_dict(torch.load(os.path.join(dir, "best_model.pt")))
        else:
            price._model.load_state_dict(torch.load(os.path.join(dir, "latest_model.pt")))
        return price


    def train_update(self, train_workload_each_perc) -> None:
        db = self._schema.name()
        assert db in ('imdb', 'stats')
        
        logger = Logger(os.path.join(self._result_folder, "train.log"))

        encode_time = 0.0
        
        data_features_all, n_join_cols_all, n_fanouts_all, n_tables_all, n_filter_cols_all = [], [], [], [], []
        true_cards_all, est_cards_all = [], []
        
        for perc, (sqls, true_cards, estimated_cards, subquery_est_cards) in train_workload_each_perc.items():
            statistics_dir_perc = f"datasets/{db}/snapshots/{perc}/statistics/price"
            
            encoder = Sql2Feature(db, self._args.bin_size, statistics_dir=statistics_dir_perc)
        
            begin_encode = time.perf_counter()
            train_rets = [encoder.create_sql_features(sql) for sql in sqls] 
            encode_time += time.perf_counter() - begin_encode
            
            data_features, n_join_cols, n_fanouts, n_tables, n_filter_cols = map(list, zip(*train_rets))
            
            data_features_all.extend(data_features)
            n_join_cols_all.extend(n_join_cols)
            n_fanouts_all.extend(n_fanouts)
            n_tables_all.extend(n_tables)
            n_filter_cols_all.extend(n_filter_cols)
            
            true_cards_all.extend(true_cards)
            est_cards_all.extend(estimated_cards)

        max_n_join_col = max(n_join_cols_all)
        max_n_fanout = max(n_fanouts_all)
        max_n_table = max(n_tables_all)
        max_n_filter_col = max(n_filter_cols_all)
        
        begin_padding = time.perf_counter()
        train_data, train_padding_masks = features_padding(self._args.bin_size, self._args.table_dim, self._args.filter_dim,
                                                           data_features_all, n_join_cols_all, n_fanouts_all, 
                                                           n_tables_all, n_filter_cols_all,
                                                           max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)

        
        encode_time += time.perf_counter() - begin_padding
        
        logger.write(f"[Encode Time] {encode_time:.2f} seconds\n")
        
        train_card = torch.FloatTensor(true_cards_all)
        train_est_card = torch.FloatTensor(est_cards_all)
        
        train_dataset = CustomDataset(list(zip(train_data, train_card, train_est_card, train_padding_masks,
                                               n_join_cols_all, n_fanouts_all, n_tables_all, n_filter_cols_all)))
        
        self._model.reset_feature_num_info(max_n_join_col, max_n_fanout, max_n_table, max_n_filter_col)
        
        begin_fit = time.perf_counter()
        
        trainer = PRICETrainer(model=self._model, device=self._device, log=logger, batch_size=self._config['batch_size'], 
                               epochs=self._config['epochs'], lr=self._config['lr'], weight_decay=self._args.wd,
                               step_size=self._args.step_size, gamma=self._args.gamma, accumulation_steps=self._config['accumulation_steps']
                               )
        trainer.train(train_dataset)
        self._best_model_state_dict = trainer._best_model
        
        fitting_time = time.perf_counter() - begin_fit
        
        logger.write(f"[Fitting Time] {fitting_time:.2f} seconds\n")
        
        from synopsis.price_statistics import PRICEStatistics
        
        begin_updating = time.perf_counter()
        latest_statistics = PRICEStatistics.construct_statistics(self._schema, f"datasets/{db}", self._args.bin_size)
        updating_time = time.perf_counter() - begin_updating
        
        latest_statistics.save(os.path.join(self._result_folder, 'statistics_updated'))
        
        total_training_time = encode_time + fitting_time
        
        self._statistics["encode_time"] = encode_time
        self._statistics["model_time"] = fitting_time
        self._statistics["training_time"] = total_training_time
        self._statistics["updating_time"] = updating_time
        
        logger.flush()
        logger.close()

    
    def predict_update(self, 
                       query: List[List[SQLQuery]], 
                       pg_est_cards: List[List[float]],
                       pg_est_cards_stale: List[List[float]],
                       ) -> Tuple[List[List[float]], List[List[float]],
                            List[List[float]], List[List[float]]]:
                            # stale card, stale time, updated card, updated time
        result_all = []
        
        for is_stale in [True, False]:
            if is_stale:
                statistics_dir = f"datasets/{self._schema.name()}/snapshots/{self._perc}/statistics/price"
                used_pg_est_cards = pg_est_cards_stale
            else:
                statistics_dir = str(os.path.join(self._result_folder, 'statistics_updated'))
                used_pg_est_cards = pg_est_cards
            
            self._encoder = Sql2Feature(self._database, self._args.bin_size, 
                                        statistics_dir=statistics_dir)
            
            estimated_cards_all = []
            estimation_times_all = []
            
            for qlist, pg_est_list in zip(query, used_pg_est_cards):
                est_cards, est_times = self.predict(qlist, pg_est_list)
                estimated_cards_all.append(est_cards)
                estimation_times_all.append(est_times)
            
            self._encoder = None
            
            result_all.extend((estimated_cards_all, estimation_times_all))
        
        return result_all[0], result_all[1], result_all[2], result_all[3]
