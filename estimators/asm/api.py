import os 
import json
import shutil
import pickle
import subprocess


from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema  
from common.logger import Logger
from typing import List, Tuple, Dict


class ASMEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._schema = schema
        self._config = config
        self._device = config["device"]
        self._env = os.environ.copy()
        if self._device[:5] == 'cuda:':
            self._env["CUDA_VISIBLE_DEVICES"] = self._device[5:].strip()
        self._result_folder = result_folder
        self._queries_file_path = os.path.join(self._result_folder, f"{schema.name()}_queries.pkl")
        self._subqueries_file_path = os.path.join(self._result_folder, f"{schema.name()}_sub_queries.pkl")
        self._predicate_dir = os.path.join(self._result_folder, "predicate")
        self._asm_data_dir = os.path.join(self._result_folder, "asm_data")
        self._model_size_dict = dict()
        self._statistics = {"model_size": None, "training_time": None}
        if 'perc' in self._config: # for update experiment only
            self._perc = self._config["perc"]

    def train(self) -> None:
        logger = Logger(os.path.join(self._result_folder, "train.log"))

        # Generate meta models        
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()
        db_name = self._schema.name().split("_")[0] if self._schema.name().startswith("imdb_") else self._schema.name()
        commands = ["python", "run_experiment.py",
                    "--dataset", dataset, 
                    "--generate_models",
                    "--data_path", "../../../datasets/" + db_name + "/{}.csv", 
                    "--model_path", str(os.path.join("../../..", self._result_folder)),
                    "--asm_data_path", str(os.path.join("../../..", self._asm_data_dir)),
                    ]
        logger.write(" ".join(commands) + "\n")
        result = subprocess.run(commands, capture_output=True, text=True,
                                cwd="estimators/asm/source", check=True)
        stdout_output = result.stdout
        gen_meta_model_time = float(stdout_output.splitlines()[-1].strip()[len("total time: "):])
        logger.write(f"gen_meta_model_time: {gen_meta_model_time}" + "\n")

        # Generate AR
        gen_ar_time = 0.0
        for table in self._schema.tables():
            commands = ["python", "AR/run.py", "--run", f"{db_name}-single-{table.name()}",
                        "--asm_data_path", str(os.path.join("../../..", self._asm_data_dir))]
            logger.write(" ".join(commands) + "\n")
            result = subprocess.run(commands, capture_output=True, text=True,
                                    cwd="estimators/asm/source", env=self._env,
                                    check=True)
            lines = result.stdout.splitlines()
            ok = 0
            for line in lines[-1::-1]:
                if "TERMINATED" in line and "NeuroCard" in line:
                    table_ar_time = float(line.split("|")[-2].strip())
                    gen_ar_time += table_ar_time
                    ok += 1
                if "ModelSize" in line:
                    self._model_size_dict[table.name()] = float(line[10 + line.find("ModelSize:"):].strip())
                    ok += 1
                if ok == 2:
                    break
            logger.write(f"gen ar for table {table.name()}, space = {self._model_size_dict[table.name()]} mb, time cost = {table_ar_time}\n")

        total_time = gen_meta_model_time + gen_ar_time
        logger.write(f"[Gen Meta Model Time]: {gen_meta_model_time}\n")
        logger.write(f"[Gen AR Time]: {gen_ar_time}\n")
        logger.write(f"[Total Time] {total_time}\n")
        logger.close()

        self._statistics["gen_meta_model_time"] = gen_meta_model_time
        self._statistics["gen_ar_time"] = gen_ar_time
        self._statistics["training_time"] = total_time


    def predict(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]]]:
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()

        all_queries: Dict[str, str] = dict() # Dict[QueryName, QueryStr]
        all_sub_plan_queries_str: Dict[str, List[Tuple[str]]] = dict() # Dict[QueryName, List[(QL, QR)]]
        sorted_subqueries_idx: List[List[int]] = []

        os.makedirs(self._predicate_dir, exist_ok=True)

        for i, subqueries in enumerate(query):
            query_name = f"{dataset}_{i+1:03}"

            sorted_idx = sorted(list(range(len(subqueries))), key=lambda x: (len(subqueries[x].nodes()), subqueries[x].nodes()[0].table().try_alias()))
            sorted_subqueries_idx.append(sorted_idx)

            all_queries[query_name] = subqueries[sorted_idx[-1]].sql()

            all_predicates = dict() # Dict[TableAlias, Tuple[TableName, ConjunctivePredicateStr, Set[JoinKeys]]]
            for node in subqueries[sorted_idx[-1]].nodes():
                all_predicates[node.table().alias()] = (node.table().name(),
                                                        " AND ".join(str(predicate) for predicate in node.predicates()),
                                                        set())
            for edge in subqueries[sorted_idx[-1]].edges():
                if edge.is_pk_fk():
                    fk = edge.foreign_key()
                    table1, table2 = fk.table.alias(), fk.referenced_table.alias()
                    column1, column2 = fk.column.name(), fk.referenced_column.name()
                else:
                    fk1, fk2 = edge.foreign_key()
                    table1, table2 = fk1.table.alias(), fk2.table.alias()
                    column1, column2 = fk1.column.name(), fk2.column.name()
                all_predicates[table1][2].add(column1)
                all_predicates[table2][2].add(column2)
            
            with open(os.path.join(self._predicate_dir, f"{query_name}.pkl"), "wb") as f:
                pickle.dump(all_predicates, f)
            
            subplan_queries_str = []
            seen_sq = set()
            for j in sorted_idx:
                subquery = subqueries[j]
                if len(subquery.nodes()) == 1:
                    seen_sq.add(subquery.nodes()[0].table().alias())
                    continue
                table_alias = [node.table().alias() for node in subquery.nodes()]
                table_alias.sort()
                for k, alias in enumerate(table_alias):
                    right_q = " ".join(table_alias[:k] + table_alias[k+1:])
                    if right_q in seen_sq:
                        subplan_queries_str.append((alias, right_q))
                        seen_sq.add(" ".join(table_alias))
                        break
            
            all_sub_plan_queries_str[query_name] = subplan_queries_str
            
        with open(self._queries_file_path, "wb") as f:
            pickle.dump(all_queries, f)
        
        with open(self._subqueries_file_path, "wb") as f:
            pickle.dump(all_sub_plan_queries_str, f)

        queries_estimated_card: List[List[float]] = [[] for _ in range(len(query))]
        queries_latencies: List[List[float]] = [[] for _ in range(len(query))]
        commands = ["python", "run_experiment.py",
                    "--dataset", dataset,
                    "--evaluate",
                    "--ar_path", str(os.path.join("../../..", self._result_folder, "{}-single-{}.tar")),
                    "--model_path", str(os.path.join("../../..", self._result_folder, f"model_{dataset}.pkl")),
                    "--config_path", "{}-single-{}_infer",
                    "--query_file", str(os.path.join("../../..", self._queries_file_path)),
                    "--query_sub_plan_file", str(os.path.join("../../..", self._subqueries_file_path)),
                    "--query_predicate_location", str(os.path.join("../../..", self._predicate_dir)),
                    "--save_folder", str(os.path.join("../../..", self._result_folder, "result")),
                    "--asm_data_path", str(os.path.join("../../..", self._asm_data_dir)),
                    ]
        subprocess.run(commands, cwd="estimators/asm/source", env=self._env, check=True)

        for i, subqueries in enumerate(query):
            query_name = f"{dataset}_{i+1:03}"
            with open(os.path.join(self._result_folder, "result", query_name), "r") as f:
                lines = f.readlines()
                cards = [float(line.strip()) for line in lines[:len(lines) - 1]]
                est_time = float(lines[-1].strip()[len("Time: "):])
                
                subquery_cards = [0.0 for _ in range(len(subqueries))]
                subquery_latency = [0.0 for _ in range(len(subqueries))]

                for j, card in enumerate(cards):
                    subquery_cards[sorted_subqueries_idx[i][j]] = card
                subquery_latency[-1] = est_time # We record the total estimation time in the last subquery
            
            queries_estimated_card[i] = subquery_cards
            queries_latencies[i] = subquery_latency

        return queries_estimated_card, queries_latencies

    def save(self) -> None:
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        ar_dir = "estimators/asm/source/AR_models"
        prefix = "imdb-light" if self._schema.name() == 'imdb' else 'imdb' if self._schema.name().startswith('imdb_') else self._schema.name()
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()
        for f in os.listdir(ar_dir):
            if f.startswith(prefix):
                shutil.move(os.path.join(ar_dir, f), self._result_folder)

        meta_model_path = os.path.join(self._result_folder, f"model_{dataset}.pkl")
        meta_model_size_mb = os.path.getsize(meta_model_path) / (1024 * 1024)
        model_size_mb = sum(self._model_size_dict.values()) + meta_model_size_mb

        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)

    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "ASMEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device
        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)
        asm = ASMEstimator(schema, config, dir)
        asm._statistics = statistics
        return asm

    
    """For update evaluation"""
    
    def train_update(self) -> None:
        logger = Logger(os.path.join(self._result_folder, "train.log"))
        perc = self._perc
        
        # Generate meta models        
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()
        db_name = self._schema.name().split("_")[0] if self._schema.name().startswith("imdb_") else self._schema.name()
        commands = ["python", "run_experiment.py",
                    "--dataset", dataset, 
                    "--generate_models",
                    "--data_path", "../../../datasets/" + db_name + "/{}.csv", 
                    "--model_path", str(os.path.join("../../..", self._result_folder)),
                    "--init_data_ratio", str(perc / 100.0),
                    "--asm_data_path", str(os.path.join("../../..", self._asm_data_dir)),
                    ]
        logger.write(" ".join(commands) + "\n")
        result = subprocess.run(commands, capture_output=True, text=True,
                                cwd="estimators/asm/source", check=True)
        stdout_output = result.stdout
        gen_meta_model_time = float(stdout_output.splitlines()[-1].strip()[len("total time: "):])
        logger.write(f"gen_meta_model_time: {gen_meta_model_time}" + "\n")

        # Generate AR
        train_ar_time, update_ar_time = 0.0, 0.0
        for table in self._schema.tables():
            commands = ["python", "AR/run.py", "--run", f"{db_name}-single-{table.name()}", 
                        "--init_data_perc", str(perc),
                        "--asm_data_path", str(os.path.join("../../..", self._asm_data_dir))]
            logger.write(" ".join(commands) + "\n")
            result = subprocess.run(commands, capture_output=True, text=True,
                                    cwd="estimators/asm/source", env=self._env,
                                    check=True)
            lines = result.stdout.splitlines()
            table_ar_time = None
            table_updating_time = None
            table_model_size = None
            for line in lines[-1::-1]:
                if table_ar_time is None and "total_train_time:" in line:
                    table_ar_time = float(line.split(":")[1].strip())
                elif table_updating_time is None and "total_updating_time:" in line:
                    table_updating_time = float(line.split(":")[1].strip())
                elif table_model_size is None and "ModelSize" in line:
                    table_model_size = float(line[10 + line.find("ModelSize:"):].strip())
                if table_ar_time is not None and table_updating_time is not None and table_model_size is not None:
                    break

            if table_ar_time is None or table_updating_time is None or table_model_size is None:
                raise RuntimeError(f"Failed to parse ASM training output for table {table.name()}:\n{result.stdout}\n{result.stderr}")

            train_ar_time += table_ar_time
            update_ar_time += table_updating_time
            self._model_size_dict[table.name()] = table_model_size
            logger.write(f"gen ar for table {table.name()}, space = {table_model_size} mb, train time cost = {table_ar_time}, update time cost = {table_updating_time}\n")

        total_time = gen_meta_model_time + train_ar_time
        logger.write(f"[Gen Meta Model Time]: {gen_meta_model_time}\n")
        logger.write(f"[Gen AR Time]: {train_ar_time}\n")
        logger.write(f"[Total Time] {total_time}\n")
        logger.close()

        self._statistics["gen_meta_model_time"] = gen_meta_model_time
        self._statistics["gen_ar_time"] = train_ar_time
        self._statistics["training_time"] = total_time
        self._statistics['updating_time'] = update_ar_time

        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        ar_dir = "estimators/asm/source/AR_models"
        prefix = "imdb" if self._schema.name() == 'imdb' else 'imdb' if self._schema.name().startswith('imdb_') else self._schema.name()

        for f in os.listdir(ar_dir):
            if f.startswith(prefix):
                shutil.move(os.path.join(ar_dir, f), self._result_folder)

        stale_model_dir = os.path.join(self._result_folder, "stale")
        os.makedirs(stale_model_dir, exist_ok=True)
        for f in os.listdir(os.path.join(ar_dir, "stale")):
            if f.startswith(prefix):
                shutil.move(os.path.join(os.path.join(ar_dir, "stale"), f), stale_model_dir)

        meta_model_path = os.path.join(self._result_folder, f"model_{dataset}.pkl")
        meta_model_size_mb = os.path.getsize(meta_model_path) / (1024 * 1024)
        model_size_mb = sum(self._model_size_dict.values()) + meta_model_size_mb

        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)
            
    def predict_update(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]],
                                                                   List[List[float]], List[List[float]]]:
                                                                   # stale card, stale time, updated card, updated time
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()

        all_queries: Dict[str, str] = dict() # Dict[QueryName, QueryStr]
        all_sub_plan_queries_str: Dict[str, List[Tuple[str]]] = dict() # Dict[QueryName, List[(QL, QR)]]
        sorted_subqueries_idx: List[List[int]] = []

        os.makedirs(self._predicate_dir, exist_ok=True)

        for i, subqueries in enumerate(query):
            query_name = f"{dataset}_{i+1:03}"

            sorted_idx = sorted(list(range(len(subqueries))), key=lambda x: (len(subqueries[x].nodes()), subqueries[x].nodes()[0].table().try_alias()))
            sorted_subqueries_idx.append(sorted_idx)

            all_queries[query_name] = subqueries[sorted_idx[-1]].sql()

            all_predicates = dict() # Dict[TableAlias, Tuple[TableName, ConjunctivePredicateStr, Set[JoinKeys]]]
            for node in subqueries[sorted_idx[-1]].nodes():
                all_predicates[node.table().alias()] = (node.table().name(),
                                                        " AND ".join(str(predicate) for predicate in node.predicates()),
                                                        set())
            for edge in subqueries[sorted_idx[-1]].edges():
                if edge.is_pk_fk():
                    fk = edge.foreign_key()
                    table1, table2 = fk.table.alias(), fk.referenced_table.alias()
                    column1, column2 = fk.column.name(), fk.referenced_column.name()
                else:
                    fk1, fk2 = edge.foreign_key()
                    table1, table2 = fk1.table.alias(), fk2.table.alias()
                    column1, column2 = fk1.column.name(), fk2.column.name()
                all_predicates[table1][2].add(column1)
                all_predicates[table2][2].add(column2)
            
            with open(os.path.join(self._predicate_dir, f"{query_name}.pkl"), "wb") as f:
                pickle.dump(all_predicates, f)
            
            subplan_queries_str = []
            seen_sq = set()
            for j in sorted_idx:
                subquery = subqueries[j]
                if len(subquery.nodes()) == 1:
                    seen_sq.add(subquery.nodes()[0].table().alias())
                    continue
                table_alias = [node.table().alias() for node in subquery.nodes()]
                table_alias.sort()
                for k, alias in enumerate(table_alias):
                    right_q = " ".join(table_alias[:k] + table_alias[k+1:])
                    if right_q in seen_sq:
                        subplan_queries_str.append((alias, right_q))
                        seen_sq.add(" ".join(table_alias))
                        break
            
            all_sub_plan_queries_str[query_name] = subplan_queries_str
            
        with open(self._queries_file_path, "wb") as f:
            pickle.dump(all_queries, f)
        
        with open(self._subqueries_file_path, "wb") as f:
            pickle.dump(all_sub_plan_queries_str, f)

        results_all = []
        for is_updated in [False, True]:
            queries_estimated_card: List[List[float]] = [[] for _ in range(len(query))]
            queries_latencies: List[List[float]] = [[] for _ in range(len(query))]
            if is_updated:
                ar_path = str(os.path.join("../../..", self._result_folder, "{}-single-{}.tar"))
            else:
                ar_path = str(os.path.join("../../..", self._result_folder, "stale/{}-single-{}.tar"))
            commands = ["python", "run_experiment.py",
                        "--dataset", dataset,
                        "--evaluate",
                        "--ar_path", ar_path,
                        "--model_path", str(os.path.join("../../..", self._result_folder, f"model_{dataset}.pkl")),
                        "--config_path", "{}-single-{}_infer",
                        "--query_file", str(os.path.join("../../..", self._queries_file_path)),
                        "--query_sub_plan_file", str(os.path.join("../../..", self._subqueries_file_path)),
                        "--query_predicate_location", str(os.path.join("../../..", self._predicate_dir)),
                        "--save_folder", str(os.path.join("../../..", self._result_folder, "result")),
                        "--asm_data_path", str(os.path.join("../../..", self._asm_data_dir)),
                        ]
            if not is_updated:
                commands.extend(["--init_data_ratio", str(self._perc / 100.0), "--evaluate_stale", "True"])
            subprocess.run(commands, cwd="estimators/asm/source", env=self._env, check=True)

            for i, subqueries in enumerate(query):
                query_name = f"{dataset}_{i+1:03}"
                with open(os.path.join(self._result_folder, "result", query_name), "r") as f:
                    lines = f.readlines()
                    cards = [float(line.strip()) for line in lines[:len(lines) - 1]]
                    est_time = float(lines[-1].strip()[len("Time: "):])
                    
                    subquery_cards = [0.0 for _ in range(len(subqueries))]
                    subquery_latency = [0.0 for _ in range(len(subqueries))]

                    for j, card in enumerate(cards):
                        subquery_cards[sorted_subqueries_idx[i][j]] = card
                    subquery_latency[-1] = est_time # We record the total estimation time in the last subquery
                
                queries_estimated_card[i] = subquery_cards
                queries_latencies[i] = subquery_latency

            results_all.extend([queries_estimated_card, queries_latencies])
        return tuple(results_all)
