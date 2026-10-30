import re
import os
import json
import shutil
import subprocess
import pandas as pd

from time import perf_counter
from typing import List, Dict, Tuple, Any
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema, SQLForeignKey
from schema.schemas import stats_schema
from common.logger import Logger
from common.value_wrapper import value_wrapper


class NeuroCardEstimator:
    _DATA_DIR_ENV = "NEUROCARD_DATA_DIR"

    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._schema = schema
        self._config = config
        self._result_folder = result_folder
        self._device = config["device"]
        self._env = os.environ.copy()
        if self._device[:5] == 'cuda:':
            self._env["CUDA_VISIBLE_DEVICES"] = self._device[5:].strip()
        self._use_transformer = config["use_transformer"]
        self._epochs = config["epochs"]
        # Older static experiment configs predate dynamic fine-tuning and do
        # not contain this option.  Static inference never uses it.
        self._max_steps = config.get("update_max_steps", 50)
        self._update_learning_rate = config.get("update_learning_rate", 2e-4)
        self._train_tuples_num = self._epochs * 512 * 2048
        config["train_tuples_num"] = self._train_tuples_num
        self._eval_psamples = config["eval_psamples"]
        if 'perc' in self._config: # For update experiment
            self._perc = self._config["perc"]
        self._statistics = {"model_size": None, "training_time": None}
        
        self._splited_schema = self._init_splited_schema(schema)

        # path info
        self._experiments_py_path = "estimators/neurocard/source/neurocard/experiments.py"
        self._temp_queries_path = "estimators/neurocard/source/neurocard/queries/temp_queries.csv"
        self._estimated_path = "estimators/neurocard/source/neurocard/card_result_fact_psample_{}.csv".format(self._eval_psamples)
        

        with open(self._temp_queries_path, "w") as file:
            pass # create an empty file

        # saved config
        self._old_data_dir_env = None
        self._had_old_data_dir_env = False
        self._old_epochs = None
        self._old_steps = None
        self._old_model_path = []
        self._old_psamples = []
        self._old_checkpoint = []

    def _init_splited_schema(self, schema: SQLSchema) -> List[List[SQLForeignKey]]:
        db = schema.name()
        db = 'stats' if db.startswith('stats') else db
        with open("estimators/neurocard/split_schema.json") as file:
            splited_schemas = json.load(file)
        join_clauses: List[List[List[str]]] = splited_schemas[db]
        foriegn_keys = []
        
        for join_clause in join_clauses:
            cur_foriegn_keys = []
            for fk in schema.foreign_keys():
                for join in join_clause:
                    if join[0] == fk.table.name() and join[1] == fk.column.name() and join[2] == fk.referenced_table.name() \
                        and join[3] == fk.referenced_column.name() or join[0] == fk.referenced_table.name() and \
                            join[1] == fk.referenced_column.name() and join[2] == fk.table.name() and join[3] == fk.column.name():
                        cur_foriegn_keys.append(fk)
                        break

            foriegn_keys.append(cur_foriegn_keys)

        return foriegn_keys

    def set_data_dir(self, db: str, perc = None) -> None:
        dataset_name = "imdb" if db.startswith("imdb") else db
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        data_dir = os.path.join(repo_root, "datasets", dataset_name)
        if perc is not None:
            data_dir = os.path.join(data_dir, "snapshots", str(perc))
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                "NeuroCard data directory does not exist: {}".format(data_dir))

        self._had_old_data_dir_env = self._DATA_DIR_ENV in self._env
        self._old_data_dir_env = self._env.get(self._DATA_DIR_ENV)
        self._env[self._DATA_DIR_ENV] = data_dir
    
    def reset_data_dir(self) -> None:
        if self._had_old_data_dir_env:
            self._env[self._DATA_DIR_ENV] = self._old_data_dir_env
        else:
            self._env.pop(self._DATA_DIR_ENV, None)

    def set_epochs(self, db: str) -> None:
        key = 'stats' if db.startswith('stats') else db
        line_index = {'imdb': 316, 'stats': 178, 'ergastf1': 178, 'imdb_m': 296, 'imdb_large':1098}
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        self._old_epochs = lines[line_index[key] - 1]
        lines[line_index[key] - 1] = re.sub(r'\d+', str(self._epochs), self._old_epochs)
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)

    def reset_epochs(self, db: str) -> None:
        key = 'stats' if db.startswith('stats') else db
        line_index = {'imdb': 316, 'stats': 178, 'ergastf1': 178, 'imdb_m': 296, 'imdb_large':1098}
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        lines[line_index[key] - 1] = self._old_epochs
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)

    def set_model_path_psamples(self, db: str, model_path: List[str]) -> None:
        key = 'stats' if db.startswith('stats') else db
        model_line_index = {'imdb': [676], 'stats': [605, 613, 621, 629, 637, 645, 653], 'ergastf1': [i * 7 + 925 for i in range(14)], 'imdb_m': [766], 'imdb_large': [1106]}
        psamples_line_index = {'imdb': [678], 'stats': [607, 615, 623, 631, 639, 647, 655], 'ergastf1': [i * 7 + 927 for i in range(14)], 'imdb_m': [768], 'imdb_large': [1108]}
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()

        self._old_model_path = []
        for i, line_index in enumerate(model_line_index[key]):
            self._old_model_path.append(lines[line_index - 1])
            lines[line_index - 1] = re.sub(r"'(.*?)'", f"'{model_path[i]}'", self._old_model_path[-1])
        self._old_psamples = []
        for i, line_index in enumerate(psamples_line_index[key]):
            self._old_psamples.append(lines[line_index - 1])
            lines[line_index - 1] = re.sub(r'\[.*?\]', f"[{self._eval_psamples}]", self._old_psamples[-1])
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)

    def reset_model_path_psamples(self, db: str) -> None:
        key = 'stats' if db.startswith('stats') else db
        model_line_index = {'imdb': [676], 'stats': [605, 613, 621, 629, 637, 645, 653], 'ergastf1': [i * 7 + 925 for i in range(14)], 'imdb_m': [766], 'imdb_large': [1106]}
        psamples_line_index = {'imdb': [678], 'stats': [607, 615, 623, 631, 639, 647, 655], 'ergastf1': [i * 7 + 927 for i in range(14)], 'imdb_m': [768], 'imdb_large': [1108]}
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        for i, line_index in enumerate(model_line_index[key]):
            lines[line_index - 1] = self._old_model_path[i]
        for i, line_index in enumerate(psamples_line_index[key]):
            lines[line_index - 1] = self._old_psamples[i]
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)
            
    def set_decreased_max_steps(self) -> None:
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        self._old_steps = lines[50 - 1]
        lines[50 - 1] = re.sub(r'\d+', str(self._max_steps), self._old_steps)
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)

    def reset_decreased_max_steps(self) -> None:
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        lines[50 - 1] = self._old_steps
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)

    def set_checkpoint_to_load(self, model_path: str) -> None:
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        line_number = 63
        new_line = "    'checkpoint_to_load': '{}',\n".format(model_path)
        self._old_checkpoint = lines[line_number - 1]
        lines[line_number - 1] = new_line
        lines[94 - 1] = "    'training_on_the_checkpoint': True,\n"
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)

    def reset_checkpoint(self) -> None:
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        lines[63 - 1] = self._old_checkpoint
        lines[94 - 1] = "    'training_on_the_checkpoint': False,\n"
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)
    
    def set_save_samples(self, path: str) -> None:
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        lines[96 - 1] = "    '_save_samples': '{}',\n".format(path)
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)
    
    def reset_save_samples(self) -> None:
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        lines[96 - 1] = "    '_save_samples': None,\n"
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)
    
    def set_load_samples(self, path: str) -> None:
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        lines[97 - 1] = "    '_load_samples': '{}',\n".format(path)
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)
    
    def reset_load_samples(self) -> None:
        with open(self._experiments_py_path, "r") as f:
            lines = f.readlines()
        lines[97 - 1] = "    '_load_samples': None,\n"
        with open(self._experiments_py_path, "w") as f:
            f.writelines(lines)
    
    
    def remove_cache_folder(self) -> None:
        if os.path.exists("./estimators/neurocard/source/neurocard/cache"):
            shutil.rmtree("./estimators/neurocard/source/neurocard/cache")

    @staticmethod
    def _file_versions(paths: List[str]) -> Dict[str, int]:
        return {
            path: os.stat(path).st_mtime_ns
            for path in paths
            if os.path.isfile(path)
        }

    @staticmethod
    def _fresh_artifact(paths: List[str], previous: Dict[str, int],
                        description: str) -> str:
        fresh = [
            path for path in paths
            if path not in previous or os.stat(path).st_mtime_ns > previous[path]
        ]
        if len(fresh) != 1:
            raise RuntimeError(
                "Expected exactly one fresh {}, found {}: {}".format(
                    description, len(fresh), fresh))
        return fresh[0]

    @staticmethod
    def _remove_if_exists(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def _one_matching_file(folder: str, predicate, description: str) -> str:
        matches = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if predicate(name)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one {}, found {}: {}".format(
                    description, len(matches), matches))
        return matches[0]
    
    @staticmethod
    def query_to_str(query: SQLQuery) -> str:
        tables = ["{} {}".format(query_node.table().name(), query_node.table().alias()) for query_node in query.nodes()]
        neurocard_value_wrapper = lambda x: x if "," not in x else '"' + x + '"'
        joins = [
            "{}.{}={}.{}".format(
                edge.foreign_key().referenced_table.alias(),
                edge.foreign_key().referenced_column.name(),
                edge.foreign_key().table.alias(),
                edge.foreign_key().column.name(),
            )
            for edge in query.edges() ]
        predicates = [
            "{}.{},{},{}".format(
                predicate.table.alias(),
                predicate.column.name(),
                predicate.operator,
                neurocard_value_wrapper(value_wrapper(predicate.value)),
            )
            for query_node in query.nodes() for predicate in query_node.predicates()
        ]
        
        return ",".join(tables) + "#" + ",".join(joins) + "#" + ",".join(predicates) + "#1\n"   # we add fake ground truth: 1

    def train(self) -> None:
        db = self._schema.name()
        log = Logger(os.path.join(self._result_folder, "train.log"))
        data_dir_set = epochs_set = False
        try:
            self.set_data_dir(db)
            data_dir_set = True
            self.set_epochs(db)
            epochs_set = True

            if db.split("_")[0] == "imdb":
                args_db_name = "job-light" if db == "imdb" else "job-m" if db == "imdb_m" else "job-large"
                commands = ["python", "run.py", "--run", args_db_name]
                log.write("[Command] " + " ".join(commands) + "\n")
                start = perf_counter()
                subprocess.run(
                    commands,
                    cwd="estimators/neurocard/source/neurocard",
                    env=self._env,
                    check=True)
                total_time = perf_counter() - start
            else:
                splited_schema_num = len(self._splited_schema)
                schema_name = self._schema.name()
                total_time = 0.0
                for i in range(splited_schema_num):
                    commands = ["python", "run.py", "--run", schema_name + str(i+1)]
                    log.write("[Command] " + " ".join(commands) + "\n")
                    start = perf_counter()
                    subprocess.run(
                        commands,
                        cwd="estimators/neurocard/source/neurocard",
                        env=self._env,
                        check=True)
                    total_time += perf_counter() - start
            log.write("[Total Time] " + str(total_time) + "\n")
        finally:
            if epochs_set:
                self.reset_epochs(db)
            if data_dir_set:
                self.reset_data_dir()
            log.close()

        self._statistics["training_time"] = total_time

    def predict(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]]]:
        grouped_queries:List[List[List[SQLQuery]]] = [[] for _ in range(len(self._splited_schema))]
        grouped_queries_idx: List[List[int]] = [[] for _ in range(len(self._splited_schema))]
        queries_estimated_card: List[List[float]] = [[] for _ in range(len(query))]
        queries_latencies: List[List[float]] = [[] for _ in range(len(query))]

        for idx, q in enumerate(query):
            outer_q = max(q, key=lambda x: len(x.nodes()))
            edges = outer_q.edges()
            assert(all(edge.is_pk_fk() for edge in edges))
            group_id = 0
            while group_id < len(grouped_queries) and len(edges) > 0:
                ok = True
                for edge in edges:
                    if not edge.foreign_key() in self._splited_schema[group_id]:
                        ok = False
                        break
                if ok:
                    grouped_queries[group_id].append(q)
                    grouped_queries_idx[group_id].append(idx)
                    break
                group_id += 1
            assert group_id < len(grouped_queries), f"Failed to group queries, {outer_q.sql()}"

        db = self._schema.name()

        model_paths = []
        for group_id in range(len(grouped_queries)):
            prefix = "imdb-" if db.split("_")[0] == "imdb" else db + str(group_id + 1) + "-"
            model_path = self._one_matching_file(
                self._result_folder,
                lambda name: name.startswith(prefix) and name.endswith(".pt"),
                "NeuroCard model for {}".format(prefix))
            model_path = str(os.path.join("../../../..", model_path))
            model_paths.append(model_path)
        
        data_dir_set = model_config_set = False
        try:
            self.set_data_dir(db)
            data_dir_set = True
            self.set_model_path_psamples(db, model_paths)
            model_config_set = True

            for group_id in range(len(grouped_queries)):
                queries = grouped_queries[group_id]
                queries_idx = grouped_queries_idx[group_id]
                with open(self._temp_queries_path, "w") as f:
                    f.writelines([self.query_to_str(x) for q in queries for x in q])

                prefix = "imdb" if db.split("_")[0] == "imdb" else db + str(group_id + 1)
                commands = ["python", "run.py", "--run", "job-light-reload" if db == "imdb" else "job-m-reload" if db == "imdb_m" else "job-large-reload" if db == "imdb_large" else prefix + "-reload"]
                self._remove_if_exists(self._estimated_path)
                subprocess.run(
                    commands,
                    cwd="estimators/neurocard/source/neurocard",
                    env=self._env,
                    check=True)
                if not os.path.isfile(self._estimated_path):
                    raise RuntimeError(
                        "NeuroCard inference completed without producing {}".
                        format(self._estimated_path))

                df = pd.read_csv(self._estimated_path)
                expected_rows = sum(len(q) for q in queries)
                if len(df) != expected_rows:
                    raise RuntimeError(
                        "NeuroCard inference returned {} rows; expected {}".
                        format(len(df), expected_rows))
                cards = [est if est >= 0 else -est for est in df["est_card"].tolist()]
                latencies = (df["time_ms"] / 1000).tolist()

                # Merge cards and latencies
                i, j = 0, 0
                while i < len(queries):
                    queries_estimated_card[queries_idx[i]] = cards[j: j+len(queries[i])]
                    queries_latencies[queries_idx[i]] = latencies[j: j+len(queries[i])]
                    j += len(queries[i])
                    i += 1
        finally:
            if model_config_set:
                self.reset_model_path_psamples(db)
            if data_dir_set:
                self.reset_data_dir()

        return queries_estimated_card, queries_latencies

    def save(self) -> None:
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)

        db = self._schema.name()
        model_dir = "estimators/neurocard/source/neurocard/models"
        model_mb = 0.0
        for i in range(len(self._splited_schema)):
            prefix = "imdb-" if db.split("_")[0] == "imdb" else db + str(i + 1) + "-"
            files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pt")]
            latest_file = max(files, key=os.path.getmtime)
            model_mb += float(re.search(r"-([\d.]+MB)-", os.path.basename(latest_file)).group(1)[:-2])
            shutil.move(latest_file, self._result_folder)
        
        self._statistics["model_size"] = model_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)
        
    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "NeuroCardEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device
        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)
        neurocard = NeuroCardEstimator(schema, config, dir)
        neurocard._statistics = statistics
        return neurocard

    """
        NOTE: How to use NeuroCard ? 
        Train:
        -- Set data dir, config: #tuples, arch
        -- Train separate models
        -- Save model
        -- Reset data dir/ config param

        Predict:
        -- Set data dir
        -- Set config param, model path, arch, #psamples including decide model to load
        -- Write query file, predict; Read result
        -- Reset data dir/ config param
    """


    """For update evaluation"""
    
    def train_update(self) -> None:
        perc = self._perc
        assert(perc in [20, 40, 60, 80])
        self._max_steps = ((100 - perc) // 20) * self._max_steps
        
        db = self._schema.name()
        log = Logger(os.path.join(self._result_folder, "train.log"))
        save_sample_path = os.path.join(self._result_folder, "samples_on_old_snapshot.csv")
        training_time, updating_time, model_size = 0.0, 0.0, 0.0
        
        if db == "imdb":
            train_args = ['job-light']
        elif db == "stats":
            train_args = ["stats"+str(i+1) for i in range(len(self._splited_schema))]
        else:
            log.close()
            raise NotImplementedError(
                "NeuroCard update is only configured for imdb and stats")

        model_dir = "estimators/neurocard/source/neurocard/models"
        training_stats_path = os.path.join(model_dir, "training_stats.txt")

        try:
            for train_arg in train_args:
                log.write("Training on " + train_arg + "...\n")
                dataset = train_arg if train_arg != "job-light" else "imdb"

                # Phase 1: materialize samples from the initial snapshot.  Both
                # table loading and JCT/index construction receive the same
                # NEUROCARD_DATA_DIR through the subprocess environment.
                self.remove_cache_folder()
                data_dir_set = epochs_set = save_samples_set = False
                try:
                    self.set_data_dir(db, perc)
                    data_dir_set = True
                    self.set_epochs(db)
                    epochs_set = True
                    self.set_save_samples(
                        os.path.join("../../../..", save_sample_path))
                    save_samples_set = True

                    log.write("Generating samples on initial snapshot...\n")
                    commands = ["python", "run.py", "--run", train_arg]
                    log.write("[Command] " + " ".join(commands) + "\n")
                    start = perf_counter()
                    subprocess.run(
                        commands,
                        cwd="estimators/neurocard/source/neurocard",
                        env=self._env,
                        check=True)
                    sample_time = perf_counter() - start
                    if not os.path.isfile(save_sample_path):
                        raise RuntimeError(
                            "Initial-snapshot sampling produced no sample file: {}".
                            format(save_sample_path))
                    training_time += sample_time
                    log.write("[Sample Time] " + str(sample_time) + "\n")
                finally:
                    if save_samples_set:
                        self.reset_save_samples()
                    if epochs_set:
                        self.reset_epochs(db)
                    if data_dir_set:
                        self.reset_data_dir()

                # Phase 2: train on the materialized initial samples, then
                # fine-tune on samples drawn from the full snapshot.
                self.remove_cache_folder()
                stale_paths = [
                    os.path.join(model_dir, name)
                    for name in os.listdir(model_dir)
                    if name.startswith(dataset + "-") and name.endswith(".pt")
                ]
                updated_paths = [
                    os.path.join(model_dir, name)
                    for name in os.listdir(model_dir)
                    if name.startswith("updated-" + dataset + "-") and
                    name.endswith(".pt")
                ]
                stale_before = self._file_versions(stale_paths)
                updated_before = self._file_versions(updated_paths)
                stats_before = self._file_versions([training_stats_path])

                data_dir_set = epochs_set = load_samples_set = steps_set = False
                original_epochs = self._epochs
                try:
                    self._epochs = original_epochs * 2
                    self.set_epochs(db)
                    epochs_set = True
                    self._epochs = original_epochs
                    self.set_data_dir(db)
                    data_dir_set = True
                    self.set_load_samples(
                        os.path.join("../../../..", save_sample_path))
                    load_samples_set = True
                    self.set_decreased_max_steps()
                    steps_set = True

                    log.write(
                        "Training model on initial samples & finetuning on full-data samples...\n")
                    commands = [
                        "python", "run.py", "--run", train_arg,
                        "--decrease_steps_after_half_epochs",
                        "--update_learning_rate",
                        str(self._update_learning_rate),
                    ]
                    log.write("[Command] " + " ".join(commands) + "\n")
                    subprocess.run(
                        commands,
                        cwd="estimators/neurocard/source/neurocard",
                        env=self._env,
                        check=True)

                    stats_file = self._fresh_artifact(
                        [training_stats_path], stats_before,
                        "training statistics file")
                    values = {}
                    with open(stats_file) as f:
                        for line in f:
                            key, separator, value = line.partition(":")
                            if separator:
                                values[key.strip()] = float(value.strip())
                    required = {"training_time", "updating_time", "model_size"}
                    missing = required.difference(values)
                    if missing:
                        raise RuntimeError(
                            "Missing fields in {}: {}".format(
                                stats_file, sorted(missing)))
                    training_time += values["training_time"]
                    updating_time += values["updating_time"]
                    model_size += values["model_size"]

                    stale_paths = [
                        os.path.join(model_dir, name)
                        for name in os.listdir(model_dir)
                        if name.startswith(dataset + "-") and
                        name.endswith(".pt")
                    ]
                    updated_paths = [
                        os.path.join(model_dir, name)
                        for name in os.listdir(model_dir)
                        if name.startswith("updated-" + dataset + "-") and
                        name.endswith(".pt")
                    ]
                    stale_model = self._fresh_artifact(
                        stale_paths, stale_before, "stale NeuroCard model")
                    updated_model = self._fresh_artifact(
                        updated_paths, updated_before,
                        "updated NeuroCard model")
                    shutil.move(stale_model, self._result_folder)
                    shutil.move(updated_model, self._result_folder)
                finally:
                    self._epochs = original_epochs
                    if steps_set:
                        self.reset_decreased_max_steps()
                    if load_samples_set:
                        self.reset_load_samples()
                    if data_dir_set:
                        self.reset_data_dir()
                    if epochs_set:
                        self.reset_epochs(db)
                    self._remove_if_exists(save_sample_path)
                    self._remove_if_exists(save_sample_path + ".lock")

            self._statistics["training_time"] = training_time
            self._statistics["updating_time"] = updating_time
            self._statistics["model_size"] = model_size
            log.write("[Training Time] " + str(training_time) + "\n")
            log.write("[Updating Time] " + str(updating_time) + "\n")
            log.write("[Model Size] " + str(model_size) + "\n")
        finally:
            log.close()
        
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)
        
    
    def predict_update(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]],
                                                                   List[List[float]], List[List[float]]]:
                                                                   # stale card, stale time, updated card, updated time
        logger = Logger(os.path.join(self._result_folder, "predict.log"), "a")
        
        grouped_queries:List[List[List[SQLQuery]]] = [[] for _ in range(len(self._splited_schema))]
        grouped_queries_idx: List[List[int]] = [[] for _ in range(len(self._splited_schema))]

        for idx, q in enumerate(query):
            outer_q = max(q, key=lambda x: len(x.nodes()))
            edges = outer_q.edges()
            assert(all(edge.is_pk_fk() for edge in edges))
            group_id = 0
            while group_id < len(grouped_queries) and len(edges) > 0:
                ok = True
                for edge in edges:
                    if not edge.foreign_key() in self._splited_schema[group_id]:
                        ok = False
                        break
                if ok:
                    grouped_queries[group_id].append(q)
                    grouped_queries_idx[group_id].append(idx)
                    break
                group_id += 1
            assert group_id < len(grouped_queries), f"Failed to group queries, {outer_q.sql()}"

        db = self._schema.name()

        result_all = []

        for is_updated in [False, True]:

            queries_estimated_card: List[List[float]] = [[] for _ in range(len(query))]
            queries_latencies: List[List[float]] = [[] for _ in range(len(query))]
            
            model_paths = []
            for group_id in range(len(grouped_queries)):
                prefix = "imdb-" if db.split("_")[0] == "imdb" else db + str(group_id + 1) + "-"
                if is_updated:
                    model_path = self._one_matching_file(
                        self._result_folder,
                        lambda name: name.startswith("updated-") and
                        prefix in name and name.endswith(".pt"),
                        "updated NeuroCard model for {}".format(prefix))
                else:
                    model_path = self._one_matching_file(
                        self._result_folder,
                        lambda name: name.startswith(prefix) and
                        name.endswith(".pt"),
                        "stale NeuroCard model for {}".format(prefix))
                model_path = str(os.path.join("../../../..", model_path))
                model_paths.append(model_path)
            
            self.remove_cache_folder()
            data_dir_set = model_config_set = False
            try:
                self.set_data_dir(db)
                data_dir_set = True
                self.set_model_path_psamples(db, model_paths)
                model_config_set = True

                for group_id in range(len(grouped_queries)):
                    queries = grouped_queries[group_id]
                    queries_idx = grouped_queries_idx[group_id]
                    with open(self._temp_queries_path, "w") as f:
                        f.writelines([self.query_to_str(x) for q in queries for x in q])

                    prefix = "imdb" if db.split("_")[0] == "imdb" else db + str(group_id + 1)
                    commands = ["python", "run.py", "--run", "job-light-reload" if db == "imdb" else "job-m-reload" if db == "imdb_m" else "job-large-reload" if db == "imdb_large" else prefix + "-reload"]

                    if not is_updated:
                        commands.extend([
                            '--init_data_ratio', str(self._perc / 100),
                        ])
                    logger.write("[Command] " + " ".join(commands) + "\n")

                    self._remove_if_exists(self._estimated_path)
                    subprocess.run(
                        commands,
                        cwd="estimators/neurocard/source/neurocard",
                        env=self._env,
                        check=True)
                    if not os.path.isfile(self._estimated_path):
                        raise RuntimeError(
                            "NeuroCard inference completed without producing {}".
                            format(self._estimated_path))

                    df = pd.read_csv(self._estimated_path)
                    expected_rows = sum(len(q) for q in queries)
                    if len(df) != expected_rows:
                        raise RuntimeError(
                            "NeuroCard inference returned {} rows; expected {}".
                            format(len(df), expected_rows))
                    cards = [est if est >= 0 else -est for est in df["est_card"].tolist()]
                    latencies = (df["time_ms"] / 1000).tolist()

                    # Merge cards and latencies
                    i, j = 0, 0
                    while i < len(queries):
                        queries_estimated_card[queries_idx[i]] = cards[j: j+len(queries[i])]
                        queries_latencies[queries_idx[i]] = latencies[j: j+len(queries[i])]
                        j += len(queries[i])
                        i += 1

                result_all.append(queries_estimated_card)
                result_all.append(queries_latencies)
            finally:
                if model_config_set:
                    self.reset_model_path_psamples(db)
                if data_dir_set:
                    self.reset_data_dir()

        logger.close()

        return result_all[0], result_all[1], result_all[2], result_all[3]
