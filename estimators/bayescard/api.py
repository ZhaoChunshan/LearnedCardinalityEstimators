import os 
import json
import subprocess

from time import perf_counter
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema  
from common.logger import Logger
from typing import List, Tuple, Dict


class BayesCardEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._schema = schema
        self._config = config
        self._result_folder = result_folder
        self._subqueries_file_path = os.path.join(self._result_folder, f"{schema.name()}_sub_queries.sql")
        self._subqueries_est_path = os.path.join(self._result_folder, f"{schema.name()}_sub_queries.sql.est")
        self._statistics = {"model_size": None, "training_time": None}
        if 'update_strategy' in self._config:
            self._update_strategy = self._config["update_strategy"]
            assert self._update_strategy in ['structure', 'incremental', 'filter']
        if 'perc' in self._config:
            self._perc = self._config["perc"]


    def train(self) -> None:
        train_log = Logger(os.path.join(self._result_folder, "train.log"))

        # Generate hdf files from csvs.
        db_name = self._schema.name().split("_")[0] if self._schema.name().startswith("imdb") else self._schema.name()
        hdf_commands = ["python", "estimators/bayescard/source/run_experiment.py", 
                        "--generate_hdf",
                        "--dataset", self._schema.name(),
                        "--csv_seperator", "|",
                        "--csv_path", str(os.path.join("./datasets", db_name)),
                        "--hdf_path", str(os.path.join(self._result_folder, "hdf")),
                        ]
        train_log.write(" ".join(hdf_commands) + "\n")
        subprocess.run(hdf_commands)
        with open(os.path.join(self._result_folder, "hdf", "build_time_hdf.txt"), 'r') as f:
            gen_hdf_time = int(f.readline().strip())
        
        # Learn BayesCard models
        learn_commands = ["python", "estimators/bayescard/source/run_experiment.py",
                          "--generate_models",
                          "--dataset", self._schema.name(),
                          "--hdf_path", str(os.path.join(self._result_folder, "hdf")),
                          "--model_path", self._result_folder,
                          "--learning_algo", self._config["learning_algo"],
                          "--max_parents", str(self._config["max_parents"]),
                          "--sample_size", str(self._config["sample_size"])
                          ]
        train_log.write(" ".join(learn_commands) + "\n")
        start_learn = perf_counter()
        result = subprocess.run(learn_commands, capture_output=True, text=True)
        learn_time = perf_counter() - start_learn
        total_time = gen_hdf_time + learn_time

        self._statistics["gen_hdf_time"] = gen_hdf_time
        self._statistics["gen_ensemble_time"] = learn_time
        self._statistics["training_time"] = total_time

        train_log.write("[Gen HDF] " + str(gen_hdf_time) + "\n")
        train_log.write("[Learn] " + str(learn_time) + "\n")
        train_log.write("[Total Time] %f\n" % total_time)
        train_log.close()


    def predict(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]]]:
        logger = Logger(os.path.join(self._result_folder, "predict.log"), mode="a")

        with open(self._subqueries_file_path, "w") as f_sub_plan:
            for subqueries in query:
                for subquery in subqueries:
                    f_sub_plan.write(str(subquery.sql()) + "||1\n") # Add fake cardinality.

        predict_commands = ["python", "estimators/bayescard/source/run_experiment.py",
                            "--dataset", self._schema.name(),
                            "--evaluate_cardinalities",
                            "--model_path", str(self._result_folder),
                            "--query_file_location", str(self._subqueries_file_path),
                            "--infer_algo", self._config["infer_algo"],
                            "--learning_algo", self._config["learning_algo"],
                            "--max_parents", str(self._config["max_parents"]),
                            ]
        logger.write(" ".join(predict_commands) + "\n")
        
        subprocess.run(predict_commands)
        est_all, latencies_all = [], []
        with open(self._subqueries_est_path, "r") as f_sub_est:
            for line in f_sub_est.readlines():
                est, latency = line.strip().split("||")
                est_all.append(float(est))
                latencies_all.append(float(latency))
        
        subqueries_est, subqueries_latencies = [], []
        index = 0
        for subqueries in query:
            subqueries_est.append(est_all[index:index+len(subqueries)])
            subqueries_latencies.append(latencies_all[index:index+len(subqueries)])
            index += len(subqueries)
        
        return subqueries_est, subqueries_latencies


    def save(self) -> None:
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        model_path_list = [os.path.join(self._result_folder, f) for f in os.listdir(self._result_folder) if f.endswith(".pkl")]
        model_size_mb = sum(os.path.getsize(model_path) / (1024 * 1024) for model_path in model_path_list)
        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)


    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "BayesCardEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device

        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)

        bayescard = BayesCardEstimator(schema, config, dir)
        bayescard._statistics = statistics
        return bayescard

    def train_update(self) -> None:
        train_log = Logger(os.path.join(self._result_folder, "train.log"))

        # Generate hdf files from csvs.
        db_name = self._schema.name().split("_")[0] if self._schema.name().startswith("imdb") else self._schema.name()
        gen_hdf_time = []
        for whole_data in [True, False]:
            if not whole_data and self._update_strategy == 'filter':
                continue
            hdf_dir_name = 'hdf' if whole_data else 'hdf_init'
            data_dir_name = db_name if whole_data else f"{db_name}/snapshots/{self._perc}"
            hdf_commands = ["python", "estimators/bayescard/source/run_experiment.py", 
                            "--generate_hdf",
                            "--dataset", self._schema.name(),
                            "--csv_seperator", "|",
                            "--csv_path", str(os.path.join("./datasets", data_dir_name)),
                            "--hdf_path", str(os.path.join(self._result_folder, hdf_dir_name)),
                            ]
            if not whole_data:
                hdf_commands.extend(
                    ['--prev_meta_data_path', str(os.path.join(self._result_folder, 'hdf', 'meta_data.pkl'))]
                )
            train_log.write(" ".join(hdf_commands) + "\n")
            subprocess.run(hdf_commands)
            with open(os.path.join(self._result_folder, hdf_dir_name, "build_time_hdf.txt"), 'r') as f:
                cur_gen_hdf_time = int(f.readline().strip())
                gen_hdf_time.append(cur_gen_hdf_time)
        train_log.write("[Gen HDF] " + str(gen_hdf_time) + "\n")
        
        # Learn BayesCard models
        learn_commands = ["python", "estimators/bayescard/source/run_experiment.py",
                          "--evaluate_update",
                          "--dataset", self._schema.name(),
                          "--hdf_path", str(os.path.join(self._result_folder, "hdf")),
                          "--model_path", self._result_folder,
                          "--learning_algo", self._config["learning_algo"],
                          "--max_parents", str(self._config["max_parents"]),
                          "--sample_size", str(self._config["sample_size"]),
                          "--update_strategy", self._update_strategy,
                          "--init_data_perc", str(self._perc),
                          ]
        if not self._update_strategy == 'filter':
            learn_commands.extend(
                ['--init_hdf_path', str(os.path.join(self._result_folder, 'hdf_init'))]
            )
        train_log.write(" ".join(learn_commands) + "\n")
        start_learn = perf_counter()
        subprocess.run(learn_commands)
        learn_time = perf_counter() - start_learn
        train_log.write("[Learn] %f\n" % learn_time)
        
        with open(os.path.join(self._result_folder, 'training_time.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    tokens = line.split(":")
                    self._statistics[tokens[0].strip()] = float(tokens[1].strip())
            self._statistics['training_time'] = self._statistics['model_training_time'] + self._statistics['training_data_time']
            self._statistics['updating_time'] = self._statistics['model_updating_time'] + self._statistics['updating_data_time']

        train_log.write("[Training Time] " + str(self._statistics['training_time']) + "\n")
        train_log.write("[Updating Time] " + str(self._statistics['updating_time']) + "\n")
        
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        updated_model_dir = os.path.join(self._result_folder, "updated")
        model_path_list = [
            os.path.join(updated_model_dir, filename)
            for filename in os.listdir(updated_model_dir)
            if filename.endswith(".pkl")
        ]
        model_size_mb = sum(os.path.getsize(model_path) for model_path in model_path_list) / (1024 * 1024)
        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)


    def predict_update(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]],
                                                                   List[List[float]], List[List[float]]]:
                                                                   # stale card, stale time, updated card, updated time
        logger = Logger(os.path.join(self._result_folder, "predict.log"), mode="a")

        with open(self._subqueries_file_path, "w") as f_sub_plan:
            for subqueries in query:
                for subquery in subqueries:
                    f_sub_plan.write(str(subquery.sql()) + "||1\n") # Add fake cardinality.
        results_all = []
        for is_updated in [False, True]:
            model_dir = os.path.join(self._result_folder, 'stale') if not is_updated else \
                os.path.join(self._result_folder, 'updated')
            predict_commands = ["python", "estimators/bayescard/source/run_experiment.py",
                                "--dataset", self._schema.name(),
                                "--evaluate_cardinalities",
                                "--model_path", str(model_dir),
                                "--query_file_location", str(self._subqueries_file_path),
                                "--infer_algo", self._config["infer_algo"],
                                "--learning_algo", self._config["learning_algo"],
                                "--max_parents", str(self._config["max_parents"]),
                                ]
            logger.write(" ".join(predict_commands) + "\n")
            
            subprocess.run(predict_commands)
            est_all, latencies_all = [], []
            with open(self._subqueries_est_path, "r") as f_sub_est:
                for line in f_sub_est.readlines():
                    est, latency = line.strip().split("||")
                    est_all.append(float(est))
                    latencies_all.append(float(latency))
            
            subqueries_est, subqueries_latencies = [], []
            index = 0
            for subqueries in query:
                subqueries_est.append(est_all[index:index+len(subqueries)])
                subqueries_latencies.append(latencies_all[index:index+len(subqueries)])
                index += len(subqueries)
            
            results_all.extend([subqueries_est, subqueries_latencies])
        return tuple(results_all)
