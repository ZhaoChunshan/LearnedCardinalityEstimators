import re
import os
import json
import subprocess
import pandas as pd

from time import perf_counter
from typing import List, Dict, Tuple
from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from common.logger import Logger


class DeepDBEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._schema = schema
        self._config = config
        self._result_folder = result_folder
        self._ensemble_strategy = config["ensemble_strategy"]
        self._ensemble_budget_factor = config["ensemble_budget_factor"]
        self._ensemble_max_no_joins = config["ensemble_max_no_joins"]
        self._ratio_min_instance_slice = config["ratio_min_instance_slice"]
        self._ratio_py_file = "estimators/deepdb/source/ensemble_creation/naive.py"
        self._saved_ratio_min_instance_slice = None # Save original ratio_min_instance_slice
        self._statistics = {"model_size": None, "training_time": None}
        
        # For update experiments
        if 'update_strategy' in self._config:
            self._update_strategy = self._config["update_strategy"]
            assert self._update_strategy in ['structure', 'incremental', 'filter']
        if 'perc' in self._config:
            self._perc = self._config["perc"]

    def set_ratio_min_instance_slice(self):
        with open(self._ratio_py_file, 'r') as f:
            lines = f.readlines()
        
        self._saved_ratio_min_instance_slice = lines[14]
        lines[14] = f"RATIO_MIN_INSTANCE_SLICE = {self._ratio_min_instance_slice}\n"

        with open(self._ratio_py_file, 'w') as f:
            f.writelines(lines)

    def reset_ratio_min_instance_slice(self):
        with open(self._ratio_py_file, 'r') as f:
            lines = f.readlines()
        
        lines[14] = self._saved_ratio_min_instance_slice
        with open(self._ratio_py_file, 'w') as f:
            f.writelines(lines)

    def train(self) -> None:
        self.set_ratio_min_instance_slice()

        log = Logger(os.path.join(self._result_folder, "train.log"))

        # Generate hdf files from csvs.
        db_name = self._schema.name().split('_')[0] if self._schema.name().startswith("imdb_") else self._schema.name()
        hdf_commands = ["python", "estimators/deepdb/source/maqp.py", 
                        "--generate_hdf",
                        "--dataset", self._schema.name(),
                        "--csv_seperator", "|",
                        "--csv_path", str(os.path.join("./datasets", db_name)),
                        "--hdf_path", str(os.path.join(self._result_folder, "hdf")),
                        "--max_rows_per_hdf_file", "100000000"]
        log.write(" ".join(hdf_commands) + "\n")
        subprocess.run(hdf_commands)
        with open(os.path.join(self._result_folder, "hdf", "build_time_hdf.txt"), 'r') as f:
            gen_hdf_time = int(f.readline().strip())

        # Generate sampled hdf files
        sample_commands = ["python", "estimators/deepdb/source/maqp.py",
                           "--generate_sampled_hdfs",
                           "--dataset", self._schema.name(),
                           "--hdf_path", str(os.path.join(self._result_folder, "hdf")),
                           "--max_rows_per_hdf_file", "100000000",
                           "--hdf_sample_size", "10000"
                           ]
        log.write(" ".join(sample_commands) + "\n")
        start_sample = perf_counter()
        subprocess.run(sample_commands)
        gen_sample_time = perf_counter() - start_sample

        # Learn ensemble
        ensemble_commands = ["python", "estimators/deepdb/source/maqp.py",
                             "--generate_ensemble",
                             "--dataset", self._schema.name(),
                             "--samples_per_spn", "100000000", "100000000", "1000000", "1000000", "1000000",
                             "--ensemble_strategy", self._ensemble_strategy,
                             "--hdf_path", str(os.path.join(self._result_folder, "hdf")),
                             "--max_rows_per_hdf_file", "100000000",
                             "--samples_rdc_ensemble_tests", "10000",
                             "--ensemble_path", str(os.path.join(self._result_folder, "spn_ensembles")),
                             "--database_name", db_name,
                             "--post_sampling_factor", "10", "10", "5", "1", "1",
                             "--ensemble_budget_factor", str(self._ensemble_budget_factor),
                             "--ensemble_max_no_joins", str(self._ensemble_max_no_joins),
                             "--pairwise_rdc_path", str(os.path.join(self._result_folder, "spn_ensembles", "pairwise_rdc.pkl"))
                             ]
        if "taskset_cpus" in self._config and self._config["taskset_cpus"] is not None:
            ensemble_commands = ["taskset", "-c", self._config["taskset_cpus"]] + ensemble_commands
        log.write(" ".join(ensemble_commands) + "\n")
        start_ensemble = perf_counter()
        subprocess.run(ensemble_commands)
        gen_ensemble_time = perf_counter() - start_ensemble
        gen_time = gen_hdf_time + gen_sample_time + gen_ensemble_time

        self.reset_ratio_min_instance_slice()

        self._statistics["gen_hdf_time"] = gen_hdf_time
        self._statistics["sample_hdf_time"] = gen_sample_time
        self._statistics["gen_ensemble_time"] = gen_ensemble_time
        self._statistics["training_time"] = gen_time

        log.write("[Gen HDF] " + str(gen_hdf_time) + "\n")
        log.write("[Gen Sampled HDF] " + str(gen_sample_time) + "\n")
        log.write("[Gen Ensemble] " + str(gen_ensemble_time) + "\n")
        log.write("[Total Time] " + str(gen_time) + "\n")
        log.close()

    def predict(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]]]:
        logger = Logger(os.path.join(self._result_folder, "predict.log"), mode="a")

        query_dir = os.path.join(self._result_folder, "query")
        query_path = os.path.join(query_dir, "queries.sql")
        ground_truth_path = os.path.join(query_dir, "ground_truth.csv")
        result_path = os.path.join(query_dir, "deep_db_estimates.csv")
        ensemble_dir = os.path.join(self._result_folder, "spn_ensembles")
        pairwise_rdc_path, ensemble_path = None, None
        for filename in os.listdir(ensemble_dir):
            if filename == "pairwise_rdc.pkl":
                pairwise_rdc_path = os.path.join(ensemble_dir, filename)
            else:
                ensemble_path = os.path.join(ensemble_dir, filename)

        os.makedirs(query_dir, exist_ok=True)

        df = pd.DataFrame(columns = ["query_no", "query", "cardinality_true"])
        i = 0
        with open(query_path, "w") as f:
            for subqueries in query:
                for subquery in subqueries:
                    f.write(re.sub(r"\/\*.*?\*\/", "", subquery.sql(use_as=False)).strip() + "\n")
                    df.loc[i] = [i, subquery.sql(), 1]
                    i += 1

        df.to_csv(ground_truth_path, index=False)

        if self._ensemble_strategy == "rdc_based":
            evaluate_commands = ["python", "estimators/deepdb/source/maqp.py",
                                "--evaluate_cardinalities",
                                "--rdc_spn_selection",
                                "--max_variants", "1",
                                "--pairwise_rdc_path", str(pairwise_rdc_path),
                                "--dataset", self._schema.name(),
                                "--target_path", str(result_path),
                                "--ensemble_location", str(ensemble_path),
                                "--query_file_location", str(query_path),
                                "--ground_truth_file_location", str(ground_truth_path),
                                ]
        else:
            evaluate_commands = ["python", "estimators/deepdb/source/maqp.py",
                                "--evaluate_cardinalities",
                                "--max_variants", "1",
                                "--dataset", self._schema.name(),
                                "--target_path", str(result_path),
                                "--ensemble_location", str(ensemble_path),
                                "--query_file_location", str(query_path),
                                "--ground_truth_file_location", str(ground_truth_path),
                                ]
        if "taskset_cpus" in self._config and self._config["taskset_cpus"] is not None:
            evaluate_commands = ["taskset", "-c", self._config["taskset_cpus"]] + evaluate_commands
        
        logger.write(" ".join(evaluate_commands) + "\n")
        subprocess.run(evaluate_commands)

        df = pd.read_csv(result_path)

        est_all = df["cardinality_predict"].tolist()
        latencies_all = (df["latency_ms"] / 1000).tolist()

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
        spn_ensembles_dir = os.path.join(self._result_folder, "spn_ensembles")
        model_size_mb = 0.0
        for file in os.listdir(spn_ensembles_dir):
            model_size_mb += os.path.getsize(os.path.join(spn_ensembles_dir, file)) / (1024 * 1024)
        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)

    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "DeepDBEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device
        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)
        deepdb = DeepDBEstimator(schema, config, dir)
        deepdb._statistics = statistics
        return deepdb

    def train_update(self) -> None:
        self.set_ratio_min_instance_slice()

        log = Logger(os.path.join(self._result_folder, "train.log"))

        # Generate hdf files from csvs.
        db_name = self._schema.name().split('_')[0] if self._schema.name().startswith("imdb_") else self._schema.name()
        gen_hdf_time = []
        for whole_data in [True, False]:
            if not whole_data and self._update_strategy == 'filter':
                continue
            hdf_dir_name = 'hdf' if whole_data else 'hdf_init'
            data_dir_name = db_name if whole_data else f"{db_name}/{db_name}_{self._perc}"
            hdf_commands = ["python", "estimators/deepdb/source/maqp.py", 
                            "--generate_hdf",
                            "--dataset", self._schema.name(),
                            "--csv_seperator", "|",
                            "--csv_path", str(os.path.join("./datasets", data_dir_name)),
                            "--hdf_path", str(os.path.join(self._result_folder, hdf_dir_name)),
                            "--max_rows_per_hdf_file", "100000000"]
            if not whole_data:
                hdf_commands.extend(
                    ['--prev_meta_data_path', str(os.path.join(self._result_folder, 'hdf', 'meta_data.pkl'))]
                )
            log.write(" ".join(hdf_commands) + "\n")
            subprocess.run(hdf_commands)
            with open(os.path.join(self._result_folder, hdf_dir_name, "build_time_hdf.txt"), 'r') as f:
                cur_gen_hdf_time = int(f.readline().strip())
                gen_hdf_time.append(cur_gen_hdf_time)
        log.write("[Gen HDF] " + str(gen_hdf_time) + "\n")
        

        # Learn ensemble
        ensemble_commands = ["python", "estimators/deepdb/source/maqp.py",
                             "--evaluate_update",
                             "--dataset", self._schema.name(),
                             "--samples_per_spn", "100000000", "100000000", "1000000", "1000000", "1000000",
                             "--ensemble_strategy", "relationship",
                             "--hdf_path", str(os.path.join(self._result_folder, "hdf")),
                             "--max_rows_per_hdf_file", "100000000",
                             "--ensemble_path", str(os.path.join(self._result_folder, "spn_ensembles")),
                             "--database_name", db_name,
                             "--post_sampling_factor", "10", "10", "5", "1", "1",
                             "--update_strategy", self._update_strategy,
                             "--init_data_perc", str(self._perc),
                             ]
        if not self._update_strategy == 'filter':
            ensemble_commands.extend(
                ['--init_hdf_path', str(os.path.join(self._result_folder, 'hdf_init'))]
            )
        if "taskset_cpus" in self._config and self._config["taskset_cpus"] is not None:
            ensemble_commands = ["taskset", "-c", self._config["taskset_cpus"]] + ensemble_commands
        log.write(" ".join(ensemble_commands) + "\n")
        start_ensemble = perf_counter()
        subprocess.run(ensemble_commands)
        gen_ensemble_time = perf_counter() - start_ensemble
        gen_time = sum(gen_hdf_time) + gen_ensemble_time

        self.reset_ratio_min_instance_slice()

        with open(os.path.join(self._result_folder, 'spn_ensembles', 'training_time.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    tokens = line.split(":")
                    self._statistics[tokens[0].strip()] = float(tokens[1].strip())
            self._statistics['training_time'] = self._statistics['model_training_time'] + self._statistics['training_data_time']
            self._statistics['updating_time'] = self._statistics['model_updating_time'] + self._statistics['updating_data_time']
            

        log.write("[Training Time] " + str(self._statistics['training_time']) + "\n")
        log.write("[Updating Time] " + str(self._statistics['updating_time']) + "\n")
        
        log.close()
        
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        spn_ensembles_dir = os.path.join(self._result_folder, "spn_ensembles", "updated")
        model_size_mb = 0.0
        for file in os.listdir(spn_ensembles_dir):
            model_size_mb += os.path.getsize(os.path.join(spn_ensembles_dir, file)) / (1024 * 1024)
        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f, indent=4)
    
    def predict_update(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]],
                                                                   List[List[float]], List[List[float]]]:
                                                                   # stale card, stale time, updated card, updated time
        logger = Logger(os.path.join(self._result_folder, "predict.log"), mode="a")

        query_dir = os.path.join(self._result_folder, "query")
        query_path = os.path.join(query_dir, "queries.sql")
        ground_truth_path = os.path.join(query_dir, "ground_truth.csv")
        result_path = os.path.join(query_dir, "deep_db_estimates.csv")
        
        os.makedirs(query_dir, exist_ok=True)

        df = pd.DataFrame(columns = ["query_no", "query", "cardinality_true"])
        i = 0
        with open(query_path, "w") as f:
            for subqueries in query:
                for subquery in subqueries:
                    f.write(re.sub(r"\/\*.*?\*\/", "", subquery.sql(use_as=False)).strip() + "\n")
                    df.loc[i] = [i, subquery.sql(), 1]
                    i += 1

        df.to_csv(ground_truth_path, index=False)

        results_all = []
        
        for is_updated in [False, True]:
            ensemble_dir = os.path.join(self._result_folder, "spn_ensembles", "updated" if is_updated else "stale")
            for filename in os.listdir(ensemble_dir):
                if filename != "pairwise_rdc.pkl" and filename.endswith(".pkl"):
                    ensemble_path = os.path.join(ensemble_dir, filename)
                    break
            evaluate_commands = ["python", "estimators/deepdb/source/maqp.py",
                                "--evaluate_cardinalities",
                                "--max_variants", "1",
                                "--dataset", self._schema.name(),
                                "--target_path", str(result_path),
                                "--ensemble_location", str(ensemble_path),
                                "--query_file_location", str(query_path),
                                "--ground_truth_file_location", str(ground_truth_path),
                                ]
            if "taskset_cpus" in self._config and self._config["taskset_cpus"] is not None:
                evaluate_commands = ["taskset", "-c", self._config["taskset_cpus"]] + evaluate_commands
            
            logger.write(" ".join(evaluate_commands) + "\n")
            subprocess.run(evaluate_commands)

            df = pd.read_csv(result_path)

            est_all = df["cardinality_predict"].tolist()
            latencies_all = (df["latency_ms"] / 1000).tolist()

            subqueries_est, subqueries_latencies = [], []
            index = 0
            for subqueries in query:
                subqueries_est.append(est_all[index:index+len(subqueries)])
                subqueries_latencies.append(latencies_all[index:index+len(subqueries)])
                index += len(subqueries)
            
            results_all.extend([subqueries_est, subqueries_latencies])
        
        return tuple(results_all)
