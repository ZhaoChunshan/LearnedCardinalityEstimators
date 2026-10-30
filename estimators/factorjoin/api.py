import os 
import json
import pickle
import subprocess


from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema  
from common.logger import Logger
from typing import List, Tuple, Dict, Set


SubqueryKey = Tuple[str, ...]


def _subquery_key(query: SQLQuery) -> SubqueryKey:
    aliases = [node.table().try_alias() for node in query.nodes()]
    if len(aliases) != len(set(aliases)):
        raise ValueError(f"FactorJoin does not support repeated table aliases in one subquery: {aliases}")
    return tuple(sorted(aliases))


def _query_adjacency(query: SQLQuery) -> Dict[str, Set[str]]:
    alias_by_table = {node.table(): node.table().try_alias() for node in query.nodes()}
    adjacency = {alias: set() for alias in alias_by_table.values()}

    for edge in query.edges():
        if edge.is_pk_fk():
            left_table = edge.foreign_key().table
            right_table = edge.foreign_key().referenced_table
        else:
            left_table = edge.foreign_key()[0].table
            right_table = edge.foreign_key()[1].table

        try:
            left_alias = alias_by_table[left_table]
            right_alias = alias_by_table[right_table]
        except KeyError as error:
            raise ValueError("A FactorJoin query edge references a table outside the query") from error

        adjacency[left_alias].add(right_alias)
        adjacency[right_alias].add(left_alias)

    return adjacency


def _query_signature(query: SQLQuery) -> Tuple[Tuple, Tuple]:
    """Return an order-independent signature for a query's joins and filters."""
    alias_by_table = {node.table(): node.table().try_alias() for node in query.nodes()}
    conditions = []

    for edge in query.edges():
        if edge.is_pk_fk():
            foreign_key = edge.foreign_key()
            endpoints = (
                (alias_by_table[foreign_key.table], foreign_key.column.name()),
                (alias_by_table[foreign_key.referenced_table], foreign_key.referenced_column.name()),
            )
        else:
            left_key, right_key = edge.foreign_key()
            endpoints = (
                (alias_by_table[left_key.table], left_key.column.name()),
                (alias_by_table[right_key.table], right_key.column.name()),
            )
        conditions.append(tuple(sorted(endpoints)))

    join_signature = tuple(sorted(conditions))
    predicate_signature = tuple(sorted(
        (node.table().try_alias(), predicate.column.name(), predicate.operator, repr(predicate.value))
        for node in query.nodes()
        for predicate in node.predicates()
    ))
    return join_signature, predicate_signature


def _is_connected(nodes: Set[str], adjacency: Dict[str, Set[str]]) -> bool:
    if not nodes:
        return False

    visited = set()
    pending = [next(iter(nodes))]
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend((adjacency[node] & nodes) - visited)
    return visited == nodes


def _build_left_deep_closure(subqueries: List[SQLQuery]) \
        -> Tuple[List[Tuple[str, str]], List[str], List[str], int]:
    """Build executable FactorJoin plans and an explicit result key for every target subquery."""
    if not subqueries:
        raise ValueError("FactorJoin received an empty query group")

    target_keys = [_subquery_key(subquery) for subquery in subqueries]
    signatures_by_key = {}
    for key, subquery in zip(target_keys, subqueries):
        signature = _query_signature(subquery)
        if key in signatures_by_key and signatures_by_key[key] != signature:
            raise ValueError(f"FactorJoin received conflicting query semantics for relation set {key}")
        signatures_by_key[key] = signature

    outer_index = max(range(len(subqueries)), key=lambda i: (len(target_keys[i]), target_keys[i]))
    outer_key = set(target_keys[outer_index])
    preferred_keys = set(target_keys)
    plans_by_key: Dict[SubqueryKey, Tuple[str, SubqueryKey]] = {}

    def add_plan(key: SubqueryKey, adjacency: Dict[str, Set[str]]) -> None:
        if len(key) == 1 or key in plans_by_key:
            return

        nodes = set(key)
        candidates = []
        for left_alias in key:
            remainder = nodes - {left_alias}
            remainder_key = tuple(sorted(remainder))
            if not (adjacency[left_alias] & remainder):
                continue
            if not _is_connected(remainder, adjacency):
                continue
            candidates.append((remainder_key not in preferred_keys, left_alias, remainder_key))

        if not candidates:
            raise ValueError(f"Cannot construct a cross-product-free left-deep plan for subquery {key}")

        _, left_alias, remainder_key = min(candidates)
        add_plan(remainder_key, adjacency)
        plans_by_key[key] = (left_alias, remainder_key)

    for key, subquery in zip(target_keys, subqueries):
        if not set(key).issubset(outer_key):
            raise ValueError(f"Subquery {key} is not contained in outer query {tuple(sorted(outer_key))}")

        adjacency = _query_adjacency(subquery)
        if len(key) > 1 and not _is_connected(set(key), adjacency):
            raise ValueError(f"FactorJoin cannot evaluate disconnected subquery {key}")

        add_plan(key, adjacency)

    ordered_plan_keys = sorted(plans_by_key, key=lambda key: (len(key), key))
    plans = [
        (plans_by_key[key][0], " ".join(plans_by_key[key][1]))
        for key in ordered_plan_keys
    ]
    result_keys = [" ".join(key) for key in ordered_plan_keys]
    target_result_keys = [" ".join(key) for key in target_keys]
    return plans, result_keys, target_result_keys, outer_index


class FactorJoinEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._schema = schema
        self._config = config
        self._result_folder = result_folder
        self._derived_query_file = os.path.join(self._result_folder, f"{schema.name()}_queries.pkl")
        self._card_est_result_dir = os.path.join(self._result_folder, f"estimation_result")
        self._statistics = {"model_size": None, "training_time": None}
        
        if 'perc' in self._config:
            self._perc = self._config["perc"]

    def train(self) -> None:
        train_log = Logger(os.path.join(self._result_folder, "train.log"))

        n_bins = self._config["n_bins"]
        n_dim_dist = self._config["n_dim_dist"]
        bucket_method = "fixed_start_key" if self._schema.name() == 'imdb' else "greedy"
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()
        commands = ["python", "estimators/factorjoin/source/run_experiment.py",
                    "--dataset", dataset, 
                    "--generate_models",
                    "--data_path", "datasets/" + self._schema.name() + "/{}.csv", "--model_path", self._result_folder,
                    "--n_dim_dist", str(n_dim_dist), 
                    "--n_bins", str(n_bins),
                    "--bucket_method", bucket_method]

        train_log.write(" ".join(commands) + "\n")
        result = subprocess.run(commands, capture_output=True, text=True)
        stdout_output = result.stdout
        lines = stdout_output.splitlines()
        prefix = "Training completed: total training time is "
        total_time = 0.0
        for line in lines:
            if line.startswith(prefix):
                total_time = float(line[len(prefix):])
                break
        self._statistics["training_time"] = total_time
        train_log.write("[Total Time] %f\n" % total_time)
        train_log.close()

    def train_update(self) -> None:
        init_data_perc = self._perc
        assert(init_data_perc in [20, 40, 60, 80])
        train_log = Logger(os.path.join(self._result_folder, "train.log"))
        
        db = self._schema.name()
        assert db in ['imdb', 'stats']
        
        import pandas as pd
        if db == 'stats':
            timestamps_df = pd.read_csv('datasets/stats/timestamps_split/timestamps_by_percentage.csv')
            datatime_str = str(timestamps_df[timestamps_df['percentage'] == init_data_perc]['datetime'].values[0])
        else:
            timestamps_df = pd.read_csv('datasets/imdb/timestamps_split/years_by_percentage.csv')
            year_int = int(timestamps_df[timestamps_df['percentage'] == init_data_perc]['year'].values[0])
            datatime_str = f"{year_int}-01-01 00:00:00"
        
        dataset = "imdb-light" if db == 'imdb' else db
        bucket_method = "fixed_start_key" if self._schema.name() == 'imdb' else "greedy"
        commands = ["python", "estimators/factorjoin/source/run_experiment.py",
                    "--dataset", dataset, 
                    "--update_evaluate",
                    "--data_path", "datasets/" + db + "/{}.csv", "--model_path", self._result_folder,
                    "--n_dim_dist", str(self._config["n_dim_dist"]), 
                    "--n_bins", str(self._config["n_bins"]),
                    "--split_date", datatime_str,
                    "--bucket_method", bucket_method]
        train_log.write(" ".join(commands) + "\n")
        
        result = subprocess.run(commands, capture_output=True, text=True)
        stdout_output = result.stdout
        lines = stdout_output.splitlines()
        
        prefix_train = "training completed, took "
        prefix_update = "updating completed, took "
        training_time, updating_time = 0.0, 0.0
        for line in lines:
            if line.startswith(prefix_train):
                training_time = float(line[len(prefix_train):-3])
            elif line.startswith(prefix_update):
                updating_time = float(line[len(prefix_update):-3])
        self._statistics["training_time"] = training_time
        self._statistics["updating_time"] = updating_time
        
        
        model_path = [os.path.join(self._result_folder, f) for f in os.listdir(self._result_folder) if f.endswith(".pkl") and "updated_model" in f][0]
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        self._statistics["model_size"] = model_size_mb
        
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f)
        
        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f)
        train_log.write("[Training Time] %f\n" % training_time)
        train_log.write("[Updating Time] %f\n" % updating_time)
        train_log.close()

    def _prepare_query_payload(self, query: List[List[SQLQuery]], dataset: str) \
            -> Tuple[List[List[str]], List[int]]:
        all_queries: Dict[str, str] = {}
        all_sub_plan_queries_str: Dict[str, List[Tuple[str, str]]] = {}
        all_sub_plan_result_keys: Dict[str, List[str]] = {}
        target_result_keys: List[List[str]] = []
        outer_indices: List[int] = []

        for i, subqueries in enumerate(query):
            query_name = f"{dataset}_{i+1:03}"
            plans, plan_result_keys, targets, outer_index = _build_left_deep_closure(subqueries)

            all_queries[query_name] = subqueries[outer_index].sql()
            all_sub_plan_queries_str[query_name] = plans
            all_sub_plan_result_keys[query_name] = plan_result_keys
            target_result_keys.append(targets)
            outer_indices.append(outer_index)

        with open(self._derived_query_file, "wb") as f:
            pickle.dump((all_queries, all_sub_plan_queries_str, all_sub_plan_result_keys), f)

        return target_result_keys, outer_indices

    def _evaluate_model(self, model_path: str, dataset: str, query: List[List[SQLQuery]],
                        target_result_keys: List[List[str]], outer_indices: List[int]) \
            -> Tuple[List[List[float]], List[List[float]]]:
        commands = ["python", "estimators/factorjoin/source/run_experiment.py",
                    "--dataset", dataset,
                    "--evaluate",
                    "--model_path", model_path,
                    "--derived_query_file", self._derived_query_file,
                    "--estimation_result_folder", self._card_est_result_dir]
        print(" ".join(commands))
        subprocess.run(commands)

        queries_estimated_card: List[List[float]] = []
        queries_latencies: List[List[float]] = []

        for i, subqueries in enumerate(query):
            query_name = f"{dataset}_{i+1:03}"
            result_path = os.path.join(self._card_est_result_dir, query_name)
            with open(result_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines or not lines[-1].startswith("Time: "):
                raise ValueError(f"Invalid FactorJoin result file: {result_path}")
            est_time = float(lines[-1][len("Time: "):])

            estimates_by_key: Dict[str, float] = {}
            for line in lines[:-1]:
                if "||" not in line:
                    raise ValueError(f"FactorJoin result is missing an explicit subquery key: {line}")
                result_key, card = line.rsplit("||", 1)
                if result_key in estimates_by_key:
                    raise ValueError(f"Duplicate FactorJoin result key {result_key} in {result_path}")
                estimates_by_key[result_key] = float(card)

            missing_keys = [key for key in target_result_keys[i] if key not in estimates_by_key]
            if missing_keys:
                raise ValueError(f"FactorJoin did not evaluate target subqueries {missing_keys} in {result_path}")

            subquery_cards = [estimates_by_key[key] for key in target_result_keys[i]]
            subquery_latency = [0.0 for _ in subqueries]
            subquery_latency[outer_indices[i]] = est_time
            queries_estimated_card.append(subquery_cards)
            queries_latencies.append(subquery_latency)

        return queries_estimated_card, queries_latencies

    def predict(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]]]:
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()
        target_result_keys, outer_indices = self._prepare_query_payload(query, dataset)

        model_paths = [os.path.join(self._result_folder, f) for f in os.listdir(self._result_folder)
                       if f.endswith(".pkl") and "_queries" not in f and 'model' in f]
        if len(model_paths) == 1:
            model_path = model_paths[0]
        else:
            model_path = [f for f in model_paths if 'updated' in f][0]

        return self._evaluate_model(model_path, dataset, query, target_result_keys, outer_indices)

    def predict_update(self, query: List[List[SQLQuery]]) -> Tuple[List[List[float]], List[List[float]],
                                                                   List[List[float]], List[List[float]]]:
        # stale card, stale time, updated card, updated time
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()
        target_result_keys, outer_indices = self._prepare_query_payload(query, dataset)

        result_all = []
        for is_updated in [False, True]:
            if is_updated:
                model_path = [os.path.join(self._result_folder, f) for f in os.listdir(self._result_folder)
                              if f.endswith(".pkl") and "_queries" not in f and 'updated_model' in f][0]
            else:
                model_path = [os.path.join(self._result_folder, f) for f in os.listdir(self._result_folder)
                              if f.endswith(".pkl") and "_queries" not in f and 'model' in f and 'updated' not in f][0]

            estimated_cards, latencies = self._evaluate_model(
                model_path, dataset, query, target_result_keys, outer_indices
            )
            result_all.append(estimated_cards)
            result_all.append(latencies)

        return tuple(result_all)
    
    def save(self) -> None:
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f)
        model_path = [os.path.join(self._result_folder, f) for f in os.listdir(self._result_folder) if f.endswith(".pkl")][0]
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        self._statistics["model_size"] = model_size_mb

        with open(os.path.join(self._result_folder, "statistics.json"), "w") as f:
            json.dump(self._statistics, f)


    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "FactorJoinEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device
        with open(os.path.join(dir, "statistics.json"), "r") as f:
            statistics = json.load(f)

        factorjoin = FactorJoinEstimator(schema, config, dir)
        factorjoin._statistics = statistics
        return factorjoin
