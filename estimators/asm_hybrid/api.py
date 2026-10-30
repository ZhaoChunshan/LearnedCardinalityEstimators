import os
import json
import pickle
import shutil
import subprocess
import sys
import numpy as np

from query.sql_query import SQLQuery
from schema.sql_schema import SQLSchema
from common.logger import Logger
from typing import Dict, List, Tuple


class ASMHybridEstimator:
    def __init__(self, schema: SQLSchema, config: Dict, result_folder: str) -> None:
        self._schema = schema
        self._config = config
        self._device = config["device"]
        self._env = os.environ.copy()
        if self._device[:5] == 'cuda:':
            self._env["CUDA_VISIBLE_DEVICES"] = self._device[5:].strip()
        self._result_folder = result_folder
        self._threshold = config.get("fail_rate_threshold", 0.99)

        # Model loaded from sibling asm/ directory
        parent = os.path.dirname(self._result_folder)     # .../asm_hybrid
        exp_name = os.path.basename(self._result_folder)   # default
        self._asm_dir = os.path.join(os.path.dirname(parent), "asm", exp_name)
        self._asm_data_dir = os.path.join(self._asm_dir, "asm_data")

        self._queries_file_path = os.path.join(self._result_folder, f"{schema.name()}_queries.pkl")
        self._subqueries_file_path = os.path.join(self._result_folder, f"{schema.name()}_sub_queries.pkl")
        self._predicate_dir = os.path.join(self._result_folder, "predicate")
        self._statistics = {"model_size": None, "training_time": 0.0}

    def predict(self, query: List[List[SQLQuery]], pg_est_cards: List[List[float]]) \
            -> Tuple[List[List[float]], List[List[float]]]:
        logger = Logger(os.path.join(self._result_folder, "predict.log"), mode='a')
                
        dataset = "imdb-light" if self._schema.name() == 'imdb' else self._schema.name()

        all_queries: Dict[str, str] = dict()
        all_sub_plan_queries_str: Dict[str, List[Tuple[str]]] = dict()
        sorted_subqueries_idx: List[List[int]] = []

        os.makedirs(self._predicate_dir, exist_ok=True)

        for i, subqueries in enumerate(query):
            query_name = f"{dataset}_{i+1:03}"

            sorted_idx = sorted(list(range(len(subqueries))),
                                key=lambda x: (len(subqueries[x].nodes()),
                                               subqueries[x].nodes()[0].table().try_alias()))
            sorted_subqueries_idx.append(sorted_idx)

            all_queries[query_name] = subqueries[sorted_idx[-1]].sql()

            all_predicates = dict()
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

        # Run ASM subprocess (model paths point to sibling asm/ directory)
        commands = [sys.executable, "run_experiment.py",
                    "--dataset", dataset,
                    "--evaluate",
                    "--ar_path", str(os.path.join("../../..", self._asm_dir, "{}-single-{}.tar")),
                    "--model_path", str(os.path.join("../../..", self._asm_dir, f"model_{dataset}.pkl")),
                    "--config_path", "{}-single-{}_infer",
                    "--query_file", str(os.path.join("../../..", self._queries_file_path)),
                    "--query_sub_plan_file", str(os.path.join("../../..", self._subqueries_file_path)),
                    "--query_predicate_location", str(os.path.join("../../..", self._predicate_dir)),
                    "--save_folder", str(os.path.join("../../..", self._result_folder, "result")),
                    "--asm_data_path", str(os.path.join("../../..", self._asm_data_dir)),
                    "--exact_support"
                    ]
        logger.write(f"Running ASM subprocess with commands: {' '.join(commands)}\n")
        subprocess.run(commands, cwd="estimators/asm/source", env=self._env, check=True)

        # Exact per-alias support of every sampled tuple.  The masks are
        # measured against the in-memory source DataFrames in ASM and already
        # respect the factorized representation of sampled join columns.
        masks_path = os.path.join(self._result_folder, "result", "exact_failure_masks.pkl")
        with open(masks_path, "rb") as f:
            exact_failure_masks: Dict[str, Dict[str, np.ndarray]] = pickle.load(f)

        # Build output
        queries_estimated_card: List[List[float]] = [[] for _ in range(len(query))]
        queries_latencies: List[List[float]] = [[] for _ in range(len(query))]

        fallback_rates: List[List[float]] = []
        fallback_decisions: List[List[bool]] = []
        alias_failure_rates: Dict[str, Dict[str, float]] = dict()

        for i, subqueries in enumerate(query):
            query_name = f"{dataset}_{i+1:03}"
            result_file = os.path.join(self._result_folder, "result", query_name)
            with open(result_file, "r") as f:
                lines = f.readlines()
                cards = [float(line.strip()) for line in lines[:len(lines) - 1]]
                est_time = float(lines[-1].strip()[len("Time: "):])

            num_sub = len(subqueries)
            if len(cards) != num_sub:
                raise ValueError(
                    f"ASM returned {len(cards)} cards for {query_name}, expected {num_sub}")

            subquery_cards = [0.0 for _ in range(num_sub)]
            subquery_latency = [0.0 for _ in range(num_sub)]
            for j, card in enumerate(cards):
                subquery_cards[sorted_subqueries_idx[i][j]] = card

            query_masks = exact_failure_masks[query_name]
            alias_failure_rates[query_name] = {
                alias: float(np.asarray(mask, dtype=np.bool_).mean())
                for alias, mask in query_masks.items()
            }
            query_rates, query_decisions = [], []
            for subquery_idx, subquery in enumerate(subqueries):
                aliases = [node.table().alias() for node in subquery.nodes()]
                # Single-table estimates are evaluated separately by the AR;
                # join-sampling failure is therefore not meaningful for them.
                if len(aliases) == 1:
                    failure_rate = 0.0
                    fallback = False
                else:
                    masks = [np.asarray(query_masks[alias], dtype=np.bool_)
                             for alias in aliases]
                    sample_sizes = {len(mask) for mask in masks}
                    if len(sample_sizes) != 1:
                        raise ValueError(
                            f"Inconsistent exact-mask lengths for {query_name}: {sample_sizes}")
                    failure_rate = float(np.logical_or.reduce(masks).mean())
                    fallback = failure_rate > self._threshold
                    if fallback:
                        subquery_cards[subquery_idx] = pg_est_cards[i][subquery_idx]
                query_rates.append(failure_rate)
                query_decisions.append(fallback)

            # ASM has already evaluated the complete parent query even when a
            # subquery is replaced, so retain its measured end-to-end latency.
            subquery_latency[sorted_subqueries_idx[i][-1]] = est_time
            queries_estimated_card[i] = subquery_cards
            queries_latencies[i] = subquery_latency
            fallback_rates.append(query_rates)
            fallback_decisions.append(query_decisions)

        # Log both alias-level evidence and the actual subquery decisions.
        logger.write("Exact Alias Failure Rate Per Query:\n")
        for query_name, rates in alias_failure_rates.items():
            formatted = " | ".join(
                f"{alias}: {rate:.6f}" for alias, rate in sorted(rates.items()))
            logger.write(f"  {query_name}  {formatted}\n")

        flat_rates = np.asarray([rate for rates in fallback_rates for rate in rates])
        logger.write("Subquery Failure Rate Percentiles: "
                     f"MIN {flat_rates.min():.4f} | "
                     f"25th {np.percentile(flat_rates, 25):.4f} | "
                     f"50th {np.percentile(flat_rates, 50):.4f} | "
                     f"75th {np.percentile(flat_rates, 75):.4f} | "
                     f"90th {np.percentile(flat_rates, 90):.4f} | "
                     f"95th {np.percentile(flat_rates, 95):.4f} | "
                     f"99th {np.percentile(flat_rates, 99):.4f} | "
                     f"MAX {flat_rates.max():.4f}\n")
        n_fallback = sum(sum(decisions) for decisions in fallback_decisions)
        n_subqueries = sum(len(decisions) for decisions in fallback_decisions)
        n_parent_fallback = sum(any(decisions) for decisions in fallback_decisions)
        logger.write(
            f"Fallback count (fail_rate > {self._threshold}): "
            f"{n_fallback} / {n_subqueries} subqueries in "
            f"{n_parent_fallback} / {len(query)} parent queries\n")
        logger.close()

        return queries_estimated_card, queries_latencies

    def save(self) -> None:
        with open(os.path.join(self._result_folder, "config.json"), "w") as f:
            json.dump({k: v for k, v in self._config.items() if k != "device"}, f, indent=4)
        # Copy statistics from sibling asm dir
        asm_stats_path = os.path.join(self._asm_dir, "statistics.json")
        if os.path.exists(asm_stats_path):
            shutil.copy(asm_stats_path, os.path.join(self._result_folder, "statistics.json"))

    @staticmethod
    def load_from_dir(schema: SQLSchema, dir: str, device: str = None) -> "ASMHybridEstimator":
        with open(os.path.join(dir, "config.json"), "r") as f:
            config = json.load(f)
        if device is not None:
            config["device"] = device
        return ASMHybridEstimator(schema, config, dir)
