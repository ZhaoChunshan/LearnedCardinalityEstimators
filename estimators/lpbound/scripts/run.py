#!/usr/bin/env python3
"""Run LpBound experiments using the parent project's datasets and workloads."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import fcntl
from itertools import islice
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from project import (ESTIMATOR_ROOT, REPO_ROOT, WORKLOADS, DATASETS, STAGES, PARAMETERS,
    EXPORT_COLUMNS, atomic_csv, atomic_json, code_signature, dataset_signature,
    digest_file, digest_object, ensure_base, export_dir, prepare_workload, source_file, work_dir)
from lp_format import parse_exported_lp
from lpbound.acyclic.lpbound import build_lpbound_statistics, estimate
from lpbound.config.lpbound_config import LpBoundConfig
from lpbound.config.benchmark_schema import load_benchmark_schema
from lpbound.duckdb_adapter.duckdb_manager import DatabaseManager

_CONFIG = _CONNECTION = _FOLDER = None


def config_for(query_set):
    return LpBoundConfig(benchmark_name=query_set, schema_path=str(work_dir(query_set) / "schema.json"), **PARAMETERS)


def raw_dir(query_set):
    return work_dir(query_set) / "raw_results"


def initialize_worker(query_set):
    global _CONFIG, _CONNECTION, _FOLDER
    _CONFIG = config_for(query_set)
    _CONNECTION = DatabaseManager(load_benchmark_schema(_CONFIG)).create_or_load_db(read_only=True)
    _FOLDER = work_dir(query_set)


def estimate_batch(rows):
    records = []
    for row in rows:
        record = dict(row)
        subquery_id = int(row["SubqueryID"])
        path = _FOLDER / "lp_programs" / f"id{subquery_id}.lp"
        try:
            value = float(estimate(row["SQL"], _CONFIG, dump_lp_program_file=str(path), connection=_CONNECTION))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid cardinality estimate {value}")
            true = int(row["TrueCardinality"])
            record.update(lpbound_Estimate=value, UpperBoundOK=value + max(1e-8, true * 1e-10) >= true,
                          LPFile=str(path.relative_to(_FOLDER)), LP_SHA256=digest_file(path), Status="ok", Error="")
        except Exception as error:
            record.update(lpbound_Estimate=math.nan, UpperBoundOK=False,
                          LPFile=str(path.relative_to(_FOLDER)), LP_SHA256="", Status="error", Error=f"{type(error).__name__}: {error}")
        records.append(record)
    return records


def collect_statistics(query_set):
    folder = raw_dir(query_set)
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    times = build_lpbound_statistics(config_for(query_set))
    wall = time.perf_counter() - start
    connection = DatabaseManager(load_benchmark_schema(config_for(query_set))).create_or_load_db(read_only=True)
    try:
        rows = connection.execute("SELECT COUNT(*) FROM norms").fetchone()[0]
    finally:
        connection.close()
    atomic_csv(folder / "statistics_collection_time.csv", pd.DataFrame([{"Benchmark": query_set, "CollectionTimeSeconds": wall,
        "TimedSQLSeconds": sum(times.values()), "PreparationAndOverheadSeconds": wall - sum(times.values()), **PARAMETERS}]))
    atomic_csv(folder / "statistics_collection_time_breakdown.csv", pd.DataFrame([{"Stage": k, "Seconds": v} for k, v in times.items()]))
    size = rows * (PARAMETERS["p_max"] + 2) * 8
    atomic_csv(folder / "statistics_space_usage.csv", pd.DataFrame([{"Benchmark": query_set, "NormRows": rows, "EstimatedBytes": size, "EstimatedMB": size / 1e6}]))
    print(f"{query_set}: statistics={wall:.3f}s, norms={rows}, size={size / 1e6:.6f} MB", flush=True)


def estimate_workload(query_set, resume, workers):
    folder = work_dir(query_set)
    (folder / "lp_programs").mkdir(parents=True, exist_ok=True)
    output_dir = raw_dir(query_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "cardinality.csv"
    journal = output_dir / "cardinality.partial.csv"
    completed = {}
    if resume:
        # Completed stages are reused by their artifact hashes. Only an interrupted
        # journal is eligible for row-level recovery; corrupted final CSVs are rebuilt.
        previous_path = journal
        if previous_path.exists():
            with previous_path.open(newline="") as previous:
                for row in csv.DictReader(previous):
                    program = folder / row.get("LPFile", "")
                    if row.get("Status") == "ok" and program.is_file() and digest_file(program) == row.get("LP_SHA256"):
                        completed[int(row["SubqueryID"])] = row
    # Rewrite the journal from validated successes; failed rows are retried.
    records = list(completed.values())
    columns = ["SubqueryID", "QueryID", "SubqueryOrdinal", "Tables", "NumTables", "SQL", "SourceSQL", "TrueCardinality", "PostgresEstimate",
               "lpbound_Estimate", "UpperBoundOK", "LPFile", "LP_SHA256", "Status", "Error"]
    metadata = json.loads((folder / "workload_metadata.json").read_text())
    start = time.perf_counter()
    with (folder / "workload.csv").open(newline="") as workload, journal.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        out.flush()
        pending = (row for row in csv.DictReader(workload) if int(row["SubqueryID"]) not in completed)

        def batches():
            while batch := list(islice(pending, 64)):
                yield batch

        def consume(batch):
            records.extend(batch)
            writer.writerows(batch)
            out.flush()
            if len(records) % 1024 < len(batch) or len(records) == metadata["subqueries"]:
                elapsed = time.perf_counter() - start
                print(f"{query_set}: estimated {len(records)}/{metadata['subqueries']} ({elapsed:.1f}s)", flush=True)

        if workers == 1:
            initialize_worker(query_set)
            try:
                for batch in batches():
                    consume(estimate_batch(batch))
            finally:
                _CONNECTION.close()
        else:
            # Keep submission bounded even for workloads with hundreds of thousands of LPs.
            from collections import deque
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(query_set,)) as pool:
                iterator = iter(batches())
                futures = deque()
                for batch in islice(iterator, workers * 2):
                    futures.append(pool.submit(estimate_batch, batch))
                while futures:
                    consume(futures.popleft().result())
                    batch = next(iterator, None)
                    if batch is not None:
                        futures.append(pool.submit(estimate_batch, batch))
    frame = pd.DataFrame(records)
    for column in ("SubqueryID", "QueryID", "SubqueryOrdinal", "NumTables", "TrueCardinality"):
        frame[column] = pd.to_numeric(frame[column])
    frame = frame.sort_values("SubqueryID").reset_index(drop=True)
    atomic_csv(result_path, frame)
    atomic_csv(output_dir / "manifest.csv", frame[["SubqueryID", "QueryID", "SubqueryOrdinal", "LPFile", "LP_SHA256", "Status"]])
    failures = frame[frame.Status != "ok"]
    if not failures.empty:
        print(failures[["SubqueryID", "Error"]].head(20).to_string(index=False))
        raise RuntimeError(f"{query_set}: {len(failures)} estimates failed; use --resume after correcting the cause")
    journal.unlink(missing_ok=True)


def prepare_cpp(query_set):
    folder = work_dir(query_set)
    manifest = pd.read_csv(raw_dir(query_set) / "manifest.csv")
    if not manifest.Status.eq("ok").all():
        raise ValueError("Cannot prepare C++ inputs for failed estimates")
    output = folder / "cpp_inputs"
    output.mkdir(parents=True, exist_ok=True)
    packed = output / "models.jsonl"
    temporary = packed.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w") as f:
            for row in manifest.itertuples(index=False):
                program = folder / row.LPFile
                if digest_file(program) != row.LP_SHA256:
                    raise ValueError(f"LP hash mismatch at SubqueryID={row.SubqueryID}")
                constraints, variables, objective = parse_exported_lp(program)
                json.dump({"subquery_id": int(row.SubqueryID), "query_id": int(row.QueryID), "subquery_ordinal": int(row.SubqueryOrdinal),
                           "constraints": constraints, "variables": variables, "objective": objective}, f, separators=(",", ":"))
                f.write("\n")
        temporary.replace(packed)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    atomic_json(output / "cpp_manifest.json", {"benchmark": query_set, "format": "jsonl", "subqueries_file": "models.jsonl", "count": len(manifest), "models_sha256": digest_file(packed)})
    print(f"{query_set}: prepared {len(manifest)} exact LP models", flush=True)


def run_timing(query_set, runs, warmup):
    executable = ESTIMATOR_ROOT / "work/build/lpbound_parallel"
    if not executable.is_file():
        raise FileNotFoundError("C++ executable is missing. Run bash estimators/lpbound/scripts/build.sh")
    output = raw_dir(query_set)
    output.mkdir(parents=True, exist_ok=True)
    manifest = work_dir(query_set) / "cpp_inputs/cpp_manifest.json"
    model_metadata = json.loads(manifest.read_text())
    if digest_file(manifest.parent / model_metadata["subqueries_file"]) != model_metadata["models_sha256"]:
        raise ValueError("C++ model input hash mismatch")
    subprocess.run([str(executable), "--sequential", "--manifest", str(manifest), "--output-dir", str(output),
                    "--warmup-runs", str(warmup), "--runs", str(runs), "--threads", "1"], check=True)


def validate_results(query_set, runs):
    folder = work_dir(query_set)
    metadata = json.loads((folder / "workload_metadata.json").read_text())
    cardinality = pd.read_csv(raw_dir(query_set) / "cardinality.csv")
    workload = pd.read_csv(folder / "workload.csv")
    expected = metadata["subqueries"]
    if len(cardinality) != expected or not np.array_equal(cardinality.SubqueryID, np.arange(1, expected + 1)):
        raise ValueError("Incomplete or reordered cardinality results")
    if not cardinality.Status.eq("ok").all() or not cardinality.UpperBoundOK.astype(str).str.lower().eq("true").all():
        raise ValueError("Failed estimates or upper-bound violations")
    for field in ("SubqueryID", "QueryID", "SubqueryOrdinal", "SQL", "SourceSQL", "TrueCardinality"):
        if not cardinality[field].equals(workload[field]):
            raise ValueError(f"Cardinality source mismatch: {field}")
    if not np.isfinite(cardinality.lpbound_Estimate).all() or not cardinality.lpbound_Estimate.gt(0).all():
        raise ValueError("Non-finite or non-positive estimates")
    timing = pd.read_csv(raw_dir(query_set) / "cpp_sequential_raw.csv")
    if len(timing) != expected * runs or timing.duplicated(["SubqueryID", "RunID"]).any():
        raise ValueError("Missing or duplicate timing records")
    if set(timing.RunID) != set(range(1, runs + 1)) or not timing.SolverStatus.eq("Optimal").all():
        raise ValueError("Unexpected runs or non-optimal solves")
    expected_ids = np.arange(1, expected + 1)
    for _, group in timing.groupby("RunID"):
        if not np.array_equal(group.SubqueryID, expected_ids):
            raise ValueError("A measured run does not cover the complete manifest in order")
    merged = timing.merge(cardinality[["SubqueryID", "QueryID", "SubqueryOrdinal", "lpbound_Estimate"]], on="SubqueryID", validate="many_to_one", suffixes=("", "_expected"))
    if not (merged.QueryID.eq(merged.QueryID_expected) & merged.SubqueryOrdinal.eq(merged.SubqueryOrdinal_expected)).all():
        raise ValueError("Timing query identity mismatch")
    error = (merged.LPObjective - np.log2(merged.lpbound_Estimate)).abs()
    if not np.isfinite(error).all() or error.gt(1e-4).any():
        raise ValueError(f"Python/C++ objective mismatch: max error {error.max()}")
    if not timing[["BuildTimeNs", "SolveTimeNs", "TotalTimeNs"]].ge(0).all().all():
        raise ValueError("Negative timing values")
    outermost = cardinality.SubqueryOrdinal.eq(cardinality.groupby("QueryID").SubqueryOrdinal.transform("max"))
    if cardinality.QueryID.nunique() != metadata["parents"] or not outermost.groupby(cardinality.QueryID).sum().eq(1).all():
        raise ValueError("Invalid parent-query coverage")
    for filename, column in (("statistics_collection_time.csv", "CollectionTimeSeconds"), ("statistics_space_usage.csv", "EstimatedMB")):
        frame = pd.read_csv(raw_dir(query_set) / filename)
        if len(frame) != 1 or not np.isfinite(frame[column].iloc[0]) or frame[column].iloc[0] < 0:
            raise ValueError(f"Invalid {filename}")
    report = {"passed": True, "parents": metadata["parents"], "subqueries": expected, "timed_solves": len(timing),
              "runs": runs, "max_objective_error": float(error.max())}
    atomic_json(folder / "validation.json", report)
    print(f"{query_set}: validation passed; cardinalities={expected}, timed_solves={len(timing)}", flush=True)
    return cardinality, timing, report


def export_results(query_set, runs, state):
    cardinality, timing, report = validate_results(query_set, runs)
    medians = timing.groupby("SubqueryID").TotalTimeNs.median()
    estimated = cardinality.lpbound_Estimate.astype(float)
    true = cardinality.TrueCardinality
    est_for_q, true_for_q = estimated.clip(lower=1), true.clip(lower=1)
    frame = pd.DataFrame({"index": cardinality.QueryID - 1,
        "is_outer_most": cardinality.SubqueryOrdinal.eq(cardinality.groupby("QueryID").SubqueryOrdinal.transform("max")),
        "estimated_card": estimated, "true_card": true,
        "q_error": np.maximum(est_for_q / true_for_q, true_for_q / est_for_q),
        "estimation_time": cardinality.SubqueryID.map(medians) / 1e9,
        "query_id": cardinality.QueryID, "sql": cardinality.SourceSQL}, columns=EXPORT_COLUMNS)
    folder = export_dir(query_set)
    atomic_csv(folder / "estimation.csv", frame)
    statistics = {"model_size": float(pd.read_csv(raw_dir(query_set) / "statistics_space_usage.csv").EstimatedMB.iloc[0]),
                  "training_time": float(pd.read_csv(raw_dir(query_set) / "statistics_collection_time.csv").CollectionTimeSeconds.iloc[0])}
    atomic_json(folder / "statistics.json", statistics)
    provenance = {"query_set": query_set, "input_fingerprint": state["fingerprint"], "inputs": state["inputs"],
        "validation": report, "parameters": PARAMETERS, "warmup_runs": state["warmup_runs"],
        "units": {"model_size": "decimal MB", "training_time": "seconds", "estimation_time": "seconds"},
        "measurement": {"estimation_time": "median of C++ build-and-solve; excludes SQL parsing, statistics fetch and JSON I/O",
                        "training_time": "wall clock statistics construction; excludes CSV import", "threads": 1, "presolve": "off"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {name: digest_file(folder / name) for name in ("estimation.csv", "statistics.json")}}
    atomic_json(folder / "run_manifest.json", provenance)
    print(f"{query_set}: exported {len(frame)} rows to {folder.relative_to(REPO_ROOT)}", flush=True)


def output_files(query_set, stage):
    folder, raw = work_dir(query_set), raw_dir(query_set)
    files = {
        "prepare": [folder / n for n in ("workload.csv", "schema.json", "workload_metadata.json")],
        "data": [ESTIMATOR_ROOT / "work/databases" / f"{DATASETS[query_set]}_base{suffix}" for suffix in (".json", ".duckdb")],
        "statistics": [folder / "statistics.duckdb"] + [raw / n for n in ("statistics_collection_time.csv", "statistics_collection_time_breakdown.csv", "statistics_space_usage.csv")],
        "estimate": [raw / "cardinality.csv", raw / "manifest.csv"],
        "prepare-cpp": [folder / "cpp_inputs/cpp_manifest.json", folder / "cpp_inputs/models.jsonl"],
        "timing": [raw / "cpp_sequential_raw.csv"],
        "validate": [folder / "validation.json"],
        "export": [export_dir(query_set) / n for n in ("estimation.csv", "statistics.json", "run_manifest.json")],
    }
    return files[stage]


def run_workload(query_set, args, code):
    folder = work_dir(query_set)
    folder.mkdir(parents=True, exist_ok=True)
    signature = dataset_signature(DATASETS[query_set])
    template = ESTIMATOR_ROOT / "config/schemas" / f"{query_set}_schema.json"
    inputs = {"workload_sha256": digest_file(source_file(query_set)), "schema_sha256": digest_file(template),
              "dataset": signature, "code": code, "parameters": PARAMETERS, "runs": args.runs, "warmup_runs": args.warmup_runs}
    fingerprint = digest_object(inputs)
    state_path = folder / "state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    matching = previous.get("fingerprint") == fingerprint
    selected = args.stage or list(STAGES)
    if args.stage and selected != ["prepare"] and not matching:
        raise ValueError("Inputs/configuration changed or no matching run exists; run the full pipeline first")
    state = previous if matching else {"fingerprint": fingerprint, "inputs": inputs, "done": {}, "runs": args.runs, "warmup_runs": args.warmup_runs}
    atomic_json(state_path, state)

    verified_hashes = {}

    def artifact_matches(stage):
        recorded = state["done"].get(stage)
        files = output_files(query_set, stage)
        if not recorded or not all(p.is_file() for p in files):
            return False
        for path in files:
            stat = path.stat()
            identity = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if identity not in verified_hashes:
                verified_hashes[identity] = digest_file(path)
            if verified_hashes[identity] != recorded.get(str(path.relative_to(ESTIMATOR_ROOT))):
                return False
        return True

    actions = {"prepare": lambda: prepare_workload(query_set), "data": lambda: ensure_base(query_set, signature),
               "statistics": lambda: collect_statistics(query_set),
               "estimate": lambda: estimate_workload(query_set, args.resume and matching, args.workers),
               "prepare-cpp": lambda: prepare_cpp(query_set), "timing": lambda: run_timing(query_set, args.runs, args.warmup_runs),
               "validate": lambda: validate_results(query_set, args.runs), "export": lambda: export_results(query_set, args.runs, state)}
    for stage in selected:
        index = STAGES.index(stage)
        if args.resume and artifact_matches(stage):
            print(f"{query_set}: reuse completed stage {stage}", flush=True)
            continue
        if stage != "prepare" and any(not artifact_matches(required) for required in STAGES[:index]):
            raise ValueError(f"{query_set}: missing or changed prerequisite; run full pipeline with --resume")
        for later in STAGES[index:]:
            state["done"].pop(later, None)
        atomic_json(state_path, state)
        print(f"{query_set}: stage {stage}", flush=True)
        actions[stage]()
        state["done"][stage] = {str(p.relative_to(ESTIMATOR_ROOT)): digest_file(p) for p in output_files(query_set, stage)}
        atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--query-set", action="append", choices=WORKLOADS)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--stage", action="append", choices=STAGES, help="Run only selected stages; default: full pipeline")
    parser.add_argument("--resume", action="store_true", help="Reuse only artifacts matching current source/configuration hashes")
    parser.add_argument("--workers", type=int, default=1, help="Python LP generation workers; C++ timing remains serial")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.runs < 1 or args.warmup_runs < 0:
        parser.error("workers/runs must be positive and warmup-runs nonnegative")
    if args.stage and args.stage != sorted(set(args.stage), key=STAGES.index):
        parser.error("stages must be unique and in pipeline order")
    code = code_signature()
    # One pipeline owns the shared import cache and workload artifacts at a time.
    lock_path = ESTIMATOR_ROOT / "work/pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another LpBound pipeline is running")
        for query_set in WORKLOADS if args.all else args.query_set:
            run_workload(query_set, args, code)


if __name__ == "__main__":
    main()
