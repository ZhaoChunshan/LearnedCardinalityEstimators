#!/usr/bin/env python3
"""Preview result copies; writing requires the explicit --write option."""
import argparse
import json
from pathlib import Path
import shutil

from project import (REPO_ROOT, ESTIMATOR_ROOT, WORKLOADS, DATASETS, EXPERIMENT_TYPES, digest_file,
                     code_signature, dataset_signature,
                     export_dir, source_file, work_dir)


def publication_plan(query_set, exp_name):
    source = export_dir(query_set)
    manifest_path = source / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    state = json.loads((work_dir(query_set) / "state.json").read_text())
    if not manifest["validation"]["passed"] or "export" not in state["done"]:
        raise ValueError(f"{query_set}: export is not validated/complete")
    if manifest["input_fingerprint"] != state["fingerprint"]:
        raise ValueError(f"{query_set}: export belongs to a different run")
    if digest_file(source_file(query_set)) != manifest["inputs"]["workload_sha256"]:
        raise ValueError(f"{query_set}: workload changed after export")
    if dataset_signature(DATASETS[query_set]) != manifest["inputs"]["dataset"]:
        raise ValueError(f"{query_set}: dataset changed after export")
    schema = ESTIMATOR_ROOT / "config/schemas" / f"{query_set}_schema.json"
    if digest_file(schema) != manifest["inputs"]["schema_sha256"] or code_signature() != manifest["inputs"]["code"]:
        raise ValueError(f"{query_set}: implementation or schema changed after export")
    for filename, expected in manifest["artifacts"].items():
        if digest_file(source / filename) != expected:
            raise ValueError(f"{query_set}: exported {filename} was modified")
    destination = REPO_ROOT / "experiments" / EXPERIMENT_TYPES[query_set] / query_set / "lpbound" / exp_name
    return [(source / filename, destination / filename) for filename in ("estimation.csv", "statistics.json")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--query-set", action="append", choices=WORKLOADS)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--exp-name", default="default")
    parser.add_argument("--write", action="store_true", help="Actually copy into experiments; without this flag only print the plan")
    parser.add_argument("--overwrite", action="store_true", help="Permit replacing different existing results when --write is used")
    args = parser.parse_args()
    if not args.exp_name or Path(args.exp_name).name != args.exp_name or args.exp_name in (".", ".."):
        parser.error("exp-name must be a single directory name")
    plan = []
    for query_set in WORKLOADS if args.all else args.query_set:
        plan.extend(publication_plan(query_set, args.exp_name))
    # Preflight every selected workload before any write.
    if args.write:
        for source, destination in plan:
            if destination.exists() and digest_file(destination) != digest_file(source) and not args.overwrite:
                raise FileExistsError(f"Existing result differs: {destination}; use --overwrite explicitly")
    for source, destination in plan:
        print(f"{'COPY' if args.write else 'PREVIEW'} {source.relative_to(REPO_ROOT)} -> {destination.relative_to(REPO_ROOT)}")
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)


if __name__ == "__main__":
    main()
