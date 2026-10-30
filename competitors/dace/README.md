# DACE

This directory contains DACE's plan search and end-to-end evaluation code for `job_light`, `stats_ceb`, and `ergastf1`.

## Environment Setup

Run the commands below from the main repository root. Plan search requires an NVIDIA GPU with CUDA support. Install CMake 3.15 or later, Make, and C/C++ compilers with C++17 support before building.

```bash
conda create -n dace python=3.9.18
conda activate dace
pip install -r competitors/dace/requirements.txt
```

Install PilotScope in the `dace` environment and prepare PostgreSQL with `pg_hint_plan` and `pg_prewarm` as described in the [main README](../../README.md#pilotscope-and-postgresql). If you already have the PilotScope source, install it in this environment:

```bash
pip install -e /path/to/PilotScopeCore
```

Build the join-order enumerator with the active environment's Python:

```bash
cmake -S competitors/dace/csrc -B competitors/dace/csrc/build \
    -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build competitors/dace/csrc/build --parallel 8
```

## Prepare Datasets and Workloads

Prepare `datasets/` and `workloads/` as described in the [main README](../../README.md#datasets-and-workloads). The three workloads use the `imdb`, `stats`, and `ergastf1` databases, respectively. Set your PostgreSQL connection details in [connection_args.py](plan_generator/connection_args.py).

Use the same PostgreSQL settings as the main end-to-end experiments, including `geqo = off` and `max_parallel_workers_per_gather = 0`. Restart PostgreSQL after changing its configuration, and ensure no extended statistics from other experiments remain.

Copy the query and warmup files from the main repository:

```bash
mkdir -p competitors/dace/workloads/{job_light,stats_ceb,ergastf1}
cp workloads/job_light/{job_light,warmup}.sql competitors/dace/workloads/job_light/
cp workloads/stats_ceb/{stats_ceb,warmup}.sql competitors/dace/workloads/stats_ceb/
cp workloads/ergastf1/{ergastf1_queries,warmup}.sql competitors/dace/workloads/ergastf1/
```

The copied workloads and generated results are ignored by Git. The pretrained model `checkpoints/DACE_ours_no_imdb.ckpt` and its matching `checkpoints/feature_statistics.json` are included; no training or fine-tuning is needed.

## Reproduce Experimental Results

With the `dace` environment activated, generate plan hints for all three workloads using the `enumerate` strategy:

```bash
for query_set in job_light stats_ceb ergastf1; do
    python competitors/dace/e2e_eval.py --query_set "$query_set" \
        --strategy enumerate --planning --num_workers 64 --limit_num 500000
done
```

`--num_workers` sets the number of PostgreSQL worker processes; reduce it to fit your machine and PostgreSQL connection limit. `--limit_num` caps the candidate join orders per query; the enumerator samples when the plan space exceeds this limit.

Then execute the queries with the generated hints. Use `PG_DATA` and `pg_ctl` from the PostgreSQL installation described in the main README to restart the database before each workload, as in the original experiment:

```bash
for query_set in job_light stats_ceb ergastf1; do
    pg_ctl stop -D "$PG_DATA"
    pg_ctl start -D "$PG_DATA" -l "$PG_DATA/postgres.log"
    python competitors/dace/e2e_eval.py --query_set "$query_set" \
        --strategy enumerate --execution --query_timeout_seconds 18000
done
```

Results are saved to `competitors/dace/workloads/{query_set}/enumerate/`:

- `{query_set}_plan_hints.csv`: Query text, selected hints, and DACE inference time.
- `{query_set}_execution_results.csv`: Execution time, inference time, and their sum (`e2e_time`), in seconds.

After completing planning and execution for all three workloads, export their results to the `experiments` folder:

```bash
python competitors/dace/export_results.py
```
