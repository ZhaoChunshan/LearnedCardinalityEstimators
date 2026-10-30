# LpBound

## Environment Setup

Run the commands below from the main repository root. Install CMake 3.16 or later, Make, and C/C++ compilers with C++17 support before building.

```bash
conda create -n lpbound python=3.11
conda activate lpbound
pip install -r estimators/lpbound/requirements.txt
bash estimators/lpbound/scripts/build.sh --jobs 8
```

## Reproduce Experimental Results

Prepare `datasets/` and `workloads/` as described in the [main README](../../README.md#datasets-and-workloads). With the `lpbound` environment activated, run all five workloads:

```bash
python estimators/lpbound/scripts/run.py --all --workers 8 --resume
```

To run a single workload, replace `--all` with `--query-set`, for example:

```bash
python estimators/lpbound/scripts/run.py --query-set job_light --workers 8 --resume
```

Supported workloads are `job_light`, `stats_ceb`, `ergastf1`, `job_m_ours`, and `job_large_ours`.

`--workers` sets the number of Python processes used to parse subqueries, read precomputed statistics, construct and solve LPs with OR-Tools, and export the LP files. This stage produces cardinality estimates and prepares the LPs for C++ measurement. The reported `estimation_time` is measured in C++ and covers LP construction and solving.

Results are saved to `estimators/lpbound/results/external_exports/<query_set>/` as `estimation.csv`, `statistics.json`, and `run_manifest.json`.

To preview copying `estimation.csv` and `statistics.json` into the corresponding `experiments/` directories:

```bash
python estimators/lpbound/scripts/publish.py --all --exp-name default
```

Add `--write` to copy the files, and `--overwrite` to replace different existing results.
