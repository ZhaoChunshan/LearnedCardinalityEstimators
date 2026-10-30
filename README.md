# How Good Are Learned Cardinality Estimators, Really? A Comprehensive Experimental Study

This repository contains the source code for the experiments in our paper, “How Good Are Learned Cardinality Estimators, Really? A Comprehensive Experimental Study,” accepted at SIGMOD 2027.

## Table of Contents

- [Project Structure](#project-structure)
- [Experimental Results and Visualization](#experimental-results-and-visualization)
- [Requirements](#requirements)
  - [PilotScope and PostgreSQL](#pilotscope-and-postgresql)
  - [Python Environments](#python-environments)
- [Datasets and Workloads](#datasets-and-workloads)
  - [Datasets](#datasets)
  - [Workloads](#workloads)
- [Reproduce Experimental Results](#reproduce-experimental-results)
  - [Overall Performance Evaluation](#overall-performance-evaluation)
    - [Evaluate Learned Cardinality Estimators](#evaluate-learned-cardinality-estimators)
    - [Evaluate Non-learning Estimators](#evaluate-non-learning-estimators)
    - [Evaluate Other ML-aided Query Optimization Methods](#evaluate-other-ml-aided-query-optimization-methods)
  - [Training Data Efficiency of Query-Driven Methods](#training-data-efficiency-of-query-driven-methods)
  - [Generalization under Query Workload Drift](#generalization-under-query-workload-drift)
  - [Scalability](#scalability)
  - [Dynamic Updates](#dynamic-updates)
- [Training Workload Generation](#training-workload-generation)
- [Source Code Origins](#source-code-origins)

## Project Structure

  + `common`: Shared utilities used throughout the project.
  + `competitors`: Other ML-aided query optimization methods evaluated in our study.
  + `cpp`: C++ code for query plan enumeration.
  + `estimators`: source code for all cardinality estimators in our study.
  + `figures`: Figures presented in the paper.
  + `plots`: Scripts for generating the paper’s figures.
  + `query`: Code for query parsing, query generation, and cardinality calculation.
  + `schema`: Dataset schema definitions and utilities for working with PostgreSQL from Python.
  + `scripts`: Useful utility scripts.
  + `synopsis`: Code for constructing database statistics, such as sample bitmaps and histograms.
  + `config.json`: Configuration for the evaluated estimators.
  + `experiment.py`: Main script for running experiments.
  + `generate_workload.py`: Main script for generating training workloads.
  + `.gitignore`: Git ignore rules.
  + `README.md`: Project documentation.

## Experimental Results and Visualization

All experimental results are available in [our data repository](https://osf.io/n5vkr/files/googledrive). The `experiments.tar.gz` archive includes all results, trained models, configurations, logs, and intermediate outputs. You can use the trained models to reproduce the cardinality estimates. Download the archive to the project root directory, then extract it:

```bash
tar -zxzf experiments.tar.gz
```

To generate all figures in the paper, install `numpy`, `pandas`, and `matplotlib`, then run:

```bash
python plots/plot.py
```

The figures are saved in `figures/`.

## Requirements

### PilotScope and PostgreSQL

We conduct the end-to-end experiments on PostgreSQL. We use [PilotScope](https://github.com/alibaba/pilotscope) to inject cardinality estimates into PostgreSQL's optimizer to generate query plans for execution. Please install PilotScope and PostgreSQL by following the [PilotScope documentation](https://woodybryant.github.io/PilotScopeDoc.io/references/install.html#install-all-components-from-scratch).

We summarize the main steps below.

#### Install PilotScope

First, install PilotScope. We recommend installing it in a separate Python environment. We will assume this environment is named `pilotscope` in subsequent steps.

```bash
# Ensure your system has the required libraries
apt-get update && apt-get install -y wget git bzip2 vim gcc openssh-server sudo

# Clone the repository
git clone --branch master https://github.com/alibaba/pilotscope.git PilotScopeCore

# Navigate to the PilotScope directory
cd PilotScopeCore

# Install the minimal package using pip
pip install -e .
```

Run `import pilotscope` in Python to verify the installation.

#### Install PostgreSQL

Next, install a modified version of PostgreSQL 13.1 that is compatible with PilotScope. First, install the requirements and download the source code:

```bash
# Ensure your system has the required libraries
apt-get update && apt-get install -y build-essential libreadline-dev zlib1g-dev bison flex gdb libssl-dev libbz2-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils libffi-dev liblzma-dev

# Clone the repository
git clone --branch pilotscope-postgresql https://github.com/alibaba/pilotscope.git PilotScopePostgreSQL
```

Then, set the installation and data paths for PostgreSQL:

```bash
# Example of setting PG_PATH
echo 'export PG_PATH=/path/to/install/PostgreSQL' >> ~/.bashrc

# Example of setting PG_DATA
echo 'export PG_DATA=/path/to/store/database/data' >> ~/.bashrc

# Update system paths
echo 'export PATH=$PG_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$PG_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Apply the changes
source ~/.bashrc
```

Compile and install PostgreSQL. We recommend compiling with the `-O2` flag for better performance:

```bash
# Navigate to the PostgreSQL source code directory
cd ~/PilotScopePostgreSQL/

# Run the configuration script
./configure --prefix=$PG_PATH --enable-depend --enable-cassert --enable-debug CFLAGS="-ggdb -O2"

# Compile and install PostgreSQL
make -j && make install
```

Install the necessary extensions for PilotScope:

```bash
# Install the PilotScope extension
sh install_extension.sh

# Recompile and install PostgreSQL
make -j && make install
```

Additionally, install the `pg_prewarm` extension, which is used for warming up PostgreSQL's buffer cache before end-to-end experiments:
```bash
# Install the pg_prewarm extension
cd ./contrib/pg_prewarm
make -j USE_PGXS=1
make USE_PGXS=1 install
```

#### Configure Database Connection Settings

Initialize the database:

```bash
cd $PG_PATH/bin
./initdb -D $PG_DATA
```

Enable the `pg_hint_plan` extension:

```bash
echo "shared_preload_libraries = 'pg_hint_plan'" >> "$PG_DATA/postgresql.conf"
```

Start the PostgreSQL database:

```
pg_ctl start -D $PG_DATA 
```

Finally, create a PostgreSQL user named `pilotscope` to simplify subsequent experiments. Run the following commands in `psql`:

```sql
CREATE USER pilotscope WITH PASSWORD 'your_password';
ALTER USER pilotscope WITH SUPERUSER CREATEDB CREATEROLE;
CREATE DATABASE pilotscope OWNER pilotscope;
```

### Python Environments

Each learned estimator requires its own Python environment. We provide a `requirements.txt` file for each model in `estimators/{estimator}/`. You can create a conda environment and install the required packages. To get started quickly, you only need to install the environments for the estimators you are interested in. Our subsequent examples use MSCN and BayesCard.

MSCN

```bash
conda create -n mscn python=3.8.19
conda activate mscn
pip install -r estimators/mscn/requirements.txt
```

RobustMSCN

```bash
conda create -n robustmscn python=3.8.19
conda activate robustmscn
pip install -r estimators/robustmscn/requirements.txt
```

JGMP

```bash
conda create -n jgmp python=3.8.19
conda activate jgmp
pip install -r estimators/jgmp/requirements.txt
```

ALECE

```bash
conda create -n alece python=3.8.19
conda activate alece
pip install -r estimators/alece/requirements.txt
```

PRICE

```bash
conda create -n price python=3.10.13
conda activate price
pip install -r estimators/price/requirements.txt 
```

DeepDB

```bash
conda create -n deepdb python=3.7.16
conda activate deepdb
pip install -r estimators/deepdb/requirements.txt 
```

NeuroCard

```bash
conda create -n neurocard python=3.7.7
conda activate neurocard
pip install -r estimators/neurocard/requirements.txt 
```

BayesCard

```bash
conda create -n bayescard python=3.7.16
conda activate bayescard
pip install -r estimators/bayescard/requirements.txt
```

FactorJoin

```bash
conda create -n factorjoin python=3.7.16
conda activate factorjoin
pip install -r estimators/factorjoin/requirements.txt
```

ASM

```bash
conda create -n asm python=3.7.16
conda activate asm
pip install -r estimators/asm/requirements.txt
```

## Datasets and Workloads

### Datasets

First, download and unpack `datasets.tar.gz` from [our data repository](https://osf.io/n5vkr/files/googledrive).
Then, import the datasets as follows:

```bash
# For imdb (this dataset is large; loading takes several minutes)
cd datasets/imdb
psql -U pilotscope
\i postgres_create_imdb.sql
\i foreign_keys.sql
\q
cd ../..

# For stats
cd datasets/stats
psql -U pilotscope
\i postgres_create_stats.sql
\i foreign_keys.sql
\q
cd ../..

# For ergastf1
cd datasets/ergastf1
psql -U pilotscope
\i postgres_create_ergastf1.sql
\i foreign_keys.sql
\q
cd ../..
```

### Workloads

Download and unpack `workloads.tar.gz` from [our data repository](https://osf.io/n5vkr/files/googledrive). Make sure the unpacked workload folder is named `workloads`. The experiments read queries, true cardinalities, and PostgreSQL estimates directly from `workloads/`. See `workloads/README.md` for details on the file formats.

## Reproduce Experimental Results

### Overall Performance Evaluation

#### Evaluate Learned Cardinality Estimators

First, run the cardinality estimation experiment to get estimates for all subqueries. Then, run the end-to-end execution experiment. The general command format is provided below. For brevity, we omit environment activation commands in subsequent examples.

```bash
# Cardinality estimation experiment
conda activate {estimator}
python experiment.py card --db {database} --query_set {queryset} --estimator {estimator} 

# End-to-end execution experiment
conda activate pilotscope
python experiment.py e2e --db {database} --query_set {queryset} --estimator {estimator}
```

For example, to test BayesCard on the `STATS-CEB` workload:

```bash
python experiment.py card --db stats --query_set stats_ceb --estimator bayescard
python experiment.py e2e --db stats --query_set stats_ceb --estimator bayescard
```

To save time on model training, you can download and unpack `experiments.tar.gz` from our [additional data repository](https://osf.io/n5vkr/files/googledrive). The archive contains fully trained models and intermediate outputs. You can then load the trained model directly for predictions:

```bash
python experiment.py card --db stats --query_set stats_ceb --estimator bayescard --load_dir experiments/overall/stats_ceb/bayescard/default
```

For end-to-end evaluations, modify the `postgresql.conf` file to disable the GEQO optimizer and parallel execution. This better reflects the plan quality:

```
geqo = off
max_parallel_workers_per_gather = 0
```
You can also tune PostgreSQL knobs (e.g., `work_mem`, `shared_buffers`, `effective_cache_size`, `maintenance_work_mem`) based on your hardware for better performance. 

To calculate Spearman’s rank correlation coefficient, first build the C++ query plan enumerator. Install CMake 3.15 or later, Make, and C/C++ compilers with C++17 support before building. Run the following commands from the project root directory in the `pilotscope` environment:

```bash
conda activate pilotscope
cmake -S cpp -B cpp/build \
    -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build cpp/build --parallel 8
```

Then, run the evaluation:

```bash
python scripts/py/compute_spearman.py --query_set {queryset} --estimator {estimator} --workers {num_workers}
# or, run all workloads and estimators
# bash scripts/sh/run_all_spearman.sh
```

#### Evaluate Non-learning Estimators
Here, we evaluate two non-learning baselines: PostgreSQL with extended statistics and LpBound.
To evaluate PostgreSQL with extended statistics, create the extended statistics in PostgreSQL. Using the ErgastF1 dataset as an example:
```bash
psql -U pilotscope -d ergastf1 -f scripts/sql/create_ergastf1_extended_statistics.sql
```

To see how `scripts/sql/create_ergastf1_extended_statistics.sql` is generated, inspect and run the following script. It computes pairwise RDC values and selects column groups for creating extended statistics:
```bash
python scripts/py/calculate_rdc.py --queryset ergastf1
```

Then, run the following command to obtain postgresql cardinality estimates, which will be saved to `experiments/overall/ergastf1/postgresql_ext/default`:
```bash
python scripts/py/get_postgres_estimation.py --queryset ergastf1
```
Finally, remember to drop the extended statistics after the evaluation to avoid affecting other experiments.

The cleanup script drops all extended statistics objects in the database specified by `-d`, including objects created for other workloads. The `job_light`, `job_m_ours`, and `job_large_ours` workloads share the `imdb` database, so running this script with `-d imdb` removes the extended statistics objects created for all three workloads.

For ErgastF1, run:

```bash
psql -U pilotscope -d ergastf1 -f scripts/sql/drop_extended_statistics.sql
```

To reproduce the results of LpBound, please follow the instructions in the [LpBound README](estimators/lpbound/README.md).

#### Evaluate Other ML-aided Query Optimization Methods

Our end-to-end evaluation also includes DACE, a learned cost model (LCM), and GenJoin, a learned query optimization (LQO) method. To reproduce their results, follow the instructions in the [DACE README](competitors/dace/README.md) and the [GenJoin README](competitors/genjoin/README.md).

### Training Data Efficiency of Query-Driven Methods

To evaluate performance across different numbers of training queries, run:

```bash
python experiment.py card --type training_data_efficiency --exp_name {training_query_num} --db {database} --query_set {queryset} --estimator {estimator}
```

For example, to test MSCN on `STATS-CEB` using 1,000 training queries:

```bash
python experiment.py card --type training_data_efficiency --exp_name 1000 --db stats --query_set stats_ceb --estimator mscn
```

To run the end-to-end evaluation, just change the `card` option to `e2e`.

### Generalization under Query Workload Drift

Our default workload generator creates random queries based on the database schema, without knowing the test workload. We also provide two additional training workloads to test generalization:

  + **Template-based workload:** Generated from query templates extracted from the test workload. Located at `workloads/{workload}/train_template`.
  + **OOD (Out-of-Distribution) workload:** Generated randomly with only 0-2 joins (differing from the test workload, which has more joins). Located at `workloads/{workload}/train_small`.

Evaluate query-driven methods in both settings. For example, testing MSCN on `STATS-CEB`:

```bash
python experiment.py card --type generalization --exp_name template --db stats --query_set stats_ceb --estimator mscn
python experiment.py card --type generalization --exp_name small --db stats --query_set stats_ceb --estimator mscn
```

Adopt the `e2e` option for end-to-end evaluations.

### Scalability

We test scalability using the `JOB-M` and `JOB-LARGE` workloads. These workloads contain string equality and comparison operators. To make them compatible, convert the string values to numbers:

```bash
python scripts/py/convert_csv_string_to_int.py
```

Run the experiments:

```bash
# JOB-M
python experiment.py card --type scalability --db imdb --query_set job_m_ours --estimator bayescard
python experiment.py e2e --type scalability --db imdb --query_set job_m_ours --estimator bayescard

# JOB-LARGE
python experiment.py card --type scalability --db imdb --query_set job_large_ours --estimator bayescard
python experiment.py e2e --type scalability --db imdb --query_set job_large_ours --estimator bayescard
```

### Dynamic Updates

We create database snapshots based on temporal attributes. First, generate snapshots with varying initial data ratios:

For the IMDb database:

```bash
cd datasets/imdb
python timestamps_by_percentage.py
python generate_snapshot.py
cd ~/LearnedCardinalityEstimators
bash scripts/sh/create_imdb_snapshot_db.sh
```

For the STATS database:

```bash
cd datasets/stats
python timestamps_by_percentage.py
python generate_snapshot.py
cd ~/LearnedCardinalityEstimators
bash scripts/sh/create_stats_snapshot_db.sh
```

Query-driven methods require training queries and true cardinalities for these snapshots. The generated workloads are placed in `workloads/job_light/dynamic` and `workloads/stats_ceb/dynamic`.

Evaluate models across different initial data ratios (`init_data_perc` = 20, 40, 60, 80):

```bash
# Cardinality estimation (creates `estimation_updated.csv` and `estimation_stale.csv`)
python experiment.py card --type dynamic --exp_name {init_data_perc} --db {database} --query_set {queryset} --estimator {estimator}

# End-to-end execution for updated model (creates `end2end_updated.csv`)
python experiment.py e2e --type dynamic --exp_name {init_data_perc} --db {database} --query_set {queryset} --estimator {estimator}

# End-to-end execution for stale model (creates `end2end_stale.csv`)
python experiment.py e2e --type dynamic --exp_name {init_data_perc} --db {database} --query_set {queryset} --estimator {estimator} --stale
```

For DeepDB and BayesCard, append the update strategy to `--exp_name`: `{init_data_perc}_structure`, `{init_data_perc}_incremental`, or `{init_data_perc}_filter`. For example, use `--exp_name 20_incremental` for incremental updates with an initial data ratio of 20%.

## Training Workload Generation

You can use the provided training workloads to reproduce our results, or generate new ones from scratch.

To run the default workload generator:

```bash
nohup python generate_workload.py --db ergastf1 --query_num 101000 --output_file ./workloads/ergastf1/train_workload.sql --process_num 32 --port 5432 >nohup_gen_ergastf1.out 2>&1 &
```

To specify a template file:

```bash
nohup python generate_workload.py --db stats --query_num 101000 --output_file ./workloads/stats_ceb/workload_from_template.sql --template_file ./workloads/stats_ceb/templates.sql --process_num 32 --port 5432 >nohup_gen_ergastf1.out 2>&1 &
```

Template files can be generated using:

```bash
python scripts/py/extract_template.py --query_set ergastf1 --output_file_location ./workloads/ergastf1/templates.sql
```

To generate only small queries (0-2 joins):

```bash
nohup python generate_workload.py --db stats --query_num 101000 --output_file ./workloads/stats_ceb/workload_small.sql --small_query --process_num 32 --port 5432 >nohup_gen_stats_small.out 2>&1 &
```

To generate workloads for dynamic database snapshots:

```bash
bash scripts/sh/generate_workload_dynamic_imdb.sh
bash scripts/sh/generate_workload_dynamic_stats.sh
```

*Note: Workload generation can take several days. Queries run in parallel, so ensure PostgreSQL parallel workers are enabled and adjust `work_mem` based on `process_num` and your available RAM.*


## Source Code Origins

Our implementation uses open-source code from the following repositories. Thanks to all the authors for sharing their work!

+ **MSCN**: Learned Cardinalities: Estimating Correlated Joins with Deep Learning (CIDR 2019)
    + Authors: Andreas Kipf, Thomas Kipf, Bernhard Radke, Viktor Leis, Peter Boncz, Alfons Kemper
    + Paper: https://www.cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf
    + Code: https://github.com/andreaskipf/learnedcardinalities
+ **RobustMSCN**: Robust Query Driven Cardinality Estimation under Changing Workloads (VLDB 2023)
    + Authors: Parimarjan Negi, Ziniu Wu, Andreas Kipf, Nesime Tatbul, Ryan Marcus, Sam Madden, Tim Kraska, and Mohammad Alizadeh
    + Code: https://github.com/learnedsystems/ceb
    + Paper: https://www.vldb.org/pvldb/vol16/p1520-negi.pdf
+ **JGMP**: Sample-Efficient Cardinality Estimation Using Geometric Deep Learning (VLDB 2024)
    + Authors: Silvan Reiner and Michael Grossniklaus
    + Code: https://github.com/dbis-ukon/jgmp
    + Paper: https://dl.acm.org/doi/10.14778/3636218.3636229
+ **ALECE**: ALECE: An Attention-based Learned Cardinality Estimator for SPJ Queries on Dynamic Workloads (VLDB 2024)
    + Authors: Pengfei Li, Wenqing Wei, Rong Zhu, Bolin Ding, Jingren Zhou, and Hua Lu
    + Code: https://github.com/pfl-cs/ALECE
    + Paper: https://www.vldb.org/pvldb/vol17/p197-li.pdf
+ **PRICE**: PRICE: A Pretrained Model for Cross-Database Cardinality Estimation (VLDB 2025)
    + Authors: Tianjing Zeng, Junwei Lan, Jiahong Ma, Wenqing Wei, Rong Zhu, Yingli Zhou, Pengfei Li, Bolin Ding, Defu Lian, Zhewei Wei, Jingren Zhou
    + Code: https://github.com/StCarmen/PRICE
    + Paper: https://www.vldb.org/pvldb/vol18/p637-zhu.pdf
+ **DeepDB**: DeepDB: Learn from Data, not from Queries! (VLDB 2020)
    + Authors: Benjamin Hilprecht, Andreas Schmidt, Moritz Kulessa, Alejandro Molina, Kristian Kersting, Carsten Binnig
    + Code: https://github.com/DataManagementLab/deepdb-public
    + Paper: https://www.vldb.org/pvldb/vol13/p992-hilprecht.pdf
+ **NeuroCard**: NeuroCard: One Cardinality Estimator for All Tables (VLDB 2021)
    + Authors: Zongheng Yang, Amog Kamsetty, Sifei Luan, Eric Liang, Yan Duan, Xi  Chen, and Ion Stoica
    + Code: https://github.com/neurocard/neurocard
    + Paper: https://vldb.org/pvldb/vol14/p61-yang.pdf
+ **BayesCard**: BayesCard: Revitalizing Bayesian Networks for Cardinality Estimation (arXiv 2021)
    + Authors: Ziniu Wu, Amir Shaikhha, Rong Zhu, Kai Zeng, Yuxing Han, Jingren Zhou
    + Code: https://github.com/wuziniu/BayesCard
    + Paper: https://arxiv.org/abs/2012.14743
+ **FactorJoin**: FactorJoin: A  New Cardinality Estimation Framework for Join Queries (SIGMOD 2023)
    + Authors: Ziniu Wu, Parimarjan Negi, Mohammad Alizadeh, Tim Kraska, and Samuel Madden
    + Code: https://github.com/wuziniu/FactorJoin
    + Paper: https://dl.acm.org/doi/10.1145/3588721
+ **ASM**: ASM: Harmonizing Autoregressive Model, Sampling, and Multi-dimensional Statistics Merging for Cardinality Estimation (SIGMOD 2024)
    + Authors: Kyoungmin Kim, Sangoh Lee, Injung Kim, and Wook-Shin Han
    + Code: https://github.com/postechdblab/ASM
    + Paper: https://dl.acm.org/doi/abs/10.1145/3639300
+ **LpBound**: LpBound: Pessimistic Cardinality Estimation Using Lp-Norms of Degree Sequences (SIGMOD 2025)
    + Authors: Haozhe Zhang, Christoph Mayer, Mahmoud Abo Khamis, Dan Olteanu, and Dan Suciu
    + Code: https://github.com/fdbresearch/LpBound
    + Paper: https://dl.acm.org/doi/10.1145/3725321
+ **DACE**: DACE: A Database-Agnostic Cost Estimator (ICDE 2024)
    + Authors: Zibo Liang, Xu Chen, Yuyang Xia, Runfan Ye, Haitian Chen, Jiandong Xie, and Kai Zheng
    + Code: https://github.com/liang-zibo/DACE
    + Paper: https://ieeexplore.ieee.org/document/10598079
+ **GenJoin**: GenJoin: Conditional Generative Plan-to-Plan Query Optimizer that Learns from Subplan Hints (SIGMOD 2026)
    + Authors: Pavel Sulimov, Claude Lehmann, and Kurt Stockinger
    + Code: https://github.com/edualc/genjoin
    + Paper: https://dl.acm.org/doi/10.1145/3749165
