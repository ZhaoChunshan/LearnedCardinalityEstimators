# GenJoin: Conditional Generative Plan-to-Plan Query Optimizer that Learns from Subplan Hints

Currently under revision

\* both authors contributed equally.

### Installation

```bash
# Create a new Venv Environment in the folder "env"
python3 -m venv env

# Activate the Venv Environment
source env/bin/activate # or use the activate_env.sh

# Install Libraries via pip
pip install -r requirements.txt

# Go and have fun programming
```

### Database setup

Please refer to [docker/job/README.md](docker/job/README.md) and [docker/stack/README.md](docker/stack/README.md)

### Train data generation

Please change the `--workload` depending on JOB or STACK data required

```bash

python3 scripts/generating/create_full_plan_data.py --workload JOB

```

### Hyperparameter tuning

For hyperparameter tuning please use [notebooks/genjoin/optuna.ipynb](notebooks/genjoin/optuna.ipynb).


### Train and test

Test scripts loop through all `run_id`. Please refer to `train_test_job.sh` and `train_test_stack.sh` for examples.

```bash

python scripts/training/train_cvae.py --workload join_order_benchmark --experiment RandomSplitA --run_id 0

python scripts/testing/test_cvae.py --workload join_order_benchmark  --experiment RandomSplitA --pg_live --dump --mode test

```
