# GenJoin

This directory contains the code to reproduce GenJoin's results on the `job_light`, `stats_ceb`, and `ergastf1` workloads. The code is based on [the official GenJoin repository](https://github.com/edualc/genjoin).

### Environment Setup

First, create a new Python environment and install the dependencies:

```bash
conda create -n genjoin python=3.10
conda activate genjoin
pip install -r competitors/genjoin/requirements.txt
```

Set `DB_PORT`, `DB_USER`, and `DB_PASSWORD` in `competitors/genjoin/.env`. For example:

```bash
DB_HOST=localhost
DB_PORT=5432

DB_USER=pilotscope
DB_PASSWORD=pilotscope
SSH_PRIVATE_KEY=
SSH_USER=
```

Copy the three workload directories from the main project:

```bash
mkdir -p competitors/genjoin/workloads
cp -a workloads/{job_light,stats_ceb,ergastf1} competitors/genjoin/workloads/
```

### Reproduce Experimental Results

Pretrained model checkpoints for all three workloads are included at `models/{dataset}/cvae/run0/cvae_default.pt`. Run the following commands for end-to-end evaluation:

```bash
cd competitors/genjoin
python scripts/testing/test_cvae.py --workload job_light --dump
python scripts/testing/test_cvae.py --workload stats_ceb --dump
python scripts/testing/test_cvae.py --workload ergastf1 --dump
cd ../..
```

All timings are reported in milliseconds. You can also generate training data and train the model from scratch. For example:

```bash
cd competitors/genjoin
python scripts/generating/create_full_plan_data.py --workload job_light
python scripts/training/train_cvae.py --workload job_light
cd ../..
```

Finally, copy the result CSV files to `experiments/competitors_results/genjoin/{dataset}/`:

```bash
for dataset in job_light stats_ceb ergastf1; do
    mkdir -p "experiments/competitors_results/genjoin/$dataset"
    cp "competitors/genjoin/data/processed/$dataset/full_execution_time_${dataset}_test.csv" \
        "experiments/competitors_results/genjoin/$dataset/"
done
```

We adapted this code for our study and added it to this repository on October 10, 2026. The modified code is distributed under [GNU GPLv3](LICENSE).
