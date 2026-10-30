#!/bin/bash
set -e

query_sets=(
    "job_light"
    "stats_ceb"
    "ergastf1"
)

estimators=(
    "true"
    "postgresql"
    "postgresql_ext"
    "lpbound"
    "mscn"
    "robustmscn"
    "jgmp"
    "alece"
    "price_zs"
    "price_ft"
    "deepdb"
    "neurocard"
    "factorjoin"
    "bayescard"
    "asm"
)

for qs in "${query_sets[@]}"; do
    for est in "${estimators[@]}"; do
        if [[ "$qs" == "ergastf1" && "$est" == "factorjoin" ]]; then
            echo "Skipping factorjoin estimator for ergastf1 query set."
            continue
        fi

        echo "=== $qs / $est ==="
        python scripts/py/compute_spearman.py \
            --query_set "$qs" \
            --estimator "$est" \
            --workers 32
    done
done

echo "All done."
