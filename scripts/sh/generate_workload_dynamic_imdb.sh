#!/bin/bash

percentages=("2" "4" "6" "8" "10" "12" "14" "16" "18" "20" "24" "28" "30" "32" "36" "40" "42" "48" "54" "56" "60" "64" "72" "80")

for p in "${percentages[@]}"; do
    echo "--- Generate: job_light ${p} ---"

    python scripts/py/generate_train_workload_for_snapshot.py --output "workloads/job_light/dynamic/${p}/train_workload.sql" --query_set job_light --snapshot ${p}
    
done

echo "All done!"

