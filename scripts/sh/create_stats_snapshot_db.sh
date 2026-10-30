#!/bin/bash

percentages=("2" "4" "6" "8" "10" "12" "14" "16" "18" "20" "24" "28" "30" "32" "36" "40" "42" "48" "54" "56" "60" "64" "72" "80")

DB_USER="pilotscope"
DB_PASSWORD="" 

for p in "${percentages[@]}"; do
    DB_NAME="stats_p${p}"
    
    pushd "datasets/stats/snapshots/${p}" > /dev/null

    echo "--- Executing: ${DB_NAME} (Folder: ${p}) ---"
    
    PGPASSWORD="$DB_PASSWORD" psql -U "$DB_USER" -f "postgres_create_stats_p${p}.sql"
    
    popd > /dev/null

done

echo "All done!"
