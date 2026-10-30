

# Building Postgres Container

```bash
docker build -t "pg_img" -f Dockerfile .
```

# Initializing Postgres Cluster

```bash
docker-compose up -d

docker exec -it job bash

# Password is set to "postgres"
su postgres

cp /postgresql.conf /pgdata/postgresql.conf

pg_ctl restart
```

After the container restarted:

```bash
docker-compose up -d

docker exec -it job bash

# Password is set to "postgres"
su postgres

# LOAD JOB INTO DB
# ==========================================
# Create DB called 'imdb'
psql -c "CREATE DATABASE imdbload;"

# Load the imdb dump into the 'imdb' DB
# (it is mounted from /data/db_content/JOB/imdb.dump)
pg_restore -c -d imdbload --verbose docker-entrypoint-initdb.d/imdb.dump
# or if importing via psql, connect to the database "psql -d imdbload" and load
# the dump with "\i <path to dump>"
psql -d imdbload -c "ANALYZE verbose;"

# Adding missing FK indexes
psql -U postgres -d imdbload -a -f docker-entrypoint-initdb.d/fkindexes.sql
```

Installing pg_hint_plan:

```bash
docker exec -it job bash

su postgres

cd docker-entrypoint-initdb.d/

# For Postgres 12
sudo git clone https://github.com/ossc-db/pg_hint_plan.git -b REL12_1_3_7
# For Postgres 14
sudo git clone https://github.com/ossc-db/pg_hint_plan.git -b REL14_1_4_2
# For Postgres 16
sudo git clone https://github.com/ossc-db/pg_hint_plan.git -b REL16_1_6_0

cd pg_hint_plan
sudo make
sudo make install

psql -U postgres

CREATE EXTENSION pg_hint_plan;

# Check that it was created
\dx;
```

When executing queries on Postgres, execute the following query first:

```sql
LOAD 'pg_hint_plan'; SET pg_hint_plan.enable_hint_tables TO off;
```

NB! Starting from Postgres 15:

```sql
LOAD 'pg_hint_plan'; SET pg_hint_plan.enable_hint_table TO off;
```

# Postgres

## Check active connections on Postgres

```sql
SELECT datid, datname, pid, usename, backend_start, xact_start, query_start, state_change, state, backend_xid, backend_xmin FROM pg_stat_activity;
```

## Kill all other conections on DB

```sql
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'imdbload'
    AND pid <> pg_backend_pid();
```

(And again as a one-liner for easier copying from an editor)
```sql
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = 'imdbload';
```