"""Repository input configuration and reproducible intermediate artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ESTIMATOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ESTIMATOR_ROOT.parents[1]
sys.path.insert(0, str(ESTIMATOR_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from query.workload import TEST_WORKLOAD_FILES  # noqa: E402

WORKLOADS = tuple(TEST_WORKLOAD_FILES)
DATASETS = {
    "job_light": "imdb", "stats_ceb": "stats", "ergastf1": "ergastf1",
    "job_m_ours": "imdb", "job_large_ours": "imdb",
}
EXPERIMENT_TYPES = {
    "job_light": "overall", "stats_ceb": "overall", "ergastf1": "overall",
    "job_m_ours": "scalability", "job_large_ours": "scalability",
}
STAGES = ("prepare", "data", "statistics", "estimate", "prepare-cpp", "timing", "validate", "export")
PARAMETERS = {"num_mcvs": 5000, "num_buckets": 128, "p_max": 10}
EXPORT_COLUMNS = ["index", "is_outer_most", "estimated_card", "true_card", "q_error", "estimation_time", "query_id", "sql"]
FROM_RE = re.compile(r"\bFROM\s+(.*?)(?:\s+WHERE\b|;|$)", re.I)
COMMENT_RE = re.compile(r"^/\*\s*\((.*?)\)\s*\*/\s*(.*)$")
PREDICATE_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*(<=|>=|=|<|>)\s*('(?:[^']|'')*'|-?\d+(?:\.\d+)?)", re.I)


def work_dir(query_set):
    return ESTIMATOR_ROOT / "work" / query_set


def export_dir(query_set):
    return ESTIMATOR_ROOT / "results/external_exports" / query_set


def source_file(query_set):
    return REPO_ROOT / "workloads" / query_set / TEST_WORKLOAD_FILES[query_set]


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def aliases(sql):
    match = FROM_RE.search(sql)
    if match is None:
        raise ValueError(f"Missing flat FROM clause: {sql}")
    return {ref.strip().split()[-1].upper(): ref.strip().split()[0].upper() for ref in match.group(1).split(",")}


def prepare_workload(query_set):
    """Stream source SQL unchanged, deriving IDs and workload-specific schema."""
    folder = work_dir(query_set)
    folder.mkdir(parents=True, exist_ok=True)
    template = ESTIMATOR_ROOT / "config/schemas" / f"{query_set}_schema.json"
    schema = json.loads(template.read_text())
    schema["name"] = query_set
    used_tables, equality, ranges = set(), {}, {}
    intervals, interval_seen = {}, {}
    prune_ranges = "range_predicate_intervals" in schema
    query_id = subquery_id = ordinal = 0
    parent_sql = last_sql = None
    singletons = 0
    output = folder / "workload.csv"
    temporary = output.with_suffix(".csv.tmp")
    columns = ["SubqueryID", "QueryID", "SubqueryOrdinal", "Tables", "NumTables", "SQL", "SourceSQL", "TrueCardinality", "PostgresEstimate"]

    def check_parent():
        if parent_sql is not None and (last_sql is None or aliases(parent_sql) != aliases(last_sql)):
            raise ValueError(f"{query_set}: final subquery of parent {query_id} is not its full relation set")

    try:
        with source_file(query_set).open() as source, temporary.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for raw in source:
                line = raw.strip()
                if not line:
                    continue
                sql_and_comment, true_card, pg_card = line.rsplit("||", 2)
                if not line.startswith("/*"):
                    check_parent()
                    parent_sql = sql_and_comment.strip()
                    query_id += 1
                    ordinal = 0
                    last_sql = None
                    continue
                match = COMMENT_RE.match(sql_and_comment.strip())
                if match is None or query_id == 0:
                    raise ValueError(f"Malformed annotated subquery: {line}")
                sql = match.group(2).strip()
                alias_map = aliases(sql)
                comment_aliases = {ref.strip().split()[-1].upper() for ref in match.group(1).split(",")}
                if comment_aliases != set(alias_map):
                    raise ValueError(f"Table-list comment disagrees with FROM: {line}")
                used_tables.update(alias_map.values())
                current_intervals = {}
                for alias, column, operator, literal in PREDICATE_RE.findall(sql):
                    table, column = alias_map[alias.upper()], column.upper()
                    if column not in schema["relations"][table]:
                        raise ValueError(f"Unknown predicate column {table}.{column}")
                    destination = equality if operator == "=" else ranges
                    destination.setdefault(table, set()).add(column)
                    if operator != "=" and prune_ranges:
                        value = literal[1:-1].replace("''", "'") if literal.startswith("'") else (float(literal) if "." in literal else int(literal))
                        interval = current_intervals.setdefault((table, column), {"lower": None, "upper": None})
                        bound = "lower" if operator in (">", ">=") else "upper"
                        old = interval[bound]
                        interval[bound] = value if old is None else (max(old, value) if bound == "lower" else min(old, value))
                for key, interval in current_intervals.items():
                    seen = interval_seen.setdefault(key, set())
                    pair = (interval["lower"], interval["upper"])
                    if pair not in seen:
                        seen.add(pair)
                        intervals.setdefault(key[0], {}).setdefault(key[1], []).append(interval)
                subquery_id += 1
                singletons += len(alias_map) == 1
                writer.writerow({"SubqueryID": subquery_id, "QueryID": query_id, "SubqueryOrdinal": ordinal,
                    "Tables": " ".join(sorted(alias_map)), "NumTables": len(alias_map), "SQL": sql,
                    "SourceSQL": sql_and_comment.strip(), "TrueCardinality": int(true_card), "PostgresEstimate": float(pg_card)})
                ordinal += 1
                last_sql = sql
            check_parent()
        if not query_id or not subquery_id:
            raise ValueError("Empty workload")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    for field in ("relations", "join_variables", "groupby_variables"):
        schema[field] = {table: value for table, value in schema[field].items() if table in used_tables}
    for field, actual in (("equality_predicate_variables", equality), ("range_predicate_variables", ranges)):
        schema[field] = {table: sorted(set(schema[field].get(table, [])) | actual.get(table, set()))
                         for table in sorted(used_tables) if schema[field].get(table) or actual.get(table)}
    schema["fk_pk_joins_dict"] = {table: [join for join in joins if join["pk_relation"] in used_tables]
                                  for table, joins in schema["fk_pk_joins_dict"].items() if table in used_tables}
    if prune_ranges:
        schema["range_predicate_intervals"] = intervals
    atomic_json(folder / "schema.json", schema)
    meta = {"query_set": query_set, "parents": query_id, "subqueries": subquery_id, "singletons": singletons,
            "source": str(source_file(query_set).relative_to(REPO_ROOT)), "source_sha256": digest_file(source_file(query_set))}
    atomic_json(folder / "workload_metadata.json", meta)
    print(f"{query_set}: parents={query_id}, subqueries={subquery_id}, singletons={singletons}", flush=True)
    return meta


def dataset_signature(dataset):
    """Hash changed CSVs; reuse hashes only when their filesystem identity matches."""
    cache_path = ESTIMATOR_ROOT / "work/databases" / f"{dataset}_source_hashes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ddl = ESTIMATOR_ROOT / "config/duckdb" / f"{dataset}.sql"
    tables = re.findall(r"CREATE\s+TABLE\s+(\w+)", ddl.read_text(), re.I)
    records = {}
    for table in tables:
        path = REPO_ROOT / "datasets" / dataset / f"{table.lower()}.csv"
        stat = path.stat()
        identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        previous = cache.get(table, {})
        sha = previous.get("sha256") if previous.get("identity") == identity else digest_file(path)
        records[table] = {"identity": identity, "sha256": sha}
    atomic_json(cache_path, records)
    return {"dataset": dataset, "ddl_sha256": digest_file(ddl), "csv_sha256": {k: v["sha256"] for k, v in records.items()},
            "csv_format": {"delimiter": "|", "header": True, "null": "NULL"}}


def ensure_base(query_set, signature):
    import duckdb
    from lpbound.duckdb_adapter.duckdb_manager import sql_identifier, sql_literal
    dataset = DATASETS[query_set]
    directory = ESTIMATOR_ROOT / "work/databases"
    path = directory / f"{dataset}_base.duckdb"
    metadata_path = path.with_suffix(".json")
    fingerprint = digest_object(signature)
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("fingerprint") == fingerprint and metadata.get("database_sha256") == digest_file(path):
            print(f"{query_set}: reuse {dataset} base database", flush=True)
            return
    temporary = path.with_suffix(".building.duckdb")
    temporary.unlink(missing_ok=True)
    temporary.with_suffix(temporary.suffix + ".wal").unlink(missing_ok=True)
    rows = {}
    with duckdb.connect(str(temporary)) as connection:
        connection.execute("SET threads=8")
        connection.execute((ESTIMATOR_ROOT / "config/duckdb" / f"{dataset}.sql").read_text())
        for table in signature["csv_sha256"]:
            source = REPO_ROOT / "datasets" / dataset / f"{table.lower()}.csv"
            with source.open(newline="") as f:
                header = next(csv.reader(f, delimiter="|"))
            expected = [r[1].lower() for r in connection.execute(f"PRAGMA table_info({sql_literal(table)})").fetchall()]
            if [col.lower() for col in header] != expected:
                raise ValueError(f"{source}: CSV header {header} != schema columns {expected}")
            connection.execute(f"COPY {sql_identifier(table)} FROM {sql_literal(str(source))} (FORMAT CSV, HEADER TRUE, DELIMITER '|', NULLSTR 'NULL')")
            rows[table] = connection.execute(f"SELECT COUNT(*) FROM {sql_identifier(table)}").fetchone()[0]
            print(f"Imported {dataset}.{table}: {rows[table]} rows", flush=True)
        connection.execute("CHECKPOINT")
    temporary.replace(path)
    atomic_json(metadata_path, {"fingerprint": fingerprint, "inputs": signature, "rows": rows, "database_sha256": digest_file(path)})


def code_signature():
    import duckdb
    import ortools
    files = list((ESTIMATOR_ROOT / "src/lpbound").rglob("*.py")) + list((ESTIMATOR_ROOT / "scripts").glob("*.py"))
    files += [ESTIMATOR_ROOT / "src/lpbound/cpp_solver/manifest_solver.cpp", ESTIMATOR_ROOT / "src/lpbound/cpp_solver/CMakeLists.txt"]
    signature = {"files": {str(p.relative_to(ESTIMATOR_ROOT)): digest_file(p) for p in sorted(files)},
                 "duckdb": duckdb.__version__, "ortools": ortools.__version__}
    executable = ESTIMATOR_ROOT / "work/build/lpbound_parallel"
    if executable.is_file():
        signature["cpp_executable_sha256"] = digest_file(executable)
    return signature
