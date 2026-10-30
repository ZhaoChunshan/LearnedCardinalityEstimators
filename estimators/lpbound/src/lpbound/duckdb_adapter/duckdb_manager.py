"""Workload-local statistics over an attached, read-only shared base database."""
import duckdb

from lpbound.config.paths import LpBoundPaths


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


class DatabaseManager:
    def __init__(self, benchmark_schema, groupby=False):
        self.benchmark_schema = benchmark_schema
        self.benchmark = benchmark_schema["name"]
        self.db_name = LpBoundPaths.WORKLOAD_TO_DB_MAP[self.benchmark]
        self.base_file = LpBoundPaths.WORK_DIR / "databases" / f"{self.db_name}_base.duckdb"
        self.duckdb_file = LpBoundPaths.WORK_DIR / self.benchmark / "statistics.duckdb"

    def create_or_load_db(self, read_only=False):
        if not self.base_file.is_file():
            raise FileNotFoundError(f"Missing base database {self.base_file}; run the data stage first")
        if read_only and not self.duckdb_file.is_file():
            raise FileNotFoundError(f"Missing statistics {self.duckdb_file}; run the statistics stage first")
        if not read_only:
            self.duckdb_file.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self.duckdb_file), read_only=read_only)
        try:
            connection.execute(f"ATTACH {sql_literal(str(self.base_file))} AS base (READ_ONLY)")
            connection.execute("SET threads=" + ("1" if read_only else "8"))
            if not read_only:
                for table in self.benchmark_schema["relations"]:
                    quoted = sql_identifier(table)
                    connection.execute(f"CREATE OR REPLACE VIEW {quoted} AS SELECT * FROM base.{quoted}")
            return connection
        except Exception:
            connection.close()
            raise
