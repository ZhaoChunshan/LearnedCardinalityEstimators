"""Paths derived from the estimator location; imports do not create files."""
from pathlib import Path


class LpBoundPaths:
    PROJ_ROOT_DIR = Path(__file__).resolve().parents[3]
    REPO_ROOT_DIR = PROJ_ROOT_DIR.parents[1]
    DATASETS_DIR = REPO_ROOT_DIR / "datasets"
    WORKLOADS_DIR = REPO_ROOT_DIR / "workloads"
    SCHEMAS_DIR = PROJ_ROOT_DIR / "config/schemas"
    WORK_DIR = PROJ_ROOT_DIR / "work"
    OUTPUT_DIR = WORK_DIR
    RESULTS_DIR = PROJ_ROOT_DIR / "results"
    GENERATED_SQL_DIR = WORK_DIR / "statistics_sql"
    WORKLOAD_TO_DB_MAP = {
        "job_light": "imdb", "stats_ceb": "stats", "ergastf1": "ergastf1",
        "job_m_ours": "imdb", "job_large_ours": "imdb",
    }
