# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: workload paths and database mappings for this study.

import os


WORKLOADS = {
    'join_order_benchmark': 'JOB/JOB',
    'join_order_benchmark_extended': 'JOB/JOB-EXTENDED',
    'join_order_benchmark_light': 'JOB/JOB-LIGHT',
    'stats': 'STATS-CEB',
    'stack': 'STACK',
    'job_light': 'job_light',
    'stats_ceb': 'stats_ceb',
    'ergastf1': 'ergastf1',
}

DATABASES = {
    'join_order_benchmark': 'imdbload',
    'join_order_benchmark_extended': 'imdbload',
    'join_order_benchmark_light': 'imdbload',
    'stack': 'so',
    'job_light': 'imdb',
    'stats_ceb': 'stats',
    'ergastf1': 'ergastf1',
}

DB_CONFIG = {
    'join_order_benchmark': {
        'DB_PORT': 5431,
        'DB_NAME': 'imdbload'
    },
    'join_order_benchmark_light': {
        'DB_PORT': 5431,
        'DB_NAME': 'imdbload'
    },
    'stack': {
        'DB_PORT': 5432,
        'DB_NAME': 'so'
    },
    'job_light': {
        'DB_PORT': 5432,
        'DB_NAME': 'imdb'
    },
    'stats_ceb': {
        'DB_PORT': 5432,
        'DB_NAME': 'stats'
    },
    'ergastf1': {
        'DB_PORT': 5432,
        'DB_NAME': 'ergastf1'
    },
}


def queries_path():
    return os.path.join('data', 'queries')
    

def db_content_path():
    return os.path.join('data', 'db_content')


def processed_path():
    return os.path.join('data', 'processed')


def models_path():
    return os.path.join('models')


def rl_model_folder_path(subfolder: str):
    return os.path.join(models_path(), 'rl', subfolder)
