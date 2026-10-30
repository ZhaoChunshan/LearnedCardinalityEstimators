# SPDX-License-Identifier: Apache-2.0
# Modified for the experimental study in this repository.
# Original project: https://github.com/neurocard/neurocard
# Changes: dataset and join configurations, query and checkpoint paths, and
# training, evaluation, and update settings for this study.

"""Experiment configurations.

EXPERIMENT_CONFIGS holds all registered experiments.

TEST_CONFIGS (defined at end of file) stores "unit tests": these are meant to
run for a short amount of time and to assert metrics are reasonable.

Experiments registered here can be launched using:

  >> python run.py --run <config> [ <more configs> ]
  >> python run.py  # Runs all tests in TEST_CONFIGS.
"""
import os

from ray import tune

EXPERIMENT_CONFIGS = {}
TEST_CONFIGS = {}

# Common config. Each key is auto set as an attribute (i.e. NeuroCard.<attr>)
# so try to avoid any name conflicts with members of that class.
BASE_CONFIG = {
    'cwd': os.getcwd(),
    'epochs_per_iteration': 1,
    'num_eval_queries_per_iteration': 100,
    'num_eval_queries_at_end': 88888888,  # End of training.
    'num_eval_queries_at_checkpoint_load': 88888888,  # Evaluate a loaded ckpt.
    'epochs': 10,
    'seed': None,
    'order_seed': None,
    'bs': 2048,
    'order': None,
    'layers': 2,
    'fc_hiddens': 128,
    'warmups': 1000,
    'constant_lr': None,
    'lr_scheduler': None,
    'custom_lr_lambda': None,
    'optimizer': 'adam',
    'residual': True,
    'direct_io': True,
    'input_encoding': 'embed',
    'output_encoding': 'embed',
    'query_filters': [5, 12],
    'force_query_cols': None,
    'embs_tied': True,
    'embed_size': 32,
    'input_no_emb_if_leq': True,
    'resmade_drop_prob': 0.,
    'decreased_max_steps':5,
    # Multi-gpu data parallel training.
    'use_data_parallel': False,

    # If set, load this checkpoint and run eval immediately. No training. Can
    # be glob patterns.
    # Example:
    # 'checkpoint_to_load': tune.grid_search([
    #     'models/*52.006*',
    #     'models/*43.590*',
    #     'models/*42.251*',
    #     'models/*41.049*',
    # ]),
    'checkpoint_to_load': None,
    # Dropout for wildcard skipping.
    'disable_learnable_unk': False,
    'per_row_dropout': True,
    'dropout': 1,
    'table_dropout': False,
    'fixed_dropout_ratio': False,
    'asserts': None,
    'special_orders': 0,
    'special_order_seed': 0,
    'join_tables': [],
    'label_smoothing': 0.0,
    'compute_test_loss': False,

    # Column factorization.
    'factorize': False,
    'factorize_blacklist': None,
    'grouped_dropout': True,
    'factorize_fanouts': False,

    # Eval.
    'eval_psamples': [100, 1000, 10000],
    'eval_join_sampling': None,  # None, or #samples/query.

    # Transformer.
    'use_transformer': False,
    'transformer_args': {},

    # Checkpoint.
    'save_checkpoint_at_end': True,
    'checkpoint_every_epoch': False,
    'training_on_the_checkpoint': False,
    # Experimental.
    '_save_samples': None,
    '_load_samples': None,
    'num_orderings': 1,
    'num_dmol': 0,
}

JOB_LIGHT_BASE = {
    'dataset': 'imdb',
    'join_tables': [
        'cast_info', 'movie_companies', 'movie_info', 'movie_keyword', 'title',
        'movie_info_idx'
    ],
    'join_keys': {
        'cast_info': ['movie_id'],
        'movie_companies': ['movie_id'],
        'movie_info': ['movie_id'],
        'movie_keyword': ['movie_id'],
        'title': ['id'],
        'movie_info_idx': ['movie_id']
    },
    # Sampling starts at this table and traverses downwards in the join tree.
    'join_root': 'title',
    # Inferred.
    'join_clauses': None,
    'join_how': 'outer',
    # Used for caching metadata.  Each join graph should have a unique name.
    'join_name': 'job-light',
    # See datasets.py.
    'use_cols': 'simple',
    'seed': 0,
    'per_row_dropout': False,
    'table_dropout': True,
    'embs_tied': True,
    # Num tuples trained =
    #   bs (batch size) * max_steps (# batches per "epoch") * epochs.
    'epochs': 1,
    'bs': 2048,
    'max_steps': 512,
    # Use this fraction of total steps as warmups.
    'warmups': 0.05,
    # Number of DataLoader workers that perform join sampling.
    'loader_workers': 8,
    # Options: factorized_sampler, fair_sampler (deprecated).
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 4,
    'layers': 4,
    # Eval:
    'compute_test_loss': True,
    'queries_csv': './queries/temp_queries.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 88888888,
    'eval_psamples': [4000],

    # Multi-order.
    'special_orders': 0,
    'order_content_only': True,
    'order_indicators_at_front': False,
}


STATS_BASE = {
    'dataset': 'stats', # Filled later
    'join_tables': [], # Filled later
    'join_keys': {}, # Filled later

    # Sampling starts at this table and traverses downwards in the join tree.
    'join_root': '', # Filled Later

    # Inferred.
    'join_clauses': None, # Filled Later
    'join_how': 'outer',
    # Used for caching metadata.  Each join graph should have a unique name.
    'join_name': '', # Filled later
    
    # See datasets.py.
    'use_cols': 'simple',
    'seed': 0,
    'per_row_dropout': False,
    'table_dropout': True,
    'embs_tied': True,
    # Num tuples trained =
    #   bs (batch size) * max_steps (# batches per "epoch") * epochs.
    'epochs': 7,
    'bs': 2048,
    'max_steps': 500,
    # Use this fraction of total steps as warmups.
    'warmups': 0.05,
    'lr_scheduler': tune.grid_search(['OneCycleLR-0.28']),
    'resmade_drop_prob': tune.grid_search([.1]),
    'label_smoothing': tune.grid_search([0]),
    # Number of DataLoader workers that perform join sampling.
    'loader_workers': 8,
    # Options: factorized_sampler, fair_sampler (deprecated).
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 4,
    'layers': 4,
    # Eval:
    'compute_test_loss': True,
    'queries_csv': './queries/temp_queries.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 88888888,
    'eval_psamples': [8000],

    # Multi-order.
    'special_orders': 0,
    'order_content_only': True,
    'order_indicators_at_front': False,

    # Transformer.
    'use_transformer': False,
    'transformer_args': {
        'num_blocks': 6,
        'd_ff': 512,
        'd_model': 128,
        'num_heads': 4,
        'num_blocks': 6,
        'd_ff': 256,
        'd_model': 64,
        'num_heads': 4,
        'use_positional_embs': tune.grid_search([False]),
        'activation': 'gelu',
        'seed': None,
    },
}

FACTORIZE = {
    'factorize': True,
    'word_size_bits': 10,
    'grouped_dropout': True,
}

JOB_M = {
    'join_tables': [
        'title', 'aka_title', 'cast_info', 'complete_cast', 'movie_companies',
        'movie_info', 'movie_info_idx', 'movie_keyword', 'movie_link',
        'kind_type', 'comp_cast_type', 'company_name', 'company_type',
        'info_type', 'keyword', 'link_type'
    ],
    'join_keys': {
        'title': ['id', 'kind_id'],
        'aka_title': ['movie_id'],
        'cast_info': ['movie_id'],
        'complete_cast': ['movie_id', 'subject_id'],
        'movie_companies': ['company_id', 'company_type_id', 'movie_id'],
        'movie_info': ['movie_id'],
        'movie_info_idx': ['info_type_id', 'movie_id'],
        'movie_keyword': ['keyword_id', 'movie_id'],
        'movie_link': ['link_type_id', 'movie_id'],
        'kind_type': ['id'],
        'comp_cast_type': ['id'],
        'company_name': ['id'],
        'company_type': ['id'],
        'info_type': ['id'],
        'keyword': ['id'],
        'link_type': ['id']
    },
    'join_clauses': [
        'title.id=aka_title.movie_id',
        'title.id=cast_info.movie_id',
        'title.id=complete_cast.movie_id',
        'title.id=movie_companies.movie_id',
        'title.id=movie_info.movie_id',
        'title.id=movie_info_idx.movie_id',
        'title.id=movie_keyword.movie_id',
        'title.id=movie_link.movie_id',
        'title.kind_id=kind_type.id',
        'comp_cast_type.id=complete_cast.subject_id',
        'company_name.id=movie_companies.company_id',
        'company_type.id=movie_companies.company_type_id',
        'movie_info_idx.info_type_id=info_type.id',
        'keyword.id=movie_keyword.keyword_id',
        'link_type.id=movie_link.link_type_id',
    ],
    'join_root': 'title',
    'join_how': 'outer',
    'join_name': 'job-m',
    'use_cols': 'multi',
    'epochs': 10,
    'bs': 1000,
    'resmade_drop_prob': 0.1,
    'max_steps': 1000,
    'loader_workers': 8,
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 16,
    'warmups': 0.15,
    # Eval:
    'compute_test_loss': False,
    'queries_csv': './queries/temp_queries.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 88888888,
    'eval_psamples': [1000],
}

JOB_M_FACTORIZED = {
    'factorize': True,
    'factorize_blacklist': [],
    'factorize_fanouts': True,
    'word_size_bits': 14,
    'bs': 2048,
    'max_steps': 512,
    'epochs': 1,
    
    'epochs_per_iteration': 1,
}

### EXPERIMENT CONFIGS ###
# Run multiple experiments concurrently by using the --run flag, ex:
# $ ./run.py --run job-light
EXPERIMENT_CONFIGS = {
    # JOB-light, NeuroCard base.
    'job-light': dict(
        dict(BASE_CONFIG, **JOB_LIGHT_BASE),
        **{
            'factorize': True,
            'grouped_dropout': True,
            'loader_workers': 4,
            'warmups': 0.05,  # Ignored.
            'lr_scheduler': tune.grid_search(['OneCycleLR-0.28']),
            'loader_workers': 4,
            'max_steps': tune.grid_search([500]),
            'epochs': 7,
            'num_eval_queries_per_iteration': 70,
            'input_no_emb_if_leq': False,
            'eval_psamples': [8000],
            'epochs_per_iteration': 1,
            'resmade_drop_prob': tune.grid_search([.1]),
            'label_smoothing': tune.grid_search([0]),
            'word_size_bits': tune.grid_search([11]),
        }),
    # JOB-light-ranges, NeuroCard base.
    'job-light-ranges': dict(
        dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **FACTORIZE),
        **{
            'queries_csv': './queries/job-light-ranges.csv',
            'use_cols': 'content',
            'num_eval_queries_per_iteration': 1000,
            # 10M tuples total.
            'max_steps': tune.grid_search([500]),
            'epochs': 10,
            # Evaluate after every 1M tuples trained.
            'epochs_per_iteration': 1,
            'loader_workers': 4,
            'eval_psamples': [8000],
            'input_no_emb_if_leq': False,
            'resmade_drop_prob': tune.grid_search([0]),
            'label_smoothing': tune.grid_search([0]),
            'fc_hiddens': 128,
            'embed_size': tune.grid_search([16]),
            'word_size_bits': tune.grid_search([14]),
            'table_dropout': False,
            'lr_scheduler': None,
            'warmups': 0.1,
        },
    ),
    # JOB-light-ranges, NeuroCard-large (Transformer).
    'job-light-ranges-large': dict(
        dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **FACTORIZE),
        **{
            'queries_csv': './queries/job-light-ranges.csv',
            'use_cols': 'content',
            'num_eval_queries_per_iteration': 1000,
            'loader_workers': 4,
            'eval_psamples': [8000],
            'input_no_emb_if_leq': False,
            'resmade_drop_prob': tune.grid_search([0]),
            'table_dropout': tune.grid_search([False]),
            'lr_scheduler': None,
            'word_size_bits': 16,
            'use_data_parallel': True,
            'bs': 2048,
            'use_transformer': True,
            'transformer_args': {
                'num_blocks': 6,
                'd_ff': 512,
                'd_model': 128,
                'num_heads': 4,
                'num_blocks': 6,
                'd_ff': 256,
                'd_model': 64,
                'num_heads': 4,
                'use_positional_embs': tune.grid_search([False]),
                'activation': 'gelu',
                'seed': None,
            },
            'max_steps': 512,
            'label_smoothing': tune.grid_search([.01]),
            'epochs_per_iteration': 10,
            'warmups': 0.15,
            'lr_scheduler': tune.grid_search([None]),
            'epochs': 10,
            'join_tables': [
                'title', 'cast_info', 'movie_companies', 'movie_info',
                'movie_keyword', 'movie_info_idx'
            ],
        },
    ),
    # JOB-M, NeuroCard.
    'job-m': dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
                  **JOB_M_FACTORIZED),
    # JOB-M, NeuroCard-large (Transformer).
    'job-m-large': dict(
        dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
             **JOB_M_FACTORIZED),
        **{
            # Launch with --gpus=4.  BS=1024 is OK when split across 4 V100
            # (16GB); OOMs when on 1 gpu.  Lower the batch size if desired.
            'bs': 1024,
            'use_transformer': True,
            'transformer_args': {
                # Transformer-Base
                # number of model parameters: 107264000 (~= 409.2MB)
                'num_blocks': 6,
                'd_ff': 2048,
                'd_model': 512,
                'num_heads': 8,
                'use_positional_embs': False,
                'activation': 'gelu',
                'seed': None,
            },
            'table_dropout': tune.grid_search([False]),
            'epochs_per_iteration': 5,
            'lr_scheduler': tune.grid_search(['wd_1e-4_0.04']),
            'epochs': 20,
            'eval_psamples': [4096],
            'num_eval_queries_per_iteration': 113,
            'max_steps': 2048,
            'epochs': 40,
            'label_smoothing': tune.grid_search([.01]),
            'sampler_batch_size': 1024 * 16,
            'use_data_parallel': True,
        }),
}

########### OUR STATS CONFIG ############
EXPERIMENT_CONFIGS['stats1'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'stats1',
        'join_tables': ["badges", "users", "comments", "posthistory", "votes", "posts", "postlinks", "tags"], # Filled later
        'join_keys': {
            "badges": ["userid"],
            "users": ["id"],
            "comments": ["userid"],
            "posthistory": ["userid"],
            "votes": ["userid"],
            "posts": ["id", "owneruserid"],
            "postlinks": ["postid"],
            "tags": ["excerptpostid"],
        }, # Filled later
        'join_clauses': [
            "badges.userid=users.id",
            "comments.userid=users.id",
            "posthistory.userid=users.id",
            "votes.userid=users.id",
            "posts.owneruserid=users.id",
            "postlinks.postid=posts.id",
            "tags.excerptpostid=posts.id",
        ],
        'join_root': 'users', # All partial schema contains users table.
        'join_name': 'stats1', # Filled later
    }
)

EXPERIMENT_CONFIGS['stats2'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'stats2',
        'join_tables': ["badges", "comments", "posts", "posthistory", "postlinks", "users", "votes"], # Filled later
        'join_keys': {
            "badges": ["userid"],
            "comments": ["postid"],
            "posts": ["id", "owneruserid"],
            "posthistory": ["postid"],
            "postlinks": ["postid"],
            "users": ["id"],
            "votes": ["postid"],
        }, # Filled later
        'join_clauses': [
            "badges.userid=users.id",
            "posts.owneruserid=users.id",
            "comments.postid=posts.id",
            "posthistory.postid=posts.id",
            "postlinks.postid=posts.id",
            "votes.postid=posts.id",
        ],
        'join_root': 'users', # All partial schema contains users table.
        'join_name': 'stats2', # Filled later
    }
)

EXPERIMENT_CONFIGS['stats3'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'stats3',
        'join_tables': ["posts", "users", "votes"], # Filled later
        'join_keys': {
            "posts": ["id"],
            "users": ["id"],
            "votes": ["postid", "userid"],
        }, # Filled later
        'join_clauses': [
            "votes.postid=posts.id",
            "votes.userid=users.id",
        ],
        'join_root': 'users', # All partial schema contains users table.
        'join_name': 'stats3', # Filled later
    }
)

EXPERIMENT_CONFIGS['stats4'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'stats4',
        'join_tables': ["badges", "comments", "posts", "posthistory", "postlinks", "users", "votes"], # Filled later
        'join_keys': {
            "badges": ["userid"],
            "comments": ["postid"],
            "posts": ["id", "owneruserid"],
            "posthistory": ["userid"],
            "postlinks": ["relatedpostid"],
            "users": ["id"],
            "votes": ["userid"],
        }, # Filled later
        'join_clauses': [
            "badges.userid=users.id",
            "comments.postid=posts.id",
            "posts.owneruserid=users.id",
            "posthistory.userid=users.id",
            "postlinks.relatedpostid=posts.id",
            "votes.userid=users.id",
        ],
        'join_root': 'users', # All partial schema contains users table.
        'join_name': 'stats4', # Filled later
    }
)

EXPERIMENT_CONFIGS['stats5'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'stats5',
        'join_tables': ["badges", "comments", "posts", "postlinks", "users", "votes"], # Filled later
        'join_keys': {
            "badges": ["userid"],
            "comments": ["postid"],
            "posts": ["id", "owneruserid"],
            "postlinks": ["relatedpostid"],
            "users": ["id"],
            "votes": ["postid"],
        }, # Filled later
        'join_clauses': [
            "badges.userid=users.id",
            "comments.postid=posts.id",
            "posts.owneruserid=users.id",
            "postlinks.relatedpostid=posts.id",
            "votes.postid=posts.id",
        ],
        'join_root': 'users', # All partial schema contains users table.
        'join_name': 'stats5', # Filled later
    }
)

EXPERIMENT_CONFIGS['stats6'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'stats6',
        'join_tables': ["comments", "posts", "postlinks", "posthistory", "users", "votes"], # Filled later
        'join_keys': {
            "comments": ["userid", "postid"],
            "posts": ["id"],
            "postlinks": ["postid"],
            "posthistory": ["postid"],
            "users": ["id"],
            "votes": ["userid"],
        }, # Filled later
        'join_clauses': [
            "comments.postid=posts.id",
            "comments.userid=users.id",
            "posthistory.postid=posts.id",
            "postlinks.postid=posts.id",
            "votes.userid=users.id",
        ],
        'join_root': 'users', # All partial schema contains users table.
        'join_name': 'stats6', # Filled later
    }
)

EXPERIMENT_CONFIGS['stats7'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'stats7',
        'join_tables': ["badges", "comments", "posts", "postlinks", "posthistory", "users", "votes"], # Filled later
        'join_keys': {
            "badges": ["userid"],
            "comments": ["userid", "postid"],
            "posts": ["id"],
            "postlinks": ["relatedpostid"],
            "posthistory": ["postid"],
            "users": ["id"],
            "votes": ["postid"],
        }, # Filled later
        'join_clauses': [
            "badges.userid=users.id",
            "comments.postid=posts.id",
            "comments.userid=users.id",
            "posthistory.postid=posts.id",
            "postlinks.relatedpostid=posts.id",
            "votes.postid=posts.id",
        ],
        'join_root': 'users', # All partial schema contains users table.
        'join_name': 'stats7', # Filled later
    }
)

EXPERIMENT_CONFIGS['stats1-reload'] = dict(
    EXPERIMENT_CONFIGS['stats1'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

EXPERIMENT_CONFIGS['stats2-reload'] = dict(
    EXPERIMENT_CONFIGS['stats2'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

EXPERIMENT_CONFIGS['stats3-reload'] = dict(
    EXPERIMENT_CONFIGS['stats3'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

EXPERIMENT_CONFIGS['stats4-reload'] = dict(
    EXPERIMENT_CONFIGS['stats4'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

EXPERIMENT_CONFIGS['stats5-reload'] = dict(
    EXPERIMENT_CONFIGS['stats5'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

EXPERIMENT_CONFIGS['stats6-reload'] = dict(
    EXPERIMENT_CONFIGS['stats6'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

EXPERIMENT_CONFIGS['stats7-reload'] = dict(
    EXPERIMENT_CONFIGS['stats7'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

######  TEST CONFIGS ######
# These are run by default if you don't specify --run.

TEST_CONFIGS['test-job-light'] = dict(
    EXPERIMENT_CONFIGS['job-light'],
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'epochs': 1,
        'asserts': {
            'fact_psample_8000_median': 4,
            'fact_psample_8000_p99': 50,
            'train_bits': 80,
        },
    })

TEST_CONFIGS['job-light-reload'] = dict(
    EXPERIMENT_CONFIGS['job-light'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt',
        ]),
        'eval_psamples': [8000],
        'asserts': {
            'fact_psample_512_median': 1.7,
            'fact_psample_512_p99': 13.5,
            'fact_psample_8000_median': 1.7,
            'fact_psample_8000_p99': 10,
        },
    })

TEST_CONFIGS['test-job-light-ranges'] = dict(
    EXPERIMENT_CONFIGS['job-light-ranges'],
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'epochs': 2,
        'num_eval_queries_at_end': 88888888,
        'num_eval_queries_per_iteration': 50,
        'asserts': {
            'fact_psample_8000_median': 4,
            'fact_psample_8000_p99': 105,
            'train_bits': 70,
        },
    })

TEST_CONFIGS['job-light-ranges-reload'] = dict(
    EXPERIMENT_CONFIGS['job-light-ranges'],
    **{
        'checkpoint_to_load': 'models/job-light-ranges-pretrained.pt',
        'eval_psamples': [512, 8000],
        # Evaluating on all queries takes a while.  Shorten the wait by
        # setting this flag (adjust asserts too) during testing:
        # 'num_eval_queries_at_checkpoint_load': 50,
        'asserts': {
            'fact_psample_512_median': 2.0,
            'fact_psample_512_p99': 400,
            'fact_psample_8000_median': 1.9,
            'fact_psample_8000_p99': 400,
        },
    })

TEST_CONFIGS['test-job-light-ranges-large'] = dict(
    EXPERIMENT_CONFIGS['job-light-ranges-large'],
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'epochs': 2,
        'num_eval_queries_at_end': 88888888,
        'num_eval_queries_per_iteration': 50,
        'asserts': {
            'fact_psample_8000_median': 3,
            'fact_psample_8000_p99': 70,
            'train_bits': 68,
        },
    })

TEST_CONFIGS['job-light-ranges-large-reload'] = dict(
    EXPERIMENT_CONFIGS['job-light-ranges-large'],
    **{
        'checkpoint_to_load': 'models/job-light-ranges-large-pretrained.pt',
        'eval_psamples': [512, 8000],
        # Evaluating on all queries takes a while.  Shorten the wait by
        # setting this flag (adjust asserts too) during testing:
        # 'num_eval_queries_at_checkpoint_load': 50,
        'asserts': {
            'fact_psample_512_median': 1.55,
            'fact_psample_512_p99': 240,
            'fact_psample_8000_median': 1.45,
            'fact_psample_8000_p99': 240,
        },
    })

TEST_CONFIGS['test-job-m'] = dict(
    EXPERIMENT_CONFIGS['job-m'],
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'epochs': 2,
        'max_steps': (1 << 20) // 8192,
        'num_eval_queries_at_end': 88888888,
        'eval_psamples': [8000],
        'asserts': {
            'fact_psample_8000_median': 3,
            'fact_psample_8000_p99': 600,
            'train_bits': 150,
        },
    })

TEST_CONFIGS['job-m-reload'] = dict(
    EXPERIMENT_CONFIGS['job-m'],
    **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-m-pretrained.pt',
        ]),
        'eval_psamples': [512, 8000],
        # Evaluating on all queries takes a while.  Shorten the wait by
        # setting this flag (adjust asserts too) during testing:
        # 'num_eval_queries_at_checkpoint_load': 10,
        'asserts': {
            'fact_psample_512_median': 3.5,
            'fact_psample_512_p99': 2410,
            'fact_psample_8000_median': 3.5,
            'fact_psample_8000_p99': 2410,
        },
    })

########### OUR ERGASTF1 CONFIG ############
EXPERIMENT_CONFIGS['ergastf11'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf11',
        'join_tables': ['pitstops', 'target', 'qualifying', 'constructors', 'driverstandings', 'drivers', 'results', 'laptimes', 'races', 'constructorstandings'],
        'join_keys': {'pitstops': ['driverid'], 'target': ['driverid'], 'qualifying': ['driverid', 'constructorid'], 'constructors': ['constructorid'], 'driverstandings': ['driverid'], 'drivers': ['driverid'], 'results': ['raceid', 'driverid'], 'laptimes': ['raceid'], 'races': ['raceid'], 'constructorstandings': ['constructorid']},
        'join_clauses': ['pitstops.driverid=drivers.driverid', 'target.driverid=drivers.driverid', 'qualifying.driverid=drivers.driverid', 'qualifying.constructorid=constructors.constructorid', 'driverstandings.driverid=drivers.driverid', 'laptimes.raceid=races.raceid', 'results.raceid=races.raceid', 'results.driverid=drivers.driverid', 'constructorstandings.constructorid=constructors.constructorid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf11',
    }
)
EXPERIMENT_CONFIGS['ergastf12'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf12',
        'join_tables': ['laptimes', 'qualifying', 'races', 'target', 'constructorstandings', 'results', 'drivers', 'driverstandings', 'pitstops', 'constructorresults', 'constructors'],
        'join_keys': {'laptimes': ['driverid'], 'qualifying': ['raceid', 'driverid', 'constructorid'], 'races': ['raceid'], 'target': ['driverid'], 'constructorstandings': ['raceid'], 'results': ['raceid'], 'drivers': ['driverid'], 'driverstandings': ['driverid'], 'pitstops': ['driverid'], 'constructorresults': ['constructorid'], 'constructors': ['constructorid']},
        'join_clauses': ['laptimes.driverid=drivers.driverid', 'target.driverid=drivers.driverid', 'qualifying.driverid=drivers.driverid', 'qualifying.raceid=races.raceid', 'results.raceid=races.raceid', 'constructorstandings.raceid=races.raceid', 'driverstandings.driverid=drivers.driverid', 'pitstops.driverid=drivers.driverid', 'constructorresults.constructorid=constructors.constructorid', 'qualifying.constructorid=constructors.constructorid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf12',
    }
)
EXPERIMENT_CONFIGS['ergastf13'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf13',
        'join_tables': ['constructorresults', 'races', 'laptimes', 'results', 'constructors', 'qualifying', 'constructorstandings', 'drivers', 'pitstops', 'driverstandings', 'target'],
        'join_keys': {'constructorresults': ['raceid', 'constructorid'], 'races': ['raceid'], 'laptimes': ['raceid'], 'results': ['constructorid'], 'constructors': ['constructorid'], 'qualifying': ['driverid', 'constructorid'], 'constructorstandings': ['constructorid'], 'drivers': ['driverid'], 'pitstops': ['raceid'], 'driverstandings': ['raceid'], 'target': ['raceid']},
        'join_clauses': ['constructorresults.raceid=races.raceid', 'laptimes.raceid=races.raceid', 'constructorresults.constructorid=constructors.constructorid', 'results.constructorid=constructors.constructorid', 'qualifying.constructorid=constructors.constructorid', 'qualifying.driverid=drivers.driverid', 'constructorstandings.constructorid=constructors.constructorid', 'pitstops.raceid=races.raceid', 'driverstandings.raceid=races.raceid', 'target.raceid=races.raceid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf13',
    }
)
EXPERIMENT_CONFIGS['ergastf14'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf14',
        'join_tables': ['drivers', 'laptimes', 'results', 'qualifying', 'target', 'pitstops', 'constructorresults', 'constructors', 'driverstandings', 'races'],
        'join_keys': {'drivers': ['driverid'], 'laptimes': ['raceid', 'driverid'], 'results': ['driverid'], 'qualifying': ['driverid', 'constructorid'], 'target': ['driverid'], 'pitstops': ['driverid'], 'constructorresults': ['constructorid'], 'constructors': ['constructorid'], 'driverstandings': ['driverid'], 'races': ['raceid']},
        'join_clauses': ['qualifying.driverid=drivers.driverid', 'laptimes.driverid=drivers.driverid', 'results.driverid=drivers.driverid', 'target.driverid=drivers.driverid', 'pitstops.driverid=drivers.driverid', 'constructorresults.constructorid=constructors.constructorid', 'qualifying.constructorid=constructors.constructorid', 'driverstandings.driverid=drivers.driverid', 'laptimes.raceid=races.raceid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf14',
    }
)
EXPERIMENT_CONFIGS['ergastf15'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf15',
        'join_tables': ['constructors', 'results', 'laptimes', 'constructorstandings', 'races', 'qualifying', 'constructorresults', 'status', 'target', 'driverstandings', 'drivers'],
        'join_keys': {'constructors': ['constructorid'], 'results': ['raceid', 'statusid', 'constructorid'], 'laptimes': ['raceid'], 'constructorstandings': ['constructorid'], 'races': ['raceid'], 'qualifying': ['driverid', 'constructorid'], 'constructorresults': ['constructorid'], 'status': ['statusid'], 'target': ['raceid'], 'driverstandings': ['driverid'], 'drivers': ['driverid']},
        'join_clauses': ['results.constructorid=constructors.constructorid', 'laptimes.raceid=races.raceid', 'results.raceid=races.raceid', 'constructorstandings.constructorid=constructors.constructorid', 'constructorresults.constructorid=constructors.constructorid', 'qualifying.constructorid=constructors.constructorid', 'results.statusid=status.statusid', 'target.raceid=races.raceid', 'driverstandings.driverid=drivers.driverid', 'qualifying.driverid=drivers.driverid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf15',
    }
)
EXPERIMENT_CONFIGS['ergastf16'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf16',
        'join_tables': ['constructorstandings', 'results', 'laptimes', 'constructorresults', 'constructors', 'races', 'qualifying', 'driverstandings', 'drivers', 'pitstops', 'target'],
        'join_keys': {'constructorstandings': ['raceid', 'constructorid'], 'results': ['driverid', 'constructorid'], 'laptimes': ['raceid'], 'constructorresults': ['raceid'], 'constructors': ['constructorid'], 'races': ['raceid'], 'qualifying': ['driverid'], 'driverstandings': ['driverid'], 'drivers': ['driverid'], 'pitstops': ['driverid'], 'target': ['raceid']},
        'join_clauses': ['constructorstandings.constructorid=constructors.constructorid', 'results.constructorid=constructors.constructorid', 'constructorstandings.raceid=races.raceid', 'laptimes.raceid=races.raceid', 'constructorresults.raceid=races.raceid', 'driverstandings.driverid=drivers.driverid', 'qualifying.driverid=drivers.driverid', 'results.driverid=drivers.driverid', 'pitstops.driverid=drivers.driverid', 'target.raceid=races.raceid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf16',
    }
)
EXPERIMENT_CONFIGS['ergastf17'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf17',
        'join_tables': ['qualifying', 'results', 'constructorresults', 'target', 'drivers', 'constructors', 'races', 'constructorstandings', 'laptimes', 'driverstandings', 'pitstops'],
        'join_keys': {'qualifying': ['raceid', 'driverid', 'constructorid'], 'results': ['driverid'], 'constructorresults': ['constructorid'], 'target': ['raceid'], 'drivers': ['driverid'], 'constructors': ['constructorid'], 'races': ['raceid'], 'constructorstandings': ['constructorid'], 'laptimes': ['driverid'], 'driverstandings': ['raceid'], 'pitstops': ['driverid']},
        'join_clauses': ['qualifying.driverid=drivers.driverid', 'results.driverid=drivers.driverid', 'constructorresults.constructorid=constructors.constructorid', 'qualifying.constructorid=constructors.constructorid', 'qualifying.raceid=races.raceid', 'target.raceid=races.raceid', 'constructorstandings.constructorid=constructors.constructorid', 'laptimes.driverid=drivers.driverid', 'driverstandings.raceid=races.raceid', 'pitstops.driverid=drivers.driverid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf17',
    }
)
EXPERIMENT_CONFIGS['ergastf18'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf18',
        'join_tables': ['drivers', 'results', 'target', 'pitstops', 'qualifying', 'constructorresults', 'constructors', 'races', 'constructorstandings', 'status', 'laptimes'],
        'join_keys': {'drivers': ['driverid'], 'results': ['driverid', 'statusid', 'constructorid'], 'target': ['driverid'], 'pitstops': ['raceid', 'driverid'], 'qualifying': ['driverid'], 'constructorresults': ['constructorid'], 'constructors': ['constructorid'], 'races': ['raceid'], 'constructorstandings': ['raceid'], 'status': ['statusid'], 'laptimes': ['driverid']},
        'join_clauses': ['qualifying.driverid=drivers.driverid', 'pitstops.driverid=drivers.driverid', 'results.driverid=drivers.driverid', 'constructorresults.constructorid=constructors.constructorid', 'results.constructorid=constructors.constructorid', 'target.driverid=drivers.driverid', 'constructorstandings.raceid=races.raceid', 'pitstops.raceid=races.raceid', 'results.statusid=status.statusid', 'laptimes.driverid=drivers.driverid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf18',
    }
)
EXPERIMENT_CONFIGS['ergastf19'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf19',
        'join_tables': ['laptimes', 'results', 'constructorresults', 'constructorstandings', 'races', 'target', 'drivers', 'status', 'qualifying', 'driverstandings', 'pitstops', 'seasons'],
        'join_keys': {'laptimes': ['raceid'], 'results': ['raceid', 'statusid'], 'constructorresults': ['raceid'], 'constructorstandings': ['raceid'], 'races': ['raceid', 'year'], 'target': ['raceid', 'driverid'], 'drivers': ['driverid'], 'status': ['statusid'], 'qualifying': ['raceid'], 'driverstandings': ['raceid'], 'pitstops': ['driverid'], 'seasons': ['year']},
        'join_clauses': ['laptimes.raceid=races.raceid', 'results.raceid=races.raceid', 'constructorresults.raceid=races.raceid', 'constructorstandings.raceid=races.raceid', 'target.raceid=races.raceid', 'target.driverid=drivers.driverid', 'results.statusid=status.statusid', 'qualifying.raceid=races.raceid', 'driverstandings.raceid=races.raceid', 'pitstops.driverid=drivers.driverid', 'races.year=seasons.year'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf19',
    }
)
EXPERIMENT_CONFIGS['ergastf110'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf110',
        'join_tables': ['driverstandings', 'laptimes', 'races', 'constructorresults', 'target', 'drivers', 'qualifying', 'constructorstandings', 'pitstops', 'results', 'status'],
        'join_keys': {'driverstandings': ['raceid', 'driverid'], 'laptimes': ['driverid'], 'races': ['raceid'], 'constructorresults': ['raceid'], 'target': ['raceid'], 'drivers': ['driverid'], 'qualifying': ['driverid'], 'constructorstandings': ['raceid'], 'pitstops': ['raceid'], 'results': ['raceid', 'statusid'], 'status': ['statusid']},
        'join_clauses': ['driverstandings.driverid=drivers.driverid', 'laptimes.driverid=drivers.driverid', 'driverstandings.raceid=races.raceid', 'target.raceid=races.raceid', 'constructorresults.raceid=races.raceid', 'qualifying.driverid=drivers.driverid', 'constructorstandings.raceid=races.raceid', 'pitstops.raceid=races.raceid', 'results.raceid=races.raceid', 'results.statusid=status.statusid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf110',
    }
)
EXPERIMENT_CONFIGS['ergastf111'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf111',
        'join_tables': ['results', 'target', 'laptimes', 'races', 'drivers', 'constructorresults', 'constructorstandings', 'qualifying', 'constructors'],
        'join_keys': {'results': ['raceid'], 'target': ['raceid', 'driverid'], 'laptimes': ['driverid'], 'races': ['raceid'], 'drivers': ['driverid'], 'constructorresults': ['constructorid'], 'constructorstandings': ['constructorid'], 'qualifying': ['driverid', 'constructorid'], 'constructors': ['constructorid']},
        'join_clauses': ['results.raceid=races.raceid', 'target.raceid=races.raceid', 'laptimes.driverid=drivers.driverid', 'target.driverid=drivers.driverid', 'constructorresults.constructorid=constructors.constructorid', 'qualifying.constructorid=constructors.constructorid', 'constructorstandings.constructorid=constructors.constructorid', 'qualifying.driverid=drivers.driverid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf111',
    }
)
EXPERIMENT_CONFIGS['ergastf112'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf112',
        'join_tables': ['laptimes', 'target', 'qualifying', 'drivers', 'races', 'driverstandings'],
        'join_keys': {'laptimes': ['raceid', 'driverid'], 'target': ['raceid'], 'qualifying': ['raceid'], 'drivers': ['driverid'], 'races': ['raceid'], 'driverstandings': ['raceid']},
        'join_clauses': ['laptimes.raceid=races.raceid', 'target.raceid=races.raceid', 'qualifying.raceid=races.raceid', 'laptimes.driverid=drivers.driverid', 'driverstandings.raceid=races.raceid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf112',
    }
)
EXPERIMENT_CONFIGS['ergastf113'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf113',
        'join_tables': ['constructorstandings', 'pitstops', 'drivers', 'qualifying', 'constructors', 'constructorresults', 'races'],
        'join_keys': {'constructorstandings': ['raceid'], 'pitstops': ['raceid', 'driverid'], 'drivers': ['driverid'], 'qualifying': ['driverid', 'constructorid'], 'constructors': ['constructorid'], 'constructorresults': ['raceid'], 'races': ['raceid']},
        'join_clauses': ['constructorresults.raceid=races.raceid', 'constructorstandings.raceid=races.raceid', 'pitstops.raceid=races.raceid', 'pitstops.driverid=drivers.driverid', 'qualifying.driverid=drivers.driverid', 'qualifying.constructorid=constructors.constructorid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf113',
    }
)
EXPERIMENT_CONFIGS['ergastf114'] = dict(
    dict(dict(BASE_CONFIG, **STATS_BASE), **FACTORIZE), **{
        'dataset': 'ergastf114',
        'join_tables': ['drivers', 'results', 'pitstops', 'races', 'constructorstandings', 'driverstandings', 'qualifying'],
        'join_keys': {'drivers': ['driverid'], 'results': ['raceid', 'driverid'], 'pitstops': ['driverid'], 'races': ['raceid'], 'constructorstandings': ['raceid'], 'driverstandings': ['raceid'], 'qualifying': ['raceid']},
        'join_clauses': ['results.driverid=drivers.driverid', 'pitstops.driverid=drivers.driverid', 'driverstandings.raceid=races.raceid', 'results.raceid=races.raceid', 'qualifying.raceid=races.raceid', 'constructorstandings.raceid=races.raceid'],
        'join_root': 'races', # All partial schema contains races table.
        'join_name': 'ergastf114',
    }
)

EXPERIMENT_CONFIGS['ergastf11-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf11'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf12-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf12'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf13-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf13'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf14-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf14'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf15-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf15'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf16-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf16'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf17-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf17'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf18-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf18'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf19-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf19'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf110-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf110'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf111-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf111'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf112-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf112'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf113-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf113'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })
EXPERIMENT_CONFIGS['ergastf114-reload'] = dict(
    EXPERIMENT_CONFIGS['ergastf114'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

JOB_LARGE = {
    'join_tables': [
        'title', 'aka_title', 'cast_info', 'complete_cast', 'movie_companies',
        'movie_info', 'movie_info_idx', 'movie_keyword', 'movie_link',
        'kind_type', 'comp_cast_type', 'company_name', 'company_type',
        'info_type', 'keyword', 'link_type', 'aka_name', 'char_name',
        'name', 'person_info', 'role_type'
    ],
    'join_keys': {
        'title': ['id', 'kind_id'],
        'aka_title': ['movie_id'],
        'cast_info': ['movie_id', 'person_id', 'role_id', 'person_role_id'],
        'complete_cast': ['movie_id', 'subject_id'],
        'movie_companies': ['company_id', 'company_type_id', 'movie_id'],
        'movie_info': ['movie_id'],
        'movie_info_idx': ['info_type_id', 'movie_id'],
        'movie_keyword': ['keyword_id', 'movie_id'],
        'movie_link': ['link_type_id', 'movie_id'],
        'kind_type': ['id'],
        'comp_cast_type': ['id'],
        'company_name': ['id'],
        'company_type': ['id'],
        'info_type': ['id'],
        'keyword': ['id'],
        'link_type': ['id'],
        'aka_name': ['person_id'],
        'char_name':['id'],
        'name':['id'],
        'person_info':['person_id'],
        'role_type':['id']
    },
    'join_clauses': ['title.id=aka_title.movie_id',
                     'title.id=movie_link.movie_id',
                     'title.id=cast_info.movie_id',
                     'title.id=movie_info.movie_id',
                     'title.id=movie_info_idx.movie_id',
                     'title.kind_id=kind_type.id',
                     'title.id=movie_keyword.movie_id',
                     'title.id=movie_companies.movie_id',
                     'title.id=complete_cast.movie_id',
                     'movie_link.link_type_id=link_type.id',
                     'cast_info.person_role_id=char_name.id',
                     'cast_info.role_id=role_type.id',
                     'cast_info.person_id=name.id',
                     'movie_info_idx.info_type_id=info_type.id',
                     'movie_keyword.keyword_id=keyword.id',
                     'movie_companies.company_id=company_name.id',
                     'movie_companies.company_type_id=company_type.id',
                     'comp_cast_type.id=complete_cast.subject_id',
                     'name.id=aka_name.person_id',
                     'name.id=person_info.person_id'],
    'join_root': 'title',
    'join_how': 'outer',
    'join_name': 'job-large',
    'use_cols': 'large',
    'epochs': 1,
    'bs': 2048,
    'resmade_drop_prob': 0.1,
    'max_steps': 512,
    'loader_workers': 8,
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 16,
    'warmups': 0.15,
    # Eval:
    'compute_test_loss': False,
    'queries_csv': './queries/temp_queries.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 88888888,
    'eval_psamples': [8000],
}
JOB_LARGE_FACTORIZED = {
    'factorize': True,
    'factorize_blacklist': [],
    'factorize_fanouts': True,
    'word_size_bits': 14,
    'bs': 2048,
    'max_steps': 512,
    'epochs': 1,
    
    'epochs_per_iteration': 1,
}
EXPERIMENT_CONFIGS['job-large'] = dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_LARGE), **JOB_LARGE_FACTORIZED)
EXPERIMENT_CONFIGS['job-large-reload'] = dict(
    EXPERIMENT_CONFIGS['job-large'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt'
        ]),
        'eval_psamples': [8000],
     })

for name in TEST_CONFIGS:
    TEST_CONFIGS[name].update({'save_checkpoint_at_end': False})
EXPERIMENT_CONFIGS.update(TEST_CONFIGS)

for sf in [10, 100]:
    for stats_id in range(1, 7+1):
        EXPERIMENT_CONFIGS[f'stats_sf{sf}{stats_id}'] = dict(EXPERIMENT_CONFIGS[f'stats{stats_id}'])
        EXPERIMENT_CONFIGS[f'stats_sf{sf}{stats_id}']['dataset'] = f"stats_sf{sf}{stats_id}"
        EXPERIMENT_CONFIGS[f'stats_sf{sf}{stats_id}']['join_name'] = f"stats_sf{sf}{stats_id}"
        EXPERIMENT_CONFIGS[f'stats_sf{sf}{stats_id}-reload'] = dict(EXPERIMENT_CONFIGS[f'stats{stats_id}-reload'])
        EXPERIMENT_CONFIGS[f'stats_sf{sf}{stats_id}-reload']['dataset'] = f"stats_sf{sf}{stats_id}"
        EXPERIMENT_CONFIGS[f'stats_sf{sf}{stats_id}-reload']['join_name'] = f"stats_sf{sf}{stats_id}"
