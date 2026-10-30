"""Generate the paper figures and result tables from experiment outputs."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Embed TrueType fonts instead of Type 3 fonts in vector exports.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

# Keep the documented ``python plots/plot.py`` entry point working while using
# normal package imports everywhere else.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots.plot_funcs import (
    plot_curve_on_ax,
    plot_grouped_bar_on_ax,
    plot_grouped_bar_with_baselines_and_variants_on_ax,
    plot_grouped_box_on_ax,
    plot_grouped_box_with_baselines_and_methods_on_ax_for_dynamic_exp,
    plot_grouped_box_with_baselines_and_variants_on_ax,
)
from plots.plot_utils import (
    format_query_size_group,
    get_cardinality_estimation_time,
    get_dynamic_execution_times,
    get_dynamic_p_errors,
    get_end_to_end_time,
    get_execution_time,
    get_dace_inference_time,
    get_dace_execution_time,
    get_genjoin_inference_time,
    get_genjoin_execution_time,
    get_model_size,
    get_p_error,
    get_q_error_and_query_size,
    get_q_error_by_group_by_method,
    get_query_size_group,
    get_real_q_error,
    get_real_q_error_wo_subquery,
    get_training_time,
    get_updating_time,
)


METHODS = [
    "postgresql",
    "postgresql_ext",
    "lpbound",
    "mscn",
    "robustmscn",
    "jgmp",
    "alece",
    "price_zs",
    "price_ft",
    "deepdb",
    "neurocard",
    "bayescard",
    "factorjoin",
    "asm",
]

METHOD_NAMES = {
    'postgresql': 'PostgreSQL',
    'postgresql_ext': 'PostgreSQL-Ext',
    'lpbound': 'LpBound',
    'true': 'TrueCard',
    'mscn': 'MSCN',
    'robustmscn': 'RobustMSCN',
    'jgmp': 'JGMP',
    'alece': 'ALECE',
    'price_zs': 'PRICE',
    'price_ft': 'PRICE-FT',
    'deepdb': 'DeepDB',
    'neurocard': 'NeuroCard',
    'bayescard': 'BayesCard',
    'factorjoin': 'FactorJoin',
    'asm': 'ASM'
}

METHOD_COLORS = {
    'postgresql': '#B3B3B3',
    'postgresql_ext': '#707070',
    'lpbound': '#456F9F',

    'mscn':       '#F6C65B',
    'robustmscn': '#EE854A',

    'price_zs':   '#F3A3A6',
    'price_ft':   '#E45C55',

    'alece':      '#8F63B8',
    'jgmp':       '#59A14F',

    'deepdb':     '#83BDE3',
    'bayescard':  '#C06C9A',

    'neurocard':  '#BA41D9',
    'factorjoin': '#A67C62',
    'asm':        '#AEBB4F',
}


QUERY_SETS = ["job_light", "stats_ceb", "ergastf1"]
QUERY_SET_NAMES = {
    "job_light": "JOB-LIGHT",
    "stats_ceb": "STATS-CEB",
    "ergastf1": "ErgastF1",
    "job_m_ours": "JOB-M",
    "job_large_ours": "JOB-LARGE",
}
OVERALL_QUERY_SETS = frozenset(QUERY_SETS)
GENERALIZATION_BASELINES = ["postgresql", "postgresql_ext", "lpbound", "price_zs"]
GENERALIZATION_METHODS = ["mscn", "robustmscn", "jgmp", "alece", "price_ft"]
GENERALIZATION_VARIANTS = ["template", "default", "small"]
GENERALIZATION_HATCHES = {"default": "", "template": "\\", "small": "."}
FIGURES_DIR = Path("figures")


def _experiment_dir(queryset, method, variant="default", study=None):
    """Build an experiment directory without duplicating path rules."""
    if study is None:
        study = "overall" if queryset in OVERALL_QUERY_SETS else "scalability"
    return Path("experiments") / study / queryset / method / variant


def _generalization_dir(queryset, method, variant):
    study = "overall" if variant == "default" else "generalization"
    return _experiment_dir(queryset, method, variant, study)


def _save_figure(figure, save_path, **kwargs):
    """Save a figure as both PDF and PNG with identical layout options."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.pop("format", None)
    for image_format in ("pdf",): #("pdf", "png")
        format_path = output_path.with_suffix(f".{image_format}")
        figure.savefig(format_path, format=image_format, **kwargs)


def _log10_grouped_values(values_by_method):
    return {
        method: [np.log10(np.asarray(values)) for values in grouped_values]
        for method, grouped_values in values_by_method.items()
    }


def _column_major_legend_handles(handles, column_count):
    """Order handles top-to-bottom within each Matplotlib legend column."""
    row_count = math.ceil(len(handles) / column_count)
    return [
        handles[row * column_count + column]
        for column in range(column_count)
        for row in range(row_count)
        if row * column_count + column < len(handles)
    ]


def _method_experiment_dirs(queryset, methods=METHODS):
    excluded_methods = (
        {"factorjoin"}
        if queryset not in {"job_light", "stats_ceb"}
        else set()
    )
    return {
        method: _experiment_dir(queryset, method)
        for method in methods
        if method not in excluded_methods
    }


def _collect_generalization_values(queryset, value_reader):
    baseline_values = {
        method: [
            value_reader(_experiment_dir(queryset, method, study="overall"))
        ]
        for method in GENERALIZATION_BASELINES
    }
    variant_values = {
        variant: {
            method: [
                value_reader(_generalization_dir(queryset, method, variant))
            ]
            for method in GENERALIZATION_METHODS
        }
        for variant in GENERALIZATION_VARIANTS
    }
    return baseline_values, variant_values


def plot_subquery_q_error_per_size(
    queryset,
    ax,
    x_label=None,
    *,
    font_size=35,
    show_y_label=None,
    non_negative_under_est=True,
):
    query_size_group = get_query_size_group(queryset)
    group_names = [format_query_size_group(group) for group in query_size_group]
    experiment_dirs = _method_experiment_dirs(queryset)
    grouped_values = get_q_error_by_group_by_method(
        METHODS, experiment_dirs, query_size_group
    )
    grouped_values = _log10_grouped_values(grouped_values)

    y_limits = {
        "job_light": (-3.5, 3.5),
        "stats_ceb": (-3.5, 3.5),
        "ergastf1": (-3.5, 3.5),
        "job_m_ours": (-5.2, 5.2),
        "job_large_ours": (-7.2, 5.2),
    }
    if show_y_label is None:
        show_y_label = queryset in {
            "job_light",
            "ergastf1",
            "job_large_ours",
        }

    plot_grouped_box_on_ax(
        ax,
        METHODS,
        group_names,
        grouped_values,
        METHOD_NAMES,
        METHOD_COLORS,
        x_label="Number of Tables" if x_label is None else x_label,
        y_lim=y_limits[queryset],
        non_negative_under_est=non_negative_under_est,
        font_size=font_size,
        show_y_label=show_y_label,
        whis=(1, 99),
    )
    
def plot_subquery_relative_error_per_size_generalization(
    queryset, ax, font_size=18
):
    query_size_group = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10]]
    group_names = [r'$|\mathcal{T}_Q|\leq$' + '3', r'$|\mathcal{T}_Q|\geq$' + '4']
    baseline_dirs = {
        method: _experiment_dir(queryset, method, study="overall")
        for method in GENERALIZATION_BASELINES
    }
    baseline_values = _log10_grouped_values(
        get_q_error_by_group_by_method(
            GENERALIZATION_BASELINES, baseline_dirs, query_size_group
        )
    )
    variant_values = {}
    for variant in GENERALIZATION_VARIANTS:
        experiment_dirs = {
            method: _generalization_dir(queryset, method, variant)
            for method in GENERALIZATION_METHODS
        }
        variant_values[variant] = _log10_grouped_values(
            get_q_error_by_group_by_method(
                GENERALIZATION_METHODS, experiment_dirs, query_size_group
            )
        )

    y_limits = {
        "job_light": (-2.5, 2),
        "stats_ceb": (-3, 3),
        "ergastf1": (-2.2, 2.2),
    }
    plot_grouped_box_with_baselines_and_variants_on_ax(
        ax,
        GENERALIZATION_BASELINES + GENERALIZATION_METHODS,
        group_names,
        GENERALIZATION_VARIANTS,
        baseline_values,
        variant_values,
        METHOD_NAMES,
        METHOD_COLORS,
        GENERALIZATION_HATCHES,
        x_label="" if queryset != "ergastf1" else "Number of Tables",
        y_label="" if queryset in {"job_light", "ergastf1"} else \
            "underest. <- log(q-error) -> overest.",
        y_lim=y_limits[queryset],
        non_negative_under_est=True,
        dataset=QUERY_SET_NAMES[queryset],
        group_spacing=2,
        show_x_ticks=queryset == "ergastf1",
        boundary_line_style=None,
        font_size=font_size,
    )


def plot_p_error_per_dataset_generalization(
    ax, queryset, font_size=18, y_tick_font_size=None, x_label_pad=None
):
    baseline_values, variant_values = _collect_generalization_values(
        queryset, get_p_error
    )
    axis_settings = {
        "job_light": ((1, 2), [1, 1.5, 2]),
        "stats_ceb": ((1, 4), [1, 2, 3, 4]),
        "ergastf1": ((1, 3), [1, 2, 3]),
    }
    y_lim, y_ticks = axis_settings[queryset]

    plot_grouped_box_with_baselines_and_variants_on_ax(
        ax,
        GENERALIZATION_BASELINES + GENERALIZATION_METHODS,
        [QUERY_SET_NAMES[queryset]],
        GENERALIZATION_VARIANTS,
        baseline_values,
        variant_values,
        METHOD_NAMES,
        METHOD_COLORS,
        GENERALIZATION_HATCHES,
        x_label="",
        y_label="P-Error" if queryset == "job_light" else None,
        y_lim=y_lim,
        y_ticks=y_ticks,
        y_ticks_label=[str(tick) for tick in y_ticks],
        apply_log=True,
        show_x_ticks=False,
        font_size=font_size,
        y_tick_font_size=y_tick_font_size,
        x_label_pad=x_label_pad,
    )


def plot_execution_time_per_dataset_generalization(
    ax, queryset, font_size=18, y_tick_font_size=None, x_label_pad=None
):
    baseline_values, variant_values = _collect_generalization_values(
        queryset,
        lambda experiment_dir: np.sum(get_execution_time(experiment_dir)),
    )
    y_limits = {
        "job_light": (900, 1300),
        "stats_ceb": (2000, 8000),
        "ergastf1": (5000, 9000),
    }

    plot_grouped_bar_with_baselines_and_variants_on_ax(
        ax,
        GENERALIZATION_BASELINES + GENERALIZATION_METHODS,
        [QUERY_SET_NAMES[queryset]],
        GENERALIZATION_VARIANTS,
        baseline_values,
        variant_values,
        METHOD_NAMES,
        METHOD_COLORS,
        GENERALIZATION_HATCHES,
        y_label=(
            "Plan Exec. Time (s)" if queryset == "job_light" else None
        ),
        y_lim=y_limits[queryset],
        font_size=font_size,
        y_tick_font_size=y_tick_font_size,
        x_label_pad=x_label_pad,
    )


# Preserve the original public names for downstream scripts.
plot_perror_per_dataset_generalization = plot_p_error_per_dataset_generalization
plot_exec_time_per_dataset_generalization = (
    plot_execution_time_per_dataset_generalization
)


def plot_p_error_with_legends():
    """Plot p-error distributions with method legends."""
    save_path = FIGURES_DIR / "p_error.pdf"
    font_size = 15
    group_names = [QUERY_SET_NAMES[queryset] for queryset in QUERY_SETS]
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.9))

    p_error_values = {method: [] for method in METHODS}
    for queryset in QUERY_SETS:
        for method in METHODS:
            experiment_dir = _experiment_dir(queryset, method)
            if queryset == "ergastf1" and method == "factorjoin":
                p_error_values[method].append(None)
            else:
                p_error_values[method].append(get_p_error(experiment_dir))

    plot_grouped_box_on_ax(
        ax,
        METHODS,
        group_names,
        p_error_values,
        METHOD_NAMES,
        METHOD_COLORS,
        spacing=1.5,
        y_lim=(1, 7),
        yticks=[1, 2, 3, 4, 5, 6, 7],
        x_label="",
        y_label="P-Error",
        apply_Log=True,
        whis=(1, 99),
        show_x_ticks=True,
        font_size=font_size,
        item_step=1.15,
        y_label_pad=20
    )

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[method], label=METHOD_NAMES[method])
        for method in METHODS
    ]
    legend_handles = _column_major_legend_handles(legend_handles, 5)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.024),
        ncol=5,
        frameon=True,
        fontsize=font_size,
        handlelength=1.6,
        handleheight=0.7,
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.2,
        borderpad=0.3
    )
    fig.subplots_adjust(
        top=0.762,
        bottom=0.067,
        left=0.075,
        right=0.995,
    )
    _save_figure(fig, save_path)
    plt.show()


def plot_spearman():
    """Plot the two Spearman correlation distributions without legends."""
    save_path = FIGURES_DIR / "spearman.pdf"
    font_size = 15
    group_names = [QUERY_SET_NAMES[queryset] for queryset in QUERY_SETS]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6))

    spearman_settings = [
        ("query_spearman", "Subquery Rank Correlation"),
        ("plan_spearman", "Plan Rank Correlation"),
    ]
    for axis_index, (column, y_label) in enumerate(spearman_settings):
        values_by_method = {method: [] for method in METHODS}
        for queryset in QUERY_SETS:
            for method in METHODS:
                csv_path = _experiment_dir(queryset, method) / "spearman.csv"
                if not csv_path.exists():
                    values_by_method[method].append(None)
                    continue
                values = pd.read_csv(csv_path)[column].dropna().to_numpy()
                values = values[np.isfinite(values)]
                values_by_method[method].append(values if values.size else None)

        plot_grouped_box_on_ax(
            axes[axis_index],
            METHODS,
            group_names,
            values_by_method,
            METHOD_NAMES,
            METHOD_COLORS,
            spacing=1.5,
            y_lim=(0.5, 1),
            yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1],
            x_label="",
            y_label=y_label,
            whis=(1, 99),
            show_x_ticks=axis_index == 1,
            font_size=font_size,
            item_step=1.15
        )
        axes[axis_index].set_yticklabels(
            [f"{tick:.1f}" for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
            fontsize=font_size,
        )

    fig.subplots_adjust(
        top=0.97,
        bottom=0.05,
        left=0.075,
        right=0.995,
        hspace=0.08,
    )
    _save_figure(fig, save_path)
    plt.show()


def plot_training_time_per_dataset(
    ax, methods, method_hatch, font_size=30, alpha=1.0, hatch_color=None
):
    group_names = [QUERY_SET_NAMES[queryset] for queryset in QUERY_SETS]
    values_by_group_by_method = {method: [] for method in methods}
    for queryset in QUERY_SETS:
        for method in methods:
            if queryset == "ergastf1" and method == "factorjoin":
                values_by_group_by_method[method].append(None)
                continue
            training_time_min = (
                get_training_time(_experiment_dir(queryset, method)) / 60.0
            )
            values_by_group_by_method[method].append(training_time_min)
    plot_grouped_bar_on_ax(ax, methods, group_names, values_by_group_by_method, METHOD_NAMES, METHOD_COLORS, 
                           y_label="Training Time (min)", y_lim=(None, 1e3), y_ticks=[1, 10, 100, 1000],
                           apply_Log=True, spacing=1.5, show_x_tick_label=False,
                           method_hatches=method_hatch, font_size=font_size,
                           alpha=alpha, hatch_color=hatch_color)

def plot_model_size_per_dataset(
    ax, methods, method_hatch, font_size=30, alpha=1.0, hatch_color=None
):
    group_names = [QUERY_SET_NAMES[queryset] for queryset in QUERY_SETS]
    values_by_group_by_method = {method: [] for method in methods}
    for queryset in QUERY_SETS:
        for method in methods:
            if queryset == "ergastf1" and method == "factorjoin":
                values_by_group_by_method[method].append(None)
                continue
            model_size_mb = get_model_size(_experiment_dir(queryset, method))
            values_by_group_by_method[method].append(model_size_mb)
    plot_grouped_bar_on_ax(ax, methods, group_names, values_by_group_by_method, METHOD_NAMES, METHOD_COLORS, 
                           y_label="Model Size (MB)", y_lim=(None, 1e3), y_ticks=[1, 10, 100, 1000], apply_Log=True, spacing=1.5, 
                           method_hatches=method_hatch, font_size=font_size,
                           alpha=alpha, hatch_color=hatch_color)

def plot_q_error_per_training_data_num_curve(queryset, ax, line_styles, perc=90, excluded_subquery=False, tick_font_size=15):
    """Plot the 95% q-error curve with varied # of training data 

    Args:
        queryset (str): the query set to be ploted
        ax (str): the ax to plot
    """
    group_names = ['500', '1000', '5000', '10000', '50000', '100000']
    xticklabels = [
        r'0.5K', 
        r'1K', 
        r'5K', 
        r'10K', 
        r'50K', 
        r'100K'
    ]
    
    values_by_group_by_method = {}
    for m in METHODS:
        values_each_g = []
        for g in group_names:
            if m == 'postgresql' or m == 'lpbound' or m == 'price_zs':
                exp_dir = f"experiments/overall/{queryset}/{m}/default/"
                
            elif m in ['mscn', 'robustmscn', 'jgmp', 'alece', 'price_ft']:
                if g == '100000':
                    exp_dir = f"experiments/overall/{queryset}/{m}/default/"
                else:
                    exp_dir = f"experiments/training_data_efficiency/{queryset}/{m}/{g}"
            else:
                continue
            if not excluded_subquery:
                values = get_real_q_error(exp_dir)
            else:
                values = get_real_q_error_wo_subquery(exp_dir)
            values_each_g.append(np.percentile(values, perc))
            
        if len(values_each_g) > 0:
            values_by_group_by_method[m] = values_each_g
    if excluded_subquery:
        y_lim = (0.99, 2e3)
        y_ticks=[1, 10, 100, 1000]
    else:
        y_lim = (0.99, 5e4)
        y_ticks=[1, 10, 100, 1000,10000]
        if queryset == 'job_light':
            y_lim = (0.99, 1e4)
            y_ticks = [1,10,100,1000,10000]
        elif queryset == 'stats_ceb':
            y_lim = (0.99, 1e3)
            y_ticks = [1,10,100,1000]
    plot_curve_on_ax(ax, METHODS, xticklabels, values_by_group_by_method, METHOD_NAMES, METHOD_COLORS, line_styles,
                     apply_log=True, y_ticks=y_ticks, y_lim=y_lim, tick_fontsize=tick_font_size)

def plot_p_error_per_training_data_num_curve(queryset, ax, line_styles, perc=90, tick_font_size=15):
    """Plot the 95% p-error curve with varied # of training data 

    Args:
        queryset (str): the query set to be ploted
        ax (str): the ax to plot
    """
    group_names = ['500', '1000', '5000', '10000', '50000', '100000']
    xticklabels = [
        r'0.5K', 
        r'1K', 
        r'5K', 
        r'10K', 
        r'50K', 
        r'100K'
    ]
    
    values_by_group_by_method = {}
    for m in METHODS:
        values_each_g = []
        for g in group_names:
            if m == 'postgresql' or m == 'lpbound' or m == 'price_zs':
                exp_dir = f"experiments/overall/{queryset}/{m}/default/"
                values_each_g.append(np.percentile(get_p_error(exp_dir), perc))
            elif m in ['mscn', 'robustmscn', 'jgmp', 'alece', 'price_ft']:
                if g == '100000':
                    exp_dir = f"experiments/overall/{queryset}/{m}/default/"
                    values_each_g.append(np.percentile(get_p_error(exp_dir), perc))
                else:
                    exp_dir = f"experiments/training_data_efficiency/{queryset}/{m}/{g}"
                    values_each_g.append(np.percentile(get_p_error(exp_dir), perc))
        if len(values_each_g) > 0:
            values_by_group_by_method[m] = values_each_g

    if queryset == 'job_light':
        y_ticks=[1,2,3,4]
        y_lim=(1,4)
    elif queryset == 'stats_ceb':
        y_ticks=[1,2,3,5,7]
        y_lim=(1,7)
    else:
        y_ticks=[1,2,3,4,5,6]
        y_lim=(1,6)
        
    plot_curve_on_ax(ax, METHODS, xticklabels, values_by_group_by_method, METHOD_NAMES, METHOD_COLORS, line_styles,
                     apply_log=True, y_ticks=y_ticks, y_tick_labels= [str(i) for i in y_ticks], y_lim=y_lim, tick_fontsize=tick_font_size)

def plot_q_error():
    """Plot q-error distributions for all three datasets vertically."""
    font_size, super_font_size = 18, 18
    save_path = FIGURES_DIR / "q_error.pdf"
    fig, axes = plt.subplots(len(QUERY_SETS), 1, figsize=(18, 11))

    for axis, queryset in zip(axes, QUERY_SETS):
        plot_subquery_q_error_per_size(
            queryset,
            axis,
            x_label="",
            font_size=font_size,
            show_y_label=False,
            non_negative_under_est=True,
        )
        axis.text(
            0.01,
            0.95,
            QUERY_SET_NAMES[queryset],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=super_font_size,
            bbox={
                "boxstyle": "square,pad=0.3",
                "facecolor": "#F0F8FF",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        )

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[method], label=METHOD_NAMES[method])
        for method in METHODS
    ]
    legend_handles = _column_major_legend_handles(legend_handles, 7)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.019),
        ncol=7,
        frameon=False,
        fontsize=font_size,
        handlelength=1.6,
        handleheight=0.7,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.supxlabel("Number of Tables", fontsize=super_font_size, y=0.0015)
    fig.supylabel(
        "underestimation <- log(q-error) -> overestimation",
        fontsize=super_font_size,
        x=0.005,
    )
    fig.subplots_adjust(
        top=0.93,
        bottom=0.05,
        left=0.04,
        right=0.98,
        hspace=0.12
    )

    _save_figure(fig, save_path)
    plt.show()


def plot_model_size_and_training_time():
    old_hatch_linewidth = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.linewidth'] = 0.8
    font_size = 15
    alpha = 0.8
    hatch_color = "black"
    save_path = FIGURES_DIR / "model_size_and_training_time.pdf"
    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    
    methods = [
        method
        for method in METHODS
        if method not in {'postgresql', 'postgresql_ext', 'lpbound', 'price_zs'}
    ]
    # hatches = ['/', '\\', '|', '-', '*', 'o', '+', '.', 'X', 'OX', 'xx']
    
    hatches = [
        '/',
        '\\',
        '|',
        '-',
        '+',
        'x',
        '//',
        '\\\\',
        '||',
        '--',
        '++',
    ]
    
    method_hatch = {m:h for m, h in zip(methods, hatches)}
    
    for i, ax in enumerate(axes.flatten()):
        if i == 0:
            plot_training_time_per_dataset(
                ax, methods, method_hatch, font_size, alpha, hatch_color
            )
        else:
            plot_model_size_per_dataset(
                ax, methods, method_hatch, font_size, alpha, hatch_color
            )
    
    legend_handles = [
        mpatches.Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor=hatch_color,
            hatch=method_hatch[m],
            linewidth=0,
            alpha=alpha,
            label=METHOD_NAMES[m] if m != 'price_ft' else 'PRICE-FT',
        ) 
        for m in methods
    ]
    
    legend_handles = _column_major_legend_handles(legend_handles, 5)
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.015),
            ncol=5, frameon=True,fontsize=font_size,handlelength=1.8,handleheight=0.8,
            labelspacing=0.3, columnspacing=1, handletextpad=0.5,)
        
    fig.subplots_adjust(
        top=0.84,
        bottom=0.07,
        left=0.08,
        right=0.995,
        hspace=0.1
    )
    
    _save_figure(fig, save_path)
    plt.show()
    
    plt.rcParams['hatch.linewidth'] = old_hatch_linewidth

def print_exec_result_table():
    def my_round(f:float):
        if f == 0.0 or f is None:
            return '--'
        elif f<100:
            return f"{f:.3f}"
        elif f<1000:
            return f"{f:.2f}"
        elif f < 10000:
            return f"{f:.1f}"
        else:
            return str(int(round(f)))
    import pandas as pd
    import math
    method_names_all = METHOD_NAMES.copy()
    method_names_all.update({
        'dace': 'DACE',
        'genjoin': 'GenJoin',
    })
    for m in METHODS + ['dace', 'genjoin', 'true']:
        res = []
        for qs in ['job_light', 'stats_ceb', 'ergastf1',]:
            if m == 'factorjoin' and qs not in ['job_light', 'stats_ceb']:
                continue
            if m == 'dace':
                plan_time = get_dace_inference_time(qs)
                exec_time = get_dace_execution_time(qs)
                e2e_time  = plan_time + exec_time
            elif m == 'genjoin':
                plan_time = get_genjoin_inference_time(qs)
                exec_time = get_genjoin_execution_time(qs)
                e2e_time  = plan_time + exec_time
            else:
                if qs not in ['job_m_ours', 'job_large_ours']:
                    path = f"./experiments/overall/{qs}/{m}/default"
                else:
                    path = f"./experiments/scalability/{qs}/{m}/default"
                plan_time = get_cardinality_estimation_time(path)
                exec_time = get_execution_time(path)
                e2e_time  = get_end_to_end_time(path)
            total_plan = my_round(np.sum(plan_time))
            total_exec = f"{round(float(np.sum(exec_time)))}"
            total_e2e = f"{round(float(np.sum(e2e_time)))}"
            res.extend([total_e2e, total_exec, total_plan])
        print(f"& {method_names_all[m]} & {' & '.join([str(r) for r in res])} \\\\")


def print_asm_hybrid_table():
    """Print LaTeX rows comparing PostgreSQL, ASM, and ASM-Hybrid Q-Error."""
    methods = ['postgresql', 'asm', 'asm_hybrid']
    method_names = {
        'postgresql': 'PostgreSQL',
        'asm': 'ASM',
        'asm_hybrid': 'ASM-Hybrid',
    }
    percentiles = [50, 80, 90, 95, 99]

    for queryset in QUERY_SETS:
        for method in methods:
            experiment_dir = _experiment_dir(queryset, method)
            q_errors = get_real_q_error(experiment_dir)
            values = [
                f"{np.percentile(q_errors, percentile):.3f}"
                for percentile in percentiles
            ]
            print(
                f"& {method_names[method]} & {' & '.join(values)} \\\\"
            )
        print()


def print_execution_time_over_training_query_num():
    def round_to_int(f: float):
        if f > 18000:
            return 'TLE'
        return str(int(round(f)))
    
    querysets = ['job_light', 'stats_ceb', 'ergastf1']
    methods = [ 'mscn', 'robustmscn', 'jgmp', 'alece', 'price_ft', 'postgresql', 'postgresql_ext', 'lpbound', 'price_zs']
    query_num = [500, 1000, 5000, 10000, 50000, 100000]
    
    result_dict = {}
    
    for qs in querysets:
        result_dict[qs] = {}
        for m in methods:
            result_dict[qs][m] = []
            for qn in query_num:
                if qn == 100000:
                    exp_dir = f"experiments/overall/{qs}/{m}/default/"
                elif m in ['postgresql', 'postgresql_ext', 'lpbound', 'price_zs']:
                    exp_dir = f"experiments/overall/{qs}/{m}/default/"
                else:
                    exp_dir = f"experiments/training_data_efficiency/{qs}/{m}/{qn}"
                exec_time = round_to_int(np.sum(get_execution_time(exp_dir)))
                result_dict[qs][m].append(exec_time)

    for qs in querysets:
        print("============{}============".format(qs))
        for m in methods:
            print(" & ".join(["     ", METHOD_NAMES[m]] + result_dict[qs][m]) + "\\\\")
    

def print_execution_time_dynamic():
    def custom_format(f: float):
        if f is None:
            return '--'
        if f > 18000:
            return 'TLE'
        return str(int(round(f)))
    
    baselines = ['postgresql', 'postgresql_ext', 'lpbound', 'true']
    methods = ['robustmscn', 'alece', 'price_zs', 'price_ft', 'deepdb_i', 'deepdb_s', 'deepdb_f', 'bayescard_i', 'bayescard_s', 'bayescard_f', 'neurocard', 'factorjoin', 'asm']
    
    method_names = {
        'deepdb_s': 'DeepDB-REF',
        'deepdb_i': 'DeepDB-INC',
        'deepdb_f': 'DeepDB-TF',
        'bayescard_s': 'BayesCard-REF',
        'bayescard_i': 'BayesCard-INC',
        'bayescard_f': 'BayesCard-TF'
    }
    for m in baselines + methods:
        if m in METHOD_NAMES:
            method_names[m] = METHOD_NAMES[m]
        
    values_by_method_by_variant_by_group_by_queryset = {}
    for queryset in ['job_light', 'stats_ceb']:
        values_by_method_by_variant_by_group_by_queryset[queryset] = {'20':{}, '40':{}, '60':{}, '80':{}, '100':{}}
        values_by_method_by_variant_by_group_by_queryset[queryset]['100']['overall'] = {}
        for m in baselines + methods:
            if m.startswith('deepdb') or m.startswith('bayescard'):
                exp_dir = f"experiments/overall/{queryset}/{m.split('_')[0]}/default"
            else:
                exp_dir = f"experiments/overall/{queryset}/{m}/default"
            exec_time = np.sum(get_execution_time(exp_dir))
            values_by_method_by_variant_by_group_by_queryset[queryset]['100']['overall'][m] = exec_time
        for g in ['20', '40', '60', '80']:
            for m in methods:
                if m.startswith('deepdb') or m.startswith('bayescard'):
                    option = 'structure' if m.split('_')[1] == 's' else 'incremental' if m.split('_')[1] == 'i' else 'filter'
                    exp_dir = f"experiments/dynamic/{queryset}/{m.split('_')[0]}/{g}_{option}/"
                else:
                    exp_dir = f"experiments/dynamic/{queryset}/{m}/{g}/"
                exec_stale, exec_dyn = get_dynamic_execution_times(exp_dir)
                exec_stale = None if exec_stale is None else np.sum(exec_stale)
                exec_dyn = None if exec_dyn is None else np.sum(exec_dyn)
                for v in ['stale', 'updated']:
                    if v not in values_by_method_by_variant_by_group_by_queryset[queryset][g]:
                        values_by_method_by_variant_by_group_by_queryset[queryset][g][v] = {}
                    values_by_method_by_variant_by_group_by_queryset[queryset][g][v][m] = exec_stale if v == 'stale' else exec_dyn

    for m in baselines + methods:
        result = [f"{method_names[m]}"]
        for queryset in ['job_light', 'stats_ceb']:
            for g in ['20', '40', '60', '80', '100']:
                if g == '100':
                    exec_time = values_by_method_by_variant_by_group_by_queryset[queryset][g]['overall'][m]
                    result.append(custom_format(exec_time))
                else:
                    exec_stale = values_by_method_by_variant_by_group_by_queryset[queryset][g]['stale'].get(m, None)
                    exec_dyn = values_by_method_by_variant_by_group_by_queryset[queryset][g]['updated'].get(m, None)
                    result.append(f"{custom_format(exec_stale)}")
                    result.append(f"{custom_format(exec_dyn)}")
        print(" & ".join(result) + " \\\\")

def print_scalability_result_table(queryset):
    methods_all = ['postgresql', 'postgresql_ext', 'lpbound', 'mscn', 'robustmscn', 'jgmp', 'alece', 'price_zs','price_ft', 'deepdb', 'neurocard', 'bayescard', 'asm', 'true']
    
    def get_time_info(experiment_dir):
        exec_time = np.sum(get_execution_time(experiment_dir))
        card_time = np.sum(get_cardinality_estimation_time(experiment_dir))
        e2e_time = np.sum(get_end_to_end_time(experiment_dir))
        
        encode_time = .0
        query_driven = any(
            method in str(experiment_dir)
            for method in [
                'mscn', 'robustmscn', 'jgmp', 'alece', 'price_zs', 'price_ft'
            ]
        )
        
        if query_driven:
            prefix = 'Encode time:'
            
            with (Path(experiment_dir) / 'predict.log').open(
                'r', encoding='utf-8'
            ) as f:
                lines = f.readlines()
                
                i = len(lines) - 1
                
                start = False
                
                while i >= 0:
                    if "======================================" in lines[i]:
                        if start:
                            break
                        else:
                            start = True
                    if start and lines[i].startswith(prefix):
                        encode_time += float(lines[i].strip().split(prefix)[1].strip())
                        
                    i -= 1    
        
        return e2e_time, exec_time, card_time, encode_time
    
    def get_q_error_p_error_percentile(experiment_dir):
        q_error = get_real_q_error(experiment_dir)
        p_error = get_p_error(experiment_dir)
        q_error_perc = []
        p_error_perc = []
        for perc in [50, 80, 90, 95, 99]:
            q_error_perc.append(np.percentile(q_error, perc))
            p_error_perc.append(np.percentile(p_error, perc))
        return q_error_perc, p_error_perc

    def get_model_size_training_time(experiment_dir):
        if any(
            method in str(experiment_dir)
            for method in ['true']
        ):
            return .0, .0
        model_size = get_model_size(experiment_dir)
        training_time = get_training_time(experiment_dir)
            
        return model_size, training_time / 60.0
    
    def format_1(n: float):
        if abs(n) < 1e-7:
            return '--'
        elif abs(n) < 1000:
            return f"{n:.2f}"  
        else:
            return f"{int(round(n))}"
    
    def format_2(n: float):
        if abs(n) < 1e-7:
            return '-'
        elif abs(n) < 10000:
            return f"{n:.2f}" 
        else:
            log_10 = math.floor(math.log10(n))
            return f"${n/(10**log_10):.2f} \\cdot 10^{log_10}$"
    
    for m in methods_all:
        experiment_dir = _experiment_dir(queryset, m, study="scalability")
        result = [METHOD_NAMES[m]]

        time_tuple = get_time_info(experiment_dir)
        for t in time_tuple:
            result.append(format_1(t))
        
        q_error_perc, p_error_perc = get_q_error_p_error_percentile(experiment_dir)
        for perc in q_error_perc + p_error_perc:
            result.append(format_2(perc))
        
        model_size, training_time = get_model_size_training_time(experiment_dir)
        result.append(format_1(model_size))
        result.append(format_2(training_time))

        print("& " + f"{' & '.join([str(r) for r in result])} \\\\")
        

def plot_training_data_efficiency_curves():
    """Plot Q-Error and P-Error training-data efficiency in one figure."""
    save_path = FIGURES_DIR / "training_data_efficiency_curves.pdf"
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    font_size = 15
    line_styles = {
        'postgresql': ('-', 's', 10),
        'lpbound': ('-', '*', 10),
        'price_zs': ('-', 'v', 10),
        'mscn': ('-', 'D', 10),
        'robustmscn': ('-', '^', 10),
        'jgmp': ('-', 'p', 11),
        'alece': ('-', 'o', 10),
        'price_ft': ('-', 'X', 12),
    }

    for column, queryset in enumerate(QUERY_SETS):
        plot_q_error_per_training_data_num_curve(
            queryset, axes[0, column], line_styles, perc=90, tick_font_size=font_size
        )
        plot_p_error_per_training_data_num_curve(
            queryset, axes[1, column], line_styles, perc=90, tick_font_size=font_size
        )

        axes[0, column].set_title(
            QUERY_SET_NAMES[queryset], fontsize=font_size
        )
        axes[0, column].tick_params(
            axis="x", which="both", bottom=False, labelbottom=False
        )

    axes[0, 0].set_ylabel("Subquery Q-Error 90%", fontsize=font_size, labelpad=2)
    axes[1, 0].set_ylabel("P-Error 90%", fontsize=font_size, labelpad=16)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = _column_major_legend_handles(handles, 5)
    labels = _column_major_legend_handles(labels, 5)
    fig.supxlabel('# of training queries', fontsize=font_size, y=0.1)
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.025),
        ncol=len(labels) // 2,
        frameon=False,
        fontsize=font_size,
        columnspacing=2,
        handletextpad=1,
        handlelength=0,
        labelspacing=0.34,
        markerscale=1.2,
    )
    
    fig.subplots_adjust(
        top=0.96,
        bottom=0.22,
        left=0.08,
        right=0.99,
        wspace=0.21,
        hspace=0.1
    )

    _save_figure(fig, save_path)
    plt.show()


def plot_training_data_efficiency_mscn_hybrid_curves():
    """Plot Q-Error training-data efficiency for four selected methods."""
    save_path = FIGURES_DIR / "training_data_efficiency_mscn_hybrid_curves.pdf"
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    font_size = 14
    methods = ['postgresql', 'mscn', 'mscn_hybrid']
    method_names = {
        **METHOD_NAMES,
        'mscn_hybrid': 'MSCN-Hybrid',
    }
    method_colors = {
        **METHOD_COLORS,
        'mscn_hybrid': '#2A9D8F',
    }
    line_styles = {
        'postgresql': ('-', 's', 10),
        'lpbound': ('-', '*', 10),
        'mscn': ('-', 'D', 10),
        'mscn_hybrid': ('-', 'P', 10),
    }
    group_names = ['500', '1000', '5000', '10000', '50000', '100000']
    xticklabels = [r'0.5K', r'1K', r'5K', r'10K', r'50K', r'100K']

    for column, queryset in enumerate(QUERY_SETS):
        values_by_group_by_method = {}
        for method in methods:
            values_each_group = []
            for group_name in group_names:
                if method in ('postgresql', 'lpbound') or group_name == '100000':
                    experiment_dir = Path(
                        f"experiments/overall/{queryset}/{method}/default"
                    )
                else:
                    experiment_dir = Path(
                        "experiments/training_data_efficiency"
                    ) / queryset / method / group_name
                values = get_real_q_error(experiment_dir)
                values_each_group.append(np.percentile(values, 90))
            values_by_group_by_method[method] = values_each_group

        if queryset == 'job_light':
            y_lim = (0.99, 1e3)
            y_ticks = [1, 10, 100, 1000]
        elif queryset == 'stats_ceb':
            y_lim = (0.99, 1e3)
            y_ticks = [1, 10, 100, 1000]
        else:
            y_lim = (0.99, 5e4)
            y_ticks = [1, 10, 100, 1000, 10000]

        plot_curve_on_ax(
            axes[column],
            methods,
            xticklabels,
            values_by_group_by_method,
            method_names,
            method_colors,
            line_styles,
            apply_log=True,
            y_ticks=y_ticks,
            y_lim=y_lim,
            tick_fontsize=font_size,
        )
        axes[column].set_title(QUERY_SET_NAMES[queryset], fontsize=font_size)

    axes[0].set_ylabel(
        "Subquery Q-Error 90%", fontsize=font_size, labelpad=1.5
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.supxlabel('# of training queries', fontsize=font_size, y=0.112)
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        frameon=False,
        fontsize=font_size,
        columnspacing=3,
        handletextpad=1,
        handlelength=0,
        markerscale=1.2,
    )

    fig.subplots_adjust(
        top=0.91,
        bottom=0.35,
        left=0.08,
        right=0.99,
        wspace=0.21,
    )

    _save_figure(fig, save_path)
    plt.show()


def plot_q_error_by_size_generalization_study():
    font_size = 15
    old_hatch_linewidth = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.linewidth'] = 0.5  
    save_path = FIGURES_DIR / "generalization_q_error.pdf"
    fig, axes = plt.subplots(3, 1, figsize=(9, 6))
    for i, ax in enumerate(axes.flatten()):
        plot_subquery_relative_error_per_size_generalization(
            QUERY_SETS[i], ax, font_size
        )
    
    methods_all = ['postgresql', 'postgresql_ext', 'lpbound', 'price_zs', 'mscn', 'robustmscn', 'jgmp', 'alece', 'price_ft']
    variants = ['Template', 'Random', 'OOD']
    variant_hatches = {'Template': '\\', 'Random': '', 'OOD': '.'}
    
    legend_handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], edgecolor='#606060', label=METHOD_NAMES[m], linewidth=1) for m in methods_all]
    #'#E3E3E3'
    variant_handles = [mpatches.Patch(facecolor='white', edgecolor='#606060', label=v, hatch=variant_hatches[v]*2, linewidth=1) for v in variants]
    legends_all = _column_major_legend_handles(legend_handles + variant_handles, 6)

    fig.legend(
        handles=legends_all, 
        ncol=6, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,         
        fontsize=font_size,
        handlelength=1.3, 
        handleheight=0.7, 
        labelspacing=0.27,
        columnspacing=0.28,
        handletextpad=0.18
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.08,
        left=0.05,
        right=0.999,
        hspace=0.08,
    )
    _save_figure(fig, save_path)
    plt.show()
    plt.rcParams['hatch.linewidth'] = old_hatch_linewidth

def plot_p_error_exec_time_generalization_study():
    font_size = 15
    y_tick_font_size = 12
    p_error_x_label_pad = 13
    plan_exec_time_x_label_pad = 2
    old_hatch_linewidth = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.linewidth'] = 0.5 
    
    save_path = FIGURES_DIR / "generalization_p_error_exec_time.pdf"
    querysets = ['job_light', 'stats_ceb', 'ergastf1']
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, ax in enumerate(axes.flatten()):
        if i // 3 == 0:
            plot_p_error_per_dataset_generalization(
                ax,
                querysets[i%3],
                font_size,
                y_tick_font_size,
                p_error_x_label_pad,
            )
        else:
            plot_execution_time_per_dataset_generalization(
                ax,
                querysets[i%3],
                font_size,
                y_tick_font_size,
                plan_exec_time_x_label_pad,
            )
    
    methods_all = ['postgresql', 'postgresql_ext', 'lpbound', 'price_zs', 'mscn', 'robustmscn', 'jgmp', 'alece', 'price_ft']
    variants = ['Template', 'Random', 'OOD']
    variant_hatches = {'Template': '\\', 'Random': '', 'OOD': '.'}
    
    legend_handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], edgecolor='#606060', label=METHOD_NAMES[m], linewidth=1) for m in methods_all]
    #'#E3E3E3'
    variant_handles = [mpatches.Patch(facecolor='white', edgecolor='#606060', label=v, hatch=variant_hatches[v]*2, linewidth=1) for v in variants]
    legends_all = _column_major_legend_handles(legend_handles + variant_handles, 6)

    fig.legend(
        handles=legends_all,
        ncol=6,
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        fontsize=font_size,
        handlelength=1.3,
        handleheight=0.7,
        columnspacing=0.28,
        handletextpad=0.18,
        labelspacing=0.27,
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.045,
        left=0.08,
        right=0.999,
        wspace=0.2,
        hspace=0.12,
    )
    _save_figure(fig, save_path)
    plt.show()
    plt.rcParams['hatch.linewidth'] = old_hatch_linewidth

def plot_updating_time_per_dataset(
    ax,
    methods,
    method_name,
    method_color,
    method_hatch,
    queryset,
    font_size=30,
    alpha=1.0,
    hatch_color=None,
):
    x_tick_pad = 1.4
    bar_linewidth = 0.5
    group_names = ["20%", "40%", "60%", "80%"]
    values_by_group_by_method = {m: [0.0 for _ in range(len(group_names))] for m in methods}
    for method in methods:
        for group_name in group_names:
            if method.startswith('deepdb') or method.startswith('bayescard'):
                method_new = method.split('_')[0]
                option = method.split('_')[1]
                exp_name = group_name[:-1] + '_' + ('structure' if option == 's' else 'incremental' if option == 'i' else 'filter')
            else:
                method_new = method
                exp_name = group_name[:-1]
            updating_time = get_updating_time(f'experiments/dynamic/{queryset}/{method_new}/{exp_name}/')
            values_by_group_by_method[method][group_names.index(group_name)] = updating_time
    if queryset == 'job_light':
        y_ticks = [1, 10, 100, 1000]
        dataset = "JOB-LIGHT"
    else:
        y_ticks = [1, 10, 100, 1000,]
        dataset = "STATS-CEB"
    plot_grouped_bar_on_ax(ax, methods, group_names, values_by_group_by_method, method_name, method_color, y_ticks=y_ticks,
                           y_label=None, apply_Log=True, spacing=3, show_x_tick_label=queryset=='stats_ceb',
                           method_hatches=method_hatch, show_group_boundary=False, dataset=dataset,
                           font_size=font_size, alpha=alpha, hatch_color=hatch_color,
                           dataset_label_bbox={
                               "boxstyle": "square,pad=0.1",
                               "facecolor": "#F0F8FF",
                               "edgecolor": "none",
                               "alpha": 0.7,
                           }, x_tick_pad=x_tick_pad,
                           bar_linewidth=bar_linewidth)

def plot_updating_time_dynamic():
    old_hatch_linewidth = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.linewidth'] = 0.5
    font_size = 14
    alpha = 0.8
    hatch_color = "black"
    save_path = FIGURES_DIR / "updating_time_dynamic.pdf"
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5))
    
    methods = ['robustmscn', 'alece', 'price_ft', 'deepdb_i', 'deepdb_s', 'deepdb_f', 'bayescard_i', 'bayescard_s', 'bayescard_f', 'neurocard', 'factorjoin', 'asm']
    
    methods_name = {
        'deepdb_s': 'DeepDB-REF',
        'deepdb_i': 'DeepDB-INC',
        'deepdb_f': 'DeepDB-TF',
        'bayescard_s': 'BayesCard-REF',
        'bayescard_i': 'BayesCard-INC',
        'bayescard_f': 'BayesCard-TF'
    }
    for m in methods:
        if m in METHOD_NAMES:
            methods_name[m] = METHOD_NAMES[m]
        
    methods_color = {
        'deepdb_s': '#83BDE3',
        'deepdb_i': '#3B8FC4',
        'deepdb_f': '#B9DDF2',
        
        'bayescard_s': '#C06C9A',
        'bayescard_i': '#93486F',
        'bayescard_f': '#E4B7CF',
    }
    
    for m in methods:
        if m in METHOD_COLORS:
            methods_color[m] = METHOD_COLORS[m]
    
    hatches = [
        '-',    # robustmscn
        '\\',   # alece
        '||',    # price_ft
        '/',    # deepdb_s
        '+',    # deepdb_i
        'x',    # deepdb_f
        '//',    # bayescard_s
        '++',    # bayescard_i
        'xx',    # bayescard_f
        '.',    # neurocard
        '\\\\',   # factorjoin
        '--'    # asm
    ]#o,*,.
    
    methods_hatch = {m:h for m, h in zip(methods, hatches)}
    querysets = ['job_light', 'stats_ceb']
    for i, ax in enumerate(axes.flatten()):
        plot_updating_time_per_dataset(
            ax,
            methods,
            methods_name,
            methods_color,
            methods_hatch,
            querysets[i],
            font_size,
            alpha,
            hatch_color,
        )
    y_label = "Updating Time (s)"
    fig.supylabel(y_label, fontsize=font_size, x=0.002)
    
    legend_handles = [
        mpatches.Patch(
            facecolor=methods_color[m],
            edgecolor=hatch_color,
            hatch=methods_hatch[m],
            label=methods_name[m],
            linewidth=0,
            alpha=alpha,
        ) 
        for m in methods
    ]
    
    legend_handles = _column_major_legend_handles(legend_handles, 4)
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.035),
            ncol=4, frameon=False, fontsize=font_size, handlelength=1.8, handleheight=0.8,
            labelspacing=0.3, columnspacing=1, handletextpad=0.5)
    
    fig.subplots_adjust(
        top=0.83,
        bottom=0.04,
        left=0.07,
        right=0.999,
        hspace=0.06,
    )
    _save_figure(fig, save_path)
    plt.show()
    plt.rcParams['hatch.linewidth'] = old_hatch_linewidth

def plot_q_error_distribution_dynamic():
    old_hatch_linewidth = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.linewidth'] = 0.5
    
    font_size, super_font_size, legend_font_size = 18, 18, 16
    save_path = FIGURES_DIR / "q_error_dynamic.pdf"
    fig, axes = plt.subplots(2, 1, figsize=(18, 6))
    
    baselines = ['postgresql', 'postgresql_ext', 'lpbound', 'price_zs']
    methods = ['robustmscn', 'alece', 'price_ft', 'deepdb_i', 'deepdb_s', 'deepdb_f', 'bayescard_i', 'bayescard_s', 'bayescard_f', 'neurocard', 'factorjoin', 'asm']
    
    method_names = {
        'deepdb_s': 'DeepDB-REF',
        'deepdb_i': 'DeepDB-INC',
        'deepdb_f': 'DeepDB-TF',
        'bayescard_s': 'BayesCard-REF',
        'bayescard_i': 'BayesCard-INC',
        'bayescard_f': 'BayesCard-TF'
    }
    for m in baselines + methods:
        if m in METHOD_NAMES:
            method_names[m] = METHOD_NAMES[m]
        
    method_colors = {
        'deepdb_s': '#83BDE3',
        'deepdb_i': '#3B8FC4',
        'deepdb_f': '#B9DDF2',
        
        'bayescard_s': '#C06C9A',
        'bayescard_i': '#93486F',
        'bayescard_f': '#E4B7CF',
    }
    
    for m in baselines + methods:
        if m in METHOD_COLORS:
            method_colors[m] = METHOD_COLORS[m]

    hatches = {
        'stale': '.',
        'updated': '/'
    }
    baseline_group_name = 'Baselines (100%)'
    group_names = ['20%', '40%', '60%', '80%']
    
    for i, queryset in enumerate(['job_light', 'stats_ceb']):
        values_by_method_by_variant_by_group = {'20':{}, '40':{}, '60':{}, '80':{}, '100':{}}
        values_by_method_by_variant_by_group['100']['overall'] = {}
        for m in baselines + methods:
            if m.startswith('deepdb') or m.startswith('bayescard'):
                if m.split('_')[1] != 's':
                    continue
                exp_dir = f"experiments/overall/{queryset}/{m.split('_')[0]}/default"
            else:
                exp_dir = f"experiments/overall/{queryset}/{m}/default"
            q_errors = get_q_error_and_query_size(exp_dir)[0]
            values_by_method_by_variant_by_group['100']['overall'][m] = np.log10(q_errors)
        for g in ['20', '40', '60', '80']:
            for m in methods:
                if m.startswith('deepdb') or m.startswith('bayescard'):
                    option = 'structure' if m.split('_')[1] == 's' else 'incremental' if m.split('_')[1] == 'i' else 'filter'
                    exp_dir = f"experiments/dynamic/{queryset}/{m.split('_')[0]}/{g}_{option}/"
                else:
                    exp_dir = f"experiments/dynamic/{queryset}/{m}/{g}/"
                for v in ['stale', 'updated']:
                    if v == 'stale':
                        df = pd.read_csv(exp_dir+"estimation_stale.csv")
                    else:
                        df = pd.read_csv(exp_dir+"estimation_updated.csv")
                    
                    estimated_cards = df['estimated_card'].to_numpy(np.float64)
                    estimated_cards[estimated_cards <= 1] = 1.0
                    true_cards = df['true_card'].to_numpy(np.float64)
                    true_cards[true_cards <= 1] = 1.0
                    q_errors = estimated_cards / true_cards 
                    
                    if v not in values_by_method_by_variant_by_group[g]:
                        values_by_method_by_variant_by_group[g][v] = {}
                    values_by_method_by_variant_by_group[g][v][m] = np.log10(q_errors)
        
        dataset_name = "JOB-LIGHT" if queryset == 'job_light' else "STATS-CEB"
        plot_grouped_box_with_baselines_and_methods_on_ax_for_dynamic_exp(axes[i], baselines, methods, 
                                                                          group_names, ['stale', 'updated'],
                                                                          baseline_group_name,
                                                                          values_by_method_by_variant_by_group,
                                                                          method_names, method_colors, hatches,
                                                                          whis=(1, 99),
                                                                          x_label="",
                                                                          y_label=None,
                                                                          show_x_ticks=i == 1,
                                                                          y_lim=(-1.5,1.5),
                                                                          dataset=dataset_name,
                                                                          boundary_line_style=None,
                                                                          fontsize_param=font_size,
                                                                          x_label_font=font_size,
                                                                          dataset_font_size=super_font_size)
    
    fig.supxlabel(
        "Initial Data Percentage",
        fontsize=super_font_size,
        y=0.0015,
    )
    fig.supylabel(
        "underest. <- log(q-error) -> overest.",
        fontsize=super_font_size,
        x=0.002,
    )
    legend_handles = [
        mpatches.Patch(
            color=method_colors[m],
            label=method_names[m],
        ) 
        for m in baselines + methods
    ]
    hatch_legend_handles = [
        mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            label=l,
            hatch=hatches[l]*2, 
            linewidth=1
        )
        for l in ['stale', 'updated']
    ]
    legend_handles += hatch_legend_handles

    
    legend_handles = _column_major_legend_handles(legend_handles, 9)
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.033),
            ncol=9, frameon=False, fontsize=legend_font_size,
            handlelength=1.6, handleheight=0.7,
            columnspacing=0.6, handletextpad=0.3)
    
    fig.subplots_adjust(
        top=0.89,
        bottom=0.09,
        left=0.03,
        right=0.999,
        hspace=0.05,
    )
    _save_figure(fig, save_path)
    plt.show()
    
    plt.rcParams['hatch.linewidth'] = old_hatch_linewidth

def plot_p_error_distribution_dynamic():
    old_hatch_linewidth = plt.rcParams['hatch.linewidth']
    plt.rcParams['hatch.linewidth'] = 0.5

    font_size, super_font_size, legend_font_size = 18, 18, 16
    save_path = FIGURES_DIR / "p_error_dynamic.pdf"
    fig, axes = plt.subplots(2, 1, figsize=(18, 6))
    
    baselines = ['postgresql', 'postgresql_ext', 'lpbound', 'price_zs']
    methods = ['robustmscn', 'alece', 'price_ft', 'deepdb_i', 'deepdb_s', 'deepdb_f', 'bayescard_i', 'bayescard_s', 'bayescard_f', 'neurocard', 'factorjoin', 'asm']
    
    method_names = {
        'deepdb_s': 'DeepDB-REF',
        'deepdb_i': 'DeepDB-INC',
        'deepdb_f': 'DeepDB-TF',
        'bayescard_s': 'BayesCard-REF',
        'bayescard_i': 'BayesCard-INC',
        'bayescard_f': 'BayesCard-TF'
    }
    for m in baselines + methods:
        if m in METHOD_NAMES:
            method_names[m] = METHOD_NAMES[m]
        
    method_colors = {
        'deepdb_s': '#83BDE3',
        'deepdb_i': '#3B8FC4',
        'deepdb_f': '#B9DDF2',
        
        'bayescard_s': '#C06C9A',
        'bayescard_i': '#93486F',
        'bayescard_f': '#E4B7CF',
    }
    
    for m in baselines + methods:
        if m in METHOD_COLORS:
            method_colors[m] = METHOD_COLORS[m]

    hatches = {
        'stale': '.',
        'updated': '/'
    }
    baseline_group_name = 'Baselines (100%)'
    group_names = ['20%', '40%', '60%', '80%']
    
    for i, queryset in enumerate(['job_light', 'stats_ceb']):
        values_by_method_by_variant_by_group = {'20':{}, '40':{}, '60':{}, '80':{}, '100':{}}
        values_by_method_by_variant_by_group['100']['overall'] = {}
        for m in baselines + methods:
            if m.startswith('deepdb') or m.startswith('bayescard'):
                if m.split('_')[1] != 's':
                    continue
                exp_dir = f"experiments/overall/{queryset}/{m.split('_')[0]}/default"
            else:
                exp_dir = f"experiments/overall/{queryset}/{m}/default"
            p_errors = get_p_error(exp_dir)
            values_by_method_by_variant_by_group['100']['overall'][m] = p_errors
        for g in ['20', '40', '60', '80']:
            for m in methods:
                if m.startswith('deepdb') or m.startswith('bayescard'):
                    option = 'structure' if m.split('_')[1] == 's' else 'incremental' if m.split('_')[1] == 'i' else 'filter'
                    exp_dir = f"experiments/dynamic/{queryset}/{m.split('_')[0]}/{g}_{option}/"
                else:
                    exp_dir = f"experiments/dynamic/{queryset}/{m}/{g}/"
                p_errors_stale, p_errors_updated = get_dynamic_p_errors(exp_dir)
                for v in ['stale', 'updated']:
                    if v not in values_by_method_by_variant_by_group[g]:
                        values_by_method_by_variant_by_group[g][v] = {}
                    values_by_method_by_variant_by_group[g][v][m] = p_errors_stale if v == 'stale' else p_errors_updated
        
        dataset_name = "JOB-LIGHT" if queryset == 'job_light' else "STATS-CEB"
        plot_grouped_box_with_baselines_and_methods_on_ax_for_dynamic_exp(axes[i], baselines, methods, 
                                                                          group_names, ['stale', 'updated'],
                                                                          baseline_group_name,
                                                                          values_by_method_by_variant_by_group,
                                                                          method_names, method_colors, hatches,
                                                                          whis=(1, 99),
                                                                          x_label="",
                                                                          y_label=None,
                                                                          show_x_ticks=i == 1,
                                                                          y_lim=(1, 6),
                                                                          y_ticks=[1, 2, 3, 4, 5],
                                                                          y_ticks_label=['1', '2', '3', '4', '5'],
                                                                          dataset=dataset_name,
                                                                          boundary_line_style=None,
                                                                          fontsize_param=font_size,
                                                                          x_label_font=font_size,
                                                                          dataset_font_size=super_font_size,
                                                                          first_group_label_x_offset=1.5,
                                                                          apply_log=True)
    
    fig.supxlabel(
        "Initial Data Percentage",
        fontsize=super_font_size,
        y=0.0015,
    )
    fig.supylabel(
        "P-Error",
        fontsize=super_font_size,
        x=0.002,
    )
    legend_handles = [
        mpatches.Patch(
            color=method_colors[m],
            label=method_names[m],
        ) 
        for m in baselines + methods
    ]
    hatch_legend_handles = [
        mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            label=l,
            hatch=hatches[l]*2,    
            linewidth=1
        )
        for l in ['stale', 'updated']
    ]
    legend_handles += hatch_legend_handles

    
    legend_handles = _column_major_legend_handles(legend_handles, 9)
    
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.033),
                ncol=9, frameon=False, fontsize=legend_font_size,
                handlelength=1.6, handleheight=0.7,
                columnspacing=0.6, handletextpad=0.3)
    
    fig.subplots_adjust(
        top=0.89,
        bottom=0.09,
        left=0.03,
        right=0.999,
        hspace=0.05,
    )
    _save_figure(fig, save_path)
    plt.show()

    plt.rcParams['hatch.linewidth'] = old_hatch_linewidth


def reproduce_all_figures():
    """Generate every figure included in the paper."""
    plot_q_error()
    plot_p_error_with_legends()
    plot_spearman()
    
    plot_model_size_and_training_time()
    
    plot_training_data_efficiency_curves()
    plot_training_data_efficiency_mscn_hybrid_curves()
    
    plot_q_error_by_size_generalization_study()
    plot_p_error_exec_time_generalization_study()
    
    plot_updating_time_dynamic()
    plot_q_error_distribution_dynamic()
    plot_p_error_distribution_dynamic()

def reproduce_all_tables():
    """Print every result table included in the paper."""
    print_exec_result_table()
    print_scalability_result_table('job_m_ours')
    print_scalability_result_table('job_large_ours')

    print_execution_time_over_training_query_num()
    print_execution_time_dynamic()
    print_asm_hybrid_table()



def parse_args(args=None):
    """Parse command-line options for selecting generated artifacts."""
    parser = argparse.ArgumentParser(
        description="Generate paper figures or print result tables."
    )
    parser.add_argument(
        "--output",
        choices=("figures", "tables", "all"),
        default="figures",
        help=(
            "Artifacts to generate: 'figures' saves plots, 'tables' prints "
            "tables, and 'all' does both."
        ),
    )
    return parser.parse_args(args)


def main(args=None):
    output = parse_args(args).output
    if output in ("figures", "all"):
        reproduce_all_figures()
    if output in ("tables", "all"):
        reproduce_all_tables()


if __name__ == '__main__':
    main()
