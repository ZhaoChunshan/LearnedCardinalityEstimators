"""Reusable Matplotlib primitives for the experiment figures.

The public functions in this module intentionally keep their original
signatures so existing plotting scripts continue to work.  Layout collection
and axis styling live in small private helpers instead of being repeated by
every chart type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from matplotlib.ticker import NullLocator


DEFAULT_EDGE_COLOR = "#000000"
DEFAULT_GRID_STYLE = {
    "color": "#808080",
    "linestyle": (0, (8, 4)),
    "linewidth": 0.8,
    "alpha": 0.65,
}
GROUP_SHADE_COLOR = "#F0F8FF"


@dataclass
class _PlotData:
    values: list[Any] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    hatches: list[str] = field(default_factory=list)
    group_centers: list[float] = field(default_factory=list)
    group_labels: list[str] = field(default_factory=list)
    boundaries: list[float] = field(default_factory=list)

    def add(self, value, position: float, color: str, hatch: str = "") -> None:
        self.values.append(value)
        self.positions.append(position)
        self.colors.append(color)
        self.hatches.append(hatch)


def _has_group_value(values_by_method, method: str, group_index: int) -> bool:
    values = values_by_method.get(method)
    return bool(values) and group_index < len(values) and values[group_index] is not None


def _style_spines(ax, linewidth: float = 1.5) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(DEFAULT_EDGE_COLOR)
        spine.set_linewidth(linewidth)


def _style_axis(ax, *, spine_width: float = 1.5, grid: bool = True) -> None:
    if grid:
        ax.grid(True, axis="y", **DEFAULT_GRID_STYLE)
    ax.set_axisbelow(True)
    _style_spines(ax, spine_width)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)


def _draw_boundaries(
    ax,
    boundaries: Iterable[float],
    *,
    linewidth: float = 1.5,
    linestyle: str | None = None,
) -> None:
    for boundary in boundaries:
        ax.axvline(
            x=boundary,
            color=DEFAULT_EDGE_COLOR,
            linewidth=linewidth,
            linestyle=linestyle or "-",
        )


def _set_x_range(ax, positions: Sequence[float]) -> None:
    if positions:
        ax.set_xlim(0, positions[-1] + 1)


def _set_group_ticks(
    ax,
    data: _PlotData,
    *,
    font_size: float,
    show_labels: bool = True,
) -> None:
    ax.set_xticks(data.group_centers)
    labels = data.group_labels if show_labels else [""] * len(data.group_labels)
    ax.set_xticklabels(labels, fontsize=font_size)


def _set_log_scale(ax, enabled: bool) -> None:
    if enabled:
        ax.set_yscale("log")


def _set_log_scale_without_minor_ticks(ax, enabled: bool) -> None:
    if enabled:
        _set_log_scale(ax, True)
        ax.yaxis.set_minor_locator(NullLocator())


def _add_dataset_label(
    ax,
    dataset: str | None,
    *,
    font_size: float,
    x: float,
    horizontal_alignment: str,
    y: float = 0.97,
    bbox: dict[str, Any] | None = None,
) -> None:
    if not dataset:
        return
    if bbox is None:
        bbox = {
            "facecolor": "white",
            "alpha": 0.6,
            "edgecolor": "gray",
            "boxstyle": "round,pad=0.1",
            "linewidth": 0.5,
        }
    ax.text(
        x,
        y,
        dataset,
        transform=ax.transAxes,
        fontsize=font_size,
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        bbox=bbox,
    )


def _integer_ticks(y_lim: tuple[float, float]) -> list[int]:
    return list(range(math.ceil(y_lim[0]), math.floor(y_lim[1]) + 1))


def _absolute_negative_tick_labels(ticks: Iterable[float]) -> list[float]:
    return [tick if tick >= 0 else -tick for tick in ticks]


def _collect_grouped_values(
    methods,
    group_names,
    values_by_group_by_method,
    method_colors,
    *,
    group_spacing: float,
    item_step: float,
    reserve_missing_space: bool,
    method_hatches=None,
) -> _PlotData:
    data = _PlotData()
    position = 1.0

    for group_index, group_name in enumerate(group_names):
        group_start = position
        for method in methods:
            if not _has_group_value(
                values_by_group_by_method, method, group_index
            ):
                if reserve_missing_space:
                    position += 1
                continue
            hatch = method_hatches[method] if method_hatches else ""
            data.add(
                values_by_group_by_method[method][group_index],
                position,
                method_colors[method],
                hatch,
            )
            position += item_step

        group_end = position - 1
        data.group_centers.append((group_start + group_end) / 2)
        data.group_labels.append(group_name)
        position += group_spacing
        if group_index != len(group_names) - 1:
            data.boundaries.append((group_end + position) / 2)

    return data


def _collect_baseline_variant_values(
    methods,
    group_names,
    variant_names,
    values_by_group_by_baseline,
    values_by_group_by_method_by_variant,
    method_colors,
    variant_hatches,
    *,
    group_spacing: float,
    variant_spacing: float,
    reserve_missing_space: bool,
) -> _PlotData:
    data = _PlotData()
    position = 1.0
    baselines = [method for method in methods if method in values_by_group_by_baseline]
    competitors = [method for method in methods if method not in baselines]

    for group_index, group_name in enumerate(group_names):
        group_start = position
        for method in baselines:
            if not _has_group_value(
                values_by_group_by_baseline, method, group_index
            ):
                if reserve_missing_space:
                    position += 1
                continue
            data.add(
                values_by_group_by_baseline[method][group_index],
                position,
                method_colors[method],
            )
            position += 1

        for method in competitors:
            position += variant_spacing
            for variant in variant_names:
                values_by_method = values_by_group_by_method_by_variant.get(
                    variant, {}
                )
                if not _has_group_value(values_by_method, method, group_index):
                    if reserve_missing_space:
                        position += 1
                    continue
                data.add(
                    values_by_method[method][group_index],
                    position,
                    method_colors[method],
                    variant_hatches[variant],
                )
                position += 1

        group_end = position - 1
        data.group_centers.append((group_start + group_end) / 2)
        data.group_labels.append(group_name)
        position += group_spacing
        if group_index != len(group_names) - 1:
            data.boundaries.append((group_end + position) / 2)

    return data


def plot_grouped_box_on_ax(
    ax,
    methods,
    group_names,
    values_by_group_by_method,
    method_names,
    method_colors,
    whis=(1, 99),
    spacing=1.2,
    x_label="Number of Tables",
    y_label="under <- log(q-error) -> over",
    y_lim=(-3, 3),
    apply_Log=False,
    spacing_for_missing_data=True,
    non_negative_under_est=False,
    show_x_ticks=True,
    yticks=None,
    font_size=35,
    show_y_label=True,
    xlabelpad=5,
    item_step=1.0,
    y_label_pad=None,
    first_group_shaded=False,
):
    del method_names  # Kept in the public API for call-site compatibility.
    data = _collect_grouped_values(
        methods,
        group_names,
        values_by_group_by_method,
        method_colors,
        group_spacing=spacing,
        item_step=item_step,
        reserve_missing_space=spacing_for_missing_data,
    )

    boxplot = ax.boxplot(
        data.values,
        positions=data.positions,
        whis=whis,
        notch=True,
        patch_artist=True,
        widths=0.9,
        boxprops={"linewidth": 1},
        whiskerprops={"linewidth": 1},
        capprops={"linewidth": 1},
        showfliers=False,
        flierprops={
            "marker": ".",
            "markerfacecolor": "black",
            "markersize": 1,
            "linestyle": "none",
        },
        medianprops={"color": "white", "linewidth": 1},
    )
    for patch, color in zip(boxplot["boxes"], data.colors):
        patch.set_facecolor(color)

    _set_group_ticks(ax, data, font_size=font_size, show_labels=show_x_ticks)
    if x_label:
        ax.set_xlabel(x_label, fontsize=font_size, labelpad=xlabelpad)
    _style_axis(ax)
    _draw_boundaries(ax, data.boundaries)
    _set_log_scale(ax, apply_Log)
    ax.set_ylim(y_lim)
    _set_x_range(ax, data.positions)

    integer_ticks = _integer_ticks(y_lim)
    is_p_error = "p-error" in y_label.lower()
    if yticks is None:
        yticks = (
            [tick for tick in integer_ticks if tick % 2 == 0 or tick == 1]
            if is_p_error
            else integer_ticks
        )
    if is_p_error:
        yticklabels = [str(tick) for tick in yticks]
    elif non_negative_under_est:
        yticklabels = _absolute_negative_tick_labels(yticks)
    else:
        yticklabels = [str(tick) for tick in yticks]

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=font_size)
    if is_p_error:
        ax.set_yticks([], minor=True)
        ax.set_yticklabels([], minor=True)
    if show_y_label:
        if y_label_pad is not None:
            ax.set_ylabel(y_label, fontsize=font_size, labelpad=y_label_pad)
        else:
            ax.set_ylabel(y_label, fontsize=font_size)


def plot_grouped_bar_on_ax(
    ax,
    methods,
    group_names,
    values_by_group_by_method,
    method_names,
    method_colors,
    spacing=0.4,
    y_label="Training Time (min)",
    x_label="",
    y_lim=None,
    y_ticks=None,
    apply_Log=False,
    show_x_tick_label=True,
    method_hatches=None,
    show_group_boundary=True,
    dataset=None,
    font_size=30,
    alpha=1.0,
    hatch_color=None,
    dataset_label_x=0.5,
    dataset_label_y=0.97,
    dataset_label_alignment="center",
    dataset_label_bbox=None,
    x_tick_pad=10,
    bar_linewidth=0,
):
    del method_names
    data = _collect_grouped_values(
        methods,
        group_names,
        values_by_group_by_method,
        method_colors,
        group_spacing=spacing,
        item_step=1,
        reserve_missing_space=True,
        method_hatches=method_hatches,
    )
    
    if method_hatches is None:
        ax.bar(
            data.positions,
            data.values,
            width=1,
            color=data.colors,
            alpha=alpha,
        )
    elif hatch_color is not None:
        ax.bar(
            data.positions,
            data.values,
            width=1,
            color=data.colors,
            edgecolor=hatch_color,
            hatch=data.hatches,
            linewidth=bar_linewidth,
            alpha=alpha,
        )
    else:
        ax.bar(
            data.positions,
            data.values,
            width=1,
            color="none",
            edgecolor=data.colors,
            hatch=data.hatches,
            linewidth=2,
            alpha=alpha,
        )

    _add_dataset_label(
        ax,
        dataset,
        font_size=font_size,
        x=dataset_label_x,
        horizontal_alignment=dataset_label_alignment,
        y=dataset_label_y,
        bbox=dataset_label_bbox,
    )
    if show_x_tick_label:
        _set_group_ticks(ax, data, font_size=font_size)
        ax.tick_params(axis="x", pad=x_tick_pad)
    else:
        ax.set_xticks([])
    _style_axis(ax)
    if show_group_boundary:
        _draw_boundaries(ax, data.boundaries)
    _set_log_scale(ax, apply_Log)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    _set_x_range(ax, data.positions)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_ylabel(y_label, fontsize=font_size)
    if x_label:
        ax.set_xlabel(x_label, fontsize=font_size)


def plot_curve_on_ax(
    ax,
    methods,
    xticklabels,
    values_by_group_by_method,
    method_names,
    method_colors,
    method_line_styles,
    apply_log=False,
    y_lim=None,
    y_ticks=None,
    y_tick_labels=None,
    tick_fontsize=20,
):
    x_positions = np.arange(len(xticklabels))
    for method in methods:
        values = values_by_group_by_method.get(method)
        if values is None:
            continue
        linestyle, marker, markersize = method_line_styles[method]
        ax.plot(
            x_positions,
            values,
            label=method_names[method],
            color=method_colors[method],
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            linewidth=2.7,
            markeredgewidth=1,
        )

    _set_log_scale(ax, apply_log)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_tick_labels is not None:
        ax.set_yticklabels(y_tick_labels)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    _style_axis(ax)
    ax.minorticks_off()


def plot_grouped_box_with_baselines_and_variants_on_ax(
    ax,
    methods,
    group_names,
    variant_names,
    values_by_group_by_baseline,
    values_by_group_by_method_by_variant,
    method_names,
    method_colors,
    variant_hatches,
    whis=(1, 99),
    group_spacing=1.2,
    variant_spacing=0.4,
    x_label="Number of Tables",
    y_label="underest. <- log(q-error) -> overest.",
    y_lim=(-3, 3),
    y_ticks=None,
    y_ticks_label=None,
    apply_log=False,
    spacing_for_missing_data=True,
    non_negative_under_est=False,
    show_x_ticks=True,
    dataset=None,
    show_group_boundary=True,
    boundary_line_style=None,
    font_size=18,
    y_tick_font_size=None,
    x_label_pad=None,
):
    del method_names
    if y_tick_font_size is None:
        y_tick_font_size = font_size
    data = _collect_baseline_variant_values(
        methods,
        group_names,
        variant_names,
        values_by_group_by_baseline,
        values_by_group_by_method_by_variant,
        method_colors,
        variant_hatches,
        group_spacing=group_spacing,
        variant_spacing=variant_spacing,
        reserve_missing_space=spacing_for_missing_data,
    )
    boxplot = ax.boxplot(
        data.values,
        positions=data.positions,
        whis=whis,
        notch=True,
        patch_artist=True,
        widths=0.9,
        boxprops={"linewidth": 1},
        whiskerprops={"linewidth": 1},
        capprops={"linewidth": 1},
        showfliers=False,
        flierprops={
            "marker": ".",
            "markerfacecolor": "black",
            "markersize": 1,
            "linestyle": "none",
        },
        medianprops={"color": "white", "linewidth": 1},
    )
    for patch, color, hatch in zip(
        boxplot["boxes"], data.colors, data.hatches
    ):
        patch.set_facecolor(color)
        patch.set_hatch(hatch * 2)

    _add_dataset_label(
        ax,
        dataset,
        font_size=font_size,
        x=0.01,
        horizontal_alignment="left",
        y=0.95,
        bbox={
            "boxstyle": "square,pad=0.3",
            "facecolor": "#F0F8FF",
            "edgecolor": "none",
            "alpha": 0.7,
        },
    )
    _set_group_ticks(ax, data, font_size=font_size, show_labels=show_x_ticks)
    if x_label:
        ax.set_xlabel(x_label, fontsize=font_size, labelpad=1)
    ax.set_axisbelow(True)
    if show_group_boundary:
        _draw_boundaries(
            ax, data.boundaries, linestyle=boundary_line_style, linewidth=1
        )
    _set_log_scale_without_minor_ticks(ax, apply_log)
    ax.set_ylim(y_lim)
    _set_x_range(ax, data.positions)

    integer_ticks = _integer_ticks(y_lim)
    default_tick_labels = (
        _absolute_negative_tick_labels(integer_ticks)
        if non_negative_under_est
        else [str(tick) for tick in integer_ticks]
    )
    if y_ticks is None and y_ticks_label is None:
        ax.set_yticks(integer_ticks)
        ax.set_yticklabels(default_tick_labels, fontsize=y_tick_font_size)
    elif y_ticks is not None:
        ax.set_yticks(y_ticks)
        if y_ticks_label is not None:
            ax.set_yticklabels(y_ticks_label, fontsize=y_tick_font_size)
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(True, axis="y", **DEFAULT_GRID_STYLE)
    if y_label:
        ax.set_ylabel(y_label, fontsize=font_size, labelpad=x_label_pad)


def plot_grouped_bar_with_baselines_and_variants_on_ax(
    ax,
    methods,
    group_names,
    variant_names,
    values_by_group_by_baseline,
    values_by_group_by_method_by_variant,
    method_names,
    method_colors,
    variant_hatches,
    group_spacing=1.2,
    variant_spacing=0.4,
    x_label="",
    y_label="Total Plan Exec. Time (s)",
    y_lim=None,
    y_ticks=None,
    y_ticks_label=None,
    apply_log=False,
    spacing_for_missing_data=True,
    font_size=18,
    y_tick_font_size=None,
    x_label_pad=None,
):
    del method_names
    if y_tick_font_size is None:
        y_tick_font_size = font_size - 2
    data = _collect_baseline_variant_values(
        methods,
        group_names,
        variant_names,
        values_by_group_by_baseline,
        values_by_group_by_method_by_variant,
        method_colors,
        variant_hatches,
        group_spacing=group_spacing,
        variant_spacing=variant_spacing,
        reserve_missing_space=spacing_for_missing_data,
    )
    bars = ax.bar(
        data.positions,
        data.values,
        width=0.9,
        color=data.colors,
        edgecolor="black",
        linewidth=1,
        alpha=0.8,
    )
    for bar, hatch in zip(bars, data.hatches):
        bar.set_hatch(hatch * 2)
        bar.set_linewidth(0.5)

    _set_group_ticks(ax, data, font_size=font_size)
    if x_label:
        ax.set_xlabel(x_label, fontsize=font_size)
    ax.grid(True, axis="y", **DEFAULT_GRID_STYLE)
    ax.set_axisbelow(True)
    _draw_boundaries(ax, data.boundaries)
    _set_log_scale_without_minor_ticks(ax, apply_log)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        if y_ticks_label is not None:
            ax.set_yticklabels(y_ticks_label, fontsize=font_size)
    _set_x_range(ax, data.positions)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, labelsize=y_tick_font_size)
    if y_label:
        ax.set_ylabel(y_label, fontsize=font_size, labelpad=x_label_pad)


def plot_grouped_box_with_baselines_and_methods_on_ax_for_dynamic_exp(
    ax,
    baselines,
    methods,
    group_names,
    variant_names,
    baseline_group_name,
    values_by_method_by_variant_by_group,
    method_names,
    method_colors,
    variant_hatches,
    whis=(5, 95),
    group_spacing=2,
    method_spacing=0.1,
    x_label="Initial Data Percentage",
    y_label="underest. <- log(q-error) -> overest.",
    y_lim=(-3, 3),
    y_ticks=None,
    y_ticks_label=None,
    apply_log=False,
    spacing_for_missing_data=True,
    non_negative_under_est=True,
    show_x_ticks=True,
    dataset=None,
    show_group_boundary=True,
    boundary_line_style=None,
    fontsize_param=18,
    x_label_font=23,
    dataset_font_size=None,
    first_group_label_x_offset=0.0,
):
    del method_names, spacing_for_missing_data
    data = _PlotData()
    position = 1.0

    overall_values = values_by_method_by_variant_by_group["100"]["overall"]
    for method in baselines + methods:
        if method not in overall_values:
            continue
        data.add(
            overall_values[method], position, method_colors[method]
        )
        position += 1 + method_spacing
    position -= method_spacing
    data.group_centers.append(
        (1 + position - 1) / 2 + first_group_label_x_offset
    )
    data.group_labels.append(baseline_group_name)
    position += group_spacing
    data.boundaries.append((position - group_spacing - 1 + position) / 2)

    dynamic_groups = ("20", "40", "60", "80")
    for group_index, group_key in enumerate(dynamic_groups):
        group_start = position
        values_by_variant = values_by_method_by_variant_by_group[group_key]
        for method in methods:
            for variant in variant_names:
                data.add(
                    values_by_variant[variant][method],
                    position,
                    method_colors[method],
                    variant_hatches[variant],
                )
                position += 1
            position += method_spacing
        position -= method_spacing
        group_end = position - 1
        data.group_centers.append((group_start + group_end) / 2)
        data.group_labels.append(group_names[group_index])
        position += group_spacing
        if group_index != len(dynamic_groups) - 1:
            data.boundaries.append((group_end + position) / 2)

    boxplot = ax.boxplot(
        data.values,
        positions=data.positions,
        whis=whis,
        notch=True,
        patch_artist=True,
        widths=0.9,
        boxprops={"linewidth": 1},
        whiskerprops={"linewidth": 1},
        capprops={"linewidth": 1},
        showfliers=True,
        flierprops={"marker": None},
        medianprops={"color": "white", "linewidth": 1},
    )
    for patch, color, hatch in zip(
        boxplot["boxes"], data.colors, data.hatches
    ):
        patch.set_facecolor(color)
        patch.set_hatch(hatch)

    font_size = fontsize_param
    if dataset_font_size is None:
        dataset_font_size = font_size + 2
    _add_dataset_label(
        ax,
        dataset,
        font_size=dataset_font_size,
        x=0.01,
        horizontal_alignment="left",
        y=0.95,
        bbox={
            "boxstyle": "square,pad=0.3",
            "facecolor": "#F0F8FF",
            "edgecolor": "none",
            "alpha": 0.7,
        },
    )
    _style_axis(ax)
    if show_group_boundary:
        _draw_boundaries(
            ax,
            data.boundaries,
            linewidth=1.5,
            linestyle=boundary_line_style,
        )
    _set_group_ticks(ax, data, font_size=font_size, show_labels=show_x_ticks)
    if x_label:
        ax.set_xlabel(x_label, fontsize=x_label_font)
    _set_log_scale_without_minor_ticks(ax, apply_log)
    ax.set_ylim(y_lim)
    _set_x_range(ax, data.positions)

    integer_ticks = _integer_ticks(y_lim)
    default_tick_labels = (
        _absolute_negative_tick_labels(integer_ticks)
        if non_negative_under_est
        else [str(tick) for tick in integer_ticks]
    )
    if y_ticks is None and y_ticks_label is None:
        ax.set_yticks(integer_ticks)
        ax.set_yticklabels(default_tick_labels, fontsize=font_size)
    elif y_ticks is not None:
        ax.set_yticks(y_ticks)
        if y_ticks_label is not None:
            ax.set_yticklabels(y_ticks_label, fontsize=font_size)
    ax.tick_params(axis="both", which="both", length=0)
    if y_label:
        ax.set_ylabel(y_label, fontsize=font_size)
