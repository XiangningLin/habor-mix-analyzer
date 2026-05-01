"""
Paper-ready figures for the quantitative comparison pipeline.

All figures are saved to FIGURE_DIR (output/quantitative/figures/).
Call `generate_all_figures(...)` from the pipeline runner.

Visual style aligned with habor-analyze (src/habor_mix_analyzer/core/plotting.py).
"""

from __future__ import annotations

import re
import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .loading import load_metric_alignment
from .config import (
    BENCHMARK_TAXONOMY,
    COLOR_BLUE, COLOR_GRAY, COLOR_GREEN, COLOR_GRID, COLOR_AXIS,
    COLOR_PURPLE, COLOR_RED,
    DPI, DOMAIN_COLORS, FIGSIZE_SINGLE, FIGSIZE_WIDE,
    FIGURE_DIR, MATCH_STATUS_COLORS, MATCH_STATUS_LABELS,
    PLOT_STYLE, SUPERDOMAIN_COLORS,
)


def _apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def _wrap(value: object, width: int = 24) -> str:
    text = str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=DPI, pad_inches=0.08)
    plt.close(fig)
    print(f"  [fig] {path}")


def _categorical_colors(n: int) -> list[str]:
    palettes = ["tab20", "tab20b", "tab20c"]
    colors: list[str] = []
    for palette in palettes:
        colors.extend(mcolors.to_hex(c) for c in plt.get_cmap(palette).colors)
    if n <= len(colors):
        return colors[:n]
    return [colors[i % len(colors)] for i in range(n)]


def _slugify(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return slug or "other"


_PROGRESS_EXCLUDED_DOMAINS = {
    "Data & Analytics",
    "Professional Domains",
    "Safety & Security",
    "Multimodal",
}

_PROGRESS_GROUP_ORDER = {
    "Software Engineering (Repo-level Issue Resolution and Testing)": 0,
    "Software Engineering (Competitive & Function-level Coding)": 1,
    "Software Engineering (Other)": 2,
    "Mathematics & Reasoning": 3,
    "Knowledge & Long Context": 4,
    "Scientific Research": 5,
    "Agents, Tools & Systems": 6,
    "Other": 7,
}


def _progress_plot_group(benchmark: str, domain: str) -> str | None:
    if domain in _PROGRESS_EXCLUDED_DOMAINS:
        return None
    if domain != "Software Engineering":
        return domain

    taxonomy = BENCHMARK_TAXONOMY.get(benchmark) or BENCHMARK_TAXONOMY.get(benchmark.replace("_", "-"))
    subdomain = taxonomy.get("subdomain") if taxonomy else ""
    if subdomain == "Repo-level Issue Resolution":
        return "Software Engineering (Repo-level Issue Resolution and Testing)"
    if subdomain == "Competitive & Function-level Coding":
        return "Software Engineering (Competitive & Function-level Coding)"
    return "Software Engineering (Other)"


# ===================================================================
# Fig 1: Scatter — Harbor score vs Doc score
# ===================================================================

def fig_harbor_vs_doc_scatter(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for status, color in MATCH_STATUS_COLORS.items():
        sub = summary[summary["match_status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub["score_doc"], sub["score_harbor"],
            c=color, label=MATCH_STATUS_LABELS.get(status, status),
            s=95, alpha=0.9, edgecolors="white", linewidths=0.5,
        )

    lims = [0, 1.05]
    ax.plot(lims, lims, color=COLOR_AXIS, linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Original benchmark score (doc)")
    ax.set_ylabel("Harbor score")
    ax.set_title("Harbor vs Original Benchmark Scores")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "harbor_vs_doc_scatter.png")


# ===================================================================
# Fig 2: Bar chart — mean delta by benchmark
# ===================================================================

def fig_delta_by_benchmark(direction_df: pd.DataFrame) -> None:
    if direction_df.empty:
        return
    _apply_style()
    df = direction_df.sort_values("mean_delta")
    labels = [_wrap(b, 28) for b in df["benchmark"]]
    fig, ax = plt.subplots(figsize=(11.5, max(5.2, 0.42 * len(df))))

    colors = []
    for d in df["direction"]:
        if d == "HARBOR_HIGHER":
            colors.append(COLOR_GREEN)
        elif d == "HARBOR_LOWER":
            colors.append(COLOR_RED)
        else:
            colors.append(COLOR_GRAY)

    ax.barh(labels, df["mean_delta"], color=colors, edgecolor="white")
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.axvline(-0.05, color=COLOR_GRAY, linewidth=0.5, linestyle=":")
    ax.axvline(0.05, color=COLOR_GRAY, linewidth=0.5, linestyle=":")
    ax.set_xlabel("Mean Δ  (Harbor − Original)")
    ax.set_title("Score Difference by Benchmark")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["mean_delta"] + (0.005 if row["mean_delta"] >= 0 else -0.005),
            i, f'n={int(row["n"])}',
            va="center", ha="left" if row["mean_delta"] >= 0 else "right",
            fontsize=9, color="gray",
        )

    fig.tight_layout()
    _save(fig, "delta_by_benchmark.png")


# ===================================================================
# Fig 3: Agent lift heatmap
# ===================================================================

def fig_agent_lift_heatmap(lift_df: pd.DataFrame) -> None:
    if lift_df.empty:
        return
    _apply_style()
    pivot = lift_df.pivot_table(
        index="model", columns="benchmark", values="agent_lift", aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(14, max(3.8, 1.0 + 0.55 * pivot.shape[0])))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.1)
    image = ax.imshow(pivot.values, cmap="BrBG", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([_wrap(col, 14) for col in pivot.columns], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([_wrap(v, 20) for v in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > vmax * 0.6 else "black")

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Agent lift over terminus-2")
    ax.set_title("Agent Lift vs Terminus-2  (Δ_agent − Δ_terminus)")
    fig.tight_layout()
    _save(fig, "agent_lift_heatmap.png")


# ===================================================================
# Fig 4: Delta distribution by match status
# ===================================================================

def fig_delta_distribution(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    statuses = [s for s in MATCH_STATUS_COLORS if s in summary["match_status"].values]
    data = [summary[summary["match_status"] == s]["delta"].dropna().values for s in statuses]
    labels = [_wrap(MATCH_STATUS_LABELS.get(s, s), 22) for s in statuses]
    colors = [MATCH_STATUS_COLORS[s] for s in statuses]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, vert=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    ax.set_ylabel("Δ  (Harbor − Original)")
    ax.set_title("Delta Distribution by Match Type")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save(fig, "delta_distribution_boxplot.png")


# ===================================================================
# Fig 5: Hardest benchmarks (lowest best-model score, colored by domain)
# ===================================================================

def fig_hardest_benchmarks(difficulty_df: pd.DataFrame) -> None:
    if difficulty_df.empty:
        return
    _apply_style()
    df = difficulty_df.sort_values("best_score", ascending=True).head(40)
    labels = [_wrap(b, 24) for b in df["benchmark"]]
    colors = [DOMAIN_COLORS.get(d, COLOR_GRAY) for d in df["domain"]]
    fig, ax = plt.subplots(figsize=(11.5, max(8.2, 0.38 * len(df))))
    ax.barh(labels, df["best_score"], color=colors, edgecolor="white")
    ax.set_xlabel("Best score across all model+agent pairs")
    ax.set_title("Benchmark Difficulty: Room for Improvement")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)
    ax.set_xlim(0, 1.05)

    for i, (_, row) in enumerate(df.iterrows()):
        label = _wrap(row["best_model"], 18)
        ax.text(row["best_score"] + 0.01, i, label,
                va="center", fontsize=8, color="gray")

    handles = []
    for domain, color in DOMAIN_COLORS.items():
        if domain in df["domain"].values:
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, ec="white", label=domain))
    if handles:
        ax.legend(handles=handles, loc="lower right", frameon=False)

    fig.tight_layout()
    _save(fig, "hardest_benchmarks.png")


# ===================================================================
# Fig 6: Domain score comparison (grouped bar)
# ===================================================================

def fig_domain_scores(domain_df: pd.DataFrame) -> None:
    if domain_df.empty:
        return
    _apply_style()
    df = domain_df.sort_values("mean_best_score")
    labels = [_wrap(d, 20) for d in df["domain"]]
    colors = [DOMAIN_COLORS.get(d, COLOR_GRAY) for d in df["domain"]]

    fig, ax = plt.subplots(figsize=(11.5, max(4, 0.8 * len(df))))
    y = np.arange(len(df))
    ax.barh(y - 0.15, df["mean_best_score"], height=0.3, color=colors,
            edgecolor="white", label="Mean best score")
    ax.barh(y + 0.15, df["mean_all_scores"], height=0.3,
            color=[c + "88" for c in colors], edgecolor="white",
            label="Mean all entries")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_title("Score Distribution by Domain")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["mean_best_score"] + 0.01, i - 0.15,
                f'n={int(row["n_benchmarks"])}',
                va="center", fontsize=9, color="gray")

    fig.tight_layout()
    _save(fig, "domain_scores.png")


# ===================================================================
# Fig 7: Superdomain comparison
# ===================================================================

def fig_coding_vs_noncoding(superdomain_df: pd.DataFrame) -> None:
    if superdomain_df.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(superdomain_df))
    colors = [SUPERDOMAIN_COLORS.get(sd, COLOR_GRAY) for sd in superdomain_df["superdomain"]]
    width = 0.3
    ax.bar(x - width / 2, superdomain_df["mean_best_score"], width,
           color=colors, edgecolor="white", label="Mean best score")
    ax.bar(x + width / 2, superdomain_df["mean_all_scores"], width,
           color=[c + "88" for c in colors], edgecolor="white",
           label="Mean all entries")

    ax.set_xticks(x)
    ax.set_xticklabels(superdomain_df["superdomain"])
    ax.set_ylabel("Score")
    ax.set_title("Score Comparison by Superdomain")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)

    for i, (_, row) in enumerate(superdomain_df.iterrows()):
        ax.text(i, row["mean_best_score"] + 0.02,
                f'n={int(row["n_benchmarks"])}',
                ha="center", fontsize=11, color=COLOR_AXIS)

    fig.tight_layout()
    _save(fig, "coding_vs_noncoding.png")


# ===================================================================
# Fig 8: Progress over time (best score trajectory)
# ===================================================================

def _harbor_best_by_benchmark(harbor_df: pd.DataFrame | None) -> dict[str, tuple[float, str]]:
    if harbor_df is None or harbor_df.empty:
        return {}

    best_by_benchmark: dict[str, tuple[float, str]] = {}
    for alignment in load_metric_alignment().values():
        if not alignment.include_in_comparison or not alignment.info_stem:
            continue
        if alignment.matrix_column not in harbor_df.columns:
            continue
        scores = pd.to_numeric(harbor_df[alignment.matrix_column], errors="coerce").dropna()
        if scores.empty:
            continue
        best = float(scores.max())
        if -0.01 <= best <= 1.01:
            best_by_benchmark[alignment.info_stem] = (best, alignment.alignment_status)
    return best_by_benchmark


def fig_progress_over_time(
    progress_df: pd.DataFrame,
    harbor_df: pd.DataFrame | None = None,
) -> None:
    if progress_df.empty:
        return
    from .cross_analysis import _domain_for

    _apply_style()
    progress_df = progress_df.copy()
    progress_df["domain"] = progress_df["benchmark"].map(lambda b: _domain_for(str(b))[0])
    progress_df["plot_group"] = progress_df.apply(
        lambda row: _progress_plot_group(str(row["benchmark"]), str(row["domain"])),
        axis=1,
    )
    progress_df = progress_df[progress_df["plot_group"].notna()].copy()
    if progress_df.empty:
        return
    progress_df["n_snapshots"] = progress_df.groupby("benchmark")["benchmark"].transform("size")
    progress_df["progress_delta"] = (
        progress_df.groupby("benchmark")["best_score"].transform("max")
        - progress_df.groupby("benchmark")["best_score"].transform("min")
    )
    progress_df["sort_group"] = progress_df["plot_group"].map(
        lambda g: _PROGRESS_GROUP_ORDER.get(str(g), len(_PROGRESS_GROUP_ORDER))
    )

    benchmark_rank = (
        progress_df[["benchmark", "plot_group", "n_snapshots", "progress_delta", "sort_group"]]
        .drop_duplicates()
        .sort_values(
            ["sort_group", "n_snapshots", "progress_delta", "benchmark"],
            ascending=[True, False, False, True],
        )
    )
    selected = benchmark_rank.copy()
    harbor_best = _harbor_best_by_benchmark(harbor_df)

    def _make_group_fig(group_name: str, group_selected: pd.DataFrame) -> plt.Figure:
        plot_df = progress_df[progress_df["benchmark"].isin(group_selected["benchmark"])].copy()
        all_dates = sorted(plot_df["date_ym"].unique())
        date_to_x = {d: i for i, d in enumerate(all_dates)}
        plot_df["_x"] = plot_df["date_ym"].map(date_to_x)

        n_benchmarks = len(group_selected)
        fig, ax = plt.subplots(figsize=(16, 4))
        bench_colors: dict[str, str] = {}
        bench_last_points: dict[str, tuple[int, float]] = {}
        variants = _categorical_colors(n_benchmarks)
        ordered_benchmarks = [str(b) for b in group_selected["benchmark"]]
        for bench, color in zip(ordered_benchmarks, variants, strict=False):
            bench_colors[bench] = color

        for bench in ordered_benchmarks:
            g = plot_df[plot_df["benchmark"] == bench].sort_values("_x")
            if g.empty:
                continue
            color = bench_colors[bench]
            bench_last_points[bench] = (int(g["_x"].iloc[-1]), float(g["best_score"].iloc[-1]))
            label_name = g["benchmark_name"].iloc[0] if "benchmark_name" in g else bench
            ax.plot(
                g["_x"], g["best_score"], "o-",
                label=_wrap(label_name, 24),
                color=color, markersize=5, linewidth=1.6, alpha=0.9,
            )

        harbor_x = len(all_dates) + 2
        for bench, (score, align_status) in harbor_best.items():
            if bench not in bench_colors:
                continue
            last_x, last_score = bench_last_points[bench]
            ax.plot(
                [last_x, harbor_x], [last_score, score],
                color=bench_colors[bench], linestyle="--", linewidth=1.4,
                alpha=0.65, zorder=3,
            )
            if align_status == "subset_or_variant":
                ax.scatter(
                    harbor_x, score,
                    marker="D", s=65, facecolors="none", edgecolors=bench_colors[bench],
                    linewidths=1.5, zorder=4,
                )
            else:
                ax.scatter(
                    harbor_x, score,
                    marker="D", s=82, color=bench_colors[bench],
                    edgecolors="white", linewidths=0.8, zorder=4,
                )

        ax.set_xlabel("Result Date")
        ax.set_ylabel("score")
        ax.set_title(group_name, fontsize=15, pad=8)
        ax.grid(color=COLOR_GRID, linewidth=0.8)
        handles, labels = ax.get_legend_handles_labels()
        if harbor_best and any(bench in bench_colors for bench in harbor_best):
            handles.append(plt.Line2D(
                [0], [0], marker="D", color="none", markerfacecolor=COLOR_AXIS,
                markeredgecolor="white", markersize=8, label="Harbor result",
            ))
            labels.append("Harbor result")
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  frameon=False, fontsize=10.5, handlelength=1.9)

        step = max(1, len(all_dates) // 10)
        tick_positions = list(range(0, len(all_dates), step))
        tick_labels = [all_dates[i] for i in tick_positions]
        if any(bench in bench_colors for bench in harbor_best):
            tick_positions.append(harbor_x)
            tick_labels.append("Harbor")
            ax.set_xlim(-0.5, harbor_x + 0.5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        fig.subplots_adjust(left=0.07, right=0.84, bottom=0.2, top=0.9)
        return fig

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = FIGURE_DIR / "progress_over_time_v2.pdf"
    with PdfPages(combined_path) as pdf:
        for group_name, group_selected in selected.groupby("plot_group", sort=False):
            fig = _make_group_fig(str(group_name), group_selected)
            group_slug = _slugify(group_name)
            single_path = FIGURE_DIR / f"progress_over_time_v2_{group_slug}.pdf"
            png_path = FIGURE_DIR / f"progress_over_time_v2_{group_slug}.png"
            fig.savefig(single_path, dpi=DPI)
            fig.savefig(png_path, dpi=DPI)
            pdf.savefig(fig, dpi=DPI)
            plt.close(fig)
            print(f"  [fig] {single_path}")
            print(f"  [fig] {png_path}")
    print(f"  [fig] {combined_path}")


# ===================================================================
# Fig 9: Domain progress bar chart
# ===================================================================

def fig_domain_progress(domain_prog_df: pd.DataFrame) -> None:
    if domain_prog_df.empty:
        return
    _apply_style()
    df = domain_prog_df.sort_values("mean_absolute_progress")
    labels = [_wrap(d, 20) for d in df["domain"]]
    colors = [DOMAIN_COLORS.get(d, COLOR_GRAY) for d in df["domain"]]

    fig, ax = plt.subplots(figsize=(11.5, max(4, 0.8 * len(df))))
    ax.barh(labels, df["mean_absolute_progress"], color=colors, edgecolor="white")
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.set_xlabel("Mean absolute score improvement\n(latest snapshot − earliest snapshot)")
    ax.set_title("Progress by Domain (Doc Historical Data)")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["mean_absolute_progress"] + 0.005, i,
                f'n={int(row["n_benchmarks"])}',
                va="center", fontsize=9, color="gray")

    fig.tight_layout()
    _save(fig, "domain_progress.png")


def _format_pct(value: float) -> str:
    return f"{value:+.0f}%" if abs(value) >= 10 else f"{value:+.1f}%"


def fig_benchmark_launch_improvement(
    launch_imp_df: pd.DataFrame,
    log_scale: bool = False,
) -> None:
    if launch_imp_df.empty or "relative_improvement_pct" not in launch_imp_df.columns:
        return
    df = launch_imp_df[
        (launch_imp_df["status"] == "ok")
        & launch_imp_df["relative_improvement_pct"].notna()
    ].copy()
    if df.empty:
        return

    _apply_style()
    df = df.sort_values("relative_improvement_pct")
    labels = [_wrap(b, 28) for b in df["benchmark"]]
    values = df["relative_improvement_pct"].astype(float)
    colors = [COLOR_GREEN if v >= 0 else COLOR_RED for v in values]

    fig, ax = plt.subplots(figsize=(12.5, max(6.5, 0.34 * len(df))))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.set_xlabel("Relative improvement vs launch best (%)")
    title = "Harbor Max vs Benchmark Launch Best by Benchmark"
    ax.set_title(f"{title} (Log Scale)" if log_scale else title)
    if log_scale:
        ax.set_xscale("symlog", linthresh=10)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)

    span = max(abs(values.min()), abs(values.max()))
    pad = max(6, span * 0.015)
    for i, value in enumerate(values):
        ax.text(
            value + (pad if value >= 0 else -pad),
            i,
            _format_pct(float(value)),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color="dimgray",
        )
    left = min(values.min() - pad * 7, -25) if log_scale else values.min() - pad * 7
    ax.set_xlim(left, values.max() + pad * 9)
    fig.tight_layout()
    _save(fig, "benchmark_launch_vs_harbor_improvement.png")


def fig_domain_launch_improvement(
    domain_launch_imp_df: pd.DataFrame,
    log_scale: bool = False,
) -> None:
    if domain_launch_imp_df.empty or "median_relative_improvement_pct" not in domain_launch_imp_df.columns:
        return

    _apply_style()
    df = domain_launch_imp_df.sort_values("median_relative_improvement_pct")
    labels = [_wrap(d, 26) for d in df["domain"]]
    values = df["median_relative_improvement_pct"].astype(float)
    colors = [COLOR_GREEN if v >= 0 else COLOR_RED for v in values]

    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.48 * len(df))))
    ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.set_xlabel("Median relative improvement vs launch best (%)")
    title = "Harbor Max vs Benchmark Launch Best by Domain"
    ax.set_title(f"{title} (Log Scale)" if log_scale else title)
    if log_scale:
        ax.set_xscale("symlog", linthresh=10)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)

    span = max(abs(values.min()), abs(values.max()))
    pad = max(4, span * 0.015)
    for i, (_, row) in enumerate(df.iterrows()):
        value = float(row["median_relative_improvement_pct"])
        ax.text(
            value + (pad if value >= 0 else -pad),
            i,
            f"{_format_pct(value)}  n={int(row['n_benchmarks'])}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            color="dimgray",
        )
    left = min(values.min() - pad * 7, -span * 0.18) if log_scale else values.min() - pad * 7
    ax.set_xlim(left, values.max() + pad * 12)
    fig.tight_layout()
    _save(fig, "domain_launch_vs_harbor_improvement.png")


# ===================================================================
# Fig 10: Overall model ranking (mean score across all benchmarks)
# ===================================================================

def fig_model_ranking(harbor_df: pd.DataFrame) -> None:
    if harbor_df.empty:
        return
    _apply_style()
    from .cross_analysis import _is_well_scaled

    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    well_scaled = [c for c in bench_cols if _is_well_scaled(harbor_df[c])]

    model_means = (
        harbor_df.groupby("model")[well_scaled]
        .mean()
        .mean(axis=1)
        .sort_values(ascending=True)
    )
    if model_means.empty:
        return

    labels = [_wrap(m, 22) for m in model_means.index]
    fig, ax = plt.subplots(figsize=(11.5, max(4, 0.55 * len(model_means))))
    bars = ax.barh(labels, model_means.values, color=COLOR_BLUE, edgecolor="white")
    ax.set_xlabel("Mean score (across well-scaled benchmarks)")
    ax.set_title("Overall Model Ranking")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)

    for i, v in enumerate(model_means.values):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9, color=COLOR_AXIS)

    fig.tight_layout()
    _save(fig, "model_ranking.png")


# ===================================================================
# Fig 11: Per-domain agent effect (terminus-2 vs others, grouped by superdomain)
# ===================================================================

def fig_agent_domain_effect(harbor_df: pd.DataFrame) -> None:
    if harbor_df.empty or "agent" not in harbor_df.columns:
        return
    _apply_style()
    from .cross_analysis import _domain_for, _is_well_scaled

    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    well_scaled = {c for c in bench_cols if _is_well_scaled(harbor_df[c])}

    records = []
    for model, group in harbor_df.groupby("model"):
        t2 = group[group["agent"] == "terminus-2"]
        others = group[group["agent"] != "terminus-2"]
        if t2.empty or others.empty:
            continue
        t2_row = t2.iloc[0]
        for _, other_row in others.iterrows():
            for col in bench_cols:
                if col not in well_scaled:
                    continue
                v_t2 = t2_row[col]
                v_other = other_row[col]
                if pd.isna(v_t2) or pd.isna(v_other):
                    continue
                domain, superdomain = _domain_for(col)
                records.append(dict(
                    model=model,
                    agent=other_row["agent"],
                    benchmark=col,
                    domain=domain,
                    superdomain=superdomain,
                    delta=float(v_other) - float(v_t2),
                ))

    if not records:
        return
    df = pd.DataFrame(records)

    agg = df.groupby(["agent", "superdomain"]).agg(
        mean_delta=("delta", "mean"),
        n=("delta", "size"),
    ).reset_index()

    agents = sorted(agg["agent"].unique())
    superdomains = sorted(agg["superdomain"].unique())

    fig, ax = plt.subplots(figsize=(10, max(4, 0.8 * len(agents))))
    y = np.arange(len(agents))
    bar_height = 0.35
    sd_colors = [SUPERDOMAIN_COLORS.get(sd, COLOR_GRAY) for sd in superdomains]

    for j, sd in enumerate(superdomains):
        vals = []
        ns = []
        for agent in agents:
            row = agg[(agg["agent"] == agent) & (agg["superdomain"] == sd)]
            vals.append(float(row["mean_delta"].iloc[0]) if not row.empty else 0)
            ns.append(int(row["n"].iloc[0]) if not row.empty else 0)
        offset = (j - (len(superdomains) - 1) / 2) * bar_height
        bars = ax.barh(y + offset, vals, height=bar_height,
                       color=sd_colors[j], edgecolor="white", label=sd)
        for i, (v, n) in enumerate(zip(vals, ns)):
            if n > 0:
                ax.text(v + (0.003 if v >= 0 else -0.003), y[i] + offset,
                        f"n={n}", va="center",
                        ha="left" if v >= 0 else "right",
                        fontsize=8, color="gray")

    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(a, 20) for a in agents])
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.set_xlabel("Mean score change vs terminus-2")
    ax.set_title("Agent Effect by Superdomain (vs Terminus-2)")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "agent_domain_effect.png")


# ===================================================================
# Public entry point
# ===================================================================

def generate_all_figures(
    summary: pd.DataFrame,
    direction: pd.DataFrame,
    lift: pd.DataFrame,
) -> None:
    print("\nGenerating comparison figures...")
    fig_harbor_vs_doc_scatter(summary)
    fig_delta_by_benchmark(direction)
    fig_agent_lift_heatmap(lift)
    fig_delta_distribution(summary)


def generate_cross_figures(
    difficulty: pd.DataFrame,
    domain_df: pd.DataFrame,
    superdomain_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    domain_prog_df: pd.DataFrame,
    benchmark_launch_imp_df: pd.DataFrame | None = None,
    domain_launch_imp_df: pd.DataFrame | None = None,
    harbor_df: pd.DataFrame | None = None,
) -> None:
    print("\nGenerating cross-analysis figures...")
    fig_hardest_benchmarks(difficulty)
    fig_domain_scores(domain_df)
    fig_coding_vs_noncoding(superdomain_df)
    fig_progress_over_time(progress_df, harbor_df)
    fig_domain_progress(domain_prog_df)
    if benchmark_launch_imp_df is not None:
        fig_benchmark_launch_improvement(benchmark_launch_imp_df)
    if domain_launch_imp_df is not None:
        fig_domain_launch_improvement(domain_launch_imp_df)
    if harbor_df is not None:
        fig_model_ranking(harbor_df)
        fig_agent_domain_effect(harbor_df)
    print(f"All figures saved to {FIGURE_DIR}/")
