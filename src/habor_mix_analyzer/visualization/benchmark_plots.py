from __future__ import annotations

from ..core import *
from matplotlib.colors import to_rgb, to_hex


def darken_color(color: str, factor: float = 0.72) -> str:
    """
    factor < 1 会变深；越小越深
    比如 0.72 / 0.65 都比较合适
    """
    r, g, b = to_rgb(color)
    return to_hex((r * factor, g * factor, b * factor))


def save_svd_spectrum_plot(raw_mat: pd.DataFrame, included_benchmarks: list[str] | None = None) -> None:
    from ..studies.benchmark_predictability import _is_pct_col, _to_logit

    key = ["model", "agent"]
    agents = raw_mat["agent"].tolist()
    bench_names = [c for c in raw_mat.columns if c not in key and raw_mat[c].dtype.kind in "fi"]
    if included_benchmarks is not None:
        bench_names = [c for c in bench_names if c in included_benchmarks]
    matrix_np = raw_mat[bench_names].values
    eps = 0.005

    def _logit_zscore(M: np.ndarray) -> np.ndarray:
        M_l = M.copy()
        is_pct = np.array([_is_pct_col(M[:, j]) for j in range(M.shape[1])])
        for j in range(M_l.shape[1]):
            valid = ~np.isnan(M_l[:, j])
            if valid.any() and is_pct[j]:
                M_l[valid, j] = _to_logit(M[valid, j], eps=eps)
        mu = np.nanmean(M_l, axis=0)
        sd = np.nanstd(M_l, axis=0)
        sd[sd == 0] = 1
        Mz = (M_l - mu) / sd
        return np.where(np.isnan(Mz), 0.0, Mz)

    terminus_mask = np.array([a == BASELINE_AGENT for a in agents])

    spectra = {}
    for label, mask in [("Baseline scaffold (Terminus-2)", terminus_mask), ("Full (all scaffolds)", np.ones(len(agents), dtype=bool))]:
        Mz = _logit_zscore(matrix_np[mask])
        _, s, _ = np.linalg.svd(Mz, full_matrices=False)
        v = (s ** 2) / (s ** 2).sum() * 100
        spectra[label] = v

    n_show = min(10, min(len(v) for v in spectra.values()))
    x = np.arange(n_show)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax2 = ax.twinx()

    colors = {"Baseline scaffold (Terminus-2)": "#4292c6", "Full (all scaffolds)": "#756bb1"}
    for i, (label, v) in enumerate(spectra.items()):
        vals = v[:n_show]
        bars = ax.bar(x + (i - 0.5) * bar_w, vals, bar_w, label=label,
                      color=colors[label], edgecolor="white", alpha=0.85)
        for j, bar in enumerate(bars):
            if vals[j] > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{vals[j]:.1f}%", ha="center", va="bottom", fontsize=8,
                        color=colors[label], fontweight="bold")

    for label, v in spectra.items():
        remaining = 100 - np.cumsum(v[:n_show])
        ax2.plot(x, remaining, "o-", color=colors[label], markersize=5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\sigma_{{{i+1}}}$" for i in range(n_show)], fontsize=11)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, max(v[0] for v in spectra.values()) * 1.2)
    ax2.set_ylabel("Remaining spectrum (%)", color="#888")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#888")

    n_sys_model = int(terminus_mask.sum())
    n_sys_full = len(agents)
    n_bench = matrix_np.shape[1]
    ax.set_title(
        f"Singular value spectrum of the {n_sys_full}×{n_bench} benchmark matrix\n"
        f"(logit-transformed, z-scored — baseline subset: {n_sys_model}×{n_bench})",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.6)

def save_key_effect_plot(effects: pd.DataFrame, group_col: str, filename: str, title: str) -> None:
    plot_df = effects.sort_values("adjusted_mean")
    labels = [wrap_text(value, width=28) for value in plot_df[group_col]]
    fig, ax = plt.subplots(figsize=(11.5, max(5.2, 0.42 * len(plot_df))))
    ax.barh(labels, plot_df["adjusted_mean"], color="#a1d99b", edgecolor="white")
    ax.axvline(0, color="#666666", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Adjusted benchmark-relative score\n(0 = benchmark median; +1 = one robust scale above median)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    save_key_figure(fig, f"benchmark_level/{filename}")
    plt.close(fig)


def save_agent_lift_heatmap(agent_by_benchmark: pd.DataFrame) -> None:
    if agent_by_benchmark.empty:
        return
    from matplotlib.colors import LinearSegmentedColormap

    agent_by_benchmark = agent_by_benchmark.copy()
    agent_by_benchmark["domain"] = agent_by_benchmark["benchmark"].map(BENCHMARK_DOMAIN).fillna("Other")
    domains_present = [d for d in DOMAIN_ORDER if d in agent_by_benchmark["domain"].values]

    ordered_benchmarks = []
    domain_boundaries = []
    domain_labels = []
    for d in domains_present:
        benchmarks = sorted(agent_by_benchmark.loc[agent_by_benchmark["domain"] == d, "benchmark"].unique())
        start = len(ordered_benchmarks)
        if start > 0:
            domain_boundaries.append(start)
        ordered_benchmarks.extend(benchmarks)
        end = len(ordered_benchmarks) - 1
        domain_labels.append((start + end) / 2.0)

    pivot = agent_by_benchmark.pivot(
        index="agent", columns="benchmark", values="mean_delta_vs_terminus",
    ).fillna(0)[ordered_benchmarks]
    agents = list(pivot.index)
    n_agents = len(agents)
    n_benchmarks = len(ordered_benchmarks)

    cmap = LinearSegmentedColormap.from_list(
        "blue_pink",
        ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#e8829b", "#c51b7d"],
    )
    vmin, vmax = -2.5, 2.5

    fig_w = max(10, 0.36 * n_benchmarks + 4.0)
    fig_h = max(5, 1.0 * n_agents + 4.0)
    fig, (ax_domain, ax) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [0.06, 1], "hspace": 0.02},
    )

    benchmark_domains = [BENCHMARK_DOMAIN.get(b, "Other") for b in ordered_benchmarks]
    for i, d in enumerate(benchmark_domains):
        ax_domain.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1, color=DOMAIN_COLORS.get(d, "#aaa"), linewidth=0))
    ax_domain.set_xlim(-0.5, n_benchmarks - 0.5)
    ax_domain.set_ylim(0, 1)
    ax_domain.set_xticks([])
    ax_domain.set_yticks([])
    ax_domain.set_frame_on(False)
    for bnd in domain_boundaries:
        ax_domain.axvline(bnd - 0.5, color="white", linewidth=2.2)

    data = pivot.to_numpy(dtype=float)
    image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(n_benchmarks))
    ax.set_xticklabels(
        [benchmark_display_name(b) for b in ordered_benchmarks],
        rotation=90, ha="center", fontsize=13,
    )
    ax.set_yticks(np.arange(n_agents))
    ax.set_yticklabels(agents, fontsize=14)

    for bnd in domain_boundaries:
        ax.axvline(bnd - 0.5, color="white", linewidth=2.2)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLORS.get(d, "#aaa"))
        for d in domains_present
    ]
    fig.legend(
        legend_handles, domains_present, loc="lower center",
        ncol=len(domains_present), fontsize=8.5, framealpha=0.9,
        bbox_to_anchor=(0.48, -0.01), handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
    )

    ax_domain.set_title("Agent Lift vs terminus-2 by Benchmark (grouped by domain)", fontsize=16, pad=10)
    cbar = fig.colorbar(image, ax=[ax_domain, ax], fraction=0.018, pad=0.015, shrink=0.75)
    cbar.set_label("Score change vs terminus-2", fontsize=13)
    fig.subplots_adjust(bottom=0.25, top=0.90, left=0.08, right=0.88)
    save_key_figure(fig, "benchmark_level/benchmark_agent_lift_heatmap.png")
    plt.close(fig)


def save_benchmark_uniqueness_plot(uniqueness: pd.DataFrame, filter_table: pd.DataFrame) -> None:
    has_medape = "benchpress_medape" in uniqueness.columns
    plot_df = uniqueness.merge(
        filter_table[["benchmark", "task_cell_missing_fraction"]],
        on="benchmark",
        how="left",
    )

    if has_medape:
        plot_df = plot_df.sort_values("benchpress_medape", ascending=True)
        labels = [wrap_text(benchmark_display_name(v), 24) for v in plot_df["benchmark"]]
        values = plot_df["benchpress_medape"].astype(float)
        values = values.replace([np.inf, -np.inf], np.nan).fillna(values[np.isfinite(values)].max() * 1.2 if np.any(np.isfinite(values)) else 1.0)

        fig, ax = plt.subplots(figsize=(11.5, max(8.2, 0.30 * len(plot_df))))
        colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(values)))
        ax.barh(labels, values, color=colors, edgecolor="white")
        ax.set_title("BenchPress Predictability: Per-Benchmark Holdout Error\n(LogitBenchReg + SVD-Logit blend)", fontsize=14)
        ax.set_xlabel("Median Absolute Percentage Error (MedAPE %)\n(higher = harder to predict / more unique)")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        fig.tight_layout()
        save_key_figure(fig, "benchmark_level/benchmark_uniqueness_vs_coverage.png")
        plt.close(fig)
    else:
        plot_df = plot_df.sort_values("cv_r2_from_other_included_benchmarks")
        labels = [wrap_text(benchmark_display_name(v), 24) for v in plot_df["benchmark"]]
        values = plot_df["cv_r2_from_other_included_benchmarks"].astype(float)
        colors = np.where(values < 0, "#fdae6b", "#9ecae1")
        fig, ax = plt.subplots(figsize=(11.5, max(8.2, 0.30 * len(plot_df))))
        ax.barh(labels, values, color=colors, edgecolor="white")
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_title("Benchmark Predictability From Other Benchmarks")
        ax.set_xlabel("Cross-validated R² predicted from other included benchmarks\n(lower = more unique / harder to reconstruct)")
        ax.set_ylabel("")
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        left_edge = min(values.min(), 0)
        right_edge = max(values.max(), 0)
        ax.set_xlim(left_edge - 0.08 * max(1, abs(left_edge)), right_edge + 0.08 * max(1, abs(right_edge)))
        fig.tight_layout()
        save_key_figure(fig, "benchmark_level/benchmark_uniqueness_vs_coverage.png")
        plt.close(fig)


def save_benchmark_cluster_heatmap(ordered_corr: pd.DataFrame) -> None:
    matrix = ordered_corr.set_index("benchmark")
    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Clustered Benchmark Similarity")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([wrap_text(benchmark_display_name(value), 13) for value in matrix.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([wrap_text(benchmark_display_name(value), 16) for value in matrix.index], fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman correlation of agent+model score profiles")
    distance_array = (1 - matrix.abs()).to_numpy(copy=True)
    np.fill_diagonal(distance_array, 0)
    labels = fcluster(linkage(squareform(distance_array, checks=False), method="average"), t=6, criterion="maxclust")
    boundaries = np.where(labels[1:] != labels[:-1])[0] + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=1.8)
        ax.axvline(boundary, color="white", linewidth=1.8)
    fig.tight_layout()
    save_key_figure(fig, "benchmark_level/benchmark_similarity_clustered_heatmap.png")
    plt.close(fig)


def save_benchmark_headroom_plot(headroom: pd.DataFrame) -> None:
    if headroom.empty:
        return
    domain_order = [
        "Software Engineering",
        "Mathematics & Reasoning",
        "Knowledge & Long Context",
        "Scientific Research",
        "Agents, Tools & Systems",
        "Data & Analytics",
        "Professional Domains",
        "Safety & Security",
        "Multimodal",
        "Other",
    ]
    domain_colors = {
        "Software Engineering": "#4292c6",
        "Mathematics & Reasoning": "#e6550d",
        "Knowledge & Long Context": "#756bb1",
        "Scientific Research": "#31a354",
        "Agents, Tools & Systems": "#d6616b",
        "Data & Analytics": "#8ca252",
        "Professional Domains": "#de9ed6",
        "Safety & Security": "#636363",
        "Multimodal": "#e7969c",
        "Other": "#aaaaaa",
    }
    headroom = headroom.copy()

    tier_markers = {"frontier": "F", "hard": "H", "medium": "M", "easy": "E", "saturated": "S", "unbounded_or_penalty": "U"}
    tier_colors_map = {"frontier": "#d62728", "hard": "#e6550d", "medium": "#756bb1", "easy": "#31a354", "saturated": "#636363", "unbounded_or_penalty": "#999999"}

    # --- Plot 1: sorted by domain, then by best_score ---
    by_domain = headroom.copy()
    by_domain["domain"] = pd.Categorical(by_domain["domain"], categories=domain_order, ordered=True)
    by_domain = by_domain.sort_values(["domain", "best_score"], ascending=[True, True])

    for suffix, plot_df in [("by_domain", by_domain), ("by_score", headroom.sort_values("best_score"))]:
        labels = [wrap_text(benchmark_display_name(row["benchmark"]), 24) for _, row in plot_df.iterrows()]
        colors = [domain_colors.get(row["domain"], "#aaaaaa") for _, row in plot_df.iterrows()]
        n = len(plot_df)

        if suffix == "by_domain":
            domain_boundaries = [0]
            for i in range(1, n):
                if plot_df.iloc[i]["domain"] != plot_df.iloc[i - 1]["domain"]:
                    domain_boundaries.append(i)
            domain_boundaries.append(n)
            mid = min(domain_boundaries, key=lambda b: abs(b - n / 2))
            if mid == 0:
                mid = domain_boundaries[1]
            elif mid == n:
                mid = domain_boundaries[-2]
            n_left = mid
            n_right = n - mid
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, max(6, 0.32 * max(n_left, n_right))),
                                                     gridspec_kw={"wspace": 0.6})
            for ax, start, end in [(ax_left, 0, mid), (ax_right, mid, n)]:
                chunk_n = end - start
                chunk_labels = labels[start:end]
                chunk_colors = colors[start:end]
                chunk_df = plot_df.iloc[start:end]
                y = np.arange(chunk_n)
                bar_h = 0.65
                ax.barh(y, chunk_df["best_score"].values, height=bar_h, color=chunk_colors, edgecolor="white", alpha=0.85)
                ax.barh(y, chunk_df["headroom"].values, left=chunk_df["best_score"].values,
                        height=bar_h, color=chunk_colors, edgecolor="white", alpha=0.2)
                for i, (_, row) in enumerate(chunk_df.iterrows()):
                    ax.text(row["best_score"] + 0.01, i, f'{row["best_score"]:.0%}', va="center", fontsize=8, color="#333333")
                ax.set_yticks(y)
                ax.set_yticklabels(chunk_labels, fontsize=13)
                ax.set_xlim(0, 1.12)
                ax.set_xlabel("Score (best system)", fontsize=11)
                ax.axvline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
                ax.grid(axis="x", color="#dddddd", linewidth=0.8)
                prev_domain = None
                for i, (_, row) in enumerate(chunk_df.iterrows()):
                    if prev_domain is not None and row["domain"] != prev_domain:
                        ax.axhline(i - 0.5, color="#cccccc", linewidth=0.6, linestyle="-")
                    prev_domain = row["domain"]

            domains_present = [d for d in domain_order if d in plot_df["domain"].values]
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=domain_colors.get(d, "#aaa")) for d in domains_present]
            fig.legend(legend_handles, domains_present, loc="lower center",
                       ncol=min(len(domains_present), 4), fontsize=10, framealpha=0.9,
                       bbox_to_anchor=(0.5, -0.02))
            fig.suptitle("Benchmark Headroom: Best System Score by Domain\n(shaded area = room for improvement)",
                         fontsize=13)
        else:
            fig, ax = plt.subplots(figsize=(14, max(8, 0.38 * n)))
            y = np.arange(n)
            ax.barh(y, plot_df["best_score"], color=colors, edgecolor="white", alpha=0.85)
            ax.barh(y, plot_df["headroom"], left=plot_df["best_score"], color=colors, edgecolor="white", alpha=0.2)
            for i, (_, row) in enumerate(plot_df.iterrows()):
                tier = row.get("difficulty_tier", "")
                marker = tier_markers.get(tier, "?")
                tc = tier_colors_map.get(tier, "#999")
                ax.text(-0.02, i, marker, va="center", ha="center", fontsize=8, fontweight="bold", color=tc)
                ax.text(row["best_score"] + 0.01, i, f'{row["best_score"]:.1%}', va="center", fontsize=7.5, color="#333333")
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlim(-0.05, 1.08)
            ax.set_xlabel("Score (best system)")
            ax.set_title("Benchmark Headroom (ranked by best score)\nDifficulty tier: F=Frontier (<5%), H=Hard (5-30%), M=Medium (30-70%), E=Easy (70-95%), S=Saturated (>95%)",
                         fontsize=13)
            ax.axvline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.grid(axis="x", color="#dddddd", linewidth=0.8)
            domains_present = [d for d in domain_order if d in plot_df["domain"].values]
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=domain_colors.get(d, "#aaa")) for d in domains_present]
            ax.legend(legend_handles, domains_present, loc="lower right", fontsize=8, framealpha=0.9)

        fig.tight_layout(rect=[0, 0.05, 1, 1] if suffix == "by_domain" else [0, 0, 1, 1])
        save_key_figure(fig, f"benchmark_level/benchmark_headroom_{suffix}.png")
        plt.close(fig)


def save_benchmark_progress_and_headroom_plot(headroom: pd.DataFrame, launch_progress: pd.DataFrame) -> None:
    if headroom.empty or launch_progress.empty:
        return

    required = {"benchmark", "matrix_column", "launch_best_score", "status"}
    if not required.issubset(launch_progress.columns):
        return

    excluded_benchmarks = {"bfcl", "medagentbench", "sldbench", "codepde"}
    label_overrides = {
        "aime": "AIME 24&25",
        "bigcodebench": "BigCodeBench-Hard",
    }

    launch_cols = ["matrix_column", "launch_best_score"]
    if "is_subset" in launch_progress.columns:
        launch_cols.append("is_subset")
    launch = launch_progress.loc[
        (launch_progress["status"] == "ok")
        & launch_progress["launch_best_score"].notna(),
        launch_cols,
    ].copy()
    if "is_subset" not in launch.columns:
        launch["is_subset"] = False
    launch["is_subset"] = launch["is_subset"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    if launch.empty:
        return

    plot_df = headroom.merge(
        launch.rename(columns={"matrix_column": "benchmark", "launch_best_score": "past_sota"}),
        on="benchmark",
        how="inner",
    )
    plot_df = plot_df[~plot_df["benchmark"].isin(excluded_benchmarks)].copy()
    if plot_df.empty:
        return

    domain_order = [
        "Software Engineering",
        "Mathematics & Reasoning",
        "Knowledge & Long Context",
        "Scientific Research",
        "Agents, Tools & Systems",
        "Data & Analytics",
        "Professional Domains",
        "Safety & Security",
        "Multimodal",
        "Other",
    ]
    domain_colors = {
        "Software Engineering": "#4292c6",
        "Mathematics & Reasoning": "#e6550d",
        "Knowledge & Long Context": "#756bb1",
        "Scientific Research": "#31a354",
        "Agents, Tools & Systems": "#d6616b",
        "Data & Analytics": "#8ca252",
        "Professional Domains": "#de9ed6",
        "Safety & Security": "#636363",
        "Multimodal": "#e7969c",
        "Other": "#aaaaaa",
    }
    plot_df = plot_df.copy()
    plot_df["domain"] = pd.Categorical(plot_df["domain"], categories=domain_order, ordered=True)
    plot_df["past_sota"] = pd.to_numeric(plot_df["past_sota"], errors="coerce").clip(lower=0, upper=1)
    plot_df["current_sota"] = pd.to_numeric(plot_df["best_score"], errors="coerce").clip(lower=0, upper=1)
    plot_df = plot_df.dropna(subset=["past_sota", "current_sota"])
    if plot_df.empty:
        return

    plot_df["headroom_width"] = (1.0 - plot_df["current_sota"]).clip(lower=0)
    plot_df = plot_df.sort_values(["domain", "current_sota"], ascending=[True, True])

    def _display_label(row: pd.Series) -> str:
        benchmark = str(row["benchmark"])
        label = label_overrides.get(benchmark, benchmark_display_name(benchmark))
        if bool(row.get("is_subset", False)):
            label = f"{label}*"
        return label

    labels = [wrap_text(_display_label(row), 24) for _, row in plot_df.iterrows()]
    colors = [domain_colors.get(str(value), "#aaaaaa") for value in plot_df["domain"]]
    n = len(plot_df)

    # Lighter past-SOTA styling
    past_edge = "#969696"
    past_hatch = "///"
    past_alpha = 0.60

    domain_boundaries = [0]
    for i in range(1, n):
        if plot_df.iloc[i]["domain"] != plot_df.iloc[i - 1]["domain"]:
            domain_boundaries.append(i)
    domain_boundaries.append(n)

    mid = min(domain_boundaries, key=lambda b: abs(b - n / 2))
    if mid == 0:
        mid = domain_boundaries[1]
    elif mid == n:
        mid = domain_boundaries[-2]

    n_left = mid
    n_right = n - mid

    with plt.rc_context():
        plt.rcdefaults()
        plt.rcParams["hatch.linewidth"] = 0.35

        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=(12, max(6, 0.32 * max(n_left, n_right))),
            gridspec_kw={"wspace": 0.42},
        )

        for ax, start, end in [(ax_left, 0, mid), (ax_right, mid, n)]:
            chunk_n = end - start
            chunk_labels = labels[start:end]
            chunk_colors = colors[start:end]
            chunk_df = plot_df.iloc[start:end]
            y = np.arange(chunk_n)
            bar_h = 0.65

            ax.barh(
                y,
                chunk_df["current_sota"].values,
                height=bar_h,
                color=chunk_colors,
                edgecolor="white",
                alpha=0.35,
                linewidth=0.45,
            )

            ax.barh(
                y,
                chunk_df["headroom_width"].values,
                left=chunk_df["current_sota"].values,
                height=bar_h,
                color=chunk_colors,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.0,
            )

            # Past SOTA at launch: same category color, but darker outline only
            for i, (_, row) in enumerate(chunk_df.iterrows()):
                outline_color = darken_color(chunk_colors[i], factor=0.72)
                ax.barh(
                    i,
                    row["past_sota"],
                    height=bar_h,
                    facecolor="none",
                    edgecolor=outline_color,
                    linewidth=0.8,
                )

            for i, (_, row) in enumerate(chunk_df.iterrows()):
                ax.text(
                    row["current_sota"] + 0.01,
                    i,
                    f'{row["current_sota"]:.0%}',
                    va="center",
                    color="#333333",
                )

            ax.set_yticks(y)
            ax.set_yticklabels(chunk_labels)
            ax.set_xlim(0, 1.12)
            ax.set_xlabel("Score")
            ax.axvline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.grid(axis="x", color="#dddddd", linewidth=0.8, linestyle="--", alpha=0.8)
            for spine in ax.spines.values():
                spine.set_visible(True)

            prev_domain = None
            for i, (_, row) in enumerate(chunk_df.iterrows()):
                if prev_domain is not None and row["domain"] != prev_domain:
                    ax.axhline(i - 0.5, color="#cccccc", linewidth=0.6, linestyle="-")
                prev_domain = row["domain"]

        domains_present = [d for d in domain_order if d in plot_df["domain"].astype(str).values]

        domain_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=domain_colors.get(d, "#aaa"),
                alpha=0.35,
            )
            for d in domains_present
        ]

        past_handle = plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor="#666666",
            linewidth=1.4,
        )

        if domain_handles:
            fig.legend(
                [past_handle, *domain_handles],
                ["Past SOTA at launch", *domains_present],
                loc="lower center",
                ncol=min(len(domains_present) + 1, 5),
                framealpha=0.9,
                bbox_to_anchor=(0.5, 0.03),
            )
        # fig.suptitle(
        #     "Benchmark Headroom: Best System Score by Domain\n(shaded area = room for improvement)",
        # )

        fig.subplots_adjust(left=0.12, right=0.965, bottom=0.185, top=0.90, wspace=0.42)

        path = OUTPUT_DIR / "quantitative" / "figures" / "bench_progress_and_headroom.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def save_benchmark_headroom_summary_plot(headroom: pd.DataFrame) -> None:
    if headroom.empty:
        return
    tier_order = ["frontier", "hard", "medium", "easy", "saturated", "unbounded_or_penalty"]
    tier_labels = {
        "frontier": "Frontier (<5%)",
        "hard": "Hard (5-30%)",
        "medium": "Medium (30-70%)",
        "easy": "Easy (70-95%)",
        "saturated": "Saturated (>95%)",
        "unbounded_or_penalty": "Unbounded",
    }
    tier_colors = {
        "frontier": "#d62728",
        "hard": "#e6550d",
        "medium": "#756bb1",
        "easy": "#31a354",
        "saturated": "#636363",
        "unbounded_or_penalty": "#999999",
    }
    domain_order = [
        "Software Engineering",
        "Mathematics & Reasoning",
        "Knowledge & Long Context",
        "Scientific Research",
        "Agents, Tools & Systems",
        "Data & Analytics",
        "Professional Domains",
        "Safety & Security",
        "Multimodal",
    ]
    headroom = headroom.copy()
    domains_present = [d for d in domain_order if d in headroom["domain"].values]
    tiers_present = [t for t in tier_order if t in headroom["difficulty_tier"].values]
    counts = headroom.groupby(["domain", "difficulty_tier"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=domains_present, columns=tiers_present, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.6 * len(domains_present))))
    x = np.arange(len(domains_present))
    bottoms = np.zeros(len(domains_present))
    for tier in tiers_present:
        vals = counts[tier].values.astype(float)
        ax.barh(x, vals, left=bottoms, color=tier_colors.get(tier, "#aaa"), edgecolor="white", label=tier_labels.get(tier, tier))
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(bottoms[i] + v / 2, i, str(int(v)), va="center", ha="center", fontsize=9, fontweight="bold", color="white")
        bottoms += vals

    ax.set_yticks(x)
    ax.set_yticklabels(domains_present, fontsize=10)
    ax.set_xlabel("Number of benchmarks")
    ax.set_title("Benchmark Difficulty Tier Distribution by Domain\nTier by mean score: Frontier (<5%), Hard (5-30%), Medium (30-70%), Easy (70-95%), Saturated (>95%)",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    save_key_figure(fig, "benchmark_level/benchmark_headroom_tier_summary.png")
    plt.close(fig)


def save_within_family_summary_plot(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    summary = summary.sort_values("model_effect_median", ascending=False)
    families = summary["family"]
    x = np.arange(len(families))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.bar(x - width / 2, summary["model_effect_median"], width, label="Model effect (median)", color="#4292c6")
    ax.bar(x + width / 2, summary["agent_effect_median"], width, label="Agent effect (median)", color="#fdae6b")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_ylabel("Score range / delta")
    ax.set_title("Within-Family: Model vs Agent Effect Size")
    ax.legend()
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    for i, row in summary.iterrows():
        med_m = row["model_effect_median"]
        med_a = row["agent_effect_median"]
        top = max(med_m, med_a)
        ratio = med_m / med_a if med_a > 1e-9 else float("inf")
        idx = list(families).index(row["family"])
        ax.text(idx, top + 0.005, f"{ratio:.1f}x", ha="center", fontsize=9)

    ax = axes[1]
    win_pct = summary["model_wins"] / summary["total_benchmarks"] * 100
    colors = ["#4292c6" if p >= 50 else "#fdae6b" for p in win_pct]
    ax.barh(families, win_pct, color=colors, edgecolor="white")
    ax.axvline(50, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("% of benchmarks where model effect > agent effect")
    ax.set_title("Within-Family: Model-Dominant Benchmark %")
    ax.set_xlim(0, 105)
    for i, (pct, total) in enumerate(zip(win_pct, summary["total_benchmarks"])):
        ax.text(pct + 1, i, f"{pct:.0f}% ({int(summary.iloc[i]['model_wins'])}/{total})", va="center", fontsize=9)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)

    fig.tight_layout()
    save_key_figure(fig, "benchmark_level/within_family_model_vs_agent_summary.png")
    plt.close(fig)


def save_within_family_detail_plot(detail: pd.DataFrame) -> None:
    if detail.empty:
        return
    multi_model_families = detail.groupby("family").filter(lambda g: g["n_models"].max() >= 2)
    if multi_model_families.empty:
        return
    families = sorted(multi_model_families["family"].unique())
    n_families = len(families)
    fig, axes = plt.subplots(1, n_families, figsize=(6 * n_families, 7), squeeze=False)

    for i, family in enumerate(families):
        ax = axes[0, i]
        fam_data = multi_model_families[multi_model_families["family"] == family].sort_values("model_over_agent_ratio", ascending=True)
        top = fam_data.tail(20)
        labels = [wrap_text(benchmark_display_name(b), 20) for b in top["benchmark"]]
        y = np.arange(len(labels))
        ax.barh(y - 0.15, top["model_effect_range"], 0.3, label="Model effect", color="#4292c6")
        ax.barh(y + 0.15, top["agent_effect_mean_delta"], 0.3, label="Agent effect", color="#fdae6b")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Score range / delta")
        ax.set_title(f"{family}")
        ax.legend(fontsize=8)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)

    fig.suptitle("Within-Family Model vs Agent Effect by Benchmark (top 20)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_key_figure(fig, "benchmark_level/within_family_model_vs_agent_detail.png")
    plt.close(fig)


def save_terminus_delta_by_model_plot(terminus_by_model: pd.DataFrame) -> None:
    if terminus_by_model.empty:
        return
    pivot = terminus_by_model.pivot(index="agent", columns="model", values="mean_delta_vs_terminus").fillna(0)
    fig, ax = plt.subplots(figsize=(11, max(3.8, 0.7 * pivot.shape[0])))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="BrBG", vmin=-2.0, vmax=2.0)
    ax.set_title("How Each Agent Changes Performance vs Terminus by Model")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([wrap_text(value, 18) for value in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([wrap_text(value, 20) for value in pivot.index])
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(DELTA_SCORE_LABEL)
    fig.tight_layout()
    save_key_figure(fig, "benchmark_level/terminus_delta_by_model_heatmap.png")
    plt.close(fig)


DOMAIN_ORDER = [
    "Software Engineering",
    "Mathematics & Reasoning",
    "Knowledge & Long Context",
    "Scientific Research",
    "Agents, Tools & Systems",
    "Data & Analytics",
    "Professional Domains",
    "Safety & Security",
    "Multimodal",
    "Other",
]

DOMAIN_COLORS = {
    "Software Engineering": "#4292c6",
    "Mathematics & Reasoning": "#e6550d",
    "Knowledge & Long Context": "#756bb1",
    "Scientific Research": "#31a354",
    "Agents, Tools & Systems": "#d6616b",
    "Data & Analytics": "#8ca252",
    "Professional Domains": "#de9ed6",
    "Safety & Security": "#636363",
    "Multimodal": "#e7969c",
    "Other": "#aaaaaa",
}


def save_domain_grouped_heatmap(corr: pd.DataFrame) -> None:
    """Heatmap with benchmarks grouped by domain, domain color sidebar."""
    matrix = corr.set_index("benchmark") if "benchmark" in corr.columns else corr
    benchmarks = list(matrix.index)
    domains = [BENCHMARK_DOMAIN.get(b, "Other") for b in benchmarks]

    domain_cat = pd.Categorical(domains, categories=DOMAIN_ORDER, ordered=True)
    order = np.argsort(domain_cat.codes, kind="stable")
    ordered_benchmarks = [benchmarks[i] for i in order]
    ordered_domains = [domains[i] for i in order]
    ordered_matrix = matrix.iloc[order, order]

    n = len(ordered_benchmarks)
    fig_h = max(12, 0.30 * n)
    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(fig_h + 1.5, fig_h),
        gridspec_kw={"width_ratios": [0.03, 1], "wspace": 0.02},
    )

    bar_colors = [DOMAIN_COLORS.get(d, "#aaaaaa") for d in ordered_domains]
    for i, c in enumerate(bar_colors):
        ax_bar.add_patch(plt.Rectangle((0, i - 0.5), 1, 1, color=c, linewidth=0))
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, n - 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.invert_yaxis()
    ax_bar.set_frame_on(False)

    data = ordered_matrix.to_numpy(dtype=float)
    image = ax_heat.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    labels = [benchmark_display_name(b) for b in ordered_benchmarks]
    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
    ax_heat.set_yticks(np.arange(n))
    ax_heat.set_yticklabels(labels, fontsize=8)

    prev_domain = None
    for i, d in enumerate(ordered_domains):
        if prev_domain is not None and d != prev_domain:
            ax_heat.axhline(i - 0.5, color="white", linewidth=2.0)
            ax_heat.axvline(i - 0.5, color="white", linewidth=2.0)
        prev_domain = d

    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.02, shrink=0.7)
    cbar.set_label("Spearman rank correlation", fontsize=11)

    domains_present = [d for d in DOMAIN_ORDER if d in ordered_domains]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLORS.get(d, "#aaa")) for d in domains_present]
    fig.legend(
        legend_handles, domains_present, loc="lower center",
        ncol=min(len(domains_present), 5), fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.52, -0.01),
    )
    fig.suptitle("Benchmark Correlation by Domain", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save_key_figure(fig, "benchmark_level/benchmark_correlation_by_domain.png")
    plt.close(fig)




def save_effective_dimensionality_plot(
    explained_variance: np.ndarray,
    dim_stats: dict[str, float],
) -> None:
    """Scree plot + cumulative variance with effective-dimensionality annotations."""
    n = len(explained_variance)
    cumulative = np.cumsum(explained_variance)
    components = np.arange(1, n + 1)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35})

    ax_top.bar(components, explained_variance, color="#6baed6", edgecolor="white", alpha=0.85)
    ax_top.set_xlabel("Principal component")
    ax_top.set_ylabel("Variance explained")
    ax_top.set_title("Per-Component Variance (Scree)", fontsize=13)
    ax_top.grid(axis="y", color="#dddddd", linewidth=0.8)
    if n > 20:
        ax_top.set_xticks(np.arange(0, n + 1, max(1, n // 10)))

    ax_bot.plot(components, cumulative, "o-", color="#2171b5", markersize=4, linewidth=1.5)
    ax_bot.axhline(0.90, color="#e6550d", linewidth=1, linestyle="--", alpha=0.7)
    ax_bot.axhline(0.95, color="#d62728", linewidth=1, linestyle="--", alpha=0.7)
    ax_bot.text(n * 0.85, 0.905, "90%", color="#e6550d", fontsize=9)
    ax_bot.text(n * 0.85, 0.955, "95%", color="#d62728", fontsize=9)

    n90 = int(dim_stats["n_components_90pct"])
    n95 = int(dim_stats["n_components_95pct"])
    pr = dim_stats["participation_ratio"]
    ax_bot.axvline(n90, color="#e6550d", linewidth=0.8, linestyle=":")
    ax_bot.axvline(n95, color="#d62728", linewidth=0.8, linestyle=":")

    ax_bot.set_xlabel("Number of components")
    ax_bot.set_ylabel("Cumulative variance explained")
    ax_bot.set_title("Cumulative Variance & Effective Dimensionality", fontsize=13)
    ax_bot.grid(axis="both", color="#dddddd", linewidth=0.8)
    if n > 20:
        ax_bot.set_xticks(np.arange(0, n + 1, max(1, n // 10)))

    n_benchmarks = int(dim_stats.get("n_benchmarks", dim_stats["total_components"]))
    n_components = int(dim_stats["total_components"])
    textbox = (
        f"Included benchmarks: {n_benchmarks}\n"
        f"Extractable components: {n_components}\n"
        f"Participation ratio: {pr:.1f}\n"
        f"Components for 90%: {n90}\n"
        f"Components for 95%: {n95}\n"
        f"Top-1 variance: {dim_stats['top1_variance']:.1%}\n"
        f"Top-3 variance: {dim_stats['top3_variance']:.1%}"
    )
    ax_bot.text(
        0.98, 0.38, textbox, transform=ax_bot.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    fig.suptitle(
        f"{n_benchmarks} Benchmarks → ~{pr:.0f} Effective Independent Dimensions "
        f"(PR={pr:.1f})",
        fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_key_figure(fig, "benchmark_level/benchmark_effective_dimensionality.png")
    plt.close(fig)


def save_greedy_selection_plot(greedy_df: pd.DataFrame) -> None:
    """Plot the greedy benchmark selection curve: max correlation to selected set at each step."""
    n = len(greedy_df)
    steps = greedy_df["step"].values
    max_corr = greedy_df["max_abs_corr_to_selected"].values
    domains = greedy_df["domain"].values
    labels = [benchmark_display_name(b) for b in greedy_df["benchmark"]]

    fig, ax = plt.subplots(figsize=(9, max(10, 0.32 * n)))

    y = np.arange(n)
    colors = [DOMAIN_COLORS.get(d, "#aaaaaa") for d in domains]
    ax.barh(y, max_corr, color=colors, edgecolor="white", alpha=0.85, height=0.7)

    for i in range(n):
        if max_corr[i] > 0.01:
            ax.text(max_corr[i] + 0.01, i, f"{max_corr[i]:.2f}", va="center", fontsize=7.5, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}. {labels[i]}" for i in range(n)], fontsize=8)
    ax.set_xlabel("Max |correlation| to already-selected benchmarks\n(lower = adds more independent information)")
    ax.set_title("Greedy Benchmark Selection Order\n(each step adds the most independent remaining benchmark)", fontsize=12)
    ax.axvline(0.7, color="#d62728", linewidth=1, linestyle="--", alpha=0.6)
    ax.axvline(0.85, color="#e6550d", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(0.71, n * 0.02, "r=0.7", color="#d62728", fontsize=8)
    ax.text(0.86, n * 0.02, "r=0.85", color="#e6550d", fontsize=8)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.invert_yaxis()

    domains_present = [d for d in DOMAIN_ORDER if d in domains]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLORS.get(d, "#aaa")) for d in domains_present]
    ax.legend(legend_handles, domains_present, loc="lower right", fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    save_key_figure(fig, "benchmark_level/benchmark_greedy_selection.png")
    plt.close(fig)
