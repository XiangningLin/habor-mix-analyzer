from __future__ import annotations

from ..core import *


def save_task_similarity_heatmap(cross_similarity: pd.DataFrame, benchmark_clusters: pd.DataFrame) -> None:
    if cross_similarity.empty:
        return
    pivot = cross_similarity.pivot(index="left_benchmark", columns="right_benchmark", values="median_abs_task_similarity")
    all_benchmarks = sorted(set(pivot.index) | set(pivot.columns))
    pivot = pivot.reindex(index=all_benchmarks, columns=all_benchmarks)
    for left in all_benchmarks:
        for right in all_benchmarks:
            if pd.isna(pivot.loc[left, right]) and right in pivot.index and left in pivot.columns:
                pivot.loc[left, right] = pivot.loc[right, left]
    pivot = pivot.fillna(0)
    distance_array = (1 - pivot.to_numpy(dtype=float)).clip(0, 1)
    np.fill_diagonal(distance_array, 0)
    if len(all_benchmarks) >= 2:
        z = linkage(squareform(distance_array, checks=False), method="average")
        order = pivot.index[leaves_list(z)].tolist()
        cluster_labels = fcluster(z, t=min(6, len(all_benchmarks)), criterion="maxclust")
        label_by_benchmark = dict(zip(pivot.index, cluster_labels))
    else:
        order = all_benchmarks
        label_by_benchmark = {benchmark: 1 for benchmark in all_benchmarks}
    pivot = pivot.loc[order, order]
    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title("Task Similarity Within and Across Benchmarks")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([wrap_text(benchmark_display_name(value), 13) for value in order], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([wrap_text(benchmark_display_name(value), 16) for value in order], fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Median absolute Spearman correlation between reliable task score profiles")
    ordered_labels = np.array([label_by_benchmark[benchmark] for benchmark in order])
    boundaries = np.where(ordered_labels[1:] != ordered_labels[:-1])[0] + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=1.8)
        ax.axvline(boundary, color="white", linewidth=1.8)
    fig.tight_layout()
    save_key_figure(fig, "task_level/task_similarity_benchmark_pair_heatmap.pdf")
    plt.close(fig)


def save_representative_task_plot(representatives: pd.DataFrame) -> None:
    top = representatives.sort_values("representativeness_score", ascending=False).groupby("benchmark").head(1)
    plot_df = top.sort_values("representativeness_score", ascending=True).tail(35)
    labels = [wrap_text(f"{benchmark_display_name(row.benchmark)} / {str(row.task_id)[:44]}", 38) for row in plot_df.itertuples()]
    fig, ax = plt.subplots(figsize=(13, 13))
    ax.barh(labels, plot_df["representativeness_score"], color="#a1d99b", edgecolor="white")
    ax.set_title("Best Single Task Representative per Benchmark")
    ax.set_xlabel("Absolute correlation with benchmark's\nreliable-task aggregate")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    save_key_figure(fig, "task_level/task_best_representatives.pdf")
    plt.close(fig)
