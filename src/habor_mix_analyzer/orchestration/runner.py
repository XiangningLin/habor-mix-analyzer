from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..core import (
    BENCHMARK_INTERMEDIATE_STUDY_DIR,
    KEY_COLUMNS,
    KEY_FIGURE_DIR,
    KEY_REPORT_DIR,
    KEY_TABLE_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    TASK_INTERMEDIATE_STUDY_DIR,
    ImputationResult,
    clean_dir,
    read_matrix,
    score_columns,
    set_plot_style,
    write_csv,
)
from ..core.config import PAPER_BENCHMARKS
from ..preprocessing.svd_imputation import robust_column_stats, normalize
from ..reporting.key_analysis_report import copy_paper_figures, write_appendix_model_agent, write_appendix_stats, write_key_analysis_reports, write_paper_stats
from ..studies.benchmark_predictability import (
    pca_for_cols,
    predictability_for_cols,
)
from ..studies.benchmark_similarity import (
    benchmark_similarity_clusters,
    domain_correlation_decomposition,
    domain_correlation_summary,
    effective_dimensionality,
    full_pca_explained_variance,
    greedy_benchmark_selection,
    pairwise_correlations,
)
from ..studies.coverage_filtering import benchmark_filter_table
from ..studies.intermediate_tables import write_tables
from ..studies.leaderboards import (
    agent_model_scores_for_cols,
    benchmark_scores_long,
)
from ..studies.model_agent_roles import (
    adjusted_group_effects,
    benchmark_headroom_by_domain,
    within_family_model_vs_agent,
    within_family_summary,
)
from ..studies.provenance import analysis_data_provenance, imputation_diagnostics_summary
from ..studies.task_alignment import task_aggregate_alignment
from ..studies.task_selection import select_harbormix_tasks, task_reliability_tables
from ..studies.task_similarity import build_goto_task_set, global_task_selection, greedy_task_selection_within_benchmark, task_predictability_holdout, task_similarity_and_representatives, unified_task_holdout
from ..studies.terminus_comparison import summarize_agent_lift, terminus_delta_by_model
from ..visualization.benchmark_plots import (
    save_agent_lift_heatmap,
    save_benchmark_cluster_heatmap,
    save_benchmark_headroom_plot,
    save_benchmark_uniqueness_plot,
    save_effective_dimensionality_plot,
    save_greedy_selection_plot,
    save_svd_spectrum_plot,
    save_within_family_detail_plot,
    save_within_family_summary_plot,
)
from ..visualization.task_plots import (
    save_representative_task_plot,
    save_task_similarity_heatmap,
)

INTERMEDIATE_TABLES = [
    "benchmark_observed_imputed_long",
    "task_item_stats",
    "task_benchmark_summary",
    "task_benchmark_matrix_from_tasks",
    "agent_model_strength_scores",
    "agent_differential_by_benchmark",
    "variance_decomposition",
    "benchmark_correlations",
    "benchmark_redundancy_pairs",
    "benchmark_predictability",
    "benchmark_latent_loadings",
    "benchmark_latent_agent_model_scores",
    "benchmark_latent_explained_variance",
]

KEY_ANALYSIS_TABLES = [
    "benchmark_filtering",
    "analysis_data_provenance",
    "imputation_diagnostics_summary",
    "benchmark_agent_model_scores",
    "benchmark_scores_long",
    "benchmark_model_adjusted_effects",
    "benchmark_agent_adjusted_effects",
    "benchmark_within_family_model_vs_agent",
    "benchmark_within_family_summary",
    "benchmark_headroom_by_domain",
    "benchmark_similarity_clusters",
    "benchmark_correlation_clustered",
    "benchmark_redundancy_pairs_filtered",
    "benchmark_domain_enriched_pairs",
    "benchmark_domain_correlation_summary",
    "benchmark_effective_dimensionality",
    "benchmark_greedy_selection",
    "benchmark_uniqueness_filtered",
    "benchmark_agent_lift_vs_terminus",
    "terminus_delta_by_model",
    "task_benchmark_reliable_summary",
    "task_to_benchmark_alignment",
    "task_within_benchmark_similarity",
    "task_cross_benchmark_similarity",
    "task_representative_tasks",
    "task_predictability_ranked",
    "task_holdout_predictability",
    "task_holdout_method_comparison",
    "task_greedy_selection",
    "task_greedy_selection_summary",
    "task_goto_set",
    "task_global_representatives",
    "task_global_greedy_selection",
    "task_holdout_unified_comparison",
    "harbormix_selected_tasks",
    "harbormix_selection_by_benchmark",
]

KEY_TABLE_SUBDIRS = {
    "analysis_data_provenance": "provenance",
    "imputation_diagnostics_summary": "provenance",
    "benchmark_agent_model_scores": "leaderboards",
    "benchmark_scores_long": "leaderboards",
    "task_benchmark_reliable_summary": "task_level",
    "task_to_benchmark_alignment": "task_level",
    "task_within_benchmark_similarity": "task_level",
    "task_cross_benchmark_similarity": "task_level",
    "task_representative_tasks": "task_level",
    "task_predictability_ranked": "task_level",
    "task_holdout_predictability": "task_level",
    "task_holdout_method_comparison": "task_level",
    "task_greedy_selection": "task_level",
    "task_greedy_selection_summary": "task_level",
    "task_goto_set": "task_level",
    "task_global_representatives": "task_level",
    "task_global_greedy_selection": "task_level",
    "task_holdout_unified_comparison": "task_level",
    "harbormix_selected_tasks": "harbormix",
    "harbormix_selection_by_benchmark": "harbormix",
    "terminus_delta_by_model": "benchmark_level",
}


def log(message: str) -> None:
    print(f"[habor-analyze] {message}", flush=True)


def clean_legacy_output_dirs() -> None:
    for path in [
        PROCESSED_DIR.parent / "generated",
        OUTPUT_DIR / "figures",
        OUTPUT_DIR / "tables",
        OUTPUT_DIR / "reports",
        OUTPUT_DIR / "intermediate",
        OUTPUT_DIR / "paper",
        OUTPUT_DIR / "studies",
    ]:
        if path.exists():
            import shutil

            shutil.rmtree(path)


def ensure_output_dirs() -> None:
    for path in [PROCESSED_DIR, BENCHMARK_INTERMEDIATE_STUDY_DIR, TASK_INTERMEDIATE_STUDY_DIR, KEY_TABLE_DIR, KEY_FIGURE_DIR, KEY_REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_step_outputs(steps: set[str]) -> None:
    clean_legacy_output_dirs()
    if "impute" in steps:
        clean_dir(PROCESSED_DIR)
    elif "intermediate" in steps:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name in INTERMEDIATE_TABLES:
            path = PROCESSED_DIR / f"{name}.csv"
            if path.exists():
                path.unlink()
    if "studies" in steps:
        for path in [BENCHMARK_INTERMEDIATE_STUDY_DIR, TASK_INTERMEDIATE_STUDY_DIR, KEY_TABLE_DIR, KEY_FIGURE_DIR, KEY_REPORT_DIR]:
            clean_dir(path)
    ensure_output_dirs()


def read_raw_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_benchmark = read_matrix(RAW_DIR / "benchmark_level_matrix.csv")
    raw_task = read_matrix(RAW_DIR / "task_level_matrix.csv")
    if not raw_benchmark[KEY_COLUMNS].equals(raw_task[KEY_COLUMNS]):
        raise ValueError("Benchmark and task matrices do not have identical agent/model rows.")
    bench_cols = [c for c in score_columns(raw_benchmark) if c in PAPER_BENCHMARKS]
    task_cols = [c for c in score_columns(raw_task) if c.split("/", 1)[0] in PAPER_BENCHMARKS]
    return raw_benchmark[KEY_COLUMNS + bench_cols], raw_task[KEY_COLUMNS + task_cols]


def _build_result(df: pd.DataFrame) -> ImputationResult:
    cols = score_columns(df)
    values = df[cols].astype(float)
    stats = robust_column_stats(values)
    normalized = pd.concat([df[KEY_COLUMNS], normalize(values, stats)], axis=1)
    cv = pd.DataFrame({"method": ["raw"], "rank": [0], "holdout_cells": [0], "rmse": [0.0], "mae": [0.0]})
    return ImputationResult(
        normalized=normalized,
        raw=df.copy(),
        stats=stats,
        cv=cv,
        best_rank=0,
        missing_fraction=float(values.isna().mean().mean()),
    )


def _load_result(prefix: str) -> ImputationResult:
    stats_path = PROCESSED_DIR / f"{prefix}_column_quality.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}; run `habor-analyze impute` first.")
    return ImputationResult(
        normalized=pd.read_csv(PROCESSED_DIR / f"{prefix}_normalized_matrix.csv"),
        raw=pd.read_csv(PROCESSED_DIR / f"{prefix}_raw_matrix.csv"),
        stats=pd.read_csv(stats_path),
        cv=pd.read_csv(PROCESSED_DIR / f"{prefix}_cv.csv"),
        best_rank=0,
        missing_fraction=0.0,
    )


def load_imputation_results():
    return _load_result("benchmark"), _load_result("task")


def load_intermediate_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for name in INTERMEDIATE_TABLES:
        path = PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run `habor-analyze intermediate` first.")
        tables[name] = pd.read_csv(path)
    return tables


def _write_result(prefix: str, result: ImputationResult) -> None:
    write_csv(result.raw, PROCESSED_DIR / f"{prefix}_raw_matrix.csv")
    write_csv(result.normalized, PROCESSED_DIR / f"{prefix}_normalized_matrix.csv")
    write_csv(result.stats, PROCESSED_DIR / f"{prefix}_column_quality.csv")
    write_csv(result.cv, PROCESSED_DIR / f"{prefix}_cv.csv")


def run_imputation_step() -> None:
    log("impute: reading raw benchmark and task matrices (filtered to PAPER_BENCHMARKS)")
    ensure_output_dirs()
    raw_benchmark, raw_task = read_raw_matrices()
    log(f"impute: {len(score_columns(raw_benchmark))} benchmarks, {len(score_columns(raw_task))} tasks")
    benchmark_result = _build_result(raw_benchmark)
    task_result = _build_result(raw_task)
    _write_result("benchmark", benchmark_result)
    _write_result("task", task_result)
    log("impute: wrote processed benchmark and task matrices (raw + normalized)")


def run_intermediate_step() -> dict[str, pd.DataFrame]:
    log("intermediate: loading processed matrices")
    ensure_output_dirs()
    raw_benchmark, raw_task = read_raw_matrices()
    benchmark_result, task_result = load_imputation_results()
    log("intermediate: building shared benchmark/task tables")
    return write_tables(raw_benchmark, benchmark_result, task_result)


def build_study_tables(
    raw_benchmark: pd.DataFrame,
    benchmark_result,
    task_result,
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    filter_table = benchmark_filter_table(benchmark_result.stats)
    included_benchmarks = filter_table[filter_table["include_in_key_analysis"]]["benchmark"].tolist()
    long_df = tables["benchmark_observed_imputed_long"]

    log("studies: building benchmark leaderboards and provenance")
    agent_model_scores = agent_model_scores_for_cols(raw_benchmark, benchmark_result, included_benchmarks)
    score_long = benchmark_scores_long(raw_benchmark, benchmark_result, included_benchmarks)
    provenance = analysis_data_provenance()
    imputation_diagnostics = imputation_diagnostics_summary(benchmark_result, task_result)
    log("studies: estimating model vs agent roles")
    model_effects = adjusted_group_effects(
        long_df[long_df["benchmark"].isin(included_benchmarks)].copy(), "model", ["agent", "benchmark"]
    )
    agent_effects = adjusted_group_effects(
        long_df[long_df["benchmark"].isin(included_benchmarks)].copy(), "agent", ["model", "benchmark"]
    )
    log("studies: estimating benchmark predictability and similarity")
    corr_filtered, corr_pairs_filtered = pairwise_correlations(benchmark_result.normalized, included_benchmarks)
    uniqueness = predictability_for_cols(benchmark_result.normalized, included_benchmarks)
    family_detail = within_family_model_vs_agent(raw_benchmark, included_benchmarks)
    family_summary = within_family_summary(family_detail)
    headroom = benchmark_headroom_by_domain(raw_benchmark, included_benchmarks)
    benchmark_clusters, benchmark_corr_ordered, _benchmark_cluster_order = benchmark_similarity_clusters(corr_filtered)
    pca_loadings, pca_agent_model_scores, pca_explained = pca_for_cols(benchmark_result.normalized, included_benchmarks)
    log("studies: domain-aware correlation decomposition and effective dimensionality")
    domain_enriched_pairs = domain_correlation_decomposition(corr_filtered, corr_pairs_filtered)
    domain_corr_summary = domain_correlation_summary(domain_enriched_pairs)
    full_explained = full_pca_explained_variance(benchmark_result.normalized, included_benchmarks)
    dim_stats = effective_dimensionality(full_explained, n_benchmarks=len(included_benchmarks))
    dim_stats_df = pd.DataFrame([dim_stats])
    full_explained_df = pd.DataFrame({"component": np.arange(1, len(full_explained) + 1), "explained_variance_ratio": full_explained})
    greedy_selection = greedy_benchmark_selection(corr_filtered)
    log("studies: estimating Terminus harness deltas")
    agent_lift_summary, agent_lift_by_benchmark = summarize_agent_lift(
        tables["agent_differential_by_benchmark"], included_benchmarks
    )
    terminus_by_model = terminus_delta_by_model(tables["agent_differential_by_benchmark"], included_benchmarks)

    log("studies: building task reliability tables")
    tasks_enriched, task_summary, frontier_tasks = task_reliability_tables(tables["task_item_stats"])
    alignment = task_aggregate_alignment(benchmark_result, task_result, tasks_enriched, included_benchmarks)
    log("studies: estimating task similarity, predictability, and representatives")
    task_within_similarity, representative_tasks, task_predictability, task_cross_similarity = task_similarity_and_representatives(
        task_result, tasks_enriched, benchmark_clusters
    )
    log("studies: within-benchmark BenchPress-style task holdout predictability")
    task_holdout_ranked, task_holdout_methods = task_predictability_holdout(task_result, tasks_enriched)
    log("studies: within-benchmark greedy task selection")
    task_greedy_sel, task_greedy_sel_summary = greedy_task_selection_within_benchmark(task_result, tasks_enriched)
    log("studies: building go-to task set")
    goto_task_set = build_goto_task_set(representative_tasks, task_greedy_sel)
    log("studies: cross-benchmark global go-to task selection")
    global_task_rep, global_task_greedy = global_task_selection(task_result, tasks_enriched)
    log("studies: unified within/cross/global BenchPress-style task holdout")
    unified_holdout = unified_task_holdout(task_result, tasks_enriched)

    # ── Task-level SVD spectrum (logit space, reliable bounded tasks) ──
    reliable_task_cols = tasks_enriched.loc[
        tasks_enriched["is_bounded_score_task"]
        & tasks_enriched["is_reliable_observed_task"]
        & (tasks_enriched["observed_std"] >= 0.05),
        "task_column",
    ].tolist()
    reliable_task_cols = [c for c in reliable_task_cols if c in task_result.raw.columns]
    X_task_raw = task_result.raw[reliable_task_cols].astype(float).fillna(0).to_numpy()
    eps_t = 0.005
    X_task_smooth = np.clip(X_task_raw, eps_t, 1 - eps_t)
    X_task_logit = np.log(X_task_smooth / (1 - X_task_smooth))
    from sklearn.decomposition import PCA as _PCA
    _pca_task = _PCA()
    _pca_task.fit(X_task_logit)
    task_svd_spectrum = pd.DataFrame({
        "component": np.arange(1, len(_pca_task.explained_variance_ratio_) + 1),
        "explained_variance_ratio": _pca_task.explained_variance_ratio_,
    })
    task_svd_spectrum.attrs["n_tasks"] = len(reliable_task_cols)
    log("studies: selecting final HaborMix task set")
    selected_tasks, scored_task_pool = select_harbormix_tasks(tasks_enriched, representative_tasks, task_predictability)
    selected_task_summary = (
        selected_tasks.groupby(["benchmark", "difficulty_tier"])
        .agg(
            selected_tasks=("task_column", "size"),
            mean_selection_score=("mix_selection_score", "mean"),
            mean_task_score=("imputed_mean", "mean"),
            mean_representative_signal=("representative_signal", "mean"),
            mean_unique_unpredictable_signal=("unique_unpredictable_signal", "mean"),
            mean_difficulty_signal=("difficulty_signal", "mean"),
            mean_strength_correlation=("strength_correlation", "mean"),
        )
        .reset_index()
        .sort_values(["selected_tasks", "mean_selection_score"], ascending=False)
    )
    study_tables = {
        "benchmark_filtering": filter_table,
        "analysis_data_provenance": provenance,
        "imputation_diagnostics_summary": imputation_diagnostics,
        "benchmark_agent_model_scores": agent_model_scores,
        "benchmark_scores_long": score_long,
        "benchmark_model_adjusted_effects": model_effects,
        "benchmark_agent_adjusted_effects": agent_effects,
        "benchmark_correlation_filtered": corr_filtered,
        "benchmark_correlation_clustered": benchmark_corr_ordered,
        "benchmark_similarity_clusters": benchmark_clusters,
        "benchmark_redundancy_pairs_filtered": corr_pairs_filtered,
        "benchmark_domain_enriched_pairs": domain_enriched_pairs,
        "benchmark_domain_correlation_summary": domain_corr_summary,
        "benchmark_effective_dimensionality": dim_stats_df,
        "benchmark_pca_full_explained_variance": full_explained_df,
        "benchmark_greedy_selection": greedy_selection,
        "benchmark_uniqueness_filtered": uniqueness,
        "benchmark_within_family_model_vs_agent": family_detail,
        "benchmark_within_family_summary": family_summary,
        "benchmark_headroom_by_domain": headroom,
        "benchmark_latent_loadings_filtered": pca_loadings,
        "benchmark_latent_agent_model_scores_filtered": pca_agent_model_scores,
        "benchmark_latent_explained_variance_filtered": pca_explained,
        "benchmark_agent_lift_vs_terminus": agent_lift_summary,
        "benchmark_agent_lift_by_benchmark": agent_lift_by_benchmark,
        "terminus_delta_by_model": terminus_by_model,
        "task_enriched_item_stats": tasks_enriched,
        "task_benchmark_reliable_summary": task_summary,
        "task_to_benchmark_alignment": alignment,
        "task_within_benchmark_similarity": task_within_similarity,
        "task_cross_benchmark_similarity": task_cross_similarity,
        "task_representative_tasks": representative_tasks,
        "task_predictability_ranked": task_predictability,
        "task_holdout_predictability": task_holdout_ranked,
        "task_holdout_method_comparison": task_holdout_methods,
        "task_svd_spectrum": task_svd_spectrum,
        "task_greedy_selection": task_greedy_sel,
        "task_greedy_selection_summary": task_greedy_sel_summary,
        "task_goto_set": goto_task_set,
        "task_global_representatives": global_task_rep,
        "task_global_greedy_selection": global_task_greedy,
        "task_holdout_unified_comparison": unified_holdout,
        "harbormix_selected_tasks": selected_tasks,
        "harbormix_scored_task_pool": scored_task_pool,
        "harbormix_selection_by_benchmark": selected_task_summary,
        "task_frontier_or_saturated_watchlist": frontier_tasks,
    }
    return study_tables, included_benchmarks


def write_study_tables(study_tables: dict[str, pd.DataFrame]) -> None:
    for name, table in study_tables.items():
        directory = BENCHMARK_INTERMEDIATE_STUDY_DIR if name.startswith("benchmark") or name.startswith("analysis") else TASK_INTERMEDIATE_STUDY_DIR
        write_csv(table, directory / f"{name}.csv")
    for name in KEY_ANALYSIS_TABLES:
        subdir = KEY_TABLE_SUBDIRS.get(name, "benchmark_level" if name.startswith("benchmark") else "task_level")
        write_csv(study_tables[name], KEY_TABLE_DIR / subdir / f"{name}.csv")


def write_study_figures(study_tables: dict[str, pd.DataFrame], raw_benchmark: pd.DataFrame) -> None:
    log("figures: writing paper figures")
    set_plot_style()
    save_within_family_summary_plot(study_tables["benchmark_within_family_summary"])
    save_within_family_detail_plot(study_tables["benchmark_within_family_model_vs_agent"])
    save_agent_lift_heatmap(study_tables["benchmark_agent_lift_by_benchmark"])
    save_benchmark_headroom_plot(study_tables["benchmark_headroom_by_domain"])
    save_benchmark_cluster_heatmap(study_tables["benchmark_correlation_clustered"])
    save_benchmark_uniqueness_plot(study_tables["benchmark_uniqueness_filtered"], study_tables["benchmark_filtering"])
    save_effective_dimensionality_plot(
        study_tables["benchmark_pca_full_explained_variance"]["explained_variance_ratio"].to_numpy(),
        study_tables["benchmark_effective_dimensionality"].iloc[0].to_dict(),
    )
    save_greedy_selection_plot(study_tables["benchmark_greedy_selection"])
    save_svd_spectrum_plot(raw_benchmark)
    save_task_similarity_heatmap(
        study_tables["task_cross_benchmark_similarity"], study_tables["benchmark_similarity_clusters"]
    )
    save_representative_task_plot(study_tables["task_representative_tasks"])


def run_studies_step() -> None:
    log("studies: loading processed and intermediate tables")
    ensure_output_dirs()
    raw_benchmark, _raw_task = read_raw_matrices()
    benchmark_result, task_result = load_imputation_results()
    tables = load_intermediate_tables()
    set_plot_style()
    study_tables, included_benchmarks = build_study_tables(
        raw_benchmark, benchmark_result, task_result, tables,
    )
    write_study_tables(study_tables)
    write_study_figures(study_tables, raw_benchmark)
    write_key_analysis_reports(study_tables, benchmark_result, task_result, included_benchmarks)
    write_paper_stats(study_tables, included_benchmarks, raw_benchmark)
    write_appendix_stats(study_tables, benchmark_result, included_benchmarks)
    write_appendix_model_agent(study_tables)
    copy_paper_figures()
    log("studies: wrote key analysis tables, figures, reports, paper/tex/ stats, and paper/figs/")


def expand_steps(steps: list[str]) -> list[str]:
    if not steps or "all" in steps:
        return ["impute", "intermediate", "studies"]
    order = ["impute", "intermediate", "studies"]
    selected = set(steps)
    return [step for step in order if step in selected]


def run_pipeline(steps: list[str] | None = None, clean: bool = False) -> None:
    ordered_steps = expand_steps(steps or ["all"])
    log(f"run: steps={ordered_steps}, clean={clean}")
    if clean:
        clean_step_outputs(set(ordered_steps))
    else:
        clean_legacy_output_dirs()
        ensure_output_dirs()
    for step in ordered_steps:
        if step == "impute":
            run_imputation_step()
        elif step == "intermediate":
            run_intermediate_step()
        elif step == "studies":
            run_studies_step()
        else:
            raise ValueError(f"Unknown pipeline step: {step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HaborMix analysis pipeline in reusable steps.")
    parser.add_argument(
        "steps",
        nargs="*",
        choices=["all", "impute", "intermediate", "studies"],
        help="Steps to run in dependency order. Default: all. Example: habor-analyze impute intermediate studies",
    )
    parser.add_argument("--clean", action="store_true", help="Clean generated outputs for the selected steps before running.")
    args = parser.parse_args()
    run_pipeline(args.steps or ["all"], clean=args.clean)


if __name__ == "__main__":
    main()
