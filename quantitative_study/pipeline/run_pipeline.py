#!/usr/bin/env python3
"""
Full quantitative analysis pipeline: Harbor vs original benchmark scores.

Usage (from habor-mix-analyzer/):
  uv run python -m quantitative_study.pipeline.run_pipeline
  uv run python -m quantitative_study.pipeline.run_pipeline --harbor-csv benchmark_level_matrix.csv
  uv run python -m quantitative_study.pipeline.run_pipeline --no-figures   # tables only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import FIGURE_DIR, OUTPUT_DIR
from .loading import load_all_docs, load_harbor_matrix, load_metric_alignment
from .analysis import (
    agent_effect_by_agent,
    agent_effect_by_status,
    agent_lift_table,
    anomaly_summary_by_benchmark,
    benchmark_direction,
    build_comparison_long,
    build_comparison_summary,
    detect_anomalies,
    rank_correlation_by_benchmark,
)
from .cross_analysis import (
    benchmark_difficulty_table,
    benchmark_launch_vs_harbor_improvement,
    domain_launch_vs_harbor_improvement,
    domain_progress_summary,
    domain_summary,
    per_model_domain_scores,
    progress_over_time,
    progress_summary,
    superdomain_summary,
)
from .figures import generate_all_figures, generate_cross_figures


def _write(df: pd.DataFrame, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print(f"  [csv] {path}  ({len(df)} rows)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Harbor vs Original benchmark analysis pipeline")
    parser.add_argument("--harbor-csv", type=Path, default=None)
    parser.add_argument("--benchmark-info-dir", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    # ---- Load ----
    print("=" * 70)
    print("Loading data...")
    harbor = load_harbor_matrix(args.harbor_csv)
    alignments = load_metric_alignment()
    docs = load_all_docs(args.benchmark_info_dir, alignments)
    n_filled = sum(1 for d in docs.values() if d.rows)
    print(f"  Harbor matrix: {harbor.shape[0]} rows × {harbor.shape[1]} cols")
    print(f"  Metric alignments: {len(alignments)} rows, "
          f"{sum(a.include_in_comparison for a in alignments.values())} included")
    print(f"  Doc JSONs: {len(docs)} total, {n_filled} with results_over_time")

    # ---- A. Compare ----
    print("\n" + "=" * 70)
    print("A. Harbor vs Doc comparison...")
    long_df = build_comparison_long(harbor, docs, alignments)
    summary = build_comparison_summary(long_df)
    _write(long_df, "harbor_vs_doc_long.csv")
    _write(summary, "harbor_vs_doc_summary.csv")

    counts = long_df["match_status"].value_counts()
    print("  match_status breakdown:")
    for status, n in counts.items():
        print(f"    {status:42s} {n:5d}")
    print(f"  Rows with delta: {summary.shape[0]}")

    # ---- B. Anomalies ----
    print("\n" + "=" * 70)
    print("B. Harbor data anomaly scan...")
    anomalies = detect_anomalies(harbor)
    anom_summary = anomaly_summary_by_benchmark(anomalies)
    _write(anomalies, "harbor_anomalies.csv")
    _write(anom_summary, "harbor_anomaly_summary.csv")

    if not anom_summary.empty:
        print("  Benchmarks with anomalous cells:")
        for _, row in anom_summary.iterrows():
            print(f"    {row['benchmark']:25s}  {int(row['n_anomalous']):3d} cells  [{row['issues']}]")

    # ---- C. Agent analysis ----
    print("\n" + "=" * 70)
    print("C. Agent capability analysis...")
    eff_status = agent_effect_by_status(summary)
    eff_agent = agent_effect_by_agent(summary)
    direction = benchmark_direction(summary)
    lift = agent_lift_table(summary)
    rank_corr = rank_correlation_by_benchmark(summary)

    _write(eff_status, "agent_effect_by_match_status.csv")
    _write(eff_agent, "agent_effect_by_agent.csv")
    _write(direction, "benchmark_direction.csv")
    _write(lift, "agent_lift_table.csv")
    _write(rank_corr, "rank_correlation_by_benchmark.csv")

    if not eff_agent.empty:
        print("  Agent systematic bias:")
        for _, row in eff_agent.iterrows():
            print(f"    {row['harbor_agent']:15s}  n={int(row['n']):3d}  mean_delta={row['mean_delta']:+.4f}")

    if not direction.empty:
        print("  Benchmark direction (>= 2 matched rows):")
        for _, row in direction[direction["n"] >= 2].iterrows():
            print(f"    {row['benchmark']:22s}  n={int(row['n']):2d}  mean_delta={row['mean_delta']:+.4f}  [{row['direction']}]")

    # ---- D. Cross-analysis (taxonomy, difficulty, progress) ----
    print("\n" + "=" * 70)
    print("D. Cross-analysis: taxonomy, difficulty, progress...")

    difficulty = benchmark_difficulty_table(harbor)
    dom_sum = domain_summary(harbor)
    superdom_sum = superdomain_summary(harbor)
    model_dom = per_model_domain_scores(harbor)
    prog = progress_over_time(args.benchmark_info_dir)
    prog_sum = progress_summary(prog)
    dom_prog = domain_progress_summary(prog_sum)
    launch_imp = benchmark_launch_vs_harbor_improvement(harbor, args.benchmark_info_dir)
    domain_launch_imp = domain_launch_vs_harbor_improvement(launch_imp)

    _write(difficulty, "benchmark_difficulty.csv")
    _write(dom_sum, "domain_summary.csv")
    _write(superdom_sum, "superdomain_summary.csv")
    _write(model_dom, "model_domain_scores.csv")
    _write(launch_imp, "benchmark_launch_vs_harbor_improvement.csv")
    _write(domain_launch_imp, "domain_launch_vs_harbor_improvement.csv")
    if not prog.empty:
        _write(prog, "progress_over_time.csv")
        _write(prog_sum, "progress_summary.csv")
        _write(dom_prog, "domain_progress.csv")

    if not dom_sum.empty:
        print("  Domain score summary:")
        for _, row in dom_sum.iterrows():
            print(f"    {row['domain']:28s}  n={int(row['n_benchmarks']):2d}  "
                  f"mean_best={row['mean_best_score']:.3f}  mean_all={row['mean_all_scores']:.3f}")

    if not superdom_sum.empty:
        print("  Superdomain summary:")
        for _, row in superdom_sum.iterrows():
            print(f"    {row['superdomain']:12s}  n={int(row['n_benchmarks']):2d}  "
                  f"mean_best={row['mean_best_score']:.3f}  mean_all={row['mean_all_scores']:.3f}")

    if not difficulty.empty:
        print("  Hardest benchmarks (lowest best-model score):")
        for _, row in difficulty.head(10).iterrows():
            print(f"    {row['benchmark']:25s}  best={row['best_score']:.3f}  "
                  f"by {row['best_model']}+{row['best_agent']}  [{row['domain']}]")

    if not dom_prog.empty:
        print("  Progress by domain (doc temporal data):")
        for _, row in dom_prog.iterrows():
            print(f"    {row['domain']:28s}  n={int(row['n_benchmarks']):2d}  "
                  f"mean_progress={row['mean_absolute_progress']:+.3f}")

    if not launch_imp.empty:
        ok_launch_imp = launch_imp[launch_imp["status"] == "ok"]
        print("  Launch baseline vs Harbor max (top relative improvements):")
        for _, row in ok_launch_imp.head(10).iterrows():
            print(f"    {row['benchmark']:25s}  "
                  f"{row['launch_best_score']:.3f} → {row['harbor_best_score']:.3f}  "
                  f"rel={row['relative_improvement_pct']:+.1f}%")

    if not domain_launch_imp.empty:
        print("  Launch baseline vs Harbor max by domain:")
        for _, row in domain_launch_imp.iterrows():
            print(f"    {row['domain']:28s}  n={int(row['n_benchmarks']):2d}  "
                  f"median_rel={row['median_relative_improvement_pct']:+.1f}%  "
                  f"mean_rel={row['mean_relative_improvement_pct']:+.1f}%")

    # ---- E. Figures ----
    if not args.no_figures:
        print("\n" + "=" * 70)
        print("E. Generating paper figures...")
        progress_for_figures = progress_over_time(
            args.benchmark_info_dir,
            min_snapshots=1,
            frontier_only=True,
        )
        generate_all_figures(
            summary=summary,
            direction=direction,
            lift=lift,
        )
        generate_cross_figures(
            difficulty=difficulty,
            domain_df=dom_sum,
            superdomain_df=superdom_sum,
            progress_df=progress_for_figures,
            domain_prog_df=dom_prog,
            benchmark_launch_imp_df=launch_imp,
            domain_launch_imp_df=domain_launch_imp,
            harbor_df=harbor,
        )

    # ---- Done ----
    print("\n" + "=" * 70)
    print(f"Pipeline complete. All outputs in {OUTPUT_DIR}/")
    if not args.no_figures:
        print(f"Figures in {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
