from __future__ import annotations

from ..core import *


def analysis_data_provenance() -> pd.DataFrame:
    rows = [
        {
            "analysis": "coverage filtering",
            "primary_matrix": "raw benchmark matrix column stats",
            "processed_output": "data/processed/intermediate/benchmark_column_quality.csv",
        },
        {
            "analysis": "agent+model aggregate leaderboard",
            "primary_matrix": "raw benchmark scores",
            "processed_output": "data/processed/intermediate/benchmark_raw_matrix.csv",
        },
        {
            "analysis": "per-benchmark mini-leaderboards",
            "primary_matrix": "raw benchmark scores",
            "processed_output": "data/processed/intermediate/benchmark_raw_matrix.csv",
        },
        {
            "analysis": "model vs agent roles",
            "primary_matrix": "benchmark-relative normalized scores",
            "processed_output": "data/processed/intermediate/benchmark_normalized_matrix.csv",
        },
        {
            "analysis": "benchmark predictability and similarity",
            "primary_matrix": "benchmark-relative normalized scores",
            "processed_output": "data/processed/intermediate/benchmark_normalized_matrix.csv",
        },
        {
            "analysis": "terminus harness deltas",
            "primary_matrix": "benchmark-relative normalized scores",
            "processed_output": "data/processed/intermediate/benchmark_normalized_matrix.csv",
        },
        {
            "analysis": "task similarity and representatives",
            "primary_matrix": "raw task scores (filtered to reliable bounded tasks)",
            "processed_output": "data/processed/intermediate/task_raw_matrix.csv",
        },
        {
            "analysis": "HaborMix candidate selection",
            "primary_matrix": "task item statistics from raw scores",
            "processed_output": "data/processed/intermediate/task_item_stats.csv",
        },
    ]
    return pd.DataFrame(rows)


def imputation_diagnostics_summary(
    benchmark_result: ImputationResult,
    task_result: ImputationResult,
) -> pd.DataFrame:
    rows = []
    for name, result in [("benchmark", benchmark_result), ("task", task_result)]:
        rows.append(
            {
                "matrix": name,
                "agent_model_rows": int(result.raw.shape[0]),
                "score_columns": int(result.raw.shape[1] - len(KEY_COLUMNS)),
                "preprocessing_method": "raw (no imputation)",
                "missing_fraction_before_processing": result.missing_fraction,
                "selected_imputation_method": "none",
                "selected_imputation_rank": 0,
                "task_imputation_method_used_for_benchmark_aggregation": np.nan,
                "task_imputation_rank_used_for_benchmark_aggregation": np.nan,
                "holdout_cells": 0,
                "holdout_rmse_scaled_score_space": np.nan,
                "holdout_mae_scaled_score_space": np.nan,
                "mae_gap_to_second_best_method": np.nan,
                "interpretation": f"Raw {name} scores used directly (data is fully observed for PAPER_BENCHMARKS).",
            }
        )
    return pd.DataFrame(rows)
