from __future__ import annotations

from ..core import *
from .intermediate_tables import corr_or_nan


# ─── Within-benchmark BenchPress-style task predictability ────────────────────


def task_predictability_holdout(
    task_result: ImputationResult,
    tasks_enriched: pd.DataFrame,
    n_folds: int = 3,
    min_tasks: int = 5,
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-benchmark BenchPress-style task predictability via holdout.

    For each benchmark with >= min_tasks reliable tasks, hides 50 % of each
    system's task scores per fold, predicts the hidden cells using three
    methods (task-mean baseline, top-k peer weighted, within-benchmark SVD),
    blends peer + SVD (0.6/0.4), and reports per-task MedAE.

    Returns (per_task_predictability, per_benchmark_method_comparison).
    """
    reliable = tasks_enriched[
        tasks_enriched["is_bounded_score_task"]
        & tasks_enriched["is_reliable_observed_task"]
        & (tasks_enriched["observed_std"] >= 0.05)
    ]

    rng = np.random.RandomState(RANDOM_SEED)
    per_task_rows: list[dict] = []
    method_rows: list[dict] = []

    for benchmark, group in reliable.groupby("benchmark"):
        task_cols = [c for c in group["task_column"] if c in task_result.raw.columns]
        if len(task_cols) < min_tasks:
            continue

        M = task_result.raw[task_cols].astype(float).to_numpy()
        n_sys, n_tasks = M.shape

        baseline_ae: dict[int, list[float]] = {j: [] for j in range(n_tasks)}
        peer_ae: dict[int, list[float]] = {j: [] for j in range(n_tasks)}
        svd_ae: dict[int, list[float]] = {j: [] for j in range(n_tasks)}
        blend_ae: dict[int, list[float]] = {j: [] for j in range(n_tasks)}

        for _fold in range(n_folds):
            M_train = M.copy()
            holdout = np.zeros((n_sys, n_tasks), dtype=bool)

            for i in range(n_sys):
                n_hide = max(1, n_tasks // 2)
                hide_idx = rng.choice(n_tasks, size=n_hide, replace=False)
                M_train[i, hide_idx] = np.nan
                holdout[i, hide_idx] = True

            col_means = np.nanmean(M_train, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)

            M_filled = M_train.copy()
            for j in range(n_tasks):
                M_filled[np.isnan(M_filled[:, j]), j] = col_means[j]

            # ── Peer prediction (top-k correlated tasks, weighted avg) ──
            corr = np.corrcoef(M_filled.T)
            np.fill_diagonal(corr, 0)
            corr = np.nan_to_num(corr)
            k = min(top_k, n_tasks - 1)

            M_peer = np.full_like(M, np.nan)
            for j in range(n_tasks):
                abs_c = np.abs(corr[j])
                top_idx = np.argsort(abs_c)[-k:]
                weights = abs_c[top_idx]
                w_sum = weights.sum()
                if w_sum < 1e-8:
                    M_peer[:, j] = col_means[j]
                else:
                    for i in range(n_sys):
                        if holdout[i, j]:
                            M_peer[i, j] = np.average(M_filled[i, top_idx], weights=weights)

            # ── Within-benchmark SVD (rank-2 soft-impute) ──
            rank = min(2, n_sys - 1, n_tasks - 1)
            M_svd = M_filled.copy()
            for _ in range(50):
                try:
                    U, s, Vt = np.linalg.svd(M_svd, full_matrices=False)
                except np.linalg.LinAlgError:
                    break
                M_approx = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
                M_svd_new = np.where(np.isnan(M_train), M_approx, M_train)
                if np.allclose(M_svd, M_svd_new, atol=1e-6):
                    break
                M_svd = M_svd_new

            # ── Collect errors ──
            alpha = 0.6
            for j in range(n_tasks):
                for i in range(n_sys):
                    if not holdout[i, j]:
                        continue
                    actual = M[i, j]
                    baseline_ae[j].append(abs(col_means[j] - actual))

                    p = M_peer[i, j] if np.isfinite(M_peer[i, j]) else col_means[j]
                    s = M_svd[i, j] if np.isfinite(M_svd[i, j]) else col_means[j]
                    peer_ae[j].append(abs(p - actual))
                    svd_ae[j].append(abs(s - actual))
                    blend_ae[j].append(abs(alpha * p + (1 - alpha) * s - actual))

        # ── Per-task aggregation ──
        all_baseline: list[float] = []
        all_peer: list[float] = []
        all_svd: list[float] = []
        all_blend: list[float] = []
        for j, task_col in enumerate(task_cols):
            task_row = reliable[reliable["task_column"] == task_col].iloc[0]
            b_med = float(np.median(baseline_ae[j])) if baseline_ae[j] else float("nan")
            p_med = float(np.median(peer_ae[j])) if peer_ae[j] else float("nan")
            s_med = float(np.median(svd_ae[j])) if svd_ae[j] else float("nan")
            bl_med = float(np.median(blend_ae[j])) if blend_ae[j] else float("nan")
            improvement = (
                (b_med - bl_med) / b_med * 100
                if b_med > 1e-8 and np.isfinite(bl_med)
                else float("nan")
            )

            per_task_rows.append({
                "benchmark": benchmark,
                "task_column": task_col,
                "task_id": task_row["task_id"],
                "holdout_baseline_medae": round(b_med, 6),
                "holdout_peer_medae": round(p_med, 6),
                "holdout_svd_medae": round(s_med, 6),
                "holdout_blend_medae": round(bl_med, 6),
                "holdout_improvement_pct": round(improvement, 1),
                "holdout_n_cells": len(baseline_ae[j]),
                "difficulty_tier": task_row["difficulty_tier"],
                "observed_mean": task_row["observed_mean"],
                "observed_std": task_row["observed_std"],
            })
            all_baseline.extend(baseline_ae[j])
            all_peer.extend(peer_ae[j])
            all_svd.extend(svd_ae[j])
            all_blend.extend(blend_ae[j])

        # ── Per-benchmark method comparison ──
        bm_baseline = float(np.median(all_baseline)) if all_baseline else float("nan")
        bm_blend = float(np.median(all_blend)) if all_blend else float("nan")
        method_rows.append({
            "benchmark": benchmark,
            "n_tasks": len(task_cols),
            "baseline_medae": round(bm_baseline, 6),
            "peer_medae": round(float(np.median(all_peer)), 6) if all_peer else float("nan"),
            "svd_medae": round(float(np.median(all_svd)), 6) if all_svd else float("nan"),
            "blend_medae": round(bm_blend, 6),
            "blend_improvement_pct": round(
                (bm_baseline - bm_blend) / bm_baseline * 100, 1
            ) if bm_baseline > 1e-8 and np.isfinite(bm_blend) else float("nan"),
        })

    per_task = pd.DataFrame(per_task_rows)
    if not per_task.empty:
        per_task = per_task.sort_values("holdout_blend_medae", ascending=False)
    methods = pd.DataFrame(method_rows)
    if not methods.empty:
        methods = methods.sort_values("blend_medae", ascending=False)

    return per_task, methods


def unified_task_holdout(
    task_result: ImputationResult,
    tasks_enriched: pd.DataFrame,
    n_folds: int = 3,
    top_k: int = 5,
) -> pd.DataFrame:
    """Unified holdout comparing within-only / cross-only / global peers.

    Uses the **same holdout mask** and **same k** for all three peer
    scopes, so the comparison is apples-to-apples.  For each held-out
    cell, predicts via:

    * task-mean baseline (column mean)
    * within-benchmark peers only (top-k from same benchmark)
    * cross-benchmark peers only (top-k from OTHER benchmarks)
    * global peers (top-k from all benchmarks)
    * global SVD rank-2 (soft-impute)
    * three blends: within+SVD, cross+SVD, global+SVD (0.6 peer + 0.4 SVD)

    Returns a per-benchmark comparison DataFrame.
    """
    reliable = tasks_enriched[
        tasks_enriched["is_bounded_score_task"]
        & tasks_enriched["is_reliable_observed_task"]
        & (tasks_enriched["observed_std"] >= 0.05)
    ]
    task_cols = [c for c in reliable["task_column"] if c in task_result.raw.columns]
    n_all = len(task_cols)
    if n_all < 10:
        return pd.DataFrame()

    M = task_result.raw[task_cols].astype(float).to_numpy()
    n_sys = M.shape[0]
    task_meta = reliable.set_index("task_column")
    task_benchmarks = np.array([task_meta.loc[c, "benchmark"] for c in task_cols])

    # Pre-build same-benchmark masks for each task
    bench_set: dict[str, np.ndarray] = {}
    for bench in np.unique(task_benchmarks):
        bench_set[bench] = np.where(task_benchmarks == bench)[0]

    rng = np.random.RandomState(RANDOM_SEED)

    # Per-task error accumulators for each method
    keys = ["baseline", "within_peer", "cross_peer", "global_peer",
            "svd", "within_blend", "cross_blend", "global_blend"]
    ae: dict[str, dict[int, list]] = {k: {j: [] for j in range(n_all)} for k in keys}

    for _fold in range(n_folds):
        M_train = M.copy()
        holdout = np.zeros((n_sys, n_all), dtype=bool)
        for i in range(n_sys):
            n_hide = max(1, n_all // 2)
            hide_idx = rng.choice(n_all, size=n_hide, replace=False)
            M_train[i, hide_idx] = np.nan
            holdout[i, hide_idx] = True

        col_means = np.nanmean(M_train, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        M_filled = M_train.copy()
        for j in range(n_all):
            M_filled[np.isnan(M_filled[:, j]), j] = col_means[j]

        # Global correlation matrix (once per fold)
        corr = np.corrcoef(M_filled.T)
        np.fill_diagonal(corr, 0)
        corr = np.nan_to_num(corr)
        abs_corr = np.abs(corr)

        # ── Peer predictions under three scopes ──
        M_within = np.full_like(M, np.nan)
        M_cross = np.full_like(M, np.nan)
        M_global = np.full_like(M, np.nan)

        for j in range(n_all):
            bench_j = task_benchmarks[j]
            same_mask = bench_set[bench_j]
            other_mask = np.where(task_benchmarks != bench_j)[0]

            # Within-benchmark peers (top-k from same benchmark, excluding self)
            same_others = same_mask[same_mask != j]
            if len(same_others) > 0:
                k_w = min(top_k, len(same_others))
                ac_within = abs_corr[j, same_others]
                top_w = same_others[np.argpartition(ac_within, -k_w)[-k_w:]]
                w_w = abs_corr[j, top_w]
                ws_w = w_w.sum()
                if ws_w > 1e-8:
                    pred_w = M_filled[:, top_w] @ (w_w / ws_w)
                    M_within[holdout[:, j], j] = pred_w[holdout[:, j]]
                else:
                    M_within[holdout[:, j], j] = col_means[j]
            else:
                M_within[holdout[:, j], j] = col_means[j]

            # Cross-benchmark peers (top-k from other benchmarks only)
            if len(other_mask) > 0:
                k_c = min(top_k, len(other_mask))
                ac_cross = abs_corr[j, other_mask]
                top_c = other_mask[np.argpartition(ac_cross, -k_c)[-k_c:]]
                w_c = abs_corr[j, top_c]
                ws_c = w_c.sum()
                if ws_c > 1e-8:
                    pred_c = M_filled[:, top_c] @ (w_c / ws_c)
                    M_cross[holdout[:, j], j] = pred_c[holdout[:, j]]
                else:
                    M_cross[holdout[:, j], j] = col_means[j]
            else:
                M_cross[holdout[:, j], j] = col_means[j]

            # Global peers (top-k from all benchmarks)
            k_g = min(top_k, n_all - 1)
            ac_global = abs_corr[j].copy()
            ac_global[j] = 0
            top_g = np.argpartition(ac_global, -k_g)[-k_g:]
            w_g = ac_global[top_g]
            ws_g = w_g.sum()
            if ws_g > 1e-8:
                pred_g = M_filled[:, top_g] @ (w_g / ws_g)
                M_global[holdout[:, j], j] = pred_g[holdout[:, j]]
            else:
                M_global[holdout[:, j], j] = col_means[j]

        # ── Global SVD rank-2 soft-impute ──
        rank = min(2, n_sys - 1, n_all - 1)
        M_svd = M_filled.copy()
        for _ in range(50):
            try:
                U, s, Vt = np.linalg.svd(M_svd, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            M_approx = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
            M_svd_new = np.where(np.isnan(M_train), M_approx, M_train)
            if np.allclose(M_svd, M_svd_new, atol=1e-6):
                break
            M_svd = M_svd_new

        # ── Collect errors ──
        alpha = 0.6
        for j in range(n_all):
            for i in range(n_sys):
                if not holdout[i, j]:
                    continue
                actual = M[i, j]
                bl = col_means[j]
                sv = M_svd[i, j] if np.isfinite(M_svd[i, j]) else bl
                pw = M_within[i, j] if np.isfinite(M_within[i, j]) else bl
                pc = M_cross[i, j] if np.isfinite(M_cross[i, j]) else bl
                pg = M_global[i, j] if np.isfinite(M_global[i, j]) else bl

                ae["baseline"][j].append(abs(bl - actual))
                ae["within_peer"][j].append(abs(pw - actual))
                ae["cross_peer"][j].append(abs(pc - actual))
                ae["global_peer"][j].append(abs(pg - actual))
                ae["svd"][j].append(abs(sv - actual))
                ae["within_blend"][j].append(abs(alpha * pw + (1 - alpha) * sv - actual))
                ae["cross_blend"][j].append(abs(alpha * pc + (1 - alpha) * sv - actual))
                ae["global_blend"][j].append(abs(alpha * pg + (1 - alpha) * sv - actual))

    # ── Per-benchmark aggregation ──
    col_to_idx = {c: i for i, c in enumerate(task_cols)}
    task_col_set = set(task_cols)
    rows: list[dict] = []
    for benchmark, group in reliable.groupby("benchmark"):
        bench_task_cols = [c for c in group["task_column"] if c in task_col_set]
        if not bench_task_cols:
            continue
        bench_idx = [col_to_idx[c] for c in bench_task_cols]
        row: dict = {"benchmark": benchmark, "n_tasks": len(bench_task_cols)}
        for method in keys:
            all_errors: list[float] = []
            for j in bench_idx:
                all_errors.extend(ae[method][j])
            row[f"{method}_medae"] = round(float(np.median(all_errors)), 6) if all_errors else float("nan")
        # Improvement percentages vs baseline
        bl_val = row["baseline_medae"]
        for method in ["within_blend", "cross_blend", "global_blend"]:
            val = row[f"{method}_medae"]
            row[f"{method}_vs_baseline_pct"] = round(
                (bl_val - val) / bl_val * 100, 1
            ) if bl_val > 1e-8 and np.isfinite(val) else float("nan")
        # Global vs within delta
        row["global_vs_within_delta"] = round(
            row["within_blend_medae"] - row["global_blend_medae"], 6
        )
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("global_vs_within_delta", ascending=False)
    return result


def greedy_task_selection_within_benchmark(
    task_result: ImputationResult,
    tasks_enriched: pd.DataFrame,
    min_tasks: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-benchmark greedy task selection to maximise independent information.

    For each benchmark with >= *min_tasks* reliable bounded tasks, compute
    pairwise Spearman correlation among task columns (using the normalized
    score matrix), then greedily pick tasks so that each new task has the
    lowest max |correlation| with the already-selected set -- mirroring
    ``greedy_benchmark_selection`` in ``benchmark_similarity.py``.

    Returns
    -------
    greedy_df : pd.DataFrame
        One row per (benchmark, step) with columns: step, benchmark,
        task_column, task_id, max_abs_corr_to_selected.
    greedy_summary : pd.DataFrame
        One row per benchmark with aggregate stats: n_tasks,
        n_tasks_below_0.7, frac_below_0.7, median_max_abs_corr,
        mean_max_abs_corr.
    """
    reliable = tasks_enriched[
        tasks_enriched["is_bounded_score_task"]
        & tasks_enriched["is_reliable_observed_task"]
        & (tasks_enriched["observed_std"] >= 0.05)
    ]

    greedy_rows: list[dict] = []
    summary_rows: list[dict] = []

    for benchmark, group in reliable.groupby("benchmark"):
        task_cols = [c for c in group["task_column"] if c in task_result.raw.columns]
        if len(task_cols) < min_tasks:
            continue

        # Pairwise Spearman correlation between tasks
        matrix = task_result.raw[task_cols].astype(float)
        corr = matrix.corr(method="spearman").to_numpy(copy=True)
        corr = np.nan_to_num(corr)
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0)
        n_tasks = len(task_cols)

        # Map task_column -> task_id for labelling
        task_id_map = group.set_index("task_column")["task_id"].to_dict()

        # First pick: task with lowest mean |correlation| to all others
        mean_abs = abs_corr.mean(axis=1)
        first = int(np.argmin(mean_abs))
        selected_idx = [first]
        remaining = set(range(n_tasks)) - {first}

        greedy_rows.append({
            "step": 1,
            "benchmark": benchmark,
            "task_column": task_cols[first],
            "task_id": task_id_map.get(task_cols[first], task_cols[first]),
            "max_abs_corr_to_selected": 0.0,
        })

        while remaining:
            best_candidate = None
            best_max_corr = 2.0
            for candidate in remaining:
                max_corr = max(abs_corr[candidate, s] for s in selected_idx)
                if max_corr < best_max_corr:
                    best_max_corr = max_corr
                    best_candidate = candidate
            selected_idx.append(best_candidate)
            remaining.remove(best_candidate)
            greedy_rows.append({
                "step": len(selected_idx),
                "benchmark": benchmark,
                "task_column": task_cols[best_candidate],
                "task_id": task_id_map.get(task_cols[best_candidate], task_cols[best_candidate]),
                "max_abs_corr_to_selected": float(best_max_corr),
            })

        # Per-benchmark summary
        corr_values = [r["max_abs_corr_to_selected"] for r in greedy_rows if r["benchmark"] == benchmark and r["step"] > 1]
        n_below_07 = sum(1 for v in corr_values if v < 0.7) + 1  # +1 for the first task (always independent)
        summary_rows.append({
            "benchmark": benchmark,
            "n_tasks": n_tasks,
            "n_tasks_below_0.7": n_below_07,
            "frac_below_0.7": round(n_below_07 / n_tasks, 4),
            "median_max_abs_corr": round(float(np.median(corr_values)), 4) if corr_values else 0.0,
            "mean_max_abs_corr": round(float(np.mean(corr_values)), 4) if corr_values else 0.0,
        })

    greedy_df = pd.DataFrame(greedy_rows)
    greedy_summary = pd.DataFrame(summary_rows)
    if not greedy_summary.empty:
        greedy_summary = greedy_summary.sort_values("n_tasks_below_0.7", ascending=False)
    return greedy_df, greedy_summary


def task_similarity_and_representatives(
    task_result: ImputationResult,
    tasks_enriched: pd.DataFrame,
    benchmark_clusters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reliable = tasks_enriched[
        tasks_enriched["is_bounded_score_task"]
        & tasks_enriched["is_reliable_observed_task"]
        & (tasks_enriched["observed_std"] >= 0.05)
    ].copy()
    reliable_columns = [col for col in reliable["task_column"] if col in task_result.raw.columns]
    matrix = task_result.raw[reliable_columns].astype(float)
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0).replace(0, np.nan)
    standardized = standardized.fillna(0)
    task_profiles = standardized.to_numpy().T
    corr = (task_profiles @ task_profiles.T) / task_profiles.shape[1]
    corr = np.clip(np.nan_to_num(corr), -1, 1)
    np.fill_diagonal(corr, 1)
    task_index = pd.Index(reliable_columns)
    bench_for_task = reliable.set_index("task_column").loc[task_index, "benchmark"]

    within_rows = []
    representative_rows = []
    predictability_rows = []
    positions_by_benchmark: dict[str, list[int]] = {}
    for position, benchmark in enumerate(bench_for_task.to_numpy()):
        positions_by_benchmark.setdefault(str(benchmark), []).append(position)

    for benchmark, positions in positions_by_benchmark.items():
        idx = np.array(positions, dtype=int)
        if len(idx) < 2:
            continue
        sub = np.abs(corr[np.ix_(idx, idx)])
        np.fill_diagonal(sub, np.nan)
        mean_peer = np.nanmean(sub, axis=1)
        max_peer = np.nanmax(sub, axis=1)
        within_rows.append(
            {
                "benchmark": benchmark,
                "n_reliable_tasks": len(idx),
                "median_abs_task_similarity_within_benchmark": float(np.nanmedian(sub)),
                "mean_abs_task_similarity_within_benchmark": float(np.nanmean(sub)),
            }
        )
        aggregate = task_profiles[idx].mean(axis=0)
        aggregate_corr = np.array([corr_or_nan(task_profiles[i], aggregate) for i in idx])
        leave_one_out_corr = []
        for task_idx in idx:
            peers = idx[idx != task_idx]
            peer_aggregate = task_profiles[peers].mean(axis=0) if len(peers) else aggregate
            leave_one_out_corr.append(corr_or_nan(task_profiles[task_idx], peer_aggregate))
        leave_one_out_corr = np.array(leave_one_out_corr, dtype=float)
        for local_pos, task_idx in enumerate(idx):
            task_col = task_index[task_idx]
            task_row = reliable[reliable["task_column"] == task_col].iloc[0]
            useful_rep = abs(leave_one_out_corr[local_pos]) * float(task_row["observed_std"])
            representative_rows.append(
                {
                    "benchmark": benchmark,
                    "task_column": task_col,
                    "task_id": task_row["task_id"],
                    "representativeness_score": float(abs(aggregate_corr[local_pos])),
                    "leave_one_out_aggregate_correlation": float(leave_one_out_corr[local_pos]),
                    "useful_representativeness_score": float(useful_rep),
                    "mean_abs_similarity_to_peer_tasks": float(mean_peer[local_pos]),
                    "difficulty_tier": task_row["difficulty_tier"],
                    "task_score": task_row["imputed_mean"],
                    "observed_mean": task_row["observed_mean"],
                    "observed_std": task_row["observed_std"],
                    "strength_correlation": task_row["strength_correlation"],
                }
            )
            predictability_rows.append(
                {
                    "benchmark": benchmark,
                    "task_column": task_col,
                    "task_id": task_row["task_id"],
                    "task_predictability_proxy_max_abs_peer_spearman": float(max_peer[local_pos]),
                    "task_unpredictability_score": float(1 - max_peer[local_pos]),
                    "observed_count": int(task_row["observed_count"]),
                    "difficulty_tier": task_row["difficulty_tier"],
                    "task_score": task_row["imputed_mean"],
                    "observed_mean": task_row["observed_mean"],
                    "observed_std": task_row["observed_std"],
                }
            )

    sampled = (
        reliable.sort_values(["observed_std", "strength_correlation", "observed_count"], ascending=False)
        .groupby("benchmark")
        .head(40)
        .reset_index(drop=True)
    )
    sampled_cols = [col for col in sampled["task_column"] if col in task_index]
    sampled_positions = task_index.get_indexer(sampled_cols)
    pair_rows = []
    for left_bench, left_group in sampled.groupby("benchmark"):
        left_idx = task_index.get_indexer(left_group["task_column"])
        for right_bench, right_group in sampled.groupby("benchmark"):
            if right_bench < left_bench:
                continue
            right_idx = task_index.get_indexer(right_group["task_column"])
            pair_corr = np.abs(corr[np.ix_(left_idx, right_idx)])
            if left_bench == right_bench:
                pair_corr = pair_corr[~np.eye(pair_corr.shape[0], dtype=bool)]
            pair_rows.append(
                {
                    "left_benchmark": left_bench,
                    "right_benchmark": right_bench,
                    "median_abs_task_similarity": float(np.nanmedian(pair_corr)) if pair_corr.size else np.nan,
                    "mean_abs_task_similarity": float(np.nanmean(pair_corr)) if pair_corr.size else np.nan,
                    "sampled_left_tasks": len(left_idx),
                    "sampled_right_tasks": len(right_idx),
                }
            )

    within = pd.DataFrame(within_rows).sort_values("median_abs_task_similarity_within_benchmark", ascending=False)
    representatives = pd.DataFrame(representative_rows).sort_values("representativeness_score", ascending=False)
    task_predictability = pd.DataFrame(predictability_rows).sort_values("task_unpredictability_score", ascending=False)
    cross = pd.DataFrame(pair_rows).sort_values("median_abs_task_similarity", ascending=False)
    return within, representatives, task_predictability, cross


# ─── Cross-benchmark global go-to task selection ────────────────────────────


def global_task_selection(
    task_result: ImputationResult,
    tasks_enriched: pd.DataFrame,
    max_greedy_steps: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select globally representative go-to tasks across ALL benchmarks.

    Flattens all reliable bounded tasks into a single pool, computes
    global representativeness (correlation with the overall aggregate
    profile), and runs greedy forward selection across the entire pool
    to identify a minimal set of tasks that maximises independent
    information.

    Returns
    -------
    global_representatives : pd.DataFrame
        One row per reliable task with global representativeness metrics,
        sorted by ``global_useful_representativeness`` descending.
    global_greedy : pd.DataFrame
        Greedy selection sequence (step, task_column, benchmark, task_id,
        max_abs_corr_to_selected), up to *max_greedy_steps* or all tasks.
    """
    reliable = tasks_enriched[
        tasks_enriched["is_bounded_score_task"]
        & tasks_enriched["is_reliable_observed_task"]
        & (tasks_enriched["observed_std"] >= 0.05)
    ].copy()
    reliable_columns = [c for c in reliable["task_column"] if c in task_result.raw.columns]
    if len(reliable_columns) < 10:
        return pd.DataFrame(), pd.DataFrame()

    matrix = task_result.raw[reliable_columns].astype(float)
    n_sys = matrix.shape[0]

    # Standardize columns for correlation computation
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0).replace(0, np.nan)
    standardized = standardized.fillna(0)
    task_profiles = standardized.to_numpy().T  # (n_tasks, n_systems)

    # Global aggregate = mean profile across all tasks
    global_aggregate = task_profiles.mean(axis=0)

    # Leave-one-out correlation with global aggregate
    from .intermediate_tables import corr_or_nan as _corr_or_nan

    task_meta = reliable.set_index("task_column")
    rep_rows: list[dict] = []
    for i, col in enumerate(reliable_columns):
        peers = np.delete(task_profiles, i, axis=0)
        peer_aggregate = peers.mean(axis=0)
        loo_corr = _corr_or_nan(task_profiles[i], peer_aggregate)
        agg_corr = _corr_or_nan(task_profiles[i], global_aggregate)
        row_meta = task_meta.loc[col]
        obs_std = float(row_meta["observed_std"])
        useful_rep = abs(loo_corr) * obs_std
        rep_rows.append({
            "benchmark": row_meta["benchmark"],
            "task_column": col,
            "task_id": row_meta["task_id"],
            "global_aggregate_correlation": float(agg_corr),
            "global_loo_correlation": float(loo_corr),
            "global_useful_representativeness": float(useful_rep),
            "observed_mean": float(row_meta["observed_mean"]),
            "observed_std": obs_std,
            "difficulty_tier": row_meta["difficulty_tier"],
        })

    global_rep = pd.DataFrame(rep_rows).sort_values("global_useful_representativeness", ascending=False)

    # ── Greedy forward selection across all tasks ──
    corr_mat = np.corrcoef(matrix.to_numpy().T)
    corr_mat = np.nan_to_num(corr_mat)
    abs_corr = np.abs(corr_mat)
    np.fill_diagonal(abs_corr, 0)
    n_tasks = len(reliable_columns)

    col_to_idx = {c: i for i, c in enumerate(reliable_columns)}
    task_id_map = task_meta["task_id"].to_dict()
    bench_map = task_meta["benchmark"].to_dict()

    # First pick: lowest mean |correlation|
    mean_abs = abs_corr.mean(axis=1)
    first = int(np.argmin(mean_abs))
    selected_idx = [first]
    remaining = set(range(n_tasks)) - {first}

    greedy_rows: list[dict] = [{
        "step": 1,
        "benchmark": bench_map.get(reliable_columns[first], ""),
        "task_column": reliable_columns[first],
        "task_id": task_id_map.get(reliable_columns[first], ""),
        "max_abs_corr_to_selected": 0.0,
    }]

    steps_limit = min(max_greedy_steps, n_tasks)
    while remaining and len(selected_idx) < steps_limit:
        best_candidate = None
        best_max_corr = 2.0
        for candidate in remaining:
            max_corr = max(abs_corr[candidate, s] for s in selected_idx)
            if max_corr < best_max_corr:
                best_max_corr = max_corr
                best_candidate = candidate
        selected_idx.append(best_candidate)
        remaining.remove(best_candidate)
        greedy_rows.append({
            "step": len(selected_idx),
            "benchmark": bench_map.get(reliable_columns[best_candidate], ""),
            "task_column": reliable_columns[best_candidate],
            "task_id": task_id_map.get(reliable_columns[best_candidate], ""),
            "max_abs_corr_to_selected": float(best_max_corr),
        })

    global_greedy = pd.DataFrame(greedy_rows)
    return global_rep, global_greedy


def build_goto_task_set(
    representative_tasks: pd.DataFrame,
    greedy_selection: pd.DataFrame,
    per_benchmark_k: int = 3,
) -> pd.DataFrame:
    """Build a go-to task set: top-k representative tasks per benchmark,
    filtered to independently informative tasks (greedy max |corr| < 0.7).

    Combines independence (greedy forward selection) with
    representativeness (leave-one-out correlation × variance) to produce
    a compact evaluation set that covers all benchmarks.
    """
    rep_score = dict(zip(
        representative_tasks["task_column"],
        representative_tasks["useful_representativeness_score"].astype(float),
    ))
    rep_meta = representative_tasks.set_index("task_column")

    # Per-benchmark independent tasks (greedy steps below 0.7 threshold)
    indep = greedy_selection[
        greedy_selection["max_abs_corr_to_selected"].astype(float) < 0.7
    ].copy()

    rows: list[dict] = []
    for benchmark, group in indep.groupby("benchmark"):
        group = group.copy()
        group["rep_score"] = group["task_column"].map(rep_score).fillna(0)
        top = group.nlargest(per_benchmark_k, "rep_score")
        for _, r in top.iterrows():
            tc = r["task_column"]
            meta = rep_meta.loc[tc] if tc in rep_meta.index else None
            rows.append({
                "benchmark": benchmark,
                "task_column": tc,
                "task_id": r["task_id"],
                "useful_representativeness_score": float(r["rep_score"]),
                "greedy_step": int(r["step"]),
                "max_abs_corr_to_selected": float(r["max_abs_corr_to_selected"]),
                "difficulty_tier": meta["difficulty_tier"] if meta is not None else "",
                "observed_mean": float(meta["observed_mean"]) if meta is not None else float("nan"),
                "observed_std": float(meta["observed_std"]) if meta is not None else float("nan"),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["benchmark", "useful_representativeness_score"],
            ascending=[True, False],
        )
    return result
