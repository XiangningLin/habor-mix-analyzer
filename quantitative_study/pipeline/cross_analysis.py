"""
Cross-analysis: combines Harbor matrix data with doc metadata to answer:
  1. What benchmarks are hardest? (lowest best-model scores)
  2. How do scores break down across taxonomy groups?
  3. How much progress did models make over time? (from doc temporal data)
  4. Harness vs model importance (from habor-analyze outputs)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_INFO_DIR, DOMAIN_MAP, MATRIX_COLUMN_TO_STEM,
)
from .loading import MetricAlignment, _apply_alignment_transform, load_metric_alignment

_STEM_TO_MATRIX: dict[str, str] = {}
for _mc, _st in MATRIX_COLUMN_TO_STEM.items():
    if _st is not None:
        _STEM_TO_MATRIX[_st] = _mc


def _domain_for(name: str) -> tuple[str, str]:
    """Look up domain by matrix column name, JSON stem, or normalized form."""
    if name in DOMAIN_MAP:
        return DOMAIN_MAP[name]
    hyphen = name.replace("_", "-")
    if hyphen in DOMAIN_MAP:
        return DOMAIN_MAP[hyphen]
    if name in _STEM_TO_MATRIX and _STEM_TO_MATRIX[name] in DOMAIN_MAP:
        return DOMAIN_MAP[_STEM_TO_MATRIX[name]]
    return ("Other", "Uncategorized")


def _is_well_scaled(series: pd.Series, lo: float = -0.01, hi: float = 1.01) -> bool:
    """True when all non-NaN values fall within [lo, hi]."""
    vals = series.dropna()
    if vals.empty:
        return False
    return bool((vals >= lo).all() and (vals <= hi).all())


# ===================================================================
# 1. Hardest benchmarks (lowest best-model score)
# ===================================================================

def benchmark_difficulty_table(harbor_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    rows: list[dict[str, Any]] = []
    for col in bench_cols:
        vals = harbor_df[col].dropna()
        if vals.empty:
            continue
        best_idx = vals.idxmax()
        domain, superdomain = _domain_for(col)
        rows.append(dict(
            benchmark=col,
            domain=domain,
            superdomain=superdomain,
            best_score=vals.max(),
            best_model=harbor_df.loc[best_idx, "model"],
            best_agent=harbor_df.loc[best_idx, "agent"],
            median_score=vals.median(),
            worst_score=vals.min(),
            n_entries=len(vals),
            spread=vals.max() - vals.min(),
        ))
    return pd.DataFrame(rows).sort_values("best_score", ascending=True)


# ===================================================================
# 2. Domain-level aggregation
# ===================================================================

def domain_summary(harbor_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    domain_scores: dict[str, list[float]] = {}
    domain_bests: dict[str, list[float]] = {}
    for col in bench_cols:
        vals = harbor_df[col].dropna()
        if vals.empty or not _is_well_scaled(vals):
            continue
        domain, _ = _domain_for(col)
        domain_scores.setdefault(domain, []).extend(vals.tolist())
        domain_bests.setdefault(domain, []).append(vals.max())

    rows = []
    for domain in sorted(domain_scores):
        all_vals = domain_scores[domain]
        bests = domain_bests[domain]
        rows.append(dict(
            domain=domain,
            n_benchmarks=len(bests),
            mean_best_score=np.mean(bests),
            mean_all_scores=np.mean(all_vals),
            median_all_scores=np.median(all_vals),
            std_all_scores=np.std(all_vals),
            min_best_score=min(bests),
        ))
    return pd.DataFrame(rows).sort_values("mean_best_score", ascending=False)


def superdomain_summary(harbor_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    sd_scores: dict[str, list[float]] = {}
    sd_bests: dict[str, list[float]] = {}
    for col in bench_cols:
        vals = harbor_df[col].dropna()
        if vals.empty or not _is_well_scaled(vals):
            continue
        _, superdomain = _domain_for(col)
        sd_scores.setdefault(superdomain, []).extend(vals.tolist())
        sd_bests.setdefault(superdomain, []).append(vals.max())

    rows = []
    for sd in sorted(sd_scores):
        all_vals = sd_scores[sd]
        bests = sd_bests[sd]
        rows.append(dict(
            superdomain=sd,
            n_benchmarks=len(bests),
            mean_best_score=np.mean(bests),
            mean_all_scores=np.mean(all_vals),
            median_all_scores=np.median(all_vals),
            std_all_scores=np.std(all_vals),
        ))
    return pd.DataFrame(rows)


def per_model_domain_scores(harbor_df: pd.DataFrame) -> pd.DataFrame:
    """Mean score per (model, domain) — averaged across benchmarks in that domain."""
    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]

    well_scaled = {c for c in bench_cols if _is_well_scaled(harbor_df[c])}
    records = []
    for _, row in harbor_df.iterrows():
        model, agent = row["model"], row["agent"]
        for col in bench_cols:
            if col not in well_scaled:
                continue
            v = row[col]
            if pd.isna(v):
                continue
            domain, superdomain = _domain_for(col)
            records.append(dict(
                model=model, agent=agent, benchmark=col,
                domain=domain, superdomain=superdomain,
                score=float(v),
            ))
    long = pd.DataFrame(records)
    if long.empty:
        return pd.DataFrame()

    grouped = long.groupby(["model", "domain"]).agg(
        mean_score=("score", "mean"),
        n_benchmarks=("benchmark", "nunique"),
    ).reset_index()
    return grouped.sort_values(["domain", "mean_score"], ascending=[True, False])


# ===================================================================
# 3. Progress over time from doc data
# ===================================================================

def _parse_date(s: str) -> str | None:
    """Normalize date strings to 'YYYY-MM' for grouping."""
    if not s:
        return None
    m = re.match(r"(\d{4})-(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return None


def _extract_best_score(
    results: list[dict], alignment: MetricAlignment,
) -> float | None:
    """Extract the best score using the reviewed alignment selector only."""
    best: float | None = None

    for entry in results:
        values: list[float] = []
        if alignment.doc_score_field == "score":
            val = entry.get("score")
            if isinstance(val, (int, float)) and alignment.doc_metric == "legacy_score":
                values.append(float(val))
        elif alignment.doc_score_field == "scores[].value":
            for score in entry.get("scores") or []:
                metric = str(score.get("metric") or "").strip()
                val = score.get("value")
                if metric == alignment.doc_metric and isinstance(val, (int, float)):
                    values.append(float(val))
        for value in values:
            try:
                transformed = _apply_alignment_transform(value, alignment.transform)
            except (ValueError, TypeError, ZeroDivisionError):
                continue
            if transformed < -0.01 or transformed > 1.01:
                continue
            if best is None or transformed > best:
                best = transformed
    return best


def progress_over_time(
    info_dir: Path | None = None,
    *,
    min_snapshots: int = 2,
    frontier_only: bool = True,
) -> pd.DataFrame:
    """Extract each benchmark's best score at each temporal snapshot.

    By default, keep the historical pipeline behavior: require at least two
    snapshots and retain only frontier-improving monthly rows. Figure scripts can
    opt into single-snapshot benchmarks or all monthly rows.
    """
    info_dir = info_dir or BENCHMARK_INFO_DIR
    alignments = load_metric_alignment()
    alignment_by_stem = {
        alignment.info_stem: alignment
        for alignment in alignments.values()
        if alignment.info_stem and alignment.include_in_comparison
    }
    rows: list[dict[str, Any]] = []
    for fp in sorted(info_dir.glob("*.json")):
        with open(fp) as fh:
            doc = json.load(fh)
        rot = doc.get("results_over_time", [])
        if len(rot) < min_snapshots:
            continue
        stem = fp.stem
        alignment = alignment_by_stem.get(stem)
        if alignment is None:
            continue
        name = doc.get("name", stem)
        for snapshot in rot:
            date_str = snapshot.get("date", "")
            ym = _parse_date(date_str)
            if not ym:
                continue
            results = snapshot.get("results", [])
            if not results:
                continue
            best = _extract_best_score(results, alignment)
            if best is not None:
                rows.append(dict(
                    benchmark=stem,
                    benchmark_name=name,
                    date_ym=ym,
                    best_score=best,
                    n_models=len(results),
                ))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["benchmark", "date_ym"])
    deduped = df.groupby(["benchmark", "date_ym"]).agg(
        benchmark_name=("benchmark_name", "first"),
        best_score=("best_score", "max"),
        n_models=("n_models", "max"),
    ).reset_index()
    deduped = deduped.sort_values(["benchmark", "date_ym"])

    if not frontier_only:
        return deduped.sort_values(["benchmark", "date_ym"])

    frontier_rows: list[dict[str, Any]] = []
    for _, g in deduped.groupby("benchmark", sort=False):
        running_best: float | None = None
        for row in g.to_dict("records"):
            score = float(row["best_score"])
            if running_best is None or score > running_best:
                frontier_rows.append(row)
                running_best = score

    if not frontier_rows:
        return pd.DataFrame(columns=deduped.columns)
    return pd.DataFrame(frontier_rows).sort_values(["benchmark", "date_ym"])


def progress_summary(progress_df: pd.DataFrame) -> pd.DataFrame:
    """Per-benchmark: earliest score, latest score, delta, months spanned."""
    if progress_df.empty:
        return pd.DataFrame()
    rows = []
    for bench, g in progress_df.groupby("benchmark"):
        g = g.sort_values("date_ym")
        earliest = g.iloc[0]
        latest = g.iloc[-1]
        delta = latest["best_score"] - earliest["best_score"]
        domain, superdomain = _domain_for(bench)
        rows.append(dict(
            benchmark=bench,
            domain=domain,
            superdomain=superdomain,
            earliest_date=earliest["date_ym"],
            latest_date=latest["date_ym"],
            earliest_best=earliest["best_score"],
            latest_best=latest["best_score"],
            absolute_progress=delta,
            n_snapshots=len(g),
        ))
    return pd.DataFrame(rows).sort_values("absolute_progress", ascending=False)


def benchmark_launch_vs_harbor_improvement(
    harbor_df: pd.DataFrame,
    info_dir: Path | None = None,
) -> pd.DataFrame:
    """Compare each benchmark's launch-time best doc score to Harbor's max score.

    The launch baseline is the earliest results_over_time snapshot containing
    the reviewed Harbor-aligned metric from benchmark_metric_alignment.csv.
    Relative improvement is (harbor_best - launch_best) / launch_best.
    """
    info_dir = info_dir or BENCHMARK_INFO_DIR
    alignments = load_metric_alignment()

    rows: list[dict[str, Any]] = []
    for alignment in alignments.values():
        domain, superdomain = _domain_for(alignment.matrix_column)
        base_row: dict[str, Any] = dict(
            benchmark=alignment.info_stem or alignment.matrix_column,
            matrix_column=alignment.matrix_column,
            benchmark_name=alignment.benchmark_name,
            domain=domain,
            superdomain=superdomain,
            is_subset=alignment.alignment_status == "subset_or_variant",
            launch_date_ym=None,
            launch_best_score=np.nan,
            launch_n_models=0,
            harbor_best_score=np.nan,
            harbor_best_model=None,
            harbor_best_agent=None,
            absolute_improvement=np.nan,
            relative_improvement=np.nan,
            relative_improvement_pct=np.nan,
            status="ok",
        )

        if not alignment.include_in_comparison:
            rows.append({**base_row, "status": "excluded_metric_alignment"})
            continue
        if not alignment.info_stem:
            rows.append({**base_row, "status": "missing_info_stem"})
            continue

        path = info_dir / f"{alignment.info_stem}.json"
        if not path.is_file():
            rows.append({**base_row, "status": "missing_benchmark_info_json"})
            continue

        with open(path) as fh:
            doc = json.load(fh)

        candidates: list[dict[str, Any]] = []
        for snapshot in doc.get("results_over_time") or []:
            ym = _parse_date(str(snapshot.get("date", "")))
            if not ym:
                continue
            results = snapshot.get("results") or []
            best = _extract_best_score(results, alignment)
            if best is None:
                continue
            candidates.append(dict(
                launch_date_ym=ym,
                launch_best_score=best,
                launch_n_models=len(results),
            ))
        if not candidates:
            rows.append({**base_row, "status": "missing_aligned_launch_score"})
            continue

        launch = pd.DataFrame(candidates).sort_values("launch_date_ym")
        launch_ym = launch["launch_date_ym"].iloc[0]
        launch_same_month = launch[launch["launch_date_ym"] == launch_ym]
        launch_best_idx = launch_same_month["launch_best_score"].idxmax()
        launch_record = launch_same_month.loc[launch_best_idx].to_dict()

        if alignment.matrix_column not in harbor_df.columns:
            rows.append({
                **base_row,
                **launch_record,
                "status": "missing_harbor_column",
            })
            continue

        harbor_scores = pd.to_numeric(harbor_df[alignment.matrix_column], errors="coerce")
        harbor_valid = harbor_scores.dropna()
        if harbor_valid.empty:
            rows.append({
                **base_row,
                **launch_record,
                "status": "missing_harbor_score",
            })
            continue

        harbor_best_idx = harbor_valid.idxmax()
        harbor_best = float(harbor_valid.loc[harbor_best_idx])
        if harbor_best < -0.01 or harbor_best > 1.01:
            rows.append({
                **base_row,
                **launch_record,
                "harbor_best_score": harbor_best,
                "status": "harbor_score_out_of_range",
            })
            continue

        launch_best = float(launch_record["launch_best_score"])
        absolute = harbor_best - launch_best
        relative = np.nan if launch_best == 0 else absolute / launch_best
        status = "ok" if not pd.isna(relative) else "zero_launch_baseline"

        rows.append({
            **base_row,
            **launch_record,
            "harbor_best_score": harbor_best,
            "harbor_best_model": harbor_df.loc[harbor_best_idx, "model"],
            "harbor_best_agent": harbor_df.loc[harbor_best_idx, "agent"],
            "absolute_improvement": absolute,
            "relative_improvement": relative,
            "relative_improvement_pct": relative * 100 if not pd.isna(relative) else np.nan,
            "status": status,
        })

    df = pd.DataFrame(rows)
    return df.sort_values(
        ["status", "relative_improvement"],
        ascending=[True, False],
        na_position="last",
    )


def domain_launch_vs_harbor_improvement(improvement_df: pd.DataFrame) -> pd.DataFrame:
    """Domain-level summary over computable launch-vs-Harbor improvements."""
    if improvement_df.empty:
        return pd.DataFrame()
    df = improvement_df[
        (improvement_df["status"] == "ok")
        & improvement_df["relative_improvement"].notna()
    ].copy()
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("domain").agg(
        superdomains=("superdomain", lambda vals: "; ".join(sorted(set(map(str, vals))))),
        n_benchmarks=("benchmark", "nunique"),
        mean_relative_improvement=("relative_improvement", "mean"),
        median_relative_improvement=("relative_improvement", "median"),
        mean_relative_improvement_pct=("relative_improvement_pct", "mean"),
        median_relative_improvement_pct=("relative_improvement_pct", "median"),
        mean_absolute_improvement=("absolute_improvement", "mean"),
        median_absolute_improvement=("absolute_improvement", "median"),
        mean_launch_best_score=("launch_best_score", "mean"),
        mean_harbor_best_score=("harbor_best_score", "mean"),
    ).reset_index()
    return grouped.sort_values("median_relative_improvement", ascending=False)


def domain_progress_summary(prog_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate progress by domain and superdomain."""
    if prog_summary.empty:
        return pd.DataFrame()
    rows = []
    for domain, g in prog_summary.groupby("domain"):
        rows.append(dict(
            domain=domain,
            superdomain=g.iloc[0]["superdomain"],
            n_benchmarks=len(g),
            mean_absolute_progress=g["absolute_progress"].mean(),
            median_absolute_progress=g["absolute_progress"].median(),
            mean_latest_best=g["latest_best"].mean(),
        ))
    return pd.DataFrame(rows).sort_values("mean_absolute_progress", ascending=False)
