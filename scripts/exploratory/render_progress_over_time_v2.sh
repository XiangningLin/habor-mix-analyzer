#!/usr/bin/env bash
set -euo pipefail

uv run python - <<'PY'
import pandas as pd
from pathlib import Path
from quantitative_study.pipeline.cross_analysis import progress_over_time
from quantitative_study.pipeline.figures import fig_progress_over_time

# Load Harbor data for the diamond marker at the end
# Defaulting to benchmark_level_matrix.csv in the raw directory
harbor_df_path = Path("data/raw/benchmark_level_matrix.csv")
harbor_df = pd.read_csv(harbor_df_path) if harbor_df_path.exists() else None

# Avoid stale per-domain files from prior taxonomy/grouping runs.
figure_dir = Path("output/quantitative/figures")
for pattern in ("progress_over_time_v2*.pdf", "progress_over_time_v2*.png"):
    for path in figure_dir.glob(pattern):
        path.unlink()

# Calculate a figure-focused progress table: include single-snapshot
# benchmarks, but keep only running-best rows so progress never declines.
progress_df = progress_over_time(min_snapshots=1, frontier_only=True)

# Generate the figure
fig_progress_over_time(progress_df, harbor_df)
print("Saved progress_over_time_v2 domain PDFs/PNGs to output/quantitative/figures/")
PY
