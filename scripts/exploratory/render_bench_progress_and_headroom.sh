#!/usr/bin/env bash
set -euo pipefail

uv run python - <<'PY'
import shutil

import pandas as pd

from src.habor_mix_analyzer.visualization.benchmark_plots import (
    save_benchmark_progress_and_headroom_plot,
)

headroom = pd.read_csv("output/key_analyses/tables/benchmark_level/benchmark_headroom_by_domain.csv")
launch = pd.read_csv("output/quantitative/benchmark_launch_vs_harbor_improvement.csv")

save_benchmark_progress_and_headroom_plot(headroom, launch)

shutil.copyfile(
    "output/quantitative/figures/bench_progress_and_headroom.pdf",
    "output/key_analyses/figures/benchmark_level/bench_progress_and_headroom.pdf",
)
PY
