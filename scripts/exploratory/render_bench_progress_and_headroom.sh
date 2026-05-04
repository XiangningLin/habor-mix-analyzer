#!/usr/bin/env bash
set -euo pipefail

uv run python - <<'PY'
import shutil
from pathlib import Path

import pandas as pd

from src.habor_mix_analyzer.visualization.benchmark_plots import (
    save_benchmark_progress_and_headroom_plot,
)

headroom = pd.read_csv("output/key_analyses/tables/benchmark_level/benchmark_headroom_by_domain.csv")
launch = pd.read_csv("output/quantitative/benchmark_launch_vs_harbor_improvement.csv")

save_benchmark_progress_and_headroom_plot(headroom, launch)

dest = Path("output/key_analyses/figures/benchmark_level/bench_progress_and_headroom.pdf")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(
    "output/quantitative/figures/bench_progress_and_headroom.pdf",
    dest,
)
PY
