from __future__ import annotations

from ..core import *
from ..core.config import PAPER_BENCHMARKS


def benchmark_filter_table(stats: pd.DataFrame) -> pd.DataFrame:
    table = stats.rename(columns={"column": "benchmark"}).copy()
    in_paper = table["benchmark"].isin(PAPER_BENCHMARKS)
    table["include_in_key_analysis"] = in_paper

    def reason(row: pd.Series) -> str:
        if row["include_in_key_analysis"]:
            return "included"
        return "excluded: not in paper benchmark list"

    table["filter_reason"] = table.apply(reason, axis=1)
    ordered = [
        "benchmark",
        "include_in_key_analysis",
        "filter_reason",
        "observed_count",
        "missing_count",
        "missing_fraction",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "negative_count",
        "gt_one_count",
    ]
    ordered = [col for col in ordered if col in table.columns]
    return table[ordered].sort_values(["include_in_key_analysis", "observed_count"], ascending=[False, False])
