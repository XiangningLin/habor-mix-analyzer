import pandas as pd

with open('src/habor_mix_analyzer/studies/coverage_filtering.py', 'r') as f:
    content = f.read()

old_logic = """    table["include_in_key_analysis"] = (
        in_paper
        & (table["observed_count"] >= min_observed)
        & (table["missing_fraction"] <= max_missing)
    )

    def reason(row: pd.Series) -> str:
        if row["include_in_key_analysis"]:
            return "included"
        if row["benchmark"] not in PAPER_BENCHMARKS:
            return "excluded: not in paper benchmark list"
        if row["observed_count"] < min_observed:
            return f"excluded: fewer than {min_observed} observed agent+model rows"
        return f"excluded: missing fraction above {max_missing:.0%}\""""

new_logic = """    has_extreme_values = (table["min"] < -2) | (table["max"] > 3)
    
    table["include_in_key_analysis"] = (
        in_paper
        & (table["observed_count"] >= min_observed)
        & (table["missing_fraction"] <= max_missing)
        & (~has_extreme_values)
    )

    def reason(row: pd.Series) -> str:
        if row["include_in_key_analysis"]:
            return "included"
        if row["benchmark"] not in PAPER_BENCHMARKS:
            return "excluded: not in paper benchmark list"
        if row["observed_count"] < min_observed:
            return f"excluded: fewer than {min_observed} observed agent+model rows"
        if row["missing_fraction"] > max_missing:
            return f"excluded: missing fraction above {max_missing:.0%}"
        if row["min"] < -2 or row["max"] > 3:
            return "excluded: extreme normalized values (likely unscaled metric)"
        return "excluded: unknown" """

content = content.replace(old_logic, new_logic)
with open('src/habor_mix_analyzer/studies/coverage_filtering.py', 'w') as f:
    f.write(content)
