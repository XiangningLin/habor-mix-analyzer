"""
Central configuration: paths, column/stem mappings, model/agent aliases,
score transforms, and figure style constants.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# Paths (relative to habor-mix-analyzer root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

HARBOR_CSV_CANDIDATES = [
    REPO_ROOT / "data" / "raw" / "benchmark_level_matrix.csv",
    REPO_ROOT / "benchmark_level_matrix.csv",
]
BENCHMARK_INFO_DIR = REPO_ROOT / "benchmark_info_jobs"
METRIC_ALIGNMENT_PATH = REPO_ROOT / "data" / "metadata" / "benchmark_metric_alignment.csv"
BENCHMARK_CATEGORIES_PATH = REPO_ROOT / "data" / "metadata" / "benchmark_categories.json"
OUTPUT_DIR = REPO_ROOT / "output" / "quantitative"
FIGURE_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Matrix column -> JSON stem overrides
# ---------------------------------------------------------------------------
MATRIX_COLUMN_TO_STEM: dict[str, str | None] = {
    "bixbench": "bix_bench",
    "crustbench": "crust_bench",
    "dacode": "da_code",
    "featurebench-modal": "featurebench",
    "financeagent_terminal": "financeagent",
    "labbench": "lab_bench",
    "mlgym": "mlgym_bench",
    "omnimath": "omni_math",
    "research-code-bench": "researchcodebench",
    "spider2": "spider_2",
    "swebench-verified": "swe_bench_verified",
    "swebench-multilingual": "swe_bench_multilingual",
    "swebenchpro": "swe_bench_pro",
    "swtbench": "swt_bench",
    "terminal-bench": "terminalbench_2_0",
    "qwen-coder": None,
    "swegym": None,
    "swesmith": "swe_smith",
}

# ---------------------------------------------------------------------------
# Model aliases: harbor_model -> list of known doc name patterns
# ---------------------------------------------------------------------------
MODEL_ALIASES: dict[str, list[str]] = {
    "gpt-5.4": ["gpt-5.4", "gpt 5.4"],
    "gpt-5-mini": ["gpt-5-mini", "gpt 5 mini", "gpt-5-mini-2025-08-07"],
    "gpt-5-nano": ["gpt-5-nano", "gpt 5 nano", "gpt-5-nano-2025-08-07"],
    "claude-haiku-4-5-20251001": ["claude-haiku-4-5", "claude haiku 4.5"],
    "claude-sonnet-4-6": ["claude-sonnet-4-6", "claude sonnet 4.6"],
    "claude-opus-4-6": ["claude-opus-4-6", "claude opus 4.6"],
    "gemini-3.1-pro-preview": [
        "gemini-3.1-pro-preview", "gemini 3.1 pro preview",
        "gemini-3.1-pro", "gemini 3.1 pro",
    ],
    "gemini-3-flash-preview": [
        "gemini-3-flash-preview", "gemini 3 flash preview",
        "gemini-3-flash", "gemini 3 flash",
    ],
    "deepseek-reasoner": ["deepseek-reasoner", "deepseek-r1", "deepseek r1"],
    "deepseek-chat": [
        "deepseek-chat", "deepseek-v3", "deepseek v3",
        "deepseek-v3.1", "deepseek v3.1",
    ],
    "kimi-k2.5": ["kimi-k2.5", "kimi k2.5"],
    "minimax-m2.5": ["minimax-m2.5", "minimax m2.5"],
    "glm-5": ["glm-5", "glm 5"],
    "mimo-v2-pro": ["mimo-v2-pro", "mimo v2 pro"],
    "qwen3-max": ["qwen3-max", "qwen 3 max", "qwen3 max", "qwen-3-max"],
}

AGENT_ALIASES: dict[str, list[str]] = {
    "terminus-2": ["terminus-2", "terminus 2", "terminus"],
    "codex": ["codex"],
    "claude-code": ["claude-code", "claude code"],
    "gemini-cli": ["gemini-cli", "gemini cli"],
    "qwen-coder": ["qwen-coder", "qwen coder"],
}

# ---------------------------------------------------------------------------
# Score transforms: map raw doc scores to [0, 1] to match Harbor scale.
# ---------------------------------------------------------------------------
TRANSFORM_LOG_SPEEDUP = {"algotune"}          # y = ln(max(1,x)) / (ln(max(1,x)) + 1)
TRANSFORM_NEG1_TO_1   = {"sldbench", "ineqmath"}  # y = (x+1) / 2

SKIP_METRICS = {"cost", "latency", "latency mean", "latency_mean"}


def transform_doc_score(score: float, stem: str) -> float:
    if stem in TRANSFORM_LOG_SPEEDUP:
        lnx = math.log(max(1.0, score))
        return lnx / (lnx + 1.0) if (lnx + 1.0) != 0 else 0.0
    if stem in TRANSFORM_NEG1_TO_1:
        return (score + 1.0) / 2.0
    return score


# ---------------------------------------------------------------------------
# Figure style — aligned with habor-analyze (core/plotting.py)
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE = (10, 8)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (11.5, 10)
DPI = 200

PLOT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "axes.axisbelow": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.35,
    "axes.facecolor": "#fbfbfb",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

# Palette consistent with habor-analyze
COLOR_GREEN = "#a1d99b"
COLOR_BLUE = "#9ecae1"
COLOR_ORANGE = "#fdae6b"
COLOR_PURPLE = "#bcbddc"
COLOR_RED = "#e74c3c"
COLOR_GRAY = "#95a5a6"
COLOR_GRID = "#dddddd"
COLOR_AXIS = "#666666"

# ---------------------------------------------------------------------------
# Benchmark taxonomy loaded from benchmark_categories.json
# domain = top-level category
# superdomain = agenticity bucket within each subdomain
# ---------------------------------------------------------------------------
_UNCATEGORIZED_DOMAIN = "Other"
_UNCATEGORIZED_SUPERDOMAIN = "Uncategorized"

BENCHMARK_KIND_LABELS = {
    "agentic": "Agentic",
    "modified-agentic": "Modified Agentic",
    "non-agentic": "Non-Agentic",
}


def _normalize_benchmark_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def resolve_stem(column: str) -> str | None:
    if column in ("model", "agent"):
        return None
    if column in MATRIX_COLUMN_TO_STEM:
        return MATRIX_COLUMN_TO_STEM[column]
    return column.replace("-", "_")


def _discover_matrix_columns() -> list[str]:
    for candidate in HARBOR_CSV_CANDIDATES:
        if not candidate.is_file():
            continue
        with candidate.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        return [column for column in header if column not in {"model", "agent"}]
    return []


def _build_stem_to_matrix() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in _discover_matrix_columns():
        stem = resolve_stem(column)
        if stem:
            mapping.setdefault(stem, column)
    for column, stem in MATRIX_COLUMN_TO_STEM.items():
        if stem:
            mapping.setdefault(stem, column)
    return mapping


STEM_TO_MATRIX = _build_stem_to_matrix()


def _build_stem_name_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for path in sorted(BENCHMARK_INFO_DIR.glob("*.json")):
        stem = path.stem
        lookup.setdefault(_normalize_benchmark_key(stem), stem)
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        name = str(data.get("name") or "").strip()
        if name:
            lookup.setdefault(_normalize_benchmark_key(name), stem)
    return lookup


STEM_NAME_LOOKUP = _build_stem_name_lookup()


def _build_matrix_name_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in _discover_matrix_columns():
        lookup.setdefault(_normalize_benchmark_key(column), column)
        stem = resolve_stem(column)
        if stem:
            lookup.setdefault(_normalize_benchmark_key(stem), column)
    for stem, column in STEM_TO_MATRIX.items():
        lookup.setdefault(_normalize_benchmark_key(stem), column)
        lookup.setdefault(_normalize_benchmark_key(column), column)
    for key, stem in STEM_NAME_LOOKUP.items():
        column = STEM_TO_MATRIX.get(stem)
        if column:
            lookup.setdefault(key, column)
    return lookup


MATRIX_NAME_LOOKUP = _build_matrix_name_lookup()


def _load_benchmark_taxonomy() -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, str]]]:
    if not BENCHMARK_CATEGORIES_PATH.is_file():
        return {}, {}

    raw = json.loads(BENCHMARK_CATEGORIES_PATH.read_text())
    domain_map: dict[str, tuple[str, str]] = {}
    benchmark_taxonomy: dict[str, dict[str, str]] = {}

    for domain_block in raw:
        domain = str(domain_block.get("domain") or "").strip()
        if not domain:
            continue
        for subdomain_block in domain_block.get("subdomains") or []:
            subdomain = str(subdomain_block.get("subdomain") or "").strip()
            benchmarks_by_kind = subdomain_block.get("benchmarks") or {}
            for kind, benchmarks in benchmarks_by_kind.items():
                superdomain = BENCHMARK_KIND_LABELS.get(kind, kind.replace("-", " ").title())
                for benchmark_name in benchmarks or []:
                    canonical = _normalize_benchmark_key(str(benchmark_name))
                    stem = STEM_NAME_LOOKUP.get(canonical)
                    matrix_column = MATRIX_NAME_LOOKUP.get(canonical)
                    if matrix_column is None and stem is not None:
                        matrix_column = STEM_TO_MATRIX.get(stem)
                    if stem is None and matrix_column is not None:
                        stem = resolve_stem(matrix_column)

                    taxonomy = {
                        "domain": domain,
                        "subdomain": subdomain,
                        "superdomain": superdomain,
                    }

                    if matrix_column:
                        domain_map[matrix_column] = (domain, superdomain)
                        benchmark_taxonomy[matrix_column] = taxonomy
                    if stem:
                        domain_map[stem] = (domain, superdomain)
                        benchmark_taxonomy[stem] = taxonomy

    return domain_map, benchmark_taxonomy


DOMAIN_MAP, BENCHMARK_TAXONOMY = _load_benchmark_taxonomy()

DOMAIN_COLORS = {
    "Software Engineering": "#74c476",
    "Mathematics & Reasoning": "#6baed6",
    "Knowledge & Long Context": "#9ecae1",
    "Scientific Research": "#9e9ac8",
    "Agents, Tools & Systems": "#fdae6b",
    "Data & Analytics": "#fdd0a2",
    "Professional Domains": "#fd8d3c",
    "Safety & Security": "#969696",
    "Multimodal": "#e377c2",
    _UNCATEGORIZED_DOMAIN: COLOR_GRAY,
}

SUPERDOMAIN_COLORS = {
    "Agentic": "#74c476",
    "Modified Agentic": "#9ecae1",
    "Non-Agentic": "#fdae6b",
    _UNCATEGORIZED_SUPERDOMAIN: COLOR_GRAY,
}


MATCH_STATUS_COLORS = {
    "matched_model_and_agent": COLOR_GREEN,
    "matched_model_only_doc_no_agent": COLOR_BLUE,
    "matched_model_agent_mismatch": COLOR_ORANGE,
}

MATCH_STATUS_LABELS = {
    "matched_model_and_agent": "Model + Agent matched",
    "matched_model_only_doc_no_agent": "Model matched (doc = bare LLM)",
    "matched_model_agent_mismatch": "Model matched (agent differs)",
}
