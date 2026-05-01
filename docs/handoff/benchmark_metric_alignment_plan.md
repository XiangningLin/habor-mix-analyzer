# Harbor Benchmark Metric Alignment Plan

## Goal

We need a human-reviewed mapping that says which external benchmark metric, if any, is comparable to each Harbor benchmark score in `data/raw/benchmark_level_matrix.csv`.

The main risk is that many `benchmark_info_jobs/*.json` files contain multiple metrics for the same benchmark. Some are leaderboard scores, some are costs or latency, some are split-level metrics, and some are not on the same scale as Harbor. The pipeline should not guess which metric to use.

The deliverable is a CSV alignment table that becomes the source of truth for Harbor-vs-external-score comparisons.

## Source Of Truth

Create and maintain:

```text
data/metadata/benchmark_metric_alignment.csv
```

One row should correspond to one Harbor benchmark column.

Use CSV because this file needs to be read, sorted, filtered, and reviewed by humans in a spreadsheet.

## Required CSV Columns

```text
matrix_column
info_stem
benchmark_name
harbor_metric
doc_score_field
doc_metric
alignment_status
transform
include_in_comparison
evidence_level
reviewer
review_date
notes
```

Column meanings:

| Column | Meaning |
|---|---|
| `matrix_column` | Exact benchmark column name from `data/raw/benchmark_level_matrix.csv`. |
| `info_stem` | Matching JSON file stem under `benchmark_info_jobs/`, without `.json`. |
| `benchmark_name` | Human-readable benchmark name from the JSON. |
| `harbor_metric` | Short description of what the Harbor score means. |
| `doc_score_field` | Where the external score comes from: usually `scores[].value`; use `score` for legacy JSON rows. |
| `doc_metric` | Exact external metric name to use. For legacy single-score rows, use `legacy_score` unless a clearer name is known. |
| `alignment_status` | Whether the selected external metric is comparable to Harbor. See allowed values below. |
| `transform` | How to map the external value `x` onto the Harbor-comparable scale. Use `identity` if no transform is needed. If not identity, write a lambda expression. |
| `include_in_comparison` | `true` only when the row should be used in Harbor-vs-external quantitative comparison. |
| `evidence_level` | Why we trust the alignment. See suggested values below. |
| `reviewer` | Name or handle of the person who reviewed the row. |
| `review_date` | Review date in `YYYY-MM-DD`. |
| `notes` | Concise explanation, especially for non-direct, transformed, or excluded cases. |

## Alignment Status Values

Use exactly one of these values:

| Status | Meaning | Usually include? |
|---|---|---|
| `direct` | Same metric, same direction, same scale. | `true` |
| `transformed` | Same underlying target, but the external score needs a scale or direction transform. | `true` if transform is clear |
| `subset_or_variant` | Same family of metric, but split, subset, judge, task set, or harness differs. | usually `false` |
| `proxy` | Related signal, but not the same metric. | `false` |
| `not_comparable` | Should not be compared. | `false` |
| `unknown` | Needs further review. | `false` |

Rule of thumb: only `direct` and well-justified `transformed` rows should have `include_in_comparison=true`.

## Transform Rules

The transform maps an external documented score `x` to the Harbor-comparable value.

If no transform is needed, write:

```text
identity
```

If a transform is needed, write a Python-style lambda expression:

```text
lambda x: ...
```

Examples:

| Case | Transform |
|---|---|
| Percent recorded as `83` but Harbor uses `0.83` | `lambda x: x / 100` |
| External reports attack success, Harbor uses defense score | `lambda x: 1 - x` |
| External score is in `[-1, 1]`, Harbor uses `[0, 1]` | `lambda x: (x + 1) / 2` |
| AlgoTune-style speedup score, where higher unbounded values need compression | `lambda x: math.log(max(1.0, x)) / (math.log(max(1.0, x)) + 1.0)` |
| Lower-is-better error metric converted to pass/fail threshold | `lambda x: 1.0 if x < 0.05 else 0.0` |

CSV note: if the lambda contains commas, quote the whole transform field.

Example:

```csv
algotune,algotune,AlgoTune,normalized speedup,scores[].value,AlgoTune Score,transformed,"lambda x: math.log(max(1.0, x)) / (math.log(max(1.0, x)) + 1.0)",true,manual_scale_match,alice,2026-04-25,"Matches current Harbor log-speedup normalization."
```

## Evidence Level Values

Suggested values:

| Value | Meaning |
|---|---|
| `metric_name_match` | Metric name and definition clearly match. |
| `paper_or_docs_match` | Benchmark paper/docs define the same metric. |
| `leaderboard_match` | Leaderboard column clearly matches Harbor score semantics. |
| `parity_experiment` | Harbor adapter parity experiment supports comparability. |
| `manual_review` | Human reviewer judged it comparable from available evidence. |
| `not_enough_info` | Insufficient evidence. |

Prefer `parity_experiment` or `paper_or_docs_match` when available.

## Example Rows

These examples show the intended shape. They are not final judgments.

```csv
matrix_column,info_stem,benchmark_name,harbor_metric,doc_score_field,doc_metric,alignment_status,transform,include_in_comparison,evidence_level,reviewer,review_date,notes
aime,aime,AIME,accuracy,scores[].value,"accuracy (pass@1, 60 questions)",direct,identity,true,metric_name_match,,,Same pass@1-style accuracy if task set matches.
codepde,codepde,CodePDE,pass rate,scores[].value,pass rate,direct,identity,true,parity_experiment,,,Use pass rate, not nRMSE.
strongreject,strongreject,StrongReject,defense score,scores[].value,Best StrongREJECT score,transformed,lambda x: 1 - x,true,manual_review,,,Harbor uses defense score; external metric is jailbreak success score.
gaia2,gaia2,GAIA2,pass@1,scores[].value,pass@1 (overall),subset_or_variant,identity,false,manual_review,,,Paper and CLI leaderboard split definitions may differ.
swebench-verified,swe_bench_verified,SWE-bench Verified,resolved rate,score,legacy_score,unknown,identity,false,not_enough_info,,,Legacy score field needs review and model/agent matching audit.
```

## Review Workflow

1. Generate an audit table from `data/raw/benchmark_level_matrix.csv` and `benchmark_info_jobs/*.json`.
2. For each Harbor `matrix_column`, list all candidate external metrics from the matching JSON.
3. Reviewer chooses exactly one `doc_metric` if a comparable metric exists.
4. Reviewer sets `alignment_status`, `transform`, and `include_in_comparison`.
5. Rows marked `transformed`, `subset_or_variant`, `proxy`, `not_comparable`, or `unknown` must include a short `notes` explanation.
6. A second person should spot-check all rows with `include_in_comparison=true`.
7. The pipeline should read only this CSV when computing Harbor-vs-external comparisons.

## Pipeline Rules After This Table Exists

The comparison code should follow these rules:

1. Do not fall back to the first numeric metric in a JSON row.
2. Do not use `evaluation.harbor_aligned_metric` as a machine-readable selector unless it has been migrated into this CSV.
3. Only compare rows where `include_in_comparison=true`.
4. Use `doc_metric` as an exact metric selector for `scores[].metric`.
5. For legacy JSON rows with a top-level `score`, use `doc_score_field=score` and `doc_metric=legacy_score`.
6. Apply `transform` to the external score before computing deltas.
7. Preserve `alignment_status`, `transform`, and `notes` in output tables so downstream readers can audit the comparison.

## Acceptance Checklist

- [ ] Every Harbor benchmark column has one row in `data/metadata/benchmark_metric_alignment.csv`.
- [ ] Every row has `matrix_column`, `info_stem`, `benchmark_name`, `alignment_status`, `transform`, and `include_in_comparison`.
- [ ] Every non-identity transform is written as `lambda x: ...`.
- [ ] Every included row has a specific `doc_metric` or documented `legacy_score`.
- [ ] Every excluded row explains why in `notes`.
- [ ] All `include_in_comparison=true` rows have been spot-checked by a second reviewer.
- [ ] Pipeline comparison no longer uses heuristic metric fallback.


# Appendix

Note from leaderboard:
Step 1 — What counts as a "valid trial": a trial that either (a) finished cleanly with reward IS NOT NULL and exception_info IS NULL, or (b) ended with one of the three tolerated exceptions: RewardFileNotFoundError, AgentTimeoutError, or VerifierTimeoutError (the task ran but didn't produce a valid reward — treated as "finished with default worst score"). Trials that have any other exception (e.g. NonZeroAgentExitCodeError, DaytonaRateLimitError, CancelledError) are not counted, even if the reward column happens to hold a value. For these exceptions we substitute a per-benchmark floor: algotune → 1.0 (no speedup), sldbench → -1, all others → 0. Trials from different owners running the same (benchmark, task, model, agent) are pooled together — owner does not split the group.
Step 2 — Pick up to 5 trials per task: for each (benchmark, task, model, agent), sort its valid trials by started_at descending and take the most recent ones. Keep all of them if there are 5 or fewer; keep only the latest 5 if there are more. Result: kept = min(valid_trial_count, 5), a number between 0 and 5.
Step 3 — Decide if the task scores:
kept ≥ 3 → task qualifies; its per-task score = average of those kept rewards (denominator = kept, i.e. 3, 4, or 5).
kept < 3 → task is treated as not run.
Step 3b — Standard deviation (±): for each qualified task, score_std = STDDEV_SAMP of the kept trial rewards measures noise / reliability.
Low std (e.g. ≤ 0.05) → score is stable across runs.
High std (e.g. ≥ 0.3) → score is noisy; differences between models may not be meaningful.
The displayed ± value is transformed to match the score's scale:
algotune: delta method — std_display ≈ std_raw / (x × (ln(x)+1)²) where x = mean raw reward.
sldbench, ineqmath: linear — std_display = std_raw × 0.5.
All others: std_display = std_raw (no transform).
Step 4 — Benchmark-level score (default view): arithmetic mean of per-task scores across qualified tasks. Exception: algotune uses the harmonic mean of per-task speedup scores (matches the AlgoTune paper). The ± shown is the average of per-task stds (typical noise level across tasks). A combo's cell is shown only if it qualified on every paper task (n_qualified_tasks ≥ paper_set_size); otherwise the cell is left blank.
Task-level view: shows the per-task scores and ± directly. Tasks with kept < 3 are omitted. Click any score cell to open a new tab showing the full trial traces (result.json, config.json, trial.log, etc.) for that (task, model, agent) — one tab per trial (up to 5).
Click benchmark-level cells to open a new tab with the job-level aggregate (from job.stats): per-reward distribution, exception counts, mean metric, plus raw stats / metrics / config. If the combo was run as multiple jobs (different phase/batch), each job gets its own tab.
Download ⬇ button on a benchmark header: exports a CSV (task_name, model, agent, trial_rank, trial_id, reward, started_at, trial_uri) listing exactly the trials that fed into the benchmark's scores — the same latest-up-to-5 set, only for tasks that qualified (kept ≥ 3). Rows where trial_uri IS NULL are still included so the set matches scoring 1-to-1; filter them out before bulk-downloading with curl/wget.
Only the 26 official model x agent combos are included.
Score transforms (to map all values to 0-1 range):
algotune: reward is a speedup factor (can be >> 1). Transformed via y = ln(max(1, x)) / (ln(max(1, x)) + 1).
sldbench, ineqmath: reward can be negative (-1 to 1). Transformed via y = (x + 1) / 2.
All other benchmarks: raw reward (already 0-1).
