# Benchmark Metric Alignment Handoff

Generated: 2026-04-25

This handoff summarizes today's manual benchmark metric alignment work. The key workflow was: read each benchmark JSON and Harbor adapter/task evidence one by one, discuss the alignment, get approval, then edit only the table/notes. We did not run the analysis pipeline.

## Main Files

- `data/metadata/benchmark_metric_alignment.csv`: main alignment table.
- `benchmark_metric_uncertainties.md`: follow-up questions for benchmarks that need owner/adapter confirmation.
- `benchmark_info_jobs/swegym.json`: new placeholder JSON added for SWE-Gym.
- `benchmark_info_jobs/swt_bench.json`: live leaderboard note/source URL clarified.

Other modified files shown by `git status` under `quantitative_study/` appear unrelated to today's manual metric-alignment pass and were not touched during this review.

## Workflow To Continue

- Do not run the existing pipeline while doing manual alignment review.
- For each benchmark, inspect the JSON, Harbor adapter metadata/parity, and current task count in `data/raw/task_level_matrix.csv`.
- Present the reasoning first, then wait for approval before editing.
- If metric semantics match but Harbor uses a subset, use `alignment_status=subset_or_variant`; decide `include_in_comparison` case by case and document the subset in notes.
- If the external JSON does not provide a same-task metric, leave `doc_metric` blank and usually exclude.
- Keep uncertain included benchmarks in `benchmark_metric_uncertainties.md` so they can be checked tomorrow.

## Decisions Already Made

### User-Edited Before This Stretch

- `dacode`: included; `doc_metric=total`. Harbor score distribution matched JSON `total`, not near-perfect `completion_rate`.
- `featbench`: excluded; `unknown`. JSON has empty `results_over_time`.
- `featurebench-modal`: excluded; `unknown`. Harbor has 185 tasks, not official Full 200 or Lite 30.

### Confirmed / Updated Today

- `financeagent_terminal`: excluded; `subset_or_variant`. Harbor uses 50-task public validation LLM-judge accuracy; paper has 537 samples and Vals leaderboard split is not confirmed.
- `gaia2`: excluded; `subset_or_variant`. Harbor parity is 100/800 stratified subset; paper/CLI split definitions differ.
- `ineqmath`: included; `transformed`. Raw reward can be -1/0/1, matrix uses `(x + 1) / 2`; `doc_metric=accuracy_answer`.
- `kumo`: excluded; `unknown`. JSON only easy/hard success rates; Harbor has 212 env/seed tasks with unclear split mapping.
- `lawbench`: excluded; `subset_or_variant`. Harbor uses original family evaluators, but current 181-task mixed zero/one-shot subset does not map cleanly to official overall metrics.
- `livecodebench`: excluded; `unknown`. Live/dynamic benchmark, no fixed external metric selected for Harbor's sampled set.
- `mlgym`: excluded; `not_comparable`. Harbor uses 11 continuous task-specific rewards; JSON reports AUP@4/heterogeneous per-task metrics.
- `pixiu`: excluded; `not_comparable`. PIXIU/FinBen uses task-specific metrics; no single overall metric maps to Harbor's uneven 103-task subset.
- `research-code-bench`: excluded; `not_comparable`. JSON/paper report scaled LoC-weighted pass@1; Harbor matrix is unweighted snippet success rate.
- `scicode`: included; `subset_or_variant`. Harbor uses 80-task without-background setup and macro per-problem sub-step accuracy; closest JSON metric is paper standard subproblem pass@1.
- `seal0`: included; `direct`. Harbor covers 111 Seal-0 tasks and original SealQA LLM-as-judge accuracy.
- `skillsbench`: excluded; `unknown`. JSON separates no-skills/curated/self-generated conditions; current matrix has 75 tasks vs 84 reported, no local adapter metadata/parity.
- `swegym`: excluded; `unknown`. Added placeholder `benchmark_info_jobs/swegym.json`. Adapter/parity targets SWE-Gym Lite 230/full 2438, but current matrix has only 3 tasks and no external result.
- `swesmith`: excluded; `not_comparable`; `doc_metric` left blank. JSON structured results are SWE-bench Verified paper comparison, not SWE-smith task scores. Harbor matrix has 100 SWE-smith tasks and adapter parity has 100-task Resolved Rate.
- `swtbench`: included; `subset_or_variant`. Metric matches `Success Rate (S)`, but current matrix has 50 tasks vs adapter parity Verified 433. Live leaderboard has modern Verified scores in same broad range, no exact model+agent overlap.
- `terminal-bench`: included; `direct`. Full 89-task matrix matches Terminal-Bench 2.0; metric is 0-1 accuracy; overlapping leaderboard model+agent rows are on the same scale.
- `usaco`: included for now; `subset_or_variant`. Metric matches pass@1/pass rate, but current matrix has 100 tasks vs adapter parity 304/full 307 and scores are much higher. Needs subset confirmation.
- `widesearch`: included for now; `subset_or_variant`. Metric implementation is strongly aligned by full 200-task parity; current matrix has 100 tasks (50 en + 50 zh) per experimenter confirmation. Scores can be higher; temporal sensitivity also noted.

## Follow-Up Questions For Tomorrow

These are also recorded in `benchmark_metric_uncertainties.md`.

- For USACO, confirm why Harbor matrix uses 100/304 tasks and whether the subset is intentionally easier.
- For WideSearch, confirm criteria for the 100/200-task subset (50 en + 50 zh) and whether it is representative.

## Notes On Evidence Used

- Task counts were checked from `data/raw/task_level_matrix.csv`.
- Score ranges were checked from `data/raw/benchmark_level_matrix.csv` for SWT-Bench, Terminal-Bench, USACO, and WideSearch.
- Local adapter evidence used:
  - `harbor/adapters/financeagent`
  - `harbor/adapters/lawbench`
  - `harbor/adapters/research-code-bench`
  - `harbor/adapters/swegym`
  - `harbor/adapters/swesmith`
  - `harbor/adapters/swtbench`
  - `harbor/adapters/usaco`
  - `harbor/adapters/widesearch`
- Live web check used for SWT-Bench leaderboard: `https://swtbench.com/?results=verified`.

## Validation Done

- `benchmark_info_jobs/swegym.json` was validated with `python3 -m json.tool`.
- `data/metadata/benchmark_metric_alignment.csv` was parsed with Python `csv.DictReader`.
- No `uv run habor-analyze` or other pipeline command was run.

## Suggested Next Step

Start tomorrow by resolving the included-but-uncertain subset benchmarks first:

1. `usaco`: confirm 100-task subset source and difficulty.
2. `widesearch`: confirm 10-task subset source and temporal sensitivity handling.
3. `swtbench`: confirm 50-task subset source.
4. `scicode`: confirm macro vs micro interpretation of the paper metric.

After those are settled, revisit the excluded uncertain items in `benchmark_metric_uncertainties.md` and only then run any comparison/pipeline code.
