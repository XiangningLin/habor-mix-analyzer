# Benchmark Metric Alignment Uncertainties

This file tracks benchmark alignment questions that need follow-up with benchmark or adapter owners before the row should be included in Harbor-vs-external comparisons.

## FinanceAgent Terminal

- Harbor column: `financeagent_terminal`
- Current status: `subset_or_variant`, excluded.
- Question: Harbor uses the 50-task public validation set and reports 0/1 LLM-judge accuracy. The paper reports metrics on the complete 537 samples, and the later Vals leaderboard has an `accuracy` field whose split is not explicitly confirmed as the same 50-task public validation set.

## FeatureBench Modal

- Harbor column: `featurebench-modal`
- Current status: `unknown`, excluded.
- Question: Harbor evaluated 185 tasks, which does not exactly match external Full (200 tasks) or Lite (30 tasks). Need the adapter maintainer to confirm the exact task subset.

## GAIA2

- Harbor column: `gaia2`
- Current status: `subset_or_variant`, excluded.
- Question: Paper `pass@1 (overall)` is a 7-split overall; live CLI leaderboard uses a 5-split overall; Harbor parity used a 100/800 stratified subset. Need a locked target split before comparison.

## KUMO

- Harbor column: `kumo`
- Current status: `subset_or_variant`, included (resolved 2026-04-26).
- Resolution: Harbor matrix = 202 KUMO(easy, t4-a6) + 10 KUMO(hard, t12-a16) = 212 tasks per `harbor/adapters/kumo/README.md`. Ratio matches paper pin (5050:250). Added `weighted_success_rate_harbor_subset = (202*easy + 10*hard)/212` to every entry in `benchmark_info_jobs/kumo.json` via `quantitative_study/scripts/add_kumo_weighted.py` and pointed the alignment CSV at the new field. Note: adapter README still recommends reporting easy/hard separately; the weighted field is for cross-benchmark Harbor-vs-external comparison only.

## LawBench

- Harbor column: `lawbench`
- Current status: `subset_or_variant`, excluded.
- Question: Harbor calls original LawBench family-specific evaluation functions, but the current matrix contains a mixed 181-task subset with 100 zero-shot and 81 one-shot tasks. It does not map cleanly to official `overall-zero-shot`, `overall-one-shot`, full 1000-task LawBench, or the 120-task parity subset.

## LiveCodeBench

- Harbor column: `livecodebench`
- Current status: `unknown`, excluded.
- Question: LiveCodeBench is a live/dynamic benchmark whose external leaderboard snapshots can change as new problems are added. The current external `pass@1 (AA)` snapshot is not guaranteed to match Harbor's fixed sampled task set, so no external metric is selected for now.

## MLGym-Bench

- Harbor column: `mlgym`
- Current status: `not_comparable`, excluded.
- Question: Harbor currently uses 11 task-specific continuous rewards aggregated as mean reward. External JSON reports official AUP@4 and heterogeneous per-task metrics. Need an agreed comparison target, such as binary pass rate, continuous mean reward parity, or official AUP@4, before inclusion.

## PIXIU

- Harbor column: `pixiu`
- Current status: `not_comparable`, excluded.
- Question: PIXIU/FinBen has task-specific metrics and no single overall external score. Current Harbor matrix contains 103 unevenly distributed tasks across 29 subcategories, while adapter parity uses 435 tasks. Need a subcategory-level or macro-average comparison design before inclusion.

## SciCode

- Harbor column: `scicode`
- Current status: `subset_or_variant`, included.
- Question: Harbor uses macro per-problem sub-step accuracy on the 80-task without-background setup. The closest JSON metric is `subproblem pass@1 (paper, standard)`, but confirm whether the paper number is macro per main problem or micro over subproblems.

## SkillsBench

- Harbor column: `skillsbench`
- Current status: `unknown`, excluded.
- Question: External results report no-skills, curated-skills, and self-generated-skills pass rates over 84 evaluated tasks. Current Harbor matrix has 75 tasks and no local adapter metadata/parity to identify the skills condition or subset.

## SWE-Gym

- Harbor column: `swegym`
- Current status: `unknown`, excluded.
- Question: No matching `benchmark_info_jobs` JSON file exists. The local adapter/parity targets SWE-Gym Lite 230-task `Resolved Rate`, and the full adapter supports 2438 tasks, but the current Harbor matrix contains only 3 `swegym` tasks. Need the source/external metric file and task-subset mapping before selecting a comparison metric.

## SWT-Bench

- Harbor column: `swtbench`
- Current status: `subset_or_variant`, included.
- Question: Metric semantics match SWT-Bench `Success Rate (S)`, and live Verified leaderboard scores are in the same general range as Harbor. However, the current Harbor matrix contains only 50 `swtbench` tasks, while adapter parity validates the full 433-task Verified split. There is no exact overlapping model+agent baseline in the current matrix.

## USACO

- Harbor column: `usaco`
- Current status: `subset_or_variant`, included.
- Question: Metric semantics match USACO pass@1/pass rate, and adapter parity validates the 304-task adapted full set against the original benchmark. Current Harbor matrix contains only 100 `usaco` tasks, not the 304/307-task set, and Harbor scores are substantially higher than published/HAL full-set results. Need to confirm the 100-task subset mapping and whether it is intentionally easier.

## WideSearch

- Harbor column: `widesearch`
- Current status: `subset_or_variant`, included.
- Question: Metric implementation is strongly aligned by full 200-task parity on `Mean Item F1`. Per experimenter confirmation, the current Harbor matrix contains 100 `widesearch` tasks (50 en + 50 zh), not the full 200. Scores can be higher on this subset, and WideSearch has some temporal sensitivity. Need to confirm the criteria for selecting these 100/200 tasks before making full-benchmark claims.
