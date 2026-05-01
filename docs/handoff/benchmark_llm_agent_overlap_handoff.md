# benchmark LLM/agent overlap 统计交接笔记

生成时间：2026-04-25

这份文件记录今天新增的 overlap 统计脚本、输出表和当前口径，方便明天继续检查/修正 `benchmark_info_jobs/*.json` 或调整统计定义。

## 用户问题

Harbor 里有一批 benchmark，并且 `data/raw/benchmark_level_matrix.csv` 记录了我们评测过的 `(model, agent)`。

`benchmark_info_jobs/*.json` 里记录了原 benchmark 发布时/leaderboard 上的 model 和 `system_description`。需要统计：

- 有多少原始 benchmark 测了 LLM，即顶层 `used_llm_or_agent` 是 `llm` 或 `both`。
- 这些 benchmark 里，有多少和 Harbor model 有 overlap。
- 有多少原始 benchmark 测了 agent，即顶层 `used_llm_or_agent` 是 `agent` 或 `both`。
- 这些 benchmark 里，有多少和 Harbor agent 有 overlap；agent 名来自 result-level `system_description`。

后来用户要求只从 benchmark 角度看，做一个 3x3 表：

|  | 有和我们 overlap | 有和我们不同 | 合计 |
|---|---:|---:|---:|
| 有测 LLM | 11 | 38 | 48 |
| 有测 agent | 12 | 40 | 51 |
| 合计 | 21 | 69 | 73 |

注意：右侧和底部的合计都是 benchmark 去重后的集合并集，不是直接把行/列相加。

## 新增/生成的文件

新增脚本：

- `quantitative_study/scripts/benchmark_llm_agent_overlap.py`

生成输出：

- `output/quantitative/benchmark_llm_agent_overlap_summary.csv`
- `output/quantitative/benchmark_llm_agent_overlap_by_benchmark.csv`
- `output/quantitative/benchmark_llm_agent_overlap_names.csv`
- `output/quantitative/benchmark_llm_agent_overlap_3x3.csv`

运行命令：

```bash
python3 quantitative_study/scripts/benchmark_llm_agent_overlap.py
```

验证命令：

```bash
python3 -m py_compile quantitative_study/scripts/benchmark_llm_agent_overlap.py
```

## 当前口径

### LLM overlap

- 只把 `system_description` 为空/null 的 result row 视为 bare LLM model。
- 用 result row 的 `model` 去匹配 Harbor matrix 的 `model` 列。
- 顶层 `used_llm_or_agent` 为 `llm` 或 `both` 的 benchmark 才进入 LLM 行。

### Agent overlap

- 只把 `system_description` 非空的 result row 视为 agent/system row。
- 用 result row 的 `system_description` 去匹配 Harbor matrix 的 `agent` 列。
- 顶层 `used_llm_or_agent` 为 `agent` 或 `both` 的 benchmark 才进入 agent 行。

### “有和我们不同”

当前定义为：在对应维度上，这个 benchmark 至少有一个外部名称没有匹配到 Harbor。

- LLM 行：至少一个 bare LLM model 名称没有匹配到 Harbor model。
- Agent 行：至少一个 `system_description` 没有匹配到 Harbor agent。
- 合计行：LLM/agent 两种情况取 benchmark 并集。

这意味着一个 benchmark 可以同时算入“有和我们 overlap”和“有和我们不同”。例如某 benchmark 有 GPT-5.4 对上了 Harbor，但也有别的模型没对上。

## 匹配规则

脚本复用 `quantitative_study/pipeline/config.py` 里的：

- `MODEL_ALIASES`
- `AGENT_ALIASES`
- `MATRIX_COLUMN_TO_STEM`

脚本内部额外加了实用 agent alias 逻辑，不改动现有 pipeline 的 score comparison 逻辑：

- 添加了 `Claude Agent SDK` 等扩展匹配，抓取原带有括号后缀的名称到 `claude-code`
- 脚本中自带了健壮的正则后缀截断匹配，会自动剥离像 `@2.1.104` 等版本后缀，将带版本的 agent 名称自动归拢到 `claude-code` / `codex` / `terminus-2`
- 模型匹配会做：
  - 小写化
  - 下划线转连字符
  - 去掉末尾括号，例如 `(xhigh)` / `(thinking high)`
  - 去掉末尾日期后缀，例如 `-2025-08-07`
  - alias 映射

## 当前统计结果

全量 `benchmark_info_jobs/*.json`：

- JSON 总数：73
- `used_llm_or_agent`: `llm=22`, `agent=25`, `both=26`

benchmark 角度：

- 有测 LLM：48 个，其中 11 个有 Harbor model overlap，38 个有非 Harbor model 名称。
- 有测 agent：51 个，其中 12 个有 Harbor agent overlap，40 个有非 Harbor agent/system 名称。
- 合计去重：73 个有效 benchmark，其中 21 个有至少一个 Harbor overlap，69 个有至少一个非 Harbor 名称。

唯一名称角度：

- 有测 LLM 的有效 benchmark 中，unique bare LLM model 名称 1295 个，其中 26 个外部名称匹配到 Harbor，归一到 8 个 Harbor model：
  - `claude-haiku-4-5-20251001`
  - `claude-opus-4-6`
  - `claude-sonnet-4-6`
  - `gemini-3-flash-preview`
  - `gemini-3.1-pro-preview`
  - `gpt-5-mini`
  - `gpt-5-nano`
  - `gpt-5.4`
- 有测 agent 的有效 benchmark 中，unique `system_description` 365 个，其中 29 个外部名称匹配到 Harbor，归一到 4 个 Harbor agent：
  - `claude-code`
  - `codex`
  - `gemini-cli`
  - `terminus-2`

只看能映射到 Harbor matrix column 的 benchmark：

- mapped benchmark：57 个
- 有测 LLM：40 个，其中 8 个有 Harbor model overlap。
- 有测 agent：36 个，其中 7 个有 Harbor agent overlap。

## 明天可继续处理

1. 检查 `benchmark_llm_agent_overlap_by_benchmark.csv` 里同时 `used_llm_or_agent=both` 但某一类 result row 数量为 0 的 benchmark。这些可能是源数据还没清理完，或顶层字段需要修正。
2. 如果用户觉得 “有和我们不同” 应该按 benchmark 是否完全无 overlap 来定义，需要改 `build_benchmark_perspective_rows()` 的集合逻辑。当前口径是“至少有一个不同名称”。
3. 如果要把该统计纳入正式 pipeline，可以把脚本逻辑移进 `quantitative_study/pipeline/analysis.py`，并在 `run_pipeline.py` 输出；今天为了避免碰已有 pipeline 改动，先做成独立脚本。
4. 当前 git working tree 里已有一些 pipeline 文件修改不是今天这个 overlap 脚本引入的；不要轻易 revert。今天新增/更新的是 overlap 脚本、四个 output CSV，以及这份 handoff doc。