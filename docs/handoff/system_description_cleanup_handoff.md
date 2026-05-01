# system_description 清理交接笔记

生成时间：2026-04-24

这份文件用于新对话继续清理 `benchmark_info_jobs/*.json`。当前任务不是重算 overlap，而是先把源数据里的 `used_llm_or_agent` 和 `system_description` 格式改对。

## 核心口径

来自 `benchmark_info_template.md`：

- `used_llm_or_agent` 只能是 `llm` / `agent` / `both`。
- `system_description` 应该写“包在 LLM 外面的 agent/scaffold”。
- 纯 LLM evaluation 没有 scaffold 时，`system_description` 应该是 `null`。
- provider / 厂商 / 机构名不是 agent，例如 `OpenAI`, `Anthropic`, `Google DeepMind`。
- prompt/method/config 不是 agent，例如 `CoT`, `Prompt`, `FC`, `RAG`, `ReAct`, `BM25`。
- 如果原 `system_description` 有有用信息但不是 agent，通常移动到同一条 result 的 `note`。
- 用户偏好：`note` 直接写具体值，例如 `"note": "FC"`，不要写 `"Original system_description: FC"` 这种前缀。
- 所有后续修改必须逐个文件、逐类值询问用户确认，不要批量自作主张。
- 尽量用局部 patch，避免 `json.dumps(..., indent=2)` 重写整个文件导致 diff 噪声。

## 已确认并已改的文件

### `benchmark_info_jobs/gpqa_diamond.json`

- 顶层是 `used_llm_or_agent: "llm"`。
- 所有 result 的 `system_description` 已改为 `null`。
- 原来的 provider/method 字段包括 `OpenAI`, `Anthropic`, `Google DeepMind`, `Few-Shot CoT`, `with search...` 等。
- 用户明确指出这不是 agent benchmark，`system_description` 应为 `null`。
- 当前验证：所有 264 条 result 的 `system_description` 都是 `null`。

### `benchmark_info_jobs/aime.json`

- 顶层是 `used_llm_or_agent: "llm"`。
- 所有 result 的 `system_description` 已改为 `null`。
- 原先 17 条非空值已移动到 result-level `note`，但这里当时用了较长前缀：`Original system_description: ...`。
- 已验证 JSON 合法。
- 注意：如果后续想统一 note 风格，可再问用户是否把这些 note 改成无前缀版本。

### `benchmark_info_jobs/arc_agi_2.json`

- 顶层是 `used_llm_or_agent: "llm"`。
- 用户手动处理过 `Refinement` 相关条目。
- 已处理：
  - `CoT` -> `system_description: null`, `note: "Original system_description: CoT"`
  - `Base LLM` -> `system_description: null`, `note: "Original system_description: Base LLM"`
  - `program synthesis (ARC Prize 2024 winner)` / `program synthesis (ARC Prize 2020 winner)` -> 直接改成 `null`，不留 note。
- 当前验证：50 条 result 的 `system_description` 都是 `null`。
- 注意：这里 note 也还带 `Original system_description:` 前缀；如需统一，后续单独询问。

### `benchmark_info_jobs/bfcl.json`

- 顶层是 `used_llm_or_agent: "llm"`。
- 非空 `system_description` 已全部改成 `null`。
- 原值已无前缀写入 result-level `note`：
  - `FC`: 66 条
  - `Prompt`: 34 条
  - `FC thinking`: 1 条
  - `Prompt + Thinking`: 1 条
- 当前验证：109 条 result 的 `system_description` 都是 `null`。

### `benchmark_info_jobs/researchcodebench.json`

- 顶层是 `used_llm_or_agent: "llm"`。
- 原 `system_description` 是模型 API / provider / cutoff / pricing 描述。
- 用户确认这些厂家描述不用留。
- 32 条 result 的 `system_description` 已直接改成 `null`，没有 result-level note。
- block-level note 仍保留，说明原字段含义。

### `benchmark_info_jobs/humanevalfix.json`

- 原顶层是 `both`，用户指出没有任何 agent。
- 已改：
  - `used_llm_or_agent`: `both` -> `llm`
  - 9 条 result 的 `system_description` -> `null`
  - 原值移入无前缀 result-level `note`，例如 `permissive`, `non-permissive, trained on OpenAI outputs`。
- 已验证 JSON 合法。

### `benchmark_info_jobs/aider_polyglot.json`

- 原顶层是 `llm`，但所有 result 的 `system_description` 都是 `aider`。
- 用户确认顶层改成 `agent` 就对了。
- 已改：
  - `used_llm_or_agent`: `llm` -> `agent`
  - `system_description: "aider"` 保留。

### `benchmark_info_jobs/officeqa.json`

- 原顶层是 `agent`，但同时有 agent rows 和 bare LLM baseline rows。
- 已改：
  - `used_llm_or_agent`: `agent` -> `both`
  - 12 条 `LLM baseline (no agent), ...` 改为 `system_description: null`
  - 原 baseline 描述写入无前缀 result-level `note`
  - 真实 agent/scaffold 保留，例如 `Claude Agent SDK (...)`, `OpenAI Codex CLI (...)`, `Gemini CLI (...)`, `zaidishahbaz1/officeqa`。

### `benchmark_info_jobs/algotune.json`

- 发现：顶层原来是 `llm`，但所有 result 的 `system_description` 都是 `AlgoTuner`。
- 用户说“我改好了”，因此应假设用户已手动把顶层改成 `agent`。
- 后续可以只读确认，不要重复修改。

## 已撤销 / 不应再动

### `benchmark_info_jobs/swe_smith.json`

- 我曾把 `OpenHands; train size 3.3k` 等拆成 agent + note，并且不小心用 JSON writer 打散了格式。
- 用户要求撤销。
- 已恢复到 HEAD 内容；当前 `git diff -- benchmark_info_jobs/swe_smith.json` 为空。
- 后续如果还要处理这个文件，必须重新询问用户，不能沿用之前方案。

## 待审阅候选清单

下面是当前只读扫描后仍建议审阅的文件。不要直接改，先向用户提出一个文件和具体方案。

### 高优先级

#### `benchmark_info_jobs/gaia.json`

- 顶层：`used_llm_or_agent: "agent"`。
- 有 3 条 `system_description: null`，其余是 agent/system 名。
- 可能应改为 `both`，因为 release paper 里有 pure LLM baseline。
- 需先展示这些 null rows 让用户确认。

#### `benchmark_info_jobs/scienceagentbench.json`

- 顶层：`both`。
- 有明显非-agent method/config：
  - `Direct Prompting (without knowledge)` 6 条
  - `Direct Prompting (with knowledge)` 6 条
- 其他看起来像 scaffold/system，可先不动：
  - `SAB Self-Debug`
  - `HAL Generalist Agent`
  - `Self-Debug (without knowledge)`
  - `Self-Debug (with knowledge)`
  - `OpenHands CodeAct (...)`
- 建议方案：只处理 `Direct Prompting...`，改 `system_description: null`，note 写原值。先问用户。

#### `benchmark_info_jobs/usaco.json`

- 顶层：`both`。
- 混合了 null、方法模块和可能的 agent/system：
  - `Reflexion`
  - `Semantic Retrieval`
  - `Episodic Retrieval`
  - `Semantic + Episodic`
  - `Semantic + Reflexion`
  - `Episodic + Reflexion`
  - `Semantic + Episodic + Reflexion`
  - `HAL Generalist Agent`
- 需要用户定口径：retrieval/reflexion 是 system/scaffold 还是 method note。

### 中优先级

#### `benchmark_info_jobs/gaia2.json`

- 顶层：`agent`。
- 值分布：
  - `ReAct (temperature=0.5, 3 runs/scenario, 200-step cap)` 14 条
  - `OpenClaw 2026.4.1` 6 条
- `OpenClaw` 像 agent；`ReAct (...)` 更像方法配置。
- 建议先问用户是否把 `ReAct (...)` 改成 null + note，并保留 `OpenClaw`。

#### `benchmark_info_jobs/crmarena.json`

- 顶层：`both`。
- 全部非空：
  - `ReAct` 9
  - `Act` 7
  - `Function Calling (prompt)` 5
  - `Function Calling` 4
- 这些更像 interaction mode/method，而不是 agent 名。
- 但 CRMArena 可能把这些当作 agent policy/scaffold，需要用户定口径。

#### `benchmark_info_jobs/browsecomp_plus.json`

- 顶层：`agent`。
- `BM25` 22 条明显像 retrieval baseline。
- 其他如 `Qwen3-Embed-8B`, `Mixedbread Search`, `Reason-ModernColBERT` 等是 search/retriever system，是否算 scaffold 需要定口径。
- 先问用户是否只处理 `BM25`。

### 低优先级 / 谨慎

#### `benchmark_info_jobs/swe_bench_verified.json`

- 顶层：`agent`。
- 多数是真 agent/scaffold：`mini-SWE-agent`, `OpenHands`, `SWE-agent`, `Moatless Tools`, etc.
- 但 `RAG` 6 条、`Tools` 5 条可能不是干净 agent 名。
- 不建议先改，除非用户明确同意 `RAG`/`Tools` 的口径。

#### `benchmark_info_jobs/spider_2.json`

- 顶层：`both`。
- 大多是 benchmark-specific system/scaffold。
- `Reflexion` 1 条被 heuristic 标成 method/config，但是否作为 system 保留需要用户判断。
- 低优先级。

### Stub / 空结果文件

这些不是 `system_description` 清理问题，但格式上也不完整：

- `benchmark_info_jobs/featbench.json`
- `benchmark_info_jobs/livecodebench.json`
- `benchmark_info_jobs/scicode.json`

它们目前 results 为空，`used_llm_or_agent` 也为空或缺失。后续如果要完整数据格式，需要单独补。

## 当前 git status 中涉及的 benchmark_info_jobs

截至这份 handoff 写入前，已修改文件包括：

- `benchmark_info_jobs/aider_polyglot.json`
- `benchmark_info_jobs/aime.json`
- `benchmark_info_jobs/algotune.json`（用户手动改）
- `benchmark_info_jobs/arc_agi_2.json`
- `benchmark_info_jobs/bfcl.json`
- `benchmark_info_jobs/gpqa_diamond.json`
- `benchmark_info_jobs/humanevalfix.json`
- `benchmark_info_jobs/officeqa.json`
- `benchmark_info_jobs/researchcodebench.json`

`benchmark_info_jobs/swe_smith.json` 已撤销，不在 modified 列表。

## 关于 overlap 输出

之前生成过 `output/overlap_review/` 下的 benchmark-level overlap 结果：

- `benchmark_overlap_summary.md`
- `benchmark_overlap_summary.csv`
- `benchmark_overlap_details.csv`
- `benchmark_overlap_unmapped_external.csv`
- `agent_field_examples.md`

注意：这些 overlap 结果是在部分源数据清理前生成的，随着 `system_description` 被修正，统计已经过期。等源数据清理完成后再重算 overlap。

旧的全局 overlap 文件已经删除，只保留 benchmark-level 结果和这个 handoff。

## 下一步建议

新对话可以从下面任一项开始，建议优先 `gaia.json`：

1. `gaia.json`: 是否改顶层 `agent` -> `both`，并保留 null baselines。
2. `scienceagentbench.json`: 是否处理 `Direct Prompting...` 为 null + note。
3. `gaia2.json`: 是否处理 `ReAct (...)` 为 null + note。
4. `crmarena.json`: 是否把 `ReAct` / `Act` / `Function Calling` 当 method 而非 agent。
5. `browsecomp_plus.json`: 是否只处理 `BM25`。
