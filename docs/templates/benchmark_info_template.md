# Template Field Types and Filling Guide

```jsonc
{
  "name":              // string, required. Official benchmark name
  "category":          // string, optional. One of:
                       //   Repo-level Software Engineering | Competitive/Function-level Programming | Math | Reasoning | Science & Research | Agentic/Interactive | Code Performance | Machine Learning | Misc
  "used_llm_or_agent": // string, required. llm | agent | both

  "links": {
    // Please collect as many of the following links as possible. If a link truly cannot be found, set it to null — but search thoroughly first: official websites, GitHub READMEs, HuggingFace pages, and paper abstract pages usually cross-link to each other.
    "website":    // string | null. Project homepage (often *.github.io, *.ai, or a HuggingFace Space)
    "leaderboard": // string | null. Leaderboard URL. If the leaderboard lives on the website, put the same URL
    "paper":      // string | null. Paper URL (arXiv / OpenReview / conference proceedings all fine)
    "github":     // string | null, optional but strongly recommended. Source code repo URL
    "dataset":    // string | null, optional but strongly recommended. Dataset URL (usually HuggingFace datasets)
  },

  "meta": {
    "release_date": // string, required. YYYY-MM or YYYY-MM-DD
    "num_tasks":    // integer | null, optional
  },

  "evaluation": {
    "primary_metric":       // string, required. e.g., "% Resolved" / "pass@1" / "accuracy" / "F1"
    "harbor_aligned_metric": // string | null. Leave null initially; fill after harbor experiments finish| https://harborsubabase.vercel.app/leaderboard can be used as reference, add notes if anything abnormal.
  },

  "results_over_time": [    // array. Release baseline is REQUIRED; current SOTA is recommended but not required. Include additional time points if available.
    {
      "date":        // string, required. YYYY-MM or YYYY-MM-DD
      "source_type": // string, required. paper | leaderboard | blog | other
      "source_url":  // string, required. URL where this data comes from
      "results": [   // array, required. (model, scores) rows at this time point — copy as many as the leaderboard shows
        {
          "model":              // string, required. Value from the leaderboard's "Model" column
          "effort":             // string | null, optional. The inference-time compute effort / thinking level used for this result.
                                //   Examples: "high", "medium", "low", "xhigh", "max", "default", "thinking", "non-thinking".
                                //   Copy from leaderboard if shown (e.g. "o3 (high)" → effort = "high").
                                //   For standard single-config runs or when the leaderboard does not distinguish effort, fill null.
          "system_description": // string | null, required. The agent/scaffold wrapped around the LLM to help it complete the task.
                                //   Examples: mini-SWE-agent, SWE-agent, Agentless, OpenHands, Claude Code, Codex.
                                //   For pure LLM evaluation (no scaffold), fill null.
          "scores": [           // array, required. One object per metric/column (pass@1, cost, latency, …). Single-number leaderboards → one element.
            {
              "metric":       // string, required. Short name — prefer `evaluation.primary_metric` for the main column, or copy the column header
              "value":        // number, required. For percentage metrics, use [0, 1] (e.g., 83% → 0.83). For other scales, put units in `unit` or top `note`
              "unit":         // string | null, optional. e.g. "ms", "USD", "tokens" when `value` is not a plain rate in [0,1]
            }
          ]
        }
      ],
      "note":        // string | null, optional. Any special notes about this experiment group
    }
  ],

  "notes":           // string | null, optional. Top-level notes (subsets / variants / known issues, etc.)
}
```

---

## `effort` Copy Rules

**Principle: Split out the thinking/compute level when the leaderboard shows it. Otherwise null.**

| Leaderboard shows | `model` | `effort` |
|---|---|---|
| "o3 (high)" | `"o3"` | `"high"` |
| "o3" (no qualifier) | `"o3"` | `null` |
| "gpt-5.4-2026-03-05 (xhigh thinking)" | `"gpt-5.4-2026-03-05"` | `"xhigh"` |
| "claude-opus-4-6-thinking-max" | `"claude-opus-4-6"` | `"max"` |
| "claude-opus-4-6 (Non-Thinking)" | `"claude-opus-4-6"` | `"non-thinking"` |
| "Gemini 2.5 Pro" (no thinking qualifier) | `"Gemini 2.5 Pro"` | `null` |

---

## `system_description` Copy Rules

**Principle: Copy the leaderboard as-is. Do not make classification judgments.**

| Leaderboard shows | `model` | `system_description` |
|---|---|---|
| Only "GPT-4o" (no scaffold column) | `"GPT-4o"` | `null` |
| "GPT-4o" + "Agentless" in two columns | `"GPT-4o"` | `"Agentless"` |
| "Claude 3.7 + mini-SWE-agent" combined | `"Claude 3.7 Sonnet"` | `"mini-SWE-agent"` |
| "Claude Code" as a product | `"Claude Code"` | `"Claude Code"` |
| "Codex" as a product | `"Codex"` | `"Codex"` |
| "h2oGPTe" as a product | `"h2oGPTe"` | `"h2oGPTe"` |
| "Magentic-1 (GPT-4o)" | `"GPT-4o"` | `"Magentic-1"` |

---

## Acceptance Checklist

- [ ] `name`, `category`, `used_llm_or_agent` all filled
- [ ] `links.website`, `links.leaderboard`, `links.paper` all filled
- [ ] `links.github`, `links.dataset` filled whenever available (most benchmarks have them)
- [ ] `meta.release_date` filled
- [ ] `evaluation.primary_metric` filled
- [ ] `results_over_time` has at least the release baseline; additional points (including current SOTA) are recommended
- [ ] Every record has `date`, `source_type`, `source_url`
- [ ] Every result has `model`, `system_description`, and a non-empty `scores` array
- [ ] `effort` filled when the leaderboard distinguishes thinking/compute levels; otherwise null
- [ ] For percentage metrics, every percentage `value` in `scores` is a float in [0, 1] (not strings, not 0–100)
- [ ] JSON is valid (check with https://jsonlint.com/)
- [ ] Filename = `{benchmark_name_lowercase_underscore}.json`
