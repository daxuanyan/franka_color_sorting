# LLM recovery-supervisor evaluation

Matched cases: 60

| Metric | Rule reference | DeepSeek supervisor |
|---|---:|---:|
| Action accuracy | 100.0% | 100.0% |
| Exact action + reason | 100.0% | 100.0% |
| Policy-valid output | 100.0% | 100.0% |
| Unsafe retry rate | 0.0% | 0.0% |
| Mean latency | 0.00 ms | 1720.93 ms |
| Total tokens | 0 | 72284 |

## Exact accuracy by category

| Category | Rule reference | DeepSeek supervisor |
|---|---:|---:|
| already_completed | 100.0% | 100.0% |
| budget_exhausted | 100.0% | 100.0% |
| retry_after_observation | 100.0% | 100.0% |
| sensor_transient | 100.0% | 100.0% |
| stale_motion | 100.0% | 100.0% |
| success | 100.0% | 100.0% |
| unknown_or_untrusted | 100.0% | 100.0% |

The evaluator performs no robot motion. Raw simulator/tool log text is excluded from the
model context; only typed feedback is provided. Every model decision is checked again by
a deterministic local transition policy before an observation or retry can be executed.
