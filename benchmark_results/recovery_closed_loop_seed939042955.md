# Matched benchmark comparison

Matched scenarios: 1

| Metric | Recovery disabled | Closed-loop recovery |
|---|---:|---:|
| Success rate | 0.0% | 100.0% |
| Success rate 95% CI | 0.0%–79.3% | 20.7%–100.0% |
| Detection rate | 100.0% | 100.0% |
| Mean duration (all trials) | 16.84 s | 29.14 s |
| Mean duration (successful trials) | — | 29.14 s |
| Mean overhead XY error | 11.33 mm | 11.33 mm |
| P95 overhead XY error | 21.97 mm | 21.97 mm |
| Mean wrist XY error | — | — |
| P95 wrist XY error | — | — |
| Recovery events | 1 | 1 |
| Recovery attempts | 0 | 1 |
| Successful recoveries | 0 | 1 |
| Wrist corrections accepted | 0 | 0 |
| Wrist corrections rejected | 0 | 0 |

All columns use identical per-trial seeds, target order, object layouts, camera poses, and lighting.

## Matched outcome analysis

| Comparison | First only succeeds | Second only succeeds | Exact McNemar p |
|---|---:|---:|---:|
| Recovery disabled vs Closed-loop recovery | 0 | 1 | 1.0000 |
