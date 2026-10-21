# Matched benchmark comparison

Matched scenarios: 10

| Metric | Calibrated gated wrist | Overhead only |
|---|---:|---:|
| Success rate | 100.0% | 90.0% |
| Success rate 95% CI | 72.2%–100.0% | 59.6%–98.2% |
| Detection rate | 100.0% | 100.0% |
| Mean duration (all trials) | 26.44 s | 25.10 s |
| Mean duration (successful trials) | 26.44 s | 25.98 s |
| Mean overhead XY error | 2.79 mm | 2.75 mm |
| P95 overhead XY error | 13.21 mm | 13.21 mm |
| Mean wrist XY error | 0.99 mm | — |
| P95 wrist XY error | 1.94 mm | — |
| Recovery events | 0 | 1 |
| Wrist corrections accepted | 13 | 0 |
| Wrist corrections rejected | 0 | 0 |

All columns use identical per-trial seeds, target order, object layouts, camera poses, and lighting.

## Matched outcome analysis

| Comparison | First only succeeds | Second only succeeds | Exact McNemar p |
|---|---:|---:|---:|
| Calibrated gated wrist vs Overhead only | 1 | 0 | 1.0000 |
