# Matched benchmark comparison

Matched scenarios: 30

| Metric | Calibrated gated wrist | Overhead only |
|---|---:|---:|
| Success rate | 100.0% | 96.7% |
| Success rate 95% CI | 88.6%–100.0% | 83.3%–99.4% |
| Detection rate | 100.0% | 100.0% |
| Mean duration (all trials) | 27.38 s | 27.08 s |
| Mean duration (successful trials) | 27.38 s | 27.39 s |
| Mean overhead XY error | 2.91 mm | 2.90 mm |
| P95 overhead XY error | 15.82 mm | 15.82 mm |
| Mean wrist XY error | 0.87 mm | — |
| P95 wrist XY error | 1.41 mm | — |
| Recovery events | 0 | 1 |
| Wrist corrections accepted | 40 | 0 |
| Wrist corrections rejected | 1 | 0 |

All columns use identical per-trial seeds, target order, object layouts, camera poses, and lighting.

## Matched outcome analysis

| Comparison | First only succeeds | Second only succeeds | Exact McNemar p |
|---|---:|---:|---:|
| Calibrated gated wrist vs Overhead only | 1 | 0 | 1.0000 |
