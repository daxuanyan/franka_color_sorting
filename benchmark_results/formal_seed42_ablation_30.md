# Matched benchmark comparison

Matched scenarios: 30

| Metric | Gated wrist | Overhead only | Ungated wrist |
|---|---:|---:|---:|
| Success rate | 96.7% | 96.7% | 73.3% |
| Success rate 95% CI | 83.3%–99.4% | 83.3%–99.4% | 55.6%–85.8% |
| Detection rate | 100.0% | 100.0% | 100.0% |
| Mean duration (all trials) | 31.39 s | 25.96 s | 24.77 s |
| Mean duration (successful trials) | 31.82 s | 26.26 s | 26.99 s |
| Mean overhead XY error | 2.95 mm | 2.96 mm | 2.95 mm |
| P95 overhead XY error | 15.79 mm | 15.79 mm | 15.79 mm |
| Mean wrist XY error | 16.59 mm | — | 17.23 mm |
| P95 wrist XY error | 19.90 mm | — | 19.94 mm |
| Recovery events | 1 | 1 | 8 |
| Wrist corrections accepted | 0 | 0 | 38 |
| Wrist corrections rejected | 41 | 0 | 0 |

All columns use identical per-trial seeds, target order, object layouts, camera poses, and lighting.

## Matched outcome analysis

| Comparison | First only succeeds | Second only succeeds | Exact McNemar p |
|---|---:|---:|---:|
| Gated wrist vs Overhead only | 0 | 0 | 1.0000 |
| Gated wrist vs Ungated wrist | 8 | 1 | 0.0391 |
| Overhead only vs Ungated wrist | 8 | 1 | 0.0391 |
