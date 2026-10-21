# Matched benchmark comparison

Matched scenarios: 1

| Metric | Overhead only | Continuous wrist servo |
|---|---:|---:|
| Success rate | 0.0% | 100.0% |
| Success rate 95% CI | 0.0%–79.3% | 20.7%–100.0% |
| Detection rate | 100.0% | 100.0% |
| Mean duration (all trials) | 16.84 s | 24.25 s |
| Mean duration (successful trials) | — | 24.25 s |
| Mean overhead XY error | 11.33 mm | 11.33 mm |
| P95 overhead XY error | 21.97 mm | 21.97 mm |
| Mean wrist XY error | — | 5.16 mm |
| P95 wrist XY error | — | 10.89 mm |
| Recovery events | 1 | 0 |
| Recovery attempts | 0 | 0 |
| Successful recoveries | 0 | 0 |
| Wrist corrections accepted | 0 | 5 |
| Wrist corrections rejected | 0 | 4 |
| Visual-servo updates | — | 5 |
| Final visual-servo XY error | — | 1.08 mm |
| Visual-servo XY improvement | — | 21.08 mm |

All columns use identical per-trial seeds, target order, object layouts, camera poses, and lighting.

## Matched outcome analysis

| Comparison | First only succeeds | Second only succeeds | Exact McNemar p |
|---|---:|---:|---:|
| Overhead only vs Continuous wrist servo | 0 | 1 | 1.0000 |
