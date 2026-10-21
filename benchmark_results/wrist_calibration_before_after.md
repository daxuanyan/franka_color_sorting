# Wrist RGB-D calibration analysis

| Metric | Before calibration | After calibration |
|---|---:|---:|
| Samples | 41 | 41 |
| Mean signed X error | -4.66 mm | -0.39 mm |
| Mean signed Y error | -15.10 mm | -0.61 mm |
| Mean signed Z error | 0.01 mm | 0.01 mm |
| Mean XY error | 16.59 mm | 0.87 mm |
| P95 XY error | 19.90 mm | 1.41 mm |
| Maximum XY error | 20.80 mm | 2.71 mm |
| Mean image coverage | 99.7% | 42.8% |
| Mean confidence | 0.000 | 0.806 |

Signed errors are estimate minus authored benchmark position; physics truth is used only for evaluation.

## Matched before/after analysis

Common observations: 41; improved: 41; worsened: 0; two-sided exact sign-test `p=9.09e-13`.
