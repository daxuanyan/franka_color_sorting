# Matched fault-injection benchmark

Matched scenarios: 30

| Metric | Baseline | Servo only | Servo + recovery |
|---|---:|---:|---:|
| Final task success | 12/30 (40.0%) | 20/30 (66.7%) | 30/30 (100.0%) |
| First-attempt success | 12/30 (40.0%) | 20/30 (66.7%) | 19/30 (63.3%) |
| Trials invoking recovery | 0 | 0 | 11 |
| Conditional recovery success | — | — | 11/11 (100.0%) |
| Mean duration | 26.30 s | 31.82 s | 37.19 s |
| Median duration | 23.89 s | 32.59 s | 36.36 s |
| Injected fault events | 30 | 178 | 197 |
| Safe exhausted stops | 18 | 10 | 0 |

## Success by injected fault

| Fault type | Baseline | Servo only | Servo + recovery |
|---|---:|---:|---:|
| dynamic_shift | 3/10 (30.0%) | 2/10 (20.0%) | 10/10 (100.0%) |
| pose_bias | 1/10 (10.0%) | 9/10 (90.0%) | 10/10 (100.0%) |
| sensor_degradation | 8/10 (80.0%) | 9/10 (90.0%) | 10/10 (100.0%) |

## Paired outcome analysis

| Comparison | First only | Second only | Exact McNemar p |
|---|---:|---:|---:|
| Baseline vs Servo only | 2 | 10 | 0.0386 |
| Baseline vs Servo + recovery | 0 | 18 | <0.0001 |
| Servo only vs Servo + recovery | 0 | 10 | 0.0020 |

All modes use identical recorded seeds, targets, object layouts, camera/light settings, and fault parameters.
Simulator truth is used for offline scoring and environment fault injection, never as a perception fallback.
