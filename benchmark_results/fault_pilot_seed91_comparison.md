# Matched fault-injection benchmark

Matched scenarios: 3

| Metric | Baseline | Servo only | Servo + recovery |
|---|---:|---:|---:|
| Final task success | 2/3 (66.7%) | 3/3 (100.0%) | 3/3 (100.0%) |
| First-attempt success | 2/3 (66.7%) | 3/3 (100.0%) | 3/3 (100.0%) |
| Trials invoking recovery | 0 | 0 | 0 |
| Conditional recovery success | — | — | — |
| Mean duration | 28.63 s | 36.37 s | 34.67 s |
| Injected fault events | 3 | 23 | 23 |
| Safe exhausted stops | 1 | 0 | 0 |

## Success by injected fault

| Fault type | Baseline | Servo only | Servo + recovery |
|---|---:|---:|---:|
| dynamic_shift | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) |
| pose_bias | 0/1 (0.0%) | 1/1 (100.0%) | 1/1 (100.0%) |
| sensor_degradation | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) |

## Paired outcome analysis

| Comparison | First only | Second only | Exact McNemar p |
|---|---:|---:|---:|
| Baseline vs Servo only | 0 | 1 | 1.0000 |
| Baseline vs Servo + recovery | 0 | 1 | 1.0000 |
| Servo only vs Servo + recovery | 0 | 0 | 1.0000 |

All modes use identical recorded seeds, targets, object layouts, camera/light settings, and fault parameters.
Simulator truth is used for offline scoring and environment fault injection, never as a perception fallback.
