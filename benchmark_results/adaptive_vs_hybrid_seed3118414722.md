# Adaptive SmolVLA versus hybrid RGB-D diagnostic

Both runs used scenario seed `3118414722`, trial index 0, the held-out object
set, and target `red`. This is a one-scene diagnostic, not a success-rate
estimate.

| Controller | Result | Time | Minimum EEF-target distance | Object displacement |
|---|---:|---:|---:|---:|
| R1-10k SmolVLA, adaptive 1/5-step re-observation | failure (`approach`) | 120.0 s simulation | 0.1734 m | 0.0011 mm |
| Generic RGB-D + shape-aware grasp + wrist servo | success | 27.3 s wall | n/a | placed in box |

The adaptive SmolVLA run made 598 commands and 414 policy inferences. It
shortened 368 chunks. The combined clipping rate was 60.4%, but most of this
was tiny finger-command clipping at the 0.04 m joint limit. Arm-step clipping
occurred on 60/598 commands (10.0%). The target was never lifted.

The matched hybrid run detected all four scene objects, achieved 1.06 mm mean
overhead XY localization error, made eight wrist-servo updates, and passed
physics placement verification without recovery. The matched outcome supports
spatial localization/control as the immediate bottleneck, while the sample
size is too small for a general success-rate claim.

Artifacts:

- `vla_r1_adaptive_diagnostic_blue_seed3118414722/trial_0000/`
  (the directory name predates target resolution; the recorded target is red)
- `hybrid_rgbd_diagnostic_red_seed3118414722/trial_0000/`
