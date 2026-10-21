# Hierarchical Hybrid VLA acceptance

Date: 2026-10-20

The configured `deepseek-flash` planner received an overhead RGB image, a
calibrated wrist RGB image, the user instruction, and trusted scene state. It
returned one locally valid `execute_hybrid_vla_plan` call:

```json
{
  "object_id": "red",
  "skill": "pick_and_place",
  "start_phase": "localize",
  "grasp_strategy": "shape_aware",
  "destination": "target_box",
  "stack_level": 0
}
```

The persistent metric executor detected all four held-out objects, generated a
shape-aware grasp, accepted four bounded wrist visual-servo corrections,
rejected three out-of-gate corrections, and physics-verified the red object in
the target box. No recovery was required.

The machine-readable trace is
`observations/hybrid_vla_cloud_acceptance.json`. This is a functional
single-scene acceptance test, not a randomized success-rate estimate. The
complete local test suite passed 90/90 tests.
