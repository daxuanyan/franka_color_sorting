# Architecture

The project separates high-level intent from metric robot control. A language
or vision-language planner chooses a bounded skill; deterministic local modules
perform perception, grasp generation, execution, and verification.

```text
Natural-language instruction
          |
          v
  LLM / offline planner ---- overhead + wrist RGB images
          |
          | validated HybridVLAPlan
          v
  RGB-D instance proposals + CLIP association
          |
          v
  point-cloud grasp geometry (size, yaw, jaw width)
          |
          v
  Franka state machine + continuous wrist visual servo
          |
          v
  physics-based placement verification
          |
          +---- failure ----> bounded re-observation and recovery
```

## Main components

| Component | Responsibility |
|---|---|
| `agent_main.py` | Natural-language entry point and planner/executor loop |
| `llm_agent/` | Typed plans, tool contracts, validation, and recovery policy |
| `franka_color_sorting.py` | Isaac Sim scene, Franka controller, cameras, and task state machine |
| `vision_localizer.py` | RGB-D unprojection, 3-D proposals, association, and wrist tracking |
| `semantic_labeler.py` | CLIP and experimental semantic backends |
| `grasp_geometry.py` | Shape-aware top-grasp geometry and safety rejection |
| `benchmark.py` | Seeded scenario generation, fault injection, and result collection |
| `vla_dataset.py` | Crash-tolerant multimodal episode recording and validation |
| `export_*_lerobot.py` | Leakage-safe LeRobot dataset conversion |
| `act_runtime.py` | ACT/SmolVLA sidecar inference and action safety filtering |
| `collect_dagger.py` | Online student/teacher correction data collection |

## Trust boundaries

- Simulator object truth is used only for offline scoring, fault injection, and
  final placement verification. It is not a perception fallback.
- The planner cannot provide metric poses, trajectories, or joint commands.
- Model outputs are schema-validated before execution.
- Perception confidence, geometry, displacement, joint limits, and retry counts
  are checked locally.
- Failed and interrupted demonstrations remain explicitly labelled and are not
  silently included in behavior-cloning exports.

## Data flow

Raw demonstrations contain synchronized overhead/wrist RGB-D, robot state,
expert action, phase, target identity, and audit-only simulator metadata. The
export pipeline creates episode-level train/validation/test splits before
producing ACT goal-conditioned or SmolVLA language-conditioned datasets. DAgger
episodes are added only to the training split, preserving the original held-out
validation and test parents.
