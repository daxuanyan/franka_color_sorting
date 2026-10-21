# Demo recording guide

Three short clips are enough to communicate the project's strongest results.
Record the Isaac Sim viewport at 1920x1080 and 30 or 60 FPS. Keep the terminal
out of frame unless its phase messages are important to the story.

## Recommended clips

### 1. End-to-end Hybrid VLA task

Show all four held-out objects, the natural-language instruction, successive
picks, and the final verified placement. Use the deterministic planner so the
recording does not depend on a cloud API:

```bash
python3 agent_main.py "把四个不规则物体全部放进目标箱" --offline --precise
```

Suggested final filename: `hybrid_vla_full_task`.

### 2. Shape-aware grasp

Use a single held-out object and keep the wrist camera or grasp orientation
visible. The cone or rotated cuboid best demonstrates that the system is not a
fixed color-block script:

```bash
~/isaacsim/python.sh franka_color_sorting.py \
  --target yellow --rgbd --test \
  --object-set heldout \
  --perception-backend generic --semantic-backend clip \
  --wrist-mode servo
```

Suggested final filename: `shape_aware_grasp`.

### 3. Dynamic-shift recovery

This committed scenario shifts the red object by about 60 mm during approach.
The clip should show the disturbance, failed or stale approach, retreat,
re-observation, and verified retry:

```bash
~/isaacsim/python.sh franka_color_sorting.py \
  --target red --rgbd --test \
  --scenario configs/demo_dynamic_shift.json \
  --rgbd-output observations/demo_dynamic_shift.png \
  --wrist-mode servo --max-recovery-attempts 1
```

Suggested final filename: `visual_servo_recovery`.

## Editing and compression

Put original captures in `media/raw/`; that directory is ignored by Git. After
installing `ffmpeg`, create a compact MP4 and an animated GIF preview:

```bash
bash scripts/prepare_demo_video.sh \
  media/raw/full_task.mp4 hybrid_vla_full_task 3 20
```

The last two arguments are the start time and duration in seconds. Repeat for
the other clips. Prefer 10–20 second previews, remove audio, and keep each GIF
below roughly 10 MB. If an MP4 is large, attach it to a GitHub Release or host
it externally instead of adding it to normal Git history.

## README gallery template

GitHub does not reliably render an inline HTML `<video>` player in a README.
Use a GIF as the visible preview and make it link to the higher-quality MP4:

```markdown
## Demos

| End-to-end Hybrid VLA | Shape-aware grasp | Dynamic-shift recovery |
|---|---|---|
| [![Hybrid VLA task](media/demos/hybrid_vla_full_task.gif)](media/demos/hybrid_vla_full_task.mp4) | [![Shape-aware grasp](media/demos/shape_aware_grasp.gif)](media/demos/shape_aware_grasp.mp4) | [![Visual-servo recovery](media/demos/visual_servo_recovery.gif)](media/demos/visual_servo_recovery.mp4) |
| Language plan to verified placement | RGB-D geometry and grasp orientation | Re-observe, replan, and retry |
```

Add this gallery below the `Highlights` section only after all referenced files
exist, so the public README never contains broken previews.
