# Franka color sorting

This Isaac Sim 6.1 demo spawns red, blue, green, and yellow rigid blocks. A
Franka Panda selects the requested color/category labels and places those
blocks into a static, collision-enabled target box.

## Run

```bash
cd ~/isaacsim
./python.sh ~/franka_color_sorting/franka_color_sorting.py --target red
```

Examples:

```bash
# Put the blue block in the box.
./python.sh ~/franka_color_sorting/franka_color_sorting.py --target blue

# Categories: warm = red + yellow; cool = blue + green.
./python.sh ~/franka_color_sorting/franka_color_sorting.py --target warm

# Choose multiple explicit colors/categories.
./python.sh ~/franka_color_sorting/franka_color_sorting.py --target red cool
```

The first version selects blocks from their Isaac Sim scene labels, which is a
reliable way to validate picking, grasping, and placing. It is not a camera
classifier yet. The selection boundary is isolated in `resolve_targets()`, so
the next step can replace it with RGB/semantic-camera detection without
changing the manipulation controller.

## Natural-language agent

`agent_main.py` adds a strictly bounded language layer above the verified
controller. The model sees text state only and may call only
`get_scene_state` and `sort_to_target_box`. Tool arguments are schema-checked
locally and simulator pose checks remain the authority for success.

With a cloud planner, the default path first launches a matching Isaac Sim
scene long enough to capture `observations/latest_overhead.png` from a fixed,
calibrated overhead RGB camera. That image—not object pose data—is sent to
DeepSeek as vision input. The model uses it to identify colors and relative
layout; local code retains responsibility for metric coordinates, collision
avoidance, and success verification. Add `--no-vision` only for API/debugging
tests that should skip image capture.

First smoke-test the full tool boundary without moving the robot:

```bash
cd ~/franka_color_sorting
python3 agent_main.py "把暖色方块堆进盒子" --offline --dry-run
```

Remove `--dry-run` to execute the verified GUI simulator task:

```bash
python3 agent_main.py "put the red and blue blocks in the box" --offline
```

The offline mode makes the same explicit plan every time. The cloud mode can
choose the permitted task based on the user instruction and return a final
explanation after the simulator's verified success or failure result. Each task
is limited to two physical attempts and planning is limited to eight turns.

For cloud tool calling, copy `.env.example` to `.env`, select `LLM_PROVIDER`,
and configure that provider's key. For DeepSeek, the minimum configuration is:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-flash
```

Then run the same command without `--offline`. The agent sends tool results
back to the model after each call, limits physical attempts to two, and limits
planning to five turns. Keep `.env` private.

## Validation

The controller checks the actual rigid-block pose after every phase; it does
not claim success merely because the end effector reached its target. To run
the finite test loop without the desktop viewport:

```bash
cd ~/isaacsim
./python.sh ~/franka_color_sorting/franka_color_sorting.py --target red --headless --test
```
