# Micro Rooster RL

RL training environments for **Micro Rooster** — a larger, rooster-shaped
sibling of Pollen Robotics' [Microduck](https://github.com/pollen-robotics/microduck)
— built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with PPO
and the [BAM](https://github.com/Rhoban/bam) actuator model.

This repo is a fork of
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl):
every original Microduck environment still ships unchanged, documented in
[docs/microduck.md](docs/microduck.md). The Python package keeps its upstream
name (`mjlab_microduck`) so upstream changes merge cleanly; rooster code uses
the `microrooster` prefix throughout. See [NOTICE](NOTICE) for attribution.

The robot is a phase-1 rough sim model generated from the Microduck walk
model: 12 Feetech STS3215 servos (`head_yaw`/`head_roll` welded), legs
stretched 15%, head half the duck's width (same mass), ~1.17 kg. Every number
in it is a guess to be replaced by CAD + scale readings.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Playing policies runs on the CPU;
training needs a CUDA GPU or Hugging Face Jobs.

```bash
git clone https://github.com/aray4702/microrooster_rl
cd microrooster_rl
uv sync
uv run --with pytest pytest tests/     # CPU tests, ~10 s
uv run mjpython scripts/view_rooster.py  # see the rooster (macOS; Linux: uv run python)
```

## Files

| Path | What |
| --- | --- |
| `src/mjlab_microduck/robot/microrooster/make_rooster.py` | generator: reads `microduck/robot_walk.xml`, writes the two XMLs below |
| `src/mjlab_microduck/robot/microrooster/robot_rooster.xml` | generated robot (meshes shared via `meshdir="../microduck/assets/"`) — edit the script, not this file |
| `src/mjlab_microduck/robot/microrooster/scene_rooster.xml` | generated scene with the `STAND` keyframe |
| `src/mjlab_microduck/robot/microrooster_constants.py` | robot cfg: STS3215 BAM actuators, HOME frame, `ROOSTER_STAND_Z` |
| `src/mjlab_microduck/tasks/microrooster_velocity_env_cfg.py` | `Mjlab-Velocity-Flat-MicroRooster` task |
| `policies/` | committed ONNX policies: the Microduck set and Micro Rooster versions ([policies/README.md](policies/README.md)) |
| `scripts/view_rooster.py` | MuJoCo viewer: STAND hold, or `--policy` to drive an ONNX |
| `tests/test_microrooster_model.py` | model compiles, keyframe height, committed XML == generator output |

## Play in the MuJoCo simulator

CPU MuJoCo on your own machine — no GPU needed. General notes:

- **macOS:** the viewer needs `mjpython` (`uv run mjpython …`); on Linux use
  `uv run python …`.
- **Keyboard commands go in the terminal** you launched from, not the viewer
  window (keys in the viewer toggle its own rendering options).
- In the viewer: left-drag rotates, right-drag pans, scroll zooms;
  double-click a body, then Ctrl+right-drag to push it.
- Close the window (or press Q in the terminal) to exit. If no window
  appears, look for the **mjpython** icon in the Dock.

### Microduck

The duck's shipped policy set is committed in `policies/microduck/` (a copy
of [`pollen-robotics/microduck-policies`](https://huggingface.co/pollen-robotics/microduck-policies);
see [policies/README.md](policies/README.md) to refresh it). Walk with the default walk policy (VelStand v7, 61-D obs, BAM XL330
actuators as in training):

```bash
uv run mjpython scripts/infer_policy.py --walking policies/microduck/velstand.onnx --new-cmd-obs
```

| Key | Action |
| --- | --- |
| ↑ / ↓ | walk forward / backward |
| ← / → | step sideways |
| A / E | turn left / right |
| SPACE | stop (stand) |
| P | random push |
| H | head mode (arrows / A / E / Z / S move the head) |
| B | body-pose mode (tilt the trunk while standing) |
| T | pause / resume the policy |
| Q | quit |

Other flags: `--lin-vel-x 0.3` starts walking without a key press;
`--scene src/mjlab_microduck/robot/microduck/scene_walk_backlash.xml` matches
VelStand's backlash training model; `--roulade policies/microduck/roulade.onnx`
(R) and `--kick-left` / `--kick-right` (K / L) add the one-shot tricks.

VelStand treats slow commands as "stand": at 0.15 m/s it stands still, at
0.3 m/s it walks ~0.11 m/s in this CPU rehearsal (plain `scene.xml`).

Model only, holding the `STAND` keyframe with the XML's position actuators
(no policy):

```bash
uv run mjpython - <<'EOF'
import time, mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene_walk.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("STAND").id)
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        t = time.time(); mujoco.mj_step(m, d); v.sync()
        time.sleep(max(0.0, m.opt.timestep - (time.time() - t)))
EOF
```

### Micro Rooster

Model only, holding its `STAND` pose (servos driven by BAM's STS3215 model as
in training). Only the feet collide, so if it falls the body sinks through
the floor:

```bash
uv run mjpython scripts/view_rooster.py
```

Policy: the trained rooster policies are committed in `policies/microrooster/`
([policies/README.md](policies/README.md) lists where each came from). Drive
the v1 velocity policy:

```bash
uv run mjpython scripts/view_rooster.py --policy policies/microrooster/velocity_v1.onnx --new-cmd-obs
uv run mjpython scripts/view_rooster.py --policy policies/microrooster/velocity_v1.onnx --new-cmd-obs --lin-vel-x 0.2  # start walking
```

A new HF Jobs run uploads its final policy as `exported/policy.onnx` in the
run's (private) checkpoint repo; fetch it with `uv run hf auth login` once,
then `uv run hf download <namespace>/<run-repo> exported/policy.onnx
--local-dir <dir>`, and copy it into `policies/microrooster/` as the next
`velocity_vN.onnx`.

`--new-cmd-obs` selects the unified command block (twist + head + body) that
`Mjlab-Velocity-Flat-MicroRooster` trains, like `infer_policy.py
--new-cmd-obs` for the duck; without it the command is the legacy 3-D twist,
and a mismatched ONNX is refused with a hint. The terminal prints the
achieved forward / lateral speed once per second.

| Mode | Keys |
| --- | --- |
| twist (default) | ↑/↓ forward/back, ←/→ strafe, A/E turn, SPACE stop |
| head (H toggles) | ↑/↓ head_pitch, Z/S neck_pitch (±0.1 rad steps, ±1.1 max), SPACE reset |
| body pose (B toggles) | ↑/↓ z, ←/→ pitch, A/E roll, Z/S yaw, SPACE reset — clamped to the ±5 mm / ±3° it trained on, so no visible effect |
| any | P push, Q quit |

The rooster's actor observation is 53-D (ang_vel 3 + gravity 3 + 12 joint
pos/vel/last action each + twist 3 + head 2 + body 6 commands) with 12-D
actions, not the duck's 61-D -> 14-D, so rooster policies are not
interchangeable with duck policies, `scripts/infer_policy.py` cannot run them,
and `uv run publish` (which requires `[1,61] -> [1,14]`) does not accept them.

## Regenerate the model

After editing the constants in `make_rooster.py`:

```bash
uv run python src/mjlab_microduck/robot/microrooster/make_rooster.py
uv run --with pytest pytest tests/test_microrooster_model.py
```

If the mass or stand height it prints changes, update `ROOSTER_STAND_Z` in
`microrooster_constants.py` — and retrain: existing policies were trained on
the old body.

## Train

The duck velocity recipe with a 2D head command and a taller spawn height;
logs to wandb project `mjlab_microrooster`, experiment `rooster_velocity`.

```bash
# smoke test first
uv run train Mjlab-Velocity-Flat-MicroRooster --env.scene.num-envs 64 --agent.max_iterations 5

# full run, local GPU
uv run train Mjlab-Velocity-Flat-MicroRooster --env.scene.num-envs 4096

# full run on Hugging Face Jobs
uv run train Mjlab-Velocity-Flat-MicroRooster --env.scene.num-envs 4096 \
  --agent.max_iterations 4000 --hf-jobs --flavor l4x1 --namespace danceone

# watch it (GPU)
uv run play Mjlab-Velocity-Flat-MicroRooster --wandb-run-path <entity/mjlab_microrooster/run_id>
```

### Resume a stopped HF run

An HF job starts in a fresh container, so the checkpoint must come from wandb
(checkpoints are saved every 250 iterations). Find the run id in the job log
(`View run at …/runs/<id>`), then:

```bash
uv run train Mjlab-Velocity-Flat-MicroRooster --env.scene.num-envs 4096 \
  --agent.resume True \
  --wandb-run-path aray4702-tt/mjlab_microrooster/<run_id> \
  --wandb-checkpoint-name model_<N>.pt \
  --agent.max_iterations <remaining> \
  --hf-jobs --namespace danceone
```

- `--agent.max_iterations` counts **additional** iterations from the
  checkpoint (3500 + 500 → stops at 4000), not a total.
- Do **not** set `MICRODUCK_WARM_START=1`: a real resume restores the step
  counter so curricula continue where they stopped.
- Keep the code unchanged until it finishes — `--hf-jobs` uploads the current
  working tree.
- Check the job log shows `Loading checkpoint from W&B: model_<N>.pt` and
  `Learning iteration <N>/…`, not `0/…`.

## Policy log

### v1 — 2026-10-08, `model_3999`

- Runs: wandb `aray4702-tt/mjlab_microrooster/nsvs6avi` (iterations 0–3649,
  stopped by HF billing) → resumed from `model_3500` as `bw9t2smj` (3500–3999).
- ONNX: `policies/microrooster/velocity_v1.onnx` (from HF
  `danceone/mjlab-velocity-flat-microrooster-20261007-203123`,
  `exported/policy.onnx`).
- Training at 3999: mean reward 65.6, episode length 806; every penalty ≤ 0;
  episode ends ≈ 1.6 falls : 3.3 timeouts; `error_vel_xy` 0.54 m/s,
  `error_vel_yaw` 1.85 rad/s.
- CPU MuJoCo (`view_rooster.py`, 10 s per command, body frame):

  | command | result |
  | --- | --- |
  | stand (0) | creeps forward ~0.10 m/s, slow right turn (−0.25 rad/s) |
  | forward 0.2 m/s | 0.20 m/s, curves right (−0.44 rad/s) |
  | forward 0.4 m/s | 0.23 m/s (apparent top speed) |
  | turn 0.8 rad/s | ≈ 0 — does not turn |

  No falls. Walks forward; standing still and turning are not learned yet.
  Head commands track (head_pitch ±0.5 → joint moves ≈ ±0.5 rad while
  walking); body pose is not trained.

## Upstream Microduck docs

Training recipes, the sim2real playbook and every Microduck task:
[docs/microduck.md](docs/microduck.md) and [AGENTS.md](AGENTS.md).

## License

Code: Apache 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The 3D model
files, including the Micro Rooster model derived from them, are licensed
under Creative Commons BY-NC-SA.
