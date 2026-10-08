# Training a Microduck VelStand policy from scratch (HF Jobs)

Goal: a new Microduck ONNX that matches the shipped `velstand.onnx` (v7,
`policies/microduck/velstand.onnx`): walks, stands still at zero command,
falls protectively and gets back up, with head and standing-body control.

How v7 was produced is documented in [velstand_policy.md](velstand_policy.md).
This plan reproduces that lineage with our own runs, because the original
checkpoints are not reachable from our wandb account (all five Pollen runs —
`441tzs6d`, `69u48n8l`, `6op8a8u8`, `fhathosb`, `j4i6yoq2` — return "not
found" from `aray4702-tt`).

## Why it is a chain, not one run

VelStand is trained with PPO **plus behavior cloning (BC) toward two frozen
teachers** (`tasks/distill.py`):

- a **walk teacher** — anchors upright frames (tilt < 25°) and is the
  warm-start for the first VelStand stage;
- a **stand-up teacher** — teaches fallen frames (tilt > 35°) and, from v7,
  standing body tilt.

So the teachers must exist first. Both are read from wandb runs; by default
`distill.py` points at Pollen's. Override them with:

| Env var | Value | Default (Pollen, not accessible) |
| --- | --- | --- |
| `MICRODUCK_WALK_EXPERT` | `<entity/project/run_id>@<model_N.pt>` | `pollen-robotics/mjlab_microduck/441tzs6d@model_3750.pt` |
| `MICRODUCK_STAND_EXPERT` | `<entity/project/run_id>@<model_N.pt>` | `pollen-robotics/mjlab_microduck/69u48n8l@model_9750.pt` |

`--hf-jobs` forwards every `MICRODUCK_*` variable to the job.

## Stages

Iteration counts are the ones the v7 lineage used. One iteration = 24 steps ×
4096 envs (~98k samples) followed by 5 PPO epochs × 4 mini-batches; it is
what `--agent.max_iterations` counts. Checkpoints are saved every 250
iterations (`model_<N>.pt`, uploaded to wandb and the job's HF repo).

| # | Stage | Task | Iterations | Starts from | Needs |
| --- | --- | --- | --- | --- | --- |
| 1 | walk teacher (≈ `alpha_walking`) | `Mjlab-Velocity-Flat-MicroDuck` | 3,750 | scratch | — |
| 2 | stand-up teacher (≈ `alpha_stand`) | `Mjlab-StandUp-Flat-MicroDuck` | 9,750 | scratch | — (run in parallel with 1) |
| 3 | VelStand v1: walk + protective fall + get-up | `Mjlab-VelStand-Flat-MicroDuck` | 6,000 | warm start from 1 | teachers 1 + 2 |
| 4 | VelStand v5/v6: rough terrain + backlash | `Mjlab-VelStand-Rough-Backlash-MicroDuck` | 3,750 | warm start from 3 | teachers 1 + 2 |
| 5 | VelStand v7: standing body control | `Mjlab-VelStand-Rough-Backlash-MicroDuck` | 1,500 | warm start from 4 | teachers 1 + 2 |

Unknowns to keep in mind:

- The lineage doc does not record whether the teachers used the Flat or Rough
  task variant; Flat is the assumption here.
- The current code is the v7 config (`ENABLE_BODY_CONTROL` and friends on),
  so stages 3–4 run with v7's toggles rather than the exact code of their
  original commits. Expected to be fine, untested.

### Commands

```bash
# 1 + 2 — submit both, they are independent
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.max_iterations 3750 --hf-jobs --namespace danceone --detach
uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.max_iterations 9750 --hf-jobs --namespace danceone --detach

# teachers for 3–5 (fill in the run ids printed in the job logs)
export MICRODUCK_WALK_EXPERT=aray4702-tt/mjlab_microduck/<run1>@model_3750.pt
export MICRODUCK_STAND_EXPERT=aray4702-tt/mjlab_microduck/<run2>@model_9750.pt

# 3 — warm start: new task, stage-1 weights, curricula restarted at 0
MICRODUCK_WARM_START=1 uv run train Mjlab-VelStand-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --wandb-run-path aray4702-tt/mjlab_microduck/<run1> \
  --wandb-checkpoint-name model_3750.pt --agent.max_iterations 6000 \
  --hf-jobs --namespace danceone --detach

# 4
MICRODUCK_WARM_START=1 uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --wandb-run-path aray4702-tt/mjlab_microduck/<run3> \
  --wandb-checkpoint-name model_5999.pt --agent.max_iterations 3750 \
  --hf-jobs --namespace danceone --detach

# 5
MICRODUCK_WARM_START=1 uv run train Mjlab-VelStand-Rough-Backlash-MicroDuck --env.scene.num-envs 4096 \
  --agent.resume True --wandb-run-path aray4702-tt/mjlab_microduck/<run4> \
  --wandb-checkpoint-name model_3750.pt --agent.max_iterations 1500 \
  --hf-jobs --namespace danceone --detach
```

Warm start ≠ resume: `MICRODUCK_WARM_START=1` keeps weights, normalizer and
optimizer but restarts the iteration and step counter, so the new task's
curricula run from stage 0. To continue a *stopped* run of the same stage,
use a plain resume instead (README, "Resume a stopped HF run").

## Budget

Total ≈ 24,750 iterations. At the ~2.4 s/iteration measured for the Micro
Rooster velocity run on `l4x1` (VelStand iterations may be slower — two
teachers run every update, and the all-collisions model has 70 contact
geoms), that is ≈ 16–17 GPU-hours. Wall clock ≈ 6.5 h (stage 2, with 1 in
parallel) + ≈ 7.5 h (stages 3–5). Every stage fits the 12 h default
`--timeout`. Budget a retry: checkpoints of the same run differ a lot in
behavior even when the metrics do not.

## Smoke tests (before spending GPU-hours)

64 envs, 5 iterations, on HF Jobs (no CUDA on the Mac). Stage 3's smoke test
uses the smoke runs of 1 and 2 as its teachers and warm start, which
exercises the whole wiring (env-var override, forwarding, wandb download, BC
setup) for cents. Stages 4/5 share stage 3's code path with the
Rough-Backlash task, smoke-tested from stage 3's smoke checkpoint.

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 \
  --hf-jobs --namespace danceone --run-name smoke-walk-teacher
uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 \
  --hf-jobs --namespace danceone --run-name smoke-stand-teacher
# then stage 3 / 4 with MICRODUCK_*_EXPERT and --wandb-run-path pointing at those runs' model_4.pt
```

Pass criteria: job exits 0; the log shows `Learning iteration 4/5` (or
`4/…` from the warm-start); for stage 3, the log shows both teachers loading
from our runs and no NaN termination.

## Picking and judging the result

- Watch per run: every `Episode_Reward/<penalty>` ≤ 0, mean reward rising,
  episode length behaving.
- Compare checkpoints, not just the last one (v7 shipped `model_1250` of a
  1,499-iteration stage). Pollen's eval tools (`claude_experiments/…`) are
  not in this repo; compare in sim against the shipped policy with
  `scripts/infer_policy.py --new-cmd-obs` (walk 0.3 / 0.4 m/s drift, turn
  in place ±1.0 rad/s, push with P, standing still at zero) and the
  sim-eval table in [velstand_policy.md](velstand_policy.md).
- Export with `uv run scripts/export.py <task> --wandb-run-path <...>` (or
  take the job's auto-exported `exported/policy.onnx`) and commit it under
  `policies/microduck/` with a new name — keep `velstand.onnx` as the
  reference.

## Run log

| # | Stage | Job | wandb run | Checkpoint used | Notes |
| --- | --- | --- | --- | --- | --- |
| S1 | smoke: walk teacher | `6ac7f1b5` ✅ | `duc0vnss` | `model_4.pt` | 2026-10-08, 64 envs × 5 iters, exit 0 |
| S2 | smoke: stand teacher | `6ac7f1bb` ✅ | `bs3ed7ur` | `model_4.pt` | 2026-10-08, 64 envs × 5 iters, exit 0 |
| S3 | smoke: VelStand Flat (warm start + teachers) | `6ac7f538` ✅ | `o5x08egh` | `model_4.pt` | warm start from S1; BC on with S1 (anchor) + S2 (stand) loaded from our runs; no NaN; ONNX auto-export OK |
| S4 | smoke: VelStand Rough-Backlash | `6ac7f66e` ✅ | `alnpb06v` | `model_4.pt` | warm start from S3; same teachers; no NaN; ONNX auto-export OK |
| 1 | walk teacher | | | | |
| 2 | stand-up teacher | | | | |
| 3 | VelStand v1 | | | | |
| 4 | VelStand rough + backlash | | | | |
| 5 | VelStand body control | | | | |
