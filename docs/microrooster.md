# Micro Rooster

A phase-1 rough sim model of a larger sibling robot, generated from the
Microduck walk model: 12 Feetech STS3215 servos (`head_yaw`/`head_roll`
welded), legs stretched 15%, ~1.17 kg. Every number in it is a guess to be
replaced by CAD + scale readings.

View the model holding its STAND pose (no policy; servos driven by BAM's
STS3215 model as in training):

```bash
uv run mjpython scripts/view_rooster.py     # macOS (the viewer needs mjpython)
uv run python scripts/view_rooster.py       # Linux
```

Double-click a body, then Ctrl+right-drag to push it. Only the feet collide,
so if it falls the body sinks through the floor.

Regenerate `robot_rooster.xml` + `scene_rooster.xml` after editing the
constants in `make_rooster.py`:

```bash
uv run python src/mjlab_microduck/robot/microduck/make_rooster.py
```

Train the walking policy (the duck velocity recipe with a 2D head command and
a taller spawn height; logs to wandb project `mjlab_microrooster`):

```bash
# smoke test first
uv run train Mjlab-Velocity-Flat-MicroRooster --env.scene.num-envs 64 --agent.max_iterations 5

# full run (add --hf-jobs to run on Hugging Face Jobs)
uv run train Mjlab-Velocity-Flat-MicroRooster --env.scene.num-envs 4096

uv run train Mjlab-Velocity-Flat-MicroRooster --env.scene.num-envs 4096 \
  --agent.max_iterations 4000 --hf-jobs --flavor l4x1 --namespace danceone

# watch it
uv run play Mjlab-Velocity-Flat-MicroRooster --wandb-run-path <entity/mjlab_microrooster/run_id>
```

The rooster's actor observation is 59-D (twist 3 + head 2 + body 6 commands),
not the duck's 61-D, so rooster policies are not interchangeable with duck
policies and `uv run publish` (which requires `[1,61] -> [1,14]`) does not
accept them.