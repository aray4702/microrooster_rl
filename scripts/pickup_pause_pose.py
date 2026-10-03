#!/usr/bin/env python3
"""Derive the pick-up detector's PAUSE POSE from a walking policy, and check it holds.

The paused robot holds a fixed pose at the policy gain, and on the floor a fixed pose tips —
HOME tips past 40° in ~1-3 s in sim. The pause pose is the policy's OWN mean standing stance at
zero command (it tips far more slowly, and it is what the policy expects to wake up in), so it
must be re-derived whenever the deployed walking policy changes.

    uv run scripts/pickup_pause_pose.py --policy logs/bench_onnx/<new_velstand>.onnx

Prints the raw mean stance, the mirrored/zeroed pose to paste into BOTH
`src/mjlab_microduck/pickup/features.py: PAUSE_POSE_REL` and microduck's
`duck-control/src/pickup.rs: PAUSE_POSE`, and the tip-over rate of HOME vs that pose when held on
the floor (want: clearly better than HOME; v1 numbers: HOME 50 % / 93 %, pose 11 % / 45 % past 40°
after 1 s / 3 s).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import pickup_datagen as pg  # noqa: E402

LEFT, RIGHT = slice(0, 5), slice(9, 14)


def zero_commands(env):
    for name in ("twist", "head_pose", "body_pose"):
        term = env.command_manager._terms[name]
        for attr in ("vel_command_b", "command", "_command", "pose_command"):
            value = getattr(term, attr, None)
            if isinstance(value, torch.Tensor):
                value[:] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=pg.PROD_POLICY)
    ap.add_argument("--num-envs", type=int, default=256)
    args = ap.parse_args()
    dev = "cuda:0"
    env = pg.build_env(args.num_envs, dev, seed=0, keep_held_tilted=True, quiet=True)
    robot = env.scene["robot"]
    policy = pg.OnnxMLP(args.policy, dev)

    def tilt():
        q = robot.data.root_link_quat_w
        return torch.rad2deg(torch.acos((1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1)))

    obs, _ = env.reset()
    acts = []
    for step in range(200):
        zero_commands(env)
        a = policy(obs["actor"])
        obs, *_ = env.step(a)
        if step >= 150:
            acts.append(a)
    mean = torch.cat(acts).mean(0)
    pose = mean.clone()
    # mirror: HOME is mirrored for every leg joint, so a symmetric stance has opposite offsets
    sym = (mean[LEFT] - mean[RIGHT]) / 2
    pose[LEFT], pose[RIGHT] = sym, -sym
    pose[0] = pose[9] = 0.0          # hip yaw: a stance has no turn in it
    pose[7] = pose[8] = 0.0          # head yaw / roll
    print("raw mean stance (target - HOME):", [round(x, 4) for x in mean.tolist()])
    print("PAUSE POSE (paste into features.py PAUSE_POSE_REL and pickup.rs PAUSE_POSE):")
    print("   ", [round(x, 3) for x in pose.tolist()])

    for name, target in (("HOME", torch.zeros(14, device=dev)), ("pause pose", pose)):
        obs, _ = env.reset()
        for _ in range(100):
            zero_commands(env)
            a = policy(obs["actor"])
            obs, *_ = env.step(a)
        start = a.clone()
        line = []
        for k in range(150):
            ramp = min(1.0, k * env.step_dt / 0.3)
            obs, *_ = env.step(start * (1 - ramp) + target * ramp)
            if k in (50, 149):
                line.append(f"{100 * (tilt() > 40).float().mean():.0f}% past 40° at {k * env.step_dt:.0f} s")
        print(f"held on the floor, {name:10s}: " + ", ".join(line))


if __name__ == "__main__":
    main()
