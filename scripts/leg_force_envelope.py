"""Measure the leg's FOOT FORCE envelope F(z, v) -- the term everything divides by.

WHY THIS EXISTS. The series force ceiling s* = F_leg/k - p predicted the measured
boot compression on two arms 3.6x apart in stiffness (2.17 vs 2.63 mm at
k=4290; 9.68 vs 8.85 mm at k=1200). But F_leg = 25 N was CALIBRATED from one
operating point, not measured, and every number downstream scales with it. This
measures it directly.

F_leg IS NOT A CONSTANT. It depends on

  * POSITION: near-straight legs have a short moment arm to the foot, so they
    push hard and slowly; a deep crouch does the opposite.
  * VELOCITY: a DC motor's torque falls linearly with speed,
    tau = kt*vin/R - (kt^2/R)*dq. The BAM M6 fit gives 0.911 N.m at stall and a
    19.1 rad/s no-load speed, so a push at half no-load speed delivers half the
    stall torque. An isometric number alone is an upper bound, not the answer.

So this reports a GRID, not a number: vertical ground-reaction force against
trunk height (which sets leg extension) and against prescribed trunk velocity
(which sets push speed).

HOW THE MEASUREMENT WORKS. The trunk is pinned kinematically -- its pose and
velocity are rewritten every step -- so the robot cannot lift off and the legs
work against the ground exactly as a bench dynamometer would load them. The
legs are OVER-COMMANDED past the extension pose, which matters: a
position-controlled servo produces force proportional to position error, so
commanding a reachable target measures the gain, not the motor. One env per
grid cell, so the whole envelope costs a single rollout.

    uv run python scripts/leg_force_envelope.py
    uv run python scripts/leg_force_envelope.py --task Mjlab-HopLocked-S50-DR-Sym-Locked-MicroDuck
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HOP_NO_PUSH", "1")
os.environ.setdefault("HOP_NO_HOLD", "1")
os.environ.setdefault("HOP_NO_DR", "1")

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-HopFree-S50-DR-Sym-K3344-MicroDuck"
SENSOR = "feet_ground_contact"
# Action indices, as in max_effort_hop: offsets from HOME_FRAME.
L_HIP, L_KNEE, L_ANKLE = 2, 3, 4
R_HIP, R_KNEE, R_ANKLE = 11, 12, 13
MIRROR = -1.0


def make_env(task: str, n: int, device: str):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = n
    keep = {"reset_base", "reset_robot_joints", "expand_bam_friction_fields",
            "reset_action_history"}
    cfg.events = type(cfg.events)({k: v for k, v in cfg.events.items() if k in keep})
    cfg.curriculum = type(cfg.curriculum)()
    cfg.terminations = type(cfg.terminations)()
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.extras.setdefault("log", {})
    return env


def grf_z(env) -> torch.Tensor:
    """Total UPWARD ground reaction force on both feet, per env.

    `feet_ground_contact` is reduce="netforce", so force is the summed contact
    force per foot in the GLOBAL frame. MuJoCo reports it pointing DOWN for a
    loaded foot (probed: -4.905 N under a 4.905 N weight), hence the sign flip.
    """
    sensor = env.scene[SENSOR]
    f = getattr(sensor.data, "force", None)
    if f is None:
        raise SystemExit(f"sensor '{SENSOR}' carries no force field")
    return (-f[:, :, 2]).sum(dim=1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--z-min", type=float, default=0.080, help="lowest trunk height (m)")
    ap.add_argument("--z-max", type=float, default=0.175, help="highest trunk height (m)")
    ap.add_argument("--n-z", type=int, default=12)
    ap.add_argument("--speeds", type=float, nargs="*",
                    default=[0.0, 0.15, 0.30, 0.50, 0.80],
                    help="prescribed UPWARD trunk speed (m/s): 0 is isometric")
    ap.add_argument("--settle", type=float, default=0.30, help="seconds per cell")
    args = ap.parse_args()

    heights = np.linspace(args.z_min, args.z_max, args.n_z)
    speeds = np.array(args.speeds, dtype=np.float32)
    n = len(heights) * len(speeds)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(args.task, n, dev)
    robot = env.scene["robot"]
    n_act = env.action_manager.total_action_dim

    # THE EXTENSION DIRECTION COMES FROM THE MODEL'S OWN KINEMATICS, not from a
    # guessed sign. max_effort_hop grids hip/knee/ankle over their true ranges to
    # find the tallest reachable stance; that offset vector is "push down", and
    # scaling it past 1.0 over-commands into servo saturation.
    from max_effort_hop import _kinematic_poses
    tall, _low, home_h = _kinematic_poses(args.task)
    d_hip, d_knee, d_ankle = tall[1], tall[2], tall[3]
    print(f"[envelope] extension pose dh={d_hip:+.3f} dk={d_knee:+.3f} da={d_ankle:+.3f} "
          f"rad -> trunk {tall[0]*1000:.1f} mm (home {home_h*1000:.1f} mm)")

    OVER = 2.5                      # push the command this far past reachable
    act = torch.zeros((n, n_act), device=dev)
    for (h, k, a_), sgn in (((L_HIP, L_KNEE, L_ANKLE), 1.0), ((R_HIP, R_KNEE, R_ANKLE), MIRROR)):
        act[:, h] = OVER * d_hip * sgn
        act[:, k] = OVER * d_knee * sgn
        act[:, a_] = OVER * d_ankle * sgn

    zz, vv = np.meshgrid(heights, speeds, indexing="ij")
    z_cell = torch.as_tensor(zz.reshape(-1), dtype=torch.float32, device=dev)
    v_cell = torch.as_tensor(vv.reshape(-1), dtype=torch.float32, device=dev)

    # A MOVING CELL STARTS BELOW ITS TARGET AND IS MEASURED AS IT PASSES THROUGH.
    # Integrating upward for the whole window instead carries the trunk 240 mm
    # off the ground at 0.8 m/s, so every moving cell read 0 N -- the feet were
    # simply in the air.
    # ISOMETRIC ONLY for the pinned sweep. Prescribing the trunk's velocity and
    # reading the ground force does NOT measure the actuator: the constraint has
    # to accelerate the trunk and leg masses too, and that inertial term lands in
    # the same reading. Measured that way, force ROSE with speed -- 110% of
    # isometric at 0.8 m/s -- which no DC motor does. The velocity dependence is
    # measured below instead, by letting the robot push freely.
    def run_isometric() -> torch.Tensor:
        env.reset()
        steps = int(args.settle / env.step_dt)
        peak = torch.zeros(n, device=dev)
        for k in range(steps):
            pose = robot.data.root_link_pose_w.clone()
            pose[:, 2] = z_cell
            pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
            robot.write_root_link_pose_to_sim(pose)
            robot.write_root_link_velocity_to_sim(
                torch.zeros_like(robot.data.root_link_vel_w))
            env.step(act)
            if k > steps // 4:
                peak = torch.maximum(peak, grf_z(env))
        return peak

    # FREE PUSH: the real operating condition. Start crouched, command extension,
    # let physics run, and record force against the speed the robot actually
    # reaches. No constraint force, so what the load cell sees is the leg.
    def run_free():
        env.reset()
        pose = robot.data.root_link_pose_w.clone()
        pose[:, 2] = z_cell
        pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        robot.write_root_link_pose_to_sim(pose)
        robot.write_root_link_velocity_to_sim(torch.zeros_like(robot.data.root_link_vel_w))
        rows = []
        for _ in range(int(0.35 / env.step_dt)):
            env.step(act)
            f = grf_z(env)
            v = robot.data.root_link_lin_vel_w[:, 2].float()
            rows.append(torch.stack([f, v], dim=1).clone())
        return torch.stack(rows)        # [T, n, 2]

    peak = run_isometric().reshape(len(heights), len(speeds)).cpu().numpy()
    trace = run_free().cpu().numpy()

    mass = float(env.sim.model.body_mass[0].sum()) if hasattr(env.sim.model, "body_mass") else 0.867
    W = mass * 9.81

    print(f"\nISOMETRIC FOOT FORCE, both feet.  body weight {W:.2f} N\n")
    print("     z (mm) |    F (N)   xBW")
    print("    " + "-" * 28)
    iso = peak[:, 0]
    for i, h in enumerate(heights):
        print(f"    {h*1000:8.1f} | {iso[i]:8.1f} {iso[i]/W:5.1f}")
    best = int(np.argmax(iso))
    print(f"\n  peak isometric {iso[best]:.1f} N at z = {heights[best]*1000:.0f} mm"
          f"  =  {iso[best]/W:.1f}x body weight")

    # Free push: bin force by the speed actually reached.
    f_all = trace[:, :, 0].reshape(-1)
    v_all = trace[:, :, 1].reshape(-1)
    keep = f_all > 0.5
    f_all, v_all = f_all[keep], v_all[keep]
    print(f"\nFREE PUSH -- force against the speed the robot actually reaches\n")
    print("     v (m/s) |  mean F   max F    xBW   samples")
    print("    " + "-" * 46)
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.65, 1.0]
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (v_all >= a_) & (v_all < b_)
        if m.sum() < 5:
            continue
        print(f"    {a_:4.2f}-{b_:4.2f} | {f_all[m].mean():7.1f} {f_all[m].max():7.1f}"
              f" {f_all[m].mean()/W:6.1f} {int(m.sum()):9d}")
    vmax = float(v_all.max()) if v_all.size else 0.0
    print(f"\n  fastest push reached          {vmax:5.2f} m/s")
    print(f"  that is a ballistic rise of   {vmax**2/(2*9.81)*1000:5.1f} mm")
    print(f"\n  F_leg for the simulator is the mean force in the band the robot")
    print(f"  actually pushes through, NOT the isometric peak.")


if __name__ == "__main__":
    main()
