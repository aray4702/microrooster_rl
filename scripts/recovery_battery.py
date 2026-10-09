#!/usr/bin/env python3
"""Headless get-up battery: does a 61D policy stand up from the ground?

    uv run scripts/recovery_battery.py a.onnx b.onnx [--episodes 20] [--seed 0]

CPU MuJoCo on the all-collisions scene (the model VelStand trains on; --scene
to change), BAM M6
actuators as in training (via infer_policy.py's loaders), all-zero 13D command
(= "stand still"). Per spawn type, N episodes with random yaw and ±0.1 rad
joint noise:

  face_down / face_up / side   — lying, orientations as in
                                 mdp.set_random_prone_orientation
  push                         — standing, settled 1 s, then a 1.0 m/s
                                 horizontal velocity kick in a random direction
                                 (VelStand's topple_push magnitude)

Recovered = tilt < 25° AND trunk z > STAND_FRAC × the policy's own settled
standing height, held for 1 s, within the 8 s budget (VelStand's
FALLEN_TIMEOUT_S). For `push`, episodes where the robot never tilts past 40°
count as "absorbed" (no fall) and are reported separately.

Meant for A/B'ing a stand teacher or a VelStand checkpoint against a known
reference (e.g. policies/microduck/alpha_stand.onnx, velstand.onnx).
"""

import argparse
import math
import os
import sys

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_policy import (  # noqa: E402
    BAM_KP_FW,
    BAM_VIN_MIN,
    DEFAULT_POSE,
    load_bam_model,
    load_mujoco_with_bam,
)

SCENE = "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"
DT, DECIMATION = 0.005, 4
BUDGET_S, HOLD_S, SETTLE_S = 8.0, 1.0, 1.0
UPRIGHT_TILT, FALLEN_TILT = math.radians(25), math.radians(40)
STAND_FRAC = 0.85
PUSH_SPEED = 1.0
SPAWNS = ("face_down", "face_up", "side", "push")


class Sim:
    def __init__(self, scene=SCENE, vin=7.4, vin_drop_gain=0.1):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            bam = load_bam_model(BAM_KP_FW, vin, 0.0)
            self.m, self.d, self.bam, _ = load_mujoco_with_bam(scene, bam, DT, vin_drop_gain, BAM_VIN_MIN)
        m = self.m
        self.qadr = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.vadr = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.gyro_adr = int(m.sensor_adr[gyro])

    def reset(self, quat, joint_noise, rng, lying):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = (0.0, 0.0, 0.12)
        self.d.qpos[3:7] = quat
        self.d.qpos[self.qadr] = DEFAULT_POSE + rng.uniform(-joint_noise, joint_noise, 14)
        self.bam.q_target[:] = self.d.qpos[self.qadr]
        mujoco.mj_forward(self.m, self.d)
        self.last_action = np.zeros(14, dtype=np.float32)
        if lying:
            # A rotated standing pose at z=0.12 rests on its feet and just tips
            # back up. Lower it onto the floor, then let it settle passively
            # (servos holding the spawn pose) so episodes start truly lying.
            while self.d.ncon == 0 and self.d.qpos[2] > 0.0:
                self.d.qpos[2] -= 0.001
                mujoco.mj_forward(self.m, self.d)
            for _ in range(int(0.5 / DT)):
                self.bam.update()
                mujoco.mj_step(self.m, self.d)

    def obs(self):
        q = self.d.xquat[self.trunk]
        w, xyz = q[0], q[1:4]
        g = np.array([0.0, 0.0, -1.0])
        t = np.cross(xyz, g) * 2
        grav = g - w * t + np.cross(xyz, t)
        return np.concatenate([
            self.d.sensordata[self.gyro_adr:self.gyro_adr + 3],
            grav,
            self.d.qpos[self.qadr] - DEFAULT_POSE,
            self.d.qvel[self.vadr],
            self.last_action,
            np.zeros(13),
        ]).astype(np.float32)[None]

    def step(self, sess, name):
        a = sess.run(None, {name: self.obs()})[0][0].astype(np.float32)
        self.last_action = a
        self.bam.q_target[:] = DEFAULT_POSE + a
        for _ in range(DECIMATION):
            self.bam.update()
            mujoco.mj_step(self.m, self.d)

    def tilt(self):
        # angle between trunk z axis and world z
        return math.acos(max(-1.0, min(1.0, self.d.xmat[self.trunk][8])))

    def z(self):
        return float(self.d.xpos[self.trunk][2])


def yaw_quat(yaw, kind, rng):
    s, cy, sy = 2 ** -0.5, math.cos(yaw / 2), math.sin(yaw / 2)
    if kind == "face_down":
        return (s * cy, -s * sy, s * cy, s * sy)
    if kind == "face_up":
        return (s * cy, s * sy, -s * cy, s * sy)
    if kind == "side":
        sg = 1.0 if rng.random() < 0.5 else -1.0
        return (s * cy, sg * s * cy, sg * s * sy, s * sy)
    return (cy, 0.0, 0.0, sy)  # upright


def stand_height(sim, sess, name):
    sim.reset(yaw_quat(0.0, "upright", None), 0.0, np.random.default_rng(0), lying=False)
    for _ in range(int(3.0 / (DT * DECIMATION))):
        sim.step(sess, name)
    return sim.z(), math.degrees(sim.tilt())


def episode(sim, sess, name, kind, rng, z_ok):
    steps_s = 1.0 / (DT * DECIMATION)
    sim.reset(yaw_quat(rng.uniform(-math.pi, math.pi), kind, rng), 0.1, rng, lying=kind != "push")
    fell = kind != "push"
    if kind == "push":
        for _ in range(int(SETTLE_S * steps_s)):
            sim.step(sess, name)
        ang = rng.uniform(-math.pi, math.pi)
        sim.d.qvel[0:2] += PUSH_SPEED * np.array([math.cos(ang), math.sin(ang)])
    held, max_tilt = 0, 0.0
    for k in range(int(BUDGET_S * steps_s)):
        sim.step(sess, name)
        tl = sim.tilt()
        max_tilt = max(max_tilt, tl)
        fell = fell or tl > FALLEN_TILT
        ok = tl < UPRIGHT_TILT and sim.z() > z_ok
        held = held + 1 if ok else 0
        if fell and held >= HOLD_S * steps_s:
            return "recovered", (k + 1 - held) / steps_s
    if not fell:
        return "absorbed", None
    return "failed", None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default=SCENE, help="scene XML (default: all-collisions, VelStand's model; "
                    "the stand-up env trains on scene.xml = groundcontact)")
    ap.add_argument("--spawns", nargs="+", default=list(SPAWNS), choices=SPAWNS)
    args = ap.parse_args()

    sim = Sim(args.scene)
    print(f"scene: {args.scene}")
    print(f"{'policy':28s} {'stand z/tilt':>14s}  " + "  ".join(f"{s:>16s}" for s in args.spawns))
    for path in args.onnx:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        z_stand, tilt_stand = stand_height(sim, sess, name)
        z_ok = STAND_FRAC * z_stand
        cells = []
        for kind in args.spawns:
            rng = np.random.default_rng(args.seed)
            res = [episode(sim, sess, name, kind, rng, z_ok) for _ in range(args.episodes)]
            rec = [t for r, t in res if r == "recovered"]
            falls = sum(r != "absorbed" for r, _ in res)
            med = f"{np.median(rec):.1f}s" if rec else "  - "
            cell = f"{len(rec)}/{falls} {med}"
            if kind == "push":
                cell += f" ({args.episodes - falls}abs)"
            cells.append(cell)
        label = os.path.basename(path)[:-5]
        print(f"{label:28s} {z_stand*1000:6.0f}mm {tilt_stand:4.0f}°  " + "  ".join(f"{c:>16s}" for c in cells), flush=True)
    print(f"\ncells: recovered/fell, median time-to-upright; recovered = tilt<25° and z>{STAND_FRAC:.2f}×stand z held {HOLD_S:.0f}s within {BUDGET_S:.0f}s")


if __name__ == "__main__":
    main()
