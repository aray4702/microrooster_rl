#!/usr/bin/env python3
"""Open the rough Micro Rooster in the MuJoCo viewer, standing or under a policy.

    uv run mjpython scripts/view_rooster.py     # macOS (viewer needs mjpython)
    uv run python scripts/view_rooster.py       # Linux

    # drive an exported Mjlab-Velocity-Flat-MicroRooster policy
    uv run mjpython scripts/view_rooster.py --policy policy.onnx --new-cmd-obs

Servos are driven by BAM's Feetech STS3215 model like in training (the XML's
own position actuators are tuned for the duck's XL330s and cannot hold the
1.17 kg rooster). Without --policy they just hold the STAND targets.

With --policy, the ONNX runs at 50 Hz on the rooster obs: ang_vel 3,
projected gravity 3, joint pos/vel/last action 12 each, then the command —
3D twist only (legacy, 45D), or with --new-cmd-obs the unified block twist 3 +
head 2 (neck_pitch, head_pitch) + body 6 (x, y, z, roll, pitch, yaw), 53D,
which is what Mjlab-Velocity-Flat-MicroRooster trains. Joint order and default
pose are read from the ONNX metadata.

Type in THIS terminal (not the viewer): arrows walk / strafe, A/E turn,
SPACE stops, P pushes, Q quits. With --new-cmd-obs, H toggles head mode
(UP/DOWN head_pitch, Z/S neck_pitch) and B toggles body-pose mode (UP/DOWN z,
LEFT/RIGHT pitch, A/E roll, Z/S yaw); SPACE resets the active mode's command.

Double-click a body, then Ctrl+right-drag to push it. Only the feet collide,
so if it falls the body sinks through the floor.
"""

import argparse
import random
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from bam.model import load_model

sys.path.insert(0, "scripts")
from infer_policy import TerminalInput, load_mujoco_with_bam  # noqa: E402

SCENE = "src/mjlab_microduck/robot/microrooster/scene_rooster.xml"
DECIMATION = 4  # 200 Hz sim, 50 Hz policy (as in training)
# Twist ranges the rooster velocity task samples from (twist command cfg).
VX_MAX, VY_MAX, WZ_MAX = 0.4, 0.3, 1.0
PUSH_MAX = 0.5  # m/s, trunk xy velocity set by P
# Head / body command limits = the final curriculum ranges the rooster task
# trained on. Body pose only ever got its tiny "keep the inputs alive" range.
HEAD_MAX, HEAD_STEP = 1.1, 0.1                     # rad
BODY_MAX_XYZ, BODY_STEP_XYZ = 0.005, 0.0025        # m
BODY_MAX_ANGLE, BODY_STEP_ANGLE = 0.05, 0.025      # rad
BODY_AXES = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}


def build():
    bam_model = load_model(motor_name="feetech_sts3215_7_4V", model="m1")
    bam_model.actuator.kp = 32.0  # STS3215 firmware default (matches training)
    bam_model.actuator.vin = 7.4
    model, data, bam_ctrl, names = load_mujoco_with_bam(
        SCENE, bam_model, timestep=0.005, vin_drop_gain=0.1, vin_min=6.0
    )
    stand = model.key("STAND")
    data.qpos[:] = stand.qpos
    mujoco.mj_forward(model, data)
    bam_ctrl.reset(data.qpos)
    # The STS3215 law slews an internal target; BAM only creates it when fitting.
    bam_model.actuator.q_target_smooth = bam_ctrl.q_target.copy()
    bam_ctrl.q_target[:] = [stand.ctrl[model.actuator(n).id] for n in names]
    return model, data, bam_ctrl, names


class RoosterPolicy:
    """ONNX velocity policy on the rooster, mirroring the training obs terms."""

    def __init__(self, path, model, data, bam_ctrl, names, new_cmd_obs=False):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        joints = meta["joint_names"].split(",")
        self.default = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
        self.action_scale = float(meta.get("action_scale", 1.0))
        self.new_cmd_obs = new_cmd_obs
        n_obs = self.session.get_inputs()[0].shape[1]
        base = 6 + 3 * len(joints)
        expected = base + (11 if new_cmd_obs else 3)
        if n_obs != expected:
            hint = {base + 11: " (pass --new-cmd-obs)",
                    base + 3: " (drop --new-cmd-obs)"}.get(n_obs, "")
            raise SystemExit(f"{path} expects a {n_obs}D obs, this layout builds "
                             f"{expected}D{hint}")

        self.model, self.data, self.bam_ctrl = model, data, bam_ctrl
        self.qpos_adr = [model.joint(j).qposadr[0] for j in joints]
        self.qvel_adr = [model.joint(j).dofadr[0] for j in joints]
        # bam_ctrl.q_target is in actuator order; actuators target same-named joints.
        self.target_idx = [_actuator_index(model, names, j) for j in joints]
        self.last_action = np.zeros(len(joints), dtype=np.float32)
        self.twist = np.zeros(3, dtype=np.float32)
        self.head = np.zeros(2, dtype=np.float32)
        self.body = np.zeros(6, dtype=np.float32)

        # Start from the policy's default pose (= training HOME frame).
        for adr, q in zip(self.qpos_adr, self.default):
            data.qpos[adr] = q
        mujoco.mj_forward(model, data)
        bam_ctrl.reset(data.qpos)
        bam_ctrl.model.actuator.q_target_smooth = bam_ctrl.q_target.copy()
        self.apply(self.default)

    def obs(self):
        d = self.data
        quat = d.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        gravity_b = rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        ang_vel_b = d.qvel[3:6]  # free-joint angular velocity is in the body frame
        joint_pos = d.qpos[self.qpos_adr] - self.default
        joint_vel = d.qvel[self.qvel_adr]
        terms = [ang_vel_b, gravity_b, joint_pos, joint_vel, self.last_action, self.twist]
        if self.new_cmd_obs:
            terms += [self.head, self.body]
        return np.concatenate(terms).astype(np.float32)[None]

    def apply(self, targets):
        for i, q in zip(self.target_idx, targets):
            self.bam_ctrl.q_target[i] = q

    def step(self):
        action = self.session.run(None, {"obs": self.obs()})[0][0]
        self.last_action = action.astype(np.float32)
        self.apply(self.default + self.action_scale * action)

    def set_twist(self, vx=None, vy=None, wz=None):
        for i, v in enumerate((vx, vy, wz)):
            if v is not None:
                self.twist[i] = v
        print(f"twist cmd: vx={self.twist[0]:+.2f} m/s  vy={self.twist[1]:+.2f} m/s  "
              f"wz={self.twist[2]:+.2f} rad/s")

    def bump_head(self, i, delta):
        self.head[i] = np.clip(self.head[i] + delta, -HEAD_MAX, HEAD_MAX)
        self.print_head()

    def print_head(self):
        print(f"head cmd: neck_pitch={self.head[0]:+.2f}  head_pitch={self.head[1]:+.2f} rad")

    def bump_body(self, axis, sign):
        i = BODY_AXES[axis]
        step, cap = ((BODY_STEP_XYZ, BODY_MAX_XYZ) if i < 3
                     else (BODY_STEP_ANGLE, BODY_MAX_ANGLE))
        self.body[i] = np.clip(self.body[i] + sign * step, -cap, cap)
        self.print_body()

    def print_body(self):
        x, y, z, roll, pitch, yaw = self.body
        print(f"body cmd: x={x * 1000:+.1f} y={y * 1000:+.1f} z={z * 1000:+.1f} mm  "
              f"roll={np.degrees(roll):+.1f} pitch={np.degrees(pitch):+.1f} "
              f"yaw={np.degrees(yaw):+.1f} deg")


def _actuator_index(model, names, joint_name):
    jid = model.joint(joint_name).id
    for i, n in enumerate(names):
        if model.actuator_trnid[model.actuator(n).id, 0] == jid:
            return i
    raise KeyError(f"no actuator drives joint {joint_name}")


def handle_key(key, policy, data, mode):
    """Apply one keypress; return the new mode ("twist" / "head" / "body"),
    or None when the user asked to quit."""
    if key == "q":
        return None
    if key in ("h", "b"):
        if not policy.new_cmd_obs:
            print("head / body modes need --new-cmd-obs")
            return mode
        new = {"h": "head", "b": "body"}[key]
        mode = "twist" if mode == new else new
        print(f"{mode} mode" + (" (body pose is barely trained: ±5 mm / ±3 deg)"
                                if mode == "body" else ""))
        return mode
    if key == "p":
        angle = random.uniform(0, 2 * np.pi)
        data.qvel[0:2] = PUSH_MAX * np.cos(angle), PUSH_MAX * np.sin(angle)
        print(f"push: {PUSH_MAX} m/s at {np.degrees(angle):.0f} deg")
    elif mode == "head":
        if key == "up":
            policy.bump_head(1, HEAD_STEP)
        elif key == "down":
            policy.bump_head(1, -HEAD_STEP)
        elif key == "z":
            policy.bump_head(0, HEAD_STEP)
        elif key == "s":
            policy.bump_head(0, -HEAD_STEP)
        elif key == " ":
            policy.head[:] = 0.0
            policy.print_head()
    elif mode == "body":
        moves = {"up": ("z", 1), "down": ("z", -1), "left": ("pitch", 1),
                 "right": ("pitch", -1), "a": ("roll", 1), "e": ("roll", -1),
                 "z": ("yaw", 1), "s": ("yaw", -1)}
        if key in moves:
            policy.bump_body(*moves[key])
        elif key == " ":
            policy.body[:] = 0.0
            policy.print_body()
    elif key == "up":
        policy.set_twist(vx=min(policy.twist[0] + 0.1, VX_MAX))
    elif key == "down":
        policy.set_twist(vx=max(policy.twist[0] - 0.1, -VX_MAX))
    elif key == "left":
        policy.set_twist(vy=min(policy.twist[1] + 0.1, VY_MAX))
    elif key == "right":
        policy.set_twist(vy=max(policy.twist[1] - 0.1, -VY_MAX))
    elif key == "a":
        policy.set_twist(wz=min(policy.twist[2] + 0.25, WZ_MAX))
    elif key == "e":
        policy.set_twist(wz=max(policy.twist[2] - 0.25, -WZ_MAX))
    elif key == " ":
        policy.set_twist(0.0, 0.0, 0.0)
    return mode


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", help="exported rooster velocity policy (.onnx)")
    ap.add_argument("--lin-vel-x", type=float, default=0.0, help="initial forward command [m/s]")
    ap.add_argument("--lin-vel-y", type=float, default=0.0, help="initial lateral command [m/s]")
    ap.add_argument("--ang-vel-z", type=float, default=0.0, help="initial yaw-rate command [rad/s]")
    ap.add_argument("--new-cmd-obs", action="store_true",
                    help="unified command obs twist 3 + head 2 + body 6 (what the rooster "
                         "velocity task trains); without it the command is the 3D twist only")
    args = ap.parse_args()

    model, data, bam_ctrl, names = build()
    policy = None
    if args.policy:
        policy = RoosterPolicy(args.policy, model, data, bam_ctrl, names,
                               new_cmd_obs=args.new_cmd_obs)
        policy.set_twist(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)
        print("Keys (type in this terminal): arrows walk/strafe, A/E turn, "
              "SPACE stop, P push, Q quit"
              + (", H head mode, B body-pose mode" if args.new_cmd_obs else ""))
    mode = "twist"

    with TerminalInput() as keys, mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.15)
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -15
        step, last_xy = 0, data.qpos[0:2].copy()
        while viewer.is_running():
            start = time.time()
            if policy is not None:
                for key in keys.get_keys():
                    mode = handle_key(key, policy, data, mode)
                    if mode is None:
                        return
                if step % DECIMATION == 0:
                    policy.step()
            bam_ctrl.update()
            mujoco.mj_step(model, data)
            step += 1
            if policy is not None and step % 200 == 0:  # once per sim second
                yaw = 2 * np.arctan2(data.qpos[6], data.qpos[3])
                c, s = np.cos(yaw), np.sin(yaw)
                dx, dy = data.qpos[0:2] - last_xy  # world displacement over 1 s
                last_xy = data.qpos[0:2].copy()
                print(f"[1s] fwd={c * dx + s * dy:+.2f} lat={-s * dx + c * dy:+.2f} m/s  "
                      f"cmd vx={policy.twist[0]:+.2f} vy={policy.twist[1]:+.2f}  "
                      f"trunk_z={data.qpos[2] * 1000:.1f} mm", flush=True)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
