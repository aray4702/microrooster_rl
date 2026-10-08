#!/usr/bin/env python3
"""Open the rough Micro Rooster in the MuJoCo viewer, holding its STAND pose.

    uv run mjpython scripts/view_rooster.py     # macOS (viewer needs mjpython)
    uv run python scripts/view_rooster.py       # Linux

No policy: the servos just hold the STAND targets, driven by BAM's Feetech
STS3215 model like in training (the XML's own position actuators are tuned for
the duck's XL330s and cannot hold the 1.17 kg rooster). Double-click a body,
then Ctrl+right-drag to push it. Only the feet collide in this model, so if it
falls the body sinks through the floor.
"""

import sys
import time

import mujoco
import mujoco.viewer
from bam.model import load_model

sys.path.insert(0, "scripts")
from infer_policy import load_mujoco_with_bam  # noqa: E402

SCENE = "src/mjlab_microduck/robot/microrooster/scene_rooster.xml"


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
    return model, data, bam_ctrl


def main():
    model, data, bam_ctrl = build()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.15)
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -15
        while viewer.is_running():
            start = time.time()
            bam_ctrl.update()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    main()
