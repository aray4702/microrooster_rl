"""Micro Rooster velocity task: the Microduck velocity task on the rooster body.

Everything (rewards, DR, curricula, PPO settings) is inherited from
make_microduck_velocity_env_cfg; only what the body forces is overridden:
  - robot entity (12 servos, STS3215 actuators, rooster masses)
  - head pose command 4D -> 2D (neck_pitch, head_pitch)
  - spawn height for the longer legs
Observation layout becomes 3 twist + 2 head + 6 body commands (59 dims instead
of the duck's 61), so rooster policies are not interchangeable with duck ones.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.robot.microrooster_constants import (
    MICROROOSTER_ROBOT_CFG,
    ROOSTER_STAND_Z,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Keep the (neck_pitch, head_pitch) columns of the duck's 4D head ranges.
_KEPT_HEAD_DIMS = (0, 1)


def _head_ranges(ranges):
    return tuple(ranges[i] for i in _KEPT_HEAD_DIMS)


def make_microrooster_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    cfg.scene.entities = {"robot": MICROROOSTER_ROBOT_CFG}

    head_cmd = cfg.commands["head_pose"]
    head_cmd.ranges = _head_ranges(head_cmd.ranges)
    for stage in cfg.curriculum["head_pose_range"].params["range_stages"]:
        stage["ranges"] = _head_ranges(stage["ranges"])

    cfg.events["reset_base"].params["pose_range"]["z"] = (
        ROOSTER_STAND_Z,
        ROOSTER_STAND_Z + 0.01,
    )
    return cfg


MicroroosterRlCfg = deepcopy(MicroduckRlCfg)
MicroroosterRlCfg.wandb_project = "mjlab_microrooster"
MicroroosterRlCfg.experiment_name = "rooster_velocity"
MicroroosterRlCfg.run_name = "rooster_velocity"
