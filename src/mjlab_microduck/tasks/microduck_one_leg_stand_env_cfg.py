"""Microduck *OneLegStand* task — commanded two feet ↔ one foot, either side.

One policy, driven by a posture command in the twist slot (sitstand design):
    cmd = [flag, side, 0]   flag ∈ {0 = two feet, 1 = one foot}
                            side ∈ {+1 = RIGHT foot down (lift the left),
                                    -1 = LEFT foot down (lift the right)}
"Two feet" is the all-zero command — the same deployment idle as every other
policy. side follows the community flamingo convention
(RemiFabre/microduck-flamingo-cycle). The command flips mid-episode with a
dwell time of a few seconds, so each episode trains weight shift + lift,
one-foot hold, set-down, side switches and two-foot rest.

Design (2026-10-06):
  - Slewed internal target (OneLegStandCommand.alpha, RAMP_S): the first
    SHIFT_FRAC of the ramp moves the CoM target from mid-feet onto the stance
    sole, the rest raises the swing-foot target to LIFT_HEIGHT. Rewards track
    the MOVING target, so lifting early / hopping onto one foot pays nothing
    (sitstand's anti-jackpot mechanism). The obs stays the raw command.
  - NO one-foot keyframe. Pre-training kinematic search (stiff-PD sim,
    scratchpad kin_search.py): with no ankle-roll joint, double-support
    lateral sway is kinematically locked; a one-foot stance with the CoM
    12 mm inside the stance sole AND every joint ≥ 0.1 rad from its limit
    exists, but needs ~27° trunk roll about the stance hip + the head as a
    lateral counterweight (head_yaw ~1.2 rad). So the reward specifies the
    OUTCOME (feet at their targets · CoM over support · upright-ish, only
    the stance foot touching the ground) and RL finds the pose. The upright
    factor is wide (~30° stays cheap) and the head command + HOME leg pose
    fade out with alpha.
  - Sim-only exploits the user flagged — joints PARKED on (virtual) hard
    limits and huge position commands. The XML joint ranges are hard stops
    that can bear load in sim; on the robot nothing equivalent exists, so a
    pose leaning on a stop doesn't transfer. Two guards, both from step 0:
      * joint_limit_proximity: qpos-side L1 in a LIMIT_MARGIN band next to
        every servo's hard range (the AGENTS.md fix for limit parking; the
        stock dof_pos_limits only fires in the last ~7.5%).
      * action_over_limit: command-side, penalises targets beyond
        hard limit + ACTION_OVERSHOOT (low-kp servos legitimately need some
        overshoot — only wild over-drive pays).
    LIMIT_MARGIN = 0.1 matches the feasibility search: the found pose sits
    right at the margin on hip_yaw / hip_roll, so a wider band would make
    the task infeasible without leaning on the penalty.
  - All-collisions model (every part collides): a non-foot ground-contact
    sensor (any body but the ankles) is penalised, so knee/head/trunk
    tripods can't stand in for balance.
  - Sim2real stack copied from sitstand/velocity: same DR, obs noise, IMU
    misalignment, encoder bias, obs delays, action-rate ramp, delayed pushes.

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

import math
from copy import deepcopy

# Symmetry (mirror loss). The 61D mirror table flips twist vy, which maps
# side → -side — exactly the left/right task mirror — so this is valid here.
# OFF by default per AGENTS.md; a candidate if one side trains worse.
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to the velocity env for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True   # FrictionDRBamActuator.friction_scale
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003           # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003           # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)    # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)      # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
# Final push magnitude is HALF velocity's ±0.3: a 4 cm-wide sole cannot absorb
# a walking-scale kick without stepping. Ramped late (see push curriculum).
VELOCITY_PUSH_RANGE                 = (-0.15, 0.15)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Task ───────────────────────────────────────────────────────────────────────
EPISODE_LENGTH_S = 12.0
# Dwell in each commanded posture before a resample may flip it: ramp (2 s) +
# a real hold.
POSTURE_DWELL_S = (3.5, 6.5)
# Probability a resample commands one foot (vs two feet).
ONE_LEG_PROB = 0.6
# Seconds for the internal target to go two feet → fully on one foot.
RAMP_S = 2.0
# Fraction of the ramp spent shifting the CoM over the stance foot before the
# swing-foot target starts rising.
SHIFT_FRAC = 0.4
# Swing-foot sole height target (m) — clearly off the floor, well inside reach.
LIFT_HEIGHT = 0.04

# Anti limit-parking guards (see module docstring).
LIMIT_MARGIN = 0.1        # rad band next to each servo's hard range
ACTION_OVERSHOOT = 0.3    # rad of command beyond the hard limit that is free

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_one_leg_stand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck OneLegStand environment configuration."""
    if rough:
        raise NotImplementedError("OneLegStand is flat-ground only (foot heights read vs env origin).")

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Every body except the two ankle bodies (which carry the soles).
    non_foot_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(mode="body", pattern=r"^(?!ankle_left$|ankle_right$).*$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, non_foot_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    # 70 collision geoms (all-collisions model): velstand's contact budget.
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "upright",
        "soft_landing",
        "dof_pos_limits",  # replaced by the wide-margin joint_limit_proximity below
    ]:
        cfg.rewards.pop(name, None)

    task = {"command_name": "twist", "lift_height": LIFT_HEIGHT, "shift_frac": SHIFT_FRAC}

    # ── Task stack (all track the SLEWED target) ──────────────────────────────
    # Feet at their targets: two feet → both planted; one foot → swing foot
    # follows the ramp up to LIFT_HEIGHT, stance foot planted.
    cfg.rewards["foot_height"] = RewardTermCfg(
        func=microduck_mdp.ols_foot_height_track,
        weight=3.0,
        params={**task, "std": 0.015},
    )
    # L1 driver: 'ignore the flag' costs ~LIFT_HEIGHT·20 = 0.8/step. ≤ 0 →
    # POSITIVE weight (self-negating convention).
    cfg.rewards["foot_height_l1"] = RewardTermCfg(
        func=microduck_mdp.ols_foot_height_l1,
        weight=20.0,
        params=task,
    )
    # Whole-body CoM over its support target (mid-feet → stance sole centre).
    # std 2 cm ≈ half the sole width.
    cfg.rewards["com_over_support"] = RewardTermCfg(
        func=microduck_mdp.ols_com_over_support,
        weight=2.0,
        params={**task, "std": 0.02},
    )
    # Multiplicative goal score: a lifted foot with the CoM outside the sole
    # (a fall in progress) or a CoM lean with both feet down both score ~0.
    cfg.rewards["composite"] = RewardTermCfg(
        func=microduck_mdp.ols_composite,
        weight=4.0,
        params={**task, "foot_std": 0.015, "com_std": 0.025, "upright_std": 0.6},
    )
    # Quiet hold at a COMPLETED posture (ramp done, feet at targets).
    cfg.rewards["stillness"] = RewardTermCfg(
        func=microduck_mdp.ols_stillness,
        weight=1.0,
        params={**task, "vel_std": 0.05, "foot_tol": 0.01},
    )
    # Two-foot rest = the deployed HOME stance; fades out with alpha.
    cfg.rewards["two_feet_pose"] = RewardTermCfg(
        func=microduck_mdp.ols_two_feet_pose,
        weight=1.0,
        params={"command_name": "twist", "joint_indices": _LEG_JOINTS, "std": 0.3},
    )
    # Head commandable on two feet, FREE on one foot (counterweight).
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.ols_head_pose_tracking,
        weight=0.75,
        params={"command_name": "twist", "std": 0.5},
    )
    # Always-on upright floor (+1 upright … 0 lying). Light: 27° roll costs
    # only 0.11·weight, so it shapes falls without taxing the required lean.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # ── Guards ────────────────────────────────────────────────────────────────
    # Joint parking on the (virtual) hard limits: qpos-side, every servo joint.
    # Cost ≥ 0 → NEGATIVE weight. 0.05 rad inside the band = -0.25/step.
    cfg.rewards["joint_limit_proximity"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limit_proximity,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
            "margin": LIMIT_MARGIN,
        },
    )
    # Huge commands past the limits. Cost ≥ 0 → NEGATIVE weight.
    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-2.0,
        params={"overshoot": ACTION_OVERSHOOT},
    )
    # Knee / head / trunk on the ground. Cost → NEGATIVE weight.
    cfg.rewards["non_foot_contact"] = RewardTermCfg(
        func=microduck_mdp.ols_non_foot_ground_contact,
        weight=-2.0,
        params={"sensor_name": non_foot_ground_cfg.name},
    )
    # Swing-foot stomps / dropping onto the stance leg. Self-negating (-|a_z|)
    # → POSITIVE weight.
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Sim2real regularisers — matched to velocity / sitstand ───────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # ── Observations (61D actor layout, identical to every other policy) ──────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel present_velocity is ~1 period old).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Head pose command (live on two feet, free on one) ─────────────────────
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )
    # Layout parity: [twist(3), head_pose(4), body_pose(6)]; body zero-padded.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )
    # Privileged: the slewed target the rewards track (actor never sees it).
    cfg.observations["critic"].terms["one_leg_state"] = ObservationTermCfg(
        func=microduck_mdp.ols_command_state, params={"command_name": "twist"},
    )

    # ── Command: [flag, side, 0] posture command in the twist slot ───────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.OneLegStandCommandCfg(
        **{
            **vars(command),
            "one_leg_prob": ONE_LEG_PROB,
            "ramp_s":       RAMP_S,
        }
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # Fall = > 60° tilt (the one-foot stance itself needs ~30°). Terminating
    # keeps samples on balancing rather than on lying around.
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name, non_foot_ground_cfg.name)},
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # match velocity

    # Spawn standing just above the measured HOME equilibrium (trunk z 0.117),
    # small joint noise.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.118, 0.125)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.03, 0.03)

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain: flat ─────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # Pushes only once the one-foot hold exists (sit/sitstand lesson: early
    # perturbation of an unconsolidated skill unlearns it).
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,         "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1000 * 24, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 1500 * 24, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 2000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    # action_rate: velocity's exact ramp (-0.1 → -1.0 by iter 1500).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 500 * 24,   "weight": -0.2},
                {"step": 750 * 24,   "weight": -0.4},
                {"step": 1000 * 24,  "weight": -0.6},
                {"step": 1250 * 24,  "weight": -0.8},
                {"step": 1500 * 24,  "weight": -1.0},
            ],
        },
    )
    # Torque-rate anti-jitter after the skill exists.
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 750 * 24,   "weight": -5e-4},
                {"step": 1250 * 24,  "weight": -1e-3},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckOneLegStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_one_leg_stand",
    run_name="microduck_one_leg_stand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=4_000,
)
