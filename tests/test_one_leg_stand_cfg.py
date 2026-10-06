"""OneLegStand cfg invariants + the command/target logic (CPU, no GPU)."""

from types import SimpleNamespace

import mujoco
import pytest
import torch

from mjlab_microduck.robot.microduck_constants import get_allcollisions_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_one_leg_stand_env_cfg as ols
from mjlab_microduck.tasks.microduck_one_leg_stand_env_cfg import (
    MicroduckOneLegStandRlCfg,
    make_microduck_one_leg_stand_env_cfg,
)

_CFG = make_microduck_one_leg_stand_env_cfg()


def test_uses_the_all_collisions_model():
    assert _CFG.scene.entities["robot"].spec_fn is get_allcollisions_spec


def test_posture_command_is_the_sitstand_style_flag():
    cmd = _CFG.commands["twist"]
    assert isinstance(cmd, microduck_mdp.OneLegStandCommandCfg)
    assert cmd.ramp_s == ols.RAMP_S
    assert set(cmd.sides) == {-1.0, 1.0}
    assert 0.0 < cmd.one_leg_prob < 1.0  # both postures (incl. the all-zero idle) trained


def test_obs_layout_is_61d_command_block():
    actor = list(_CFG.observations["actor"].terms)
    assert actor[-2:] == ["head_command", "body_command"]
    assert _CFG.observations["actor"].terms["body_command"].params == {"dim": 6}
    # the slewed target is privileged — the actor must not see it
    assert "one_leg_state" not in actor
    assert "one_leg_state" in _CFG.observations["critic"].terms


@pytest.mark.parametrize(
    "name",
    ["foot_height", "foot_height_l1", "com_over_support", "composite", "stillness",
     "two_feet_pose", "head_pose_tracking", "upright_linear", "gentle_motion"],
)
def test_task_terms_and_self_negating_penalties_take_positive_weights(name):
    # foot_height_l1 and gentle_motion return ≤ 0: a negative weight would
    # double-negate into a reward for the violation (AGENTS.md sign rule).
    assert _CFG.rewards[name].weight > 0.0


@pytest.mark.parametrize(
    "name",
    ["joint_limit_proximity", "action_over_limit", "non_foot_contact", "self_collisions",
     "action_rate_l2", "body_ang_vel", "angular_momentum"],
)
def test_cost_functions_take_negative_weights(name):
    assert _CFG.rewards[name].weight < 0.0


def test_limit_guards_cover_every_servo_from_step_0():
    lp = _CFG.rewards["joint_limit_proximity"]
    assert lp.params["margin"] == ols.LIMIT_MARGIN
    assert lp.params["asset_cfg"].joint_names == (r"^(?!passive_).*",)
    assert _CFG.rewards["action_over_limit"].params["overshoot"] == ols.ACTION_OVERSHOOT
    # not ramped in later: a parking habit learned early is hard to unlearn
    for c in _CFG.curriculum.values():
        assert c.params.get("reward_name") not in ("joint_limit_proximity", "action_over_limit")


def test_leg_joint_indices_resolve_to_legs_on_the_model():
    model = get_allcollisions_spec().compile()
    names = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    servo = [n for n in names if not n.startswith("passive_")]
    assert len(servo) == 14
    legs = [servo[i] for i in ols._LEG_JOINTS]
    assert all(("hip" in n or "knee" in n or "ankle" in n) for n in legs)
    # the foot sites the rewards read exist
    assert model.site("left_foot") is not None and model.site("right_foot") is not None


def test_symmetry_off_by_default():
    assert MicroduckOneLegStandRlCfg.algorithm.symmetry_cfg is None


# ── command slew ──────────────────────────────────────────────────────────────


def _t(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_slew_ramps_up_at_constant_rate_and_adopts_side():
    a, s = _t(0.0), _t(0.0)
    a, s = microduck_mdp.ols_slew(a, s, _t(1.0), _t(1.0), 0.1)
    assert s.item() == 1.0 and a.item() == pytest.approx(0.1)
    for _ in range(20):
        a, s = microduck_mdp.ols_slew(a, s, _t(1.0), _t(1.0), 0.1)
    assert a.item() == 1.0


def test_flag_drop_ramps_down_the_lifted_side():
    a, s = _t(1.0), _t(-1.0)
    # flag 0 → side slot is 0 in the obs; the blend must still refer to the
    # foot that is in the air while it comes down
    a, s = microduck_mdp.ols_slew(a, s, _t(0.0), _t(0.0), 0.25)
    assert a.item() == pytest.approx(0.75) and s.item() == -1.0


def test_side_switch_goes_through_two_feet():
    a, s = _t(1.0), _t(1.0)
    steps = 0
    while s.item() != -1.0:
        prev = a.item()
        a, s = microduck_mdp.ols_slew(a, s, _t(1.0), _t(-1.0), 0.25)
        if s.item() != -1.0:
            assert a.item() <= prev  # never ramps up on the old side
        steps += 1
        assert steps < 20
    assert a.item() == pytest.approx(0.0) or a.item() == pytest.approx(0.25)


# ── per-foot targets ─────────────────────────────────────────────────────────


def _env_with(alpha, side):
    term = SimpleNamespace(alpha=_t(alpha), active_side=_t(side))
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: term))


def test_side_plus_one_lifts_the_left_foot():
    tl, tr, shift = microduck_mdp._ols_lift_targets(_env_with(1.0, 1.0), "twist", 0.04, 0.4)
    assert tl.item() == pytest.approx(0.04) and tr.item() == 0.0 and shift.item() == 1.0
    tl, tr, _ = microduck_mdp._ols_lift_targets(_env_with(1.0, -1.0), "twist", 0.04, 0.4)
    assert tr.item() == pytest.approx(0.04) and tl.item() == 0.0


def test_weight_shift_precedes_the_lift():
    tl, tr, shift = microduck_mdp._ols_lift_targets(_env_with(0.3, 1.0), "twist", 0.04, 0.4)
    assert tl.item() == 0.0 and tr.item() == 0.0  # both feet still down
    assert 0.0 < shift.item() < 1.0
    tl, _, shift = microduck_mdp._ols_lift_targets(_env_with(0.7, 1.0), "twist", 0.04, 0.4)
    assert shift.item() == 1.0 and 0.0 < tl.item() < 0.04


def test_two_feet_targets_are_zero():
    tl, tr, shift = microduck_mdp._ols_lift_targets(_env_with(0.0, 1.0), "twist", 0.04, 0.4)
    assert tl.item() == tr.item() == shift.item() == 0.0
