"""Patch 6: interval-event timers carry across episode resets (pre-mjlab-1.6 push timing).

mjlab 1.6 resamples function-based interval timers on every reset, which delays
the first push of each episode by interval_range_s[0] and starves short episodes
of pushes. Every policy up to 2026-10 trained with the countdown carried over.
"""

from types import SimpleNamespace

import torch
from mjlab.managers.event_manager import EventManager

from mjlab_microduck.tasks import mdp


class _ClassTerm:
    def reset(self, env_ids=None):
        pass


def _stub_manager(num_envs=8):
    def term(func, is_global_time=False):
        return SimpleNamespace(func=func, is_global_time=is_global_time, interval_range_s=(3.0, 6.0))

    fn_term = term(lambda env, env_ids: None)  # e.g. push_robot
    global_term = term(lambda env, env_ids: None, is_global_time=True)
    class_term = term(_ClassTerm())
    mgr = object.__new__(EventManager)
    mgr._env = SimpleNamespace(num_envs=num_envs, device="cpu")
    mgr._mode_term_cfgs = {"interval": [fn_term, global_term, class_term]}
    mgr._mode_class_term_cfgs = {"interval": [class_term]}
    # 0.5 s left: well below interval_range_s[0], so any 1.6-style resample is visible.
    mgr._interval_term_time_left = [torch.full((num_envs,), 0.5) for _ in range(3)]
    return mgr


def test_patch_is_installed():
    assert EventManager.reset is mdp._event_reset_carry_interval_timers


def test_function_term_timer_carries_across_partial_and_full_reset():
    mgr = _stub_manager()
    mgr.reset(torch.tensor([1, 3]))
    mgr.reset(None)
    assert torch.equal(mgr._interval_term_time_left[0], torch.full((8,), 0.5))


def test_global_time_term_is_untouched_by_reset():
    mgr = _stub_manager()
    mgr.reset(torch.tensor([0, 2]))
    assert torch.equal(mgr._interval_term_time_left[1], torch.full((8,), 0.5))


def test_class_term_keeps_mjlab_16_resample_on_reset():
    mgr = _stub_manager()
    mgr.reset(torch.tensor([0, 2]))
    t = mgr._interval_term_time_left[2]
    assert (t[[0, 2]] >= 3.0).all() and (t[[0, 2]] <= 6.0).all()
    assert torch.equal(t[[1, 3, 4, 5, 6, 7]], torch.full((6,), 0.5))
