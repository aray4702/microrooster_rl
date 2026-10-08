from pathlib import Path

import pytest

from mjlab_microduck.tasks import distill

_ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_the_pollen_teachers(monkeypatch):
    monkeypatch.delenv(distill.WALK_EXPERT_ENV, raising=False)
    monkeypatch.delenv(distill.STAND_EXPERT_ENV, raising=False)
    cfg = distill.default_bc_cfg()
    assert cfg["wandb_run_path"] == "pollen-robotics/mjlab_microduck/69u48n8l"
    assert cfg["checkpoint_name"] == "model_9750.pt"
    assert cfg["anchor_wandb_run_path"] == "pollen-robotics/mjlab_microduck/441tzs6d"
    assert cfg["anchor_checkpoint_name"] == "model_3750.pt"


def test_env_overrides_route_each_teacher(monkeypatch):
    monkeypatch.setenv(distill.WALK_EXPERT_ENV, "me/mjlab_microduck/walk01@model_3750.pt")
    monkeypatch.setenv(distill.STAND_EXPERT_ENV, "me/mjlab_microduck/stand01@model_9750.pt")
    cfg = distill.default_bc_cfg()
    # stand expert = the fallen-frame teacher; walk expert = the upright anchor
    assert (cfg["wandb_run_path"], cfg["checkpoint_name"]) == (
        "me/mjlab_microduck/stand01", "model_9750.pt")
    assert (cfg["anchor_wandb_run_path"], cfg["anchor_checkpoint_name"]) == (
        "me/mjlab_microduck/walk01", "model_3750.pt")


@pytest.mark.parametrize("value", [
    "me/mjlab_microduck/walk01",              # no checkpoint
    "walk01@model_3750.pt",                   # not a full run path
    "me/mjlab_microduck/walk01@3750",         # not a .pt name
])
def test_malformed_override_fails_loudly(monkeypatch, value):
    monkeypatch.setenv(distill.WALK_EXPERT_ENV, value)
    with pytest.raises(ValueError, match=distill.WALK_EXPERT_ENV):
        distill.default_bc_cfg()


def test_hf_jobs_forwards_microduck_vars():
    # A teacher override that stays on the laptop would make the job silently
    # fall back to the unreachable defaults.
    src = (_ROOT / "src/mjlab_microduck/hf_jobs.py").read_text()
    assert 'k.startswith("MICRODUCK_")' in src
