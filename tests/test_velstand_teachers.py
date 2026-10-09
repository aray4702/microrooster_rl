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
    "policies/microduck/no_such_policy.onnx",  # ONNX path that does not exist
])
def test_malformed_override_fails_loudly(monkeypatch, value):
    monkeypatch.setenv(distill.WALK_EXPERT_ENV, value)
    with pytest.raises(ValueError, match=distill.WALK_EXPERT_ENV):
        distill.default_bc_cfg()


def test_onnx_override_is_a_local_teacher(monkeypatch):
    # Exported policies in the repo ship to HF Jobs with the source tarball.
    monkeypatch.chdir(_ROOT)
    monkeypatch.setenv(distill.STAND_EXPERT_ENV, "policies/microduck/alpha_stand.onnx")
    monkeypatch.setenv(distill.WALK_EXPERT_ENV, "me/mjlab_microduck/walk01@model_3749.pt")
    cfg = distill.default_bc_cfg()
    assert cfg["checkpoint_path"] == "policies/microduck/alpha_stand.onnx"
    assert cfg["wandb_run_path"] is None
    assert distill._resolve_checkpoint(cfg) == Path("policies/microduck/alpha_stand.onnx")
    assert (cfg["anchor_wandb_run_path"], cfg["anchor_checkpoint_path"]) == ("me/mjlab_microduck/walk01", None)


def _velstand_actor():
    import torch
    from rsl_rl.models.mlp_model import MLPModel
    from tensordict import TensorDict

    return MLPModel(
        TensorDict({"actor": torch.zeros(1, 61)}, batch_size=[1]), {"actor": ["actor"]}, "actor", 14,
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def test_onnx_teacher_reproduces_the_exported_policy():
    # The rebuilt expert must act exactly like the ONNX it came from — normalizer
    # included (Div = std + eps) — or BC would clone a different policy.
    import numpy as np
    import onnxruntime as ort
    import torch
    from tensordict import TensorDict

    path = _ROOT / "policies/microduck/alpha_stand.onnx"
    actor = _velstand_actor()
    expert, _ = distill._load_teacher(path, actor, "cpu")
    obs = torch.randn(32, 61, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        got = expert(TensorDict({"actor": obs}, batch_size=[32])).numpy()
    sess = ort.InferenceSession(str(path))
    want = np.concatenate([sess.run(None, {"obs": obs[i:i + 1].numpy()})[0] for i in range(32)])
    np.testing.assert_allclose(got, want, atol=1e-4)
    assert not any(p.requires_grad for p in expert.parameters())


def test_hf_jobs_forwards_microduck_vars():
    # A teacher override that stays on the laptop would make the job silently
    # fall back to the unreachable defaults.
    src = (_ROOT / "src/mjlab_microduck/hf_jobs.py").read_text()
    assert 'k.startswith("MICRODUCK_")' in src
