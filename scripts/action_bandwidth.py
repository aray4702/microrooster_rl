"""Is the policy using the control bandwidth it already has?

The case for 100 Hz rests on the policy being resolution-limited. That is
testable without retraining anything: look at the spectrum of the commanded
action. If its power rolls off well below Nyquist (25 Hz at a 50 Hz control
rate), the policy is not asking for bandwidth it does not have, and doubling
the rate buys nothing. If power piles up against Nyquist, it is clipped by the
rate and more would help.

Context that makes the answer likely: the BAM actuator delay is 15-30 ms, so
the 20 ms control period at 50 Hz already sits inside the plant's latency.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HOP_NO_PUSH", "1")
os.environ.setdefault("HOP_NO_HOLD", "1")
os.environ.setdefault("HOP_NO_DR", "1")

import numpy as np
import onnxruntime as ort
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ARMS = [
    ("exports/hopfree_k4290_fixed_138d5t8o.onnx",
     "Mjlab-HopFree-S50-DR-Sym-K3344-MicroDuck", "sprung k=4290"),
    ("exports/hoplocked_fixed_8bs9qhaf.onnx",
     "Mjlab-HopLocked-S50-DR-Sym-Locked-MicroDuck", "rigid"),
]
STEPS, ENVS, WARMUP = 512, 16, 64
LEG = [2, 3, 4, 11, 12, 13]          # hip_pitch, knee, ankle both sides


def run(onnx, task):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = ENVS
    keep = {"reset_base", "reset_robot_joints", "expand_bam_friction_fields",
            "reset_action_history"}
    cfg.events = type(cfg.events)({k: v for k, v in cfg.events.items() if k in keep})
    cfg.curriculum = type(cfg.curriculum)()
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.extras.setdefault("log", {})
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    obs, _, _, _, _ = env.step(torch.zeros((ENVS, env.action_manager.total_action_dim)))
    acts = []
    for _ in range(STEPS):
        o = obs["actor"].detach().cpu().numpy().astype(np.float32)
        a = np.concatenate([sess.run(None, {iname: o[e:e + 1]})[0]
                            for e in range(ENVS)], axis=0)
        obs, _, _, _, _ = env.step(torch.as_tensor(a))
        acts.append(a.copy())
    return np.array(acts), env.step_dt      # [T, ENVS, A]


for onnx, task, tag in ARMS:
    if not os.path.exists(onnx):
        print(f"{tag}: {onnx} missing")
        continue
    A, dt = run(onnx, task)
    A = A[WARMUP:, :, LEG]                       # leg joints only
    fs = 1.0 / dt
    T = A.shape[0]

    # Per-step change, as a fraction of the action's own range.
    d = np.abs(np.diff(A, axis=0))
    rng = A.max(axis=0) - A.min(axis=0)
    frac = (d.mean(axis=0) / np.maximum(rng, 1e-6)).mean()

    # Spectrum of the commanded action, averaged over envs and joints.
    win = np.hanning(T)[:, None, None]
    P = np.abs(np.fft.rfft((A - A.mean(axis=0)) * win, axis=0)) ** 2
    P = P.mean(axis=(1, 2))
    f = np.fft.rfftfreq(T, dt)
    cum = np.cumsum(P) / P.sum()

    def fq(q):
        return float(f[np.searchsorted(cum, q)])

    nyq = fs / 2
    above_half = P[f > nyq / 2].sum() / P.sum()

    print(f"\n{tag}   ({fs:.0f} Hz control, Nyquist {nyq:.0f} Hz)")
    print(f"  mean |delta action| per step   {frac*100:5.1f}% of the joint's range")
    print(f"  median action frequency        {fq(0.50):5.1f} Hz")
    print(f"  90% of action power below      {fq(0.90):5.1f} Hz")
    print(f"  99% of action power below      {fq(0.99):5.1f} Hz")
    print(f"  power above half-Nyquist       {above_half*100:5.1f}%")
    verdict = ("CLIPPED by the control rate" if above_half > 0.25
               else "comfortably inside the current rate")
    print(f"  -> {verdict}")
