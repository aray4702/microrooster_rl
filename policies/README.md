# Policies

Exported ONNX policies, committed so they play right after a clone (see
"Play in the MuJoCo simulator" in the [README](../README.md)). Each is under
1 MB. The obs normalizer is baked in (exported via `scripts/export.py` /
the HF Jobs auto-export), so they run as-is.

## `microduck/`

The policy set the Microduck ships with: an unmodified copy of
[`pollen-robotics/microduck-policies`](https://huggingface.co/pollen-robotics/microduck-policies)
at revision `6677ec007142bbaf05123a468f6ae58d13808786` (Apache-2.0, Pollen
Robotics). `manifest.json` describes each file; `velstand.onnx` is the
default walk policy. 61-D obs -> 14-D actions; run with
`scripts/infer_policy.py --new-cmd-obs`.

To refresh from the Hub:

```bash
uv run hf download pollen-robotics/microduck-policies --local-dir policies/microduck \
  --include "*.onnx" "manifest.json"
```

## `microrooster/`

| File | Task | Source | Notes |
| --- | --- | --- | --- |
| `velocity_v1.onnx` | `Mjlab-Velocity-Flat-MicroRooster` | wandb `aray4702-tt/mjlab_microrooster/bw9t2smj`, `model_3999` (resumed from `nsvs6avi@3500`); HF `danceone/mjlab-velocity-flat-microrooster-20261007-203123`, `exported/policy.onnx` | walks forward ~0.2 m/s; does not turn or stand still yet (README policy log) |

53-D obs -> 12-D actions; run with
`scripts/view_rooster.py --policy <file> --new-cmd-obs`. Add new versions as
`velocity_v2.onnx`, … with a row here, and keep old ones for comparison.
