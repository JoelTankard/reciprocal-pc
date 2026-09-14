# Phase827 Checkpoint

The retained release checkpoint is:

| Property | Value |
| --- | --- |
| Windows path | `R:/reciprocal-pc/matched-accuracy/phase827-matched-accuracy-20260911-v1/inputs/phase827/phase827-step005000.pt` |
| SHA-256 | `ab27c225ee0b5d082de0e6b772dca441fb55f4aa89c9a5734cf475f0545ad2ba` |
| Update | `5,000` |
| Token exposures | `10,240,000` |
| Validation BPB | `1.4867165192023712` |
| Exact-token accuracy | `0.23939993647535976` |
| Global backprops | `0` |
| Forward-weight transpose reads | `0` |

The checkpoint is not committed because it is a large binary artifact. Verify
its hash before loading.

```bash
shasum -a 256 phase827-step005000.pt
```

Load through the public adapter:

```python
from pfc import load_reciprocal_pc_checkpoint

model, trainer, metadata = load_reciprocal_pc_checkpoint(
    "phase827-step005000.pt",
    map_location="cpu",
)
```

The adapter constructs the exact release configuration, loads both model and
trainer state strictly, and rejects incompatible configuration or untied
lexical state. Preserving trainer state is required for an exact training
resume; loading only model weights is sufficient for inference.
