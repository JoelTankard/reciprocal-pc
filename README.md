# ReciprocalPC

ReciprocalPC is a recurrent language model trained with strictly local reciprocal
credit. The released model performs no end-to-end backpropagation, no exact
block VJP, and no forward-weight transpose read.

This repository is the clean Phase827 release. It contains the model, its local
trainer, checkpoint loader, focused behavioral tests, and the measured release
contract. Historical experiment machinery is intentionally excluded.

## Released model

| Property | Value |
| --- | ---: |
| Width | 512 |
| Recurrent layers | 2 |
| FFN width | 1,040 |
| Vocabulary | 32,768 |
| Context length | 2,048 tokens |
| Learned parameters | 23,930,434 |
| Weight form | grouped ternary |
| Global backprops | 0 |
| Forward-weight transpose reads | 0 |

The retained 5,000-update checkpoint reached validation BPB `1.4867165192` and
exact-token accuracy `0.2393999365` after `10,240,000` token exposures. These are
the measured Phase827 release values; later research architectures and their
50k curves are not mixed into this release.

## Install

Python 3.10 or newer is required.

```bash
uv sync --extra dev
uv run pytest -q
```

## Use

```python
import torch
from pfc import create_reciprocal_pc, load_reciprocal_pc_checkpoint

model, trainer = create_reciprocal_pc()
logits = model(torch.tensor([[32759, 1, 2, 3]], dtype=torch.long))

model, trainer, metadata = load_reciprocal_pc_checkpoint(
    "phase827-step005000.pt",
    map_location="cpu",
)
```

`create_reciprocal_pc()` always builds the released architecture. The loader
rejects checkpoints whose model configuration, local-training configuration,
projected observation matrix, or tied lexical endpoint differ from the release
contract.

## Documentation

- `docs/MODELS.md`: architecture, inference, and local-credit path.
- `docs/CHECKPOINTS.md`: retained checkpoint identity and loading contract.
- `docs/EXPERIMENT_RULES.md`: fair-comparison and strict-credit gates.
- `docs/path-to-gpt-2.md`: staged path from this research model to a chat model.
- `configs/reciprocalpc-release-manifest.json`: source identity and release metrics.

Internal `stage_reciprocal` keys and historical checkpoint tags remain unchanged
for compatibility. The public model name is `ReciprocalPC`.
