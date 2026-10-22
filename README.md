# Golden · ReciprocalPC

**Golden** is the current public ReciprocalPC model: a strictly local recurrent
language model with a causal additive predictive-RLS inference branch. The
ordinary-backpropagation comparison model is **baseline** (NanoChat, 36,700,242
parameters). Golden's training uses no global backpropagation, no exact block
VJP credit, and no forward-weight transpose read.

Golden is a **base language model with token APIs**, not a chat-trained model.

## Current release: Golden, 2026-10-22

| Property | Value |
| --- | ---: |
| Native width / recurrent layers / FFN width | 512 / 2 / 1,040 |
| Vocabulary / training context | 32,768 / 2,048 tokens |
| Learned parameters | 23,930,435 (23,930,434 native + one RLS gain) |
| Native training updates / token exposures | 50,000 / 102,400,000 |
| Released weights | inference-only, 20% interpolation toward update 49,000 |
| Measured BPB | `1.2595738817841562` |
| Exact-token accuracy | `0.29519411664101247` (29.5194%) |
| Accuracy denominator | 60,410 correct / 204,645 positive-byte targets |
| Global backprops / forward-weight transpose reads | 0 / 0 |

These are measured on 100 **reused development rows**, not a sealed test. The
user explicitly authorized promotion despite reused data and token-length
subgroup regressions. The aggregate improvement over the unaveraged 50k RLS
reference is small: 55 more correct targets and BPB lower by
`0.00032149648120638474`. This is a promotion decision, not evidence that every
subgroup improved or that generalization is established.

The model is **selectively ternary, not all-ternary**. Grouped ternary matrices
coexist with dense embedding/decoder, other dense parameters, and FP32 training
carriers. The 96,134,229-byte inference artifact is a real PyTorch state export,
not a claim of fully packed two-bit deployment. RLS adds computation and fixed
per-lane state; fewer parameters do not prove faster inference.

## Install

Python 3.10 or newer is required.

```bash
uv sync --extra dev
```

## Load the promoted inference weights on CPU

Download **`golden-inference.pt`** from GitHub release **`golden-20261022`**:

<https://github.com/JoelTankard/reciprocal-pc/releases/download/golden-20261022/golden-inference.pt>

Save it to a local file and verify
SHA-256 `31932202917638a4d4f5464ecd340f296de1949bf58fa5dcbf89fdbfcbbe2e85`.

Download the matched tokenizer assets from the same release:

- [`tokenizer.pkl`](https://github.com/JoelTankard/reciprocal-pc/releases/download/golden-20261022/tokenizer.pkl),
  SHA-256 `387cfc082b0bee45467774fd6f1310a922ad170886a58ccddcb468f275e06a6c`.
- [`token_bytes.pt`](https://github.com/JoelTankard/reciprocal-pc/releases/download/golden-20261022/token_bytes.pt),
  SHA-256 `ebe12e9681bb92108e84af09fa628eb7a3d2140339defc4cf80b867f8a916e7c`.

Use these frozen assets, not an arbitrary tokenizer; only load trusted,
hash-verified pickle/PyTorch files.

```python
import torch
from pfc import create_golden, load_golden_checkpoint

# Architecture only, randomly initialized; not the measured release weights.
untrained_model = create_golden(device="cpu")

# Actual promoted artifact: no trainer restore or archived run imports required.
model, metadata = load_golden_checkpoint("golden-inference.pt", map_location="cpu")
tokens = torch.tensor([[32759, 1, 2, 3]], dtype=torch.long)
output = model.prefill(tokens)
logits = output["logits"]
next_output = model.step(torch.tensor([4]), output["state"])
```

`forward`/`prefill` return logits and caller-owned persistent state. Pass that
state to continue a sequence, or omit it to start independent contexts. The
causal branch never receives target labels. See [the model contract](docs/MODELS.md)
for state and training APIs.

## Training and measured optimization

Training remains the original **unaveraged** native architecture plus the local
RLS gain update. Use `golden_local_update` for the optimized shared-prefill update;
`create_reciprocal_pc` and `load_reciprocal_pc_checkpoint` remain the native
training/historical-checkpoint APIs. The interpolated release is **never a
resumable training checkpoint**: optimizer moments and reciprocal feedback were
not reconciled by interpolation.

A short paired A40 benchmark measured **1.268× internal update throughput**
(`1.2676003549193338×`, 40 timed updates per arm across two rounds) by sharing one
native prefill instead of doing two. Model/trainer/gain-optimizer/RNG state and
outputs were exact in the retained proof. This compares Golden execution paths,
**not baseline training speed**, and excludes validation, checkpointing and setup.
There is no new inference-speed claim or new 50k optimized training run.

## Release evidence and documentation

- [Model and local-credit contract](docs/MODELS.md)
- [Artifact identity and loading](docs/CHECKPOINTS.md)
- [Baseline comparison boundaries](docs/BASELINE.md)
- [Measured findings and promotion decision](docs/FINDINGS.md)
- [Experiment rules](docs/EXPERIMENT_RULES.md)
- [Longer-term chat-model roadmap](docs/path-to-gpt-2.md)
- [Self-contained release manifest](configs/reciprocalpc-release-manifest.json)
- [Internal update-speed evidence](configs/golden-update-speed-evidence.json)

The older Phase827 **5k** scores (BPB `1.4867165192`, accuracy `0.2393999365`)
are historical, not current Golden scores. Source provenance for the native
runtime is `ec993a3cdcbcf677b1d199a06695a63e779bbefa`; public packaging and RLS
integration are described separately in the manifest. No archived run tree or
machine-specific absolute path is needed to use this release.
