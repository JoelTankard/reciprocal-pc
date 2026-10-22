# Golden Artifacts and Loading

## Public release downloads

Download the three matched assets from GitHub release **`golden-20261022`**.

| Asset | SHA-256 |
| --- | --- |
| `golden-inference.pt` | `31932202917638a4d4f5464ecd340f296de1949bf58fa5dcbf89fdbfcbbe2e85` |
| `tokenizer.pkl` | `387cfc082b0bee45467774fd6f1310a922ad170886a58ccddcb468f275e06a6c` |
| `token_bytes.pt` | `ebe12e9681bb92108e84af09fa628eb7a3d2140339defc4cf80b867f8a916e7c` |

- <https://github.com/JoelTankard/reciprocal-pc/releases/download/golden-20261022/golden-inference.pt>
- <https://github.com/JoelTankard/reciprocal-pc/releases/download/golden-20261022/tokenizer.pkl>
- <https://github.com/JoelTankard/reciprocal-pc/releases/download/golden-20261022/token_bytes.pt>

The tokenizer and byte table are the frozen assets used by the measured model,
not interchangeable with an arbitrary tokenizer. Golden is a base language
model with token APIs, not chat-trained. Pickle/PyTorch files should only be
loaded from a trusted, hash-verified source.

The model export is exactly **96,134,229 bytes**, with schema
`golden-drift-inference-only.v1` and payload fields `schema`, `model_config`,
`model`, `branch`, `transformation`. It contains native weights and the RLS gain
and fixed projection. This is a dense-state PyTorch serialization with selected
ternary-valued matrices, **not** an all-ternary packed deployment artifact.

```bash
shasum -a 256 golden-inference.pt tokenizer.pkl token_bytes.pt
```

```python
from pfc import load_golden_checkpoint

model, payload = load_golden_checkpoint("golden-inference.pt", map_location="cpu")
```

The public loader constructs Golden directly, checks the release schema and
configuration, restores native and branch state, and refreshes packed execution
state. It does not require trainer restore, archived experiment imports, CUDA,
or machine-absolute paths. The original GPU export/reload witness had exact
same-device logits; that witness is not a claim of bitwise CPU/CUDA equality.

The repository validator checks the release hash and exact reload logits at
64- and 2,048-token lengths by default:

```bash
PYTHONPATH=src python tools/validate_release_checkpoint.py golden-inference.pt --device cpu
```

To additionally exercise the optimized strict-local training helper on a
**fresh unaveraged model**, add `--training-update`. Its training length defaults
to 128; use `--training-length 2048` for the full training context. This is not a
resume of the inference artifact. Native multihorizon training requires more
than 64 token positions. These commands are validation instructions, not a
claim that they have been run as part of editing this documentation.

## Interpolation provenance and resume prohibition

| Input | Update | SHA-256 |
| --- | ---: | --- |
| Later unaveraged RLS checkpoint | 50,000 | `c376f9e9e737c9018ae00a87406f1753ad3d44ba231670ac0743108d1dbda095` |
| Earlier anchor | 49,000 | `5e2b8d4e418ef3491535554910f2155bfcf7339fc8066610aaaa0e3d87e18ac7` |

Each source full-state checkpoint was `509,586,635` bytes. The native source
commit was `ec993a3cdcbcf677b1d199a06695a63e779bbefa`. The transformation is
`lerp(later, anchor, 0.2)`: ternary-owned weights interpolate FP32 carriers and
then use the existing grouped quantizer; other learned floating weights
interpolate directly. RLS gain `0.057484474033117294` and its projection remain
unchanged. No new training updates were performed.

**Never resume `golden-inference.pt`.** Optimizer moments, reciprocal feedback,
schedule, RNG and cursor were not reconciled into a training continuation. The
native `load_reciprocal_pc_checkpoint` API is retained for its separate native
checkpoint contract and rejects inference-only exports. Training continues from
complete **unaveraged** state, not from the promoted interpolation.

## Historical release, not current Golden

The old Phase827 checkpoint SHA-256 was
`ab27c225ee0b5d082de0e6b772dca441fb55f4aa89c9a5734cf475f0545ad2ba` at 5,000
updates / 10,240,000 exposures, with BPB `1.4867165192023712` and accuracy
`0.23939993647535976`. Those values and the old source commit are retained only
under `historical_phase827_5k` in the release manifest. They must not be presented
as current Golden measurements or spliced into its 50k curve.
