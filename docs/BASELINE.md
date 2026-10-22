# Baseline Comparison Contract

**Golden** is the current local-credit ReciprocalPC release. **baseline** is the
ordinary-backpropagation NanoChat reference, not an exact-credit PC oracle.

| Property | Golden | baseline |
| --- | --- | --- |
| Learned parameters | 23,930,435 (native + one scalar) | 36,700,242 |
| Native / reference source | `ec993a3cdcbcf677b1d199a06695a63e779bbefa` | `78af72aab915566427a094b84403033d041e10c1` |
| Training objective/credit | Strict local reciprocal + output-local RLS gain | Ordinary backpropagation |
| Released model state | 50k-to-49k 20% interpolation, inference-only | Existing NanoChat raw 50k reference |

The frozen 50k quality protocol uses the same 32,768-token tokenizer, seed 42,
training rows/order, 2,048-token sequence, batch one, 2,048 tokens/update, 50,000
updates and 102,400,000 training target-position exposures. Training is one pass
through the frozen positions, not replay. Evaluation reuses 100 frozen Phase574
development rows. Primary BPB and accuracy both exclude zero-byte targets:
204,645 positive-byte targets out of 204,800 seen targets.

Frozen identity:

- Training concatenation SHA-256: `6f88470f7d73b244f74b171551d31c06e26b8e556a61b1e2dc2516a8bbf4f5c5`.
- Validation shard SHA-256: `161fbb63dc59cd7e757f46d8e00e6394d293dec87ede8a9e8a5593279d348ad6`.
- Tokenizer SHA-256: `387cfc082b0bee45467774fd6f1310a922ad170886a58ccddcb468f275e06a6c`.
- Token-byte table SHA-256: `ebe12e9681bb92108e84af09fa628eb7a3d2140339defc4cf80b867f8a916e7c`.

Equal exposures do **not** imply equal parameter count, arithmetic or hardware.
The existing baseline used BF16 compute on RTX 3090; the native path uses FP32
parameters/carriers and selected grouped-ternary deployed matrices. Golden's
interpolated inference score is explicitly transformed, not a replacement for a
raw checkpoint in a raw-vs-raw comparison. Do not infer a parameter-matched win,
learning-efficiency win, chat capability or sealed-test generalization from it.

## Current measured interpolation comparison

| Model state | BPB | Exact-token accuracy | Correct / positive-byte targets |
| --- | ---: | ---: | ---: |
| Unaveraged 50k RLS reference | `1.2598953782653626` | `0.294925358547729` | 60,355 / 204,645 |
| Promoted Golden interpolation | `1.2595738817841562` | `0.29519411664101247` | 60,410 / 204,645 |

This table compares two Golden states, **not NanoChat scores**. It does not
silently substitute historical baseline values into a current comparison.

## Speed boundary

The `1.2676003549193338x` A40 result compares shared-prefill Golden updates with
unchanged Golden updates only. It is not baseline speed, not inference speed,
and not a new 50k run. Different training hardware and arithmetic prohibit an
external speed claim from these runs.

## Historical 5k comparison

The previous release reported native Phase827 BPB `1.4867165192`, accuracy
`0.2393999365`, against matched GPT BPB `1.5625470909`, accuracy `0.2420239928`,
after 10,240,000 exposures. Those historical scores are **not current Golden or
current 50k baseline scores**. Never splice that schedule into the 50k curve.
