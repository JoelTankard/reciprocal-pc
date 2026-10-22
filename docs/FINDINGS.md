# Current Golden Findings and Promotion Decision

## 2026-10-22: user-authorized promotion

The user explicitly authorized crowning the measured short-lookback
interpolated candidate as **Golden**, and publishing its inference artifact.
This supersedes the experiment summary's original
`aggregate_development_gate_passed_not_promoted` status. It does **not** convert
the reused development evaluation into a sealed test, remove subgroup
regressions, or make the artifact resumable. Release downloads and immutable
asset identities are listed in [CHECKPOINTS.md](CHECKPOINTS.md).

Native training completed 50,000 updates at 2,048 target positions/update:
102,400,000 exposures, one pass through frozen positions. The promoted weights
move 20% from update 50,000 toward update 49,000, with carrier interpolation and
the original quantizer for ternary-owned matrices, direct interpolation for
other floating learned parameters, and unchanged RLS gain/projection. This
requires no additional training updates and is not the original full PR recipe.

## Aggregate measured result

| State | BPB | Accuracy | Correct |
| --- | ---: | ---: | ---: |
| Unaveraged 50k RLS reference | `1.2598953782653626` | `0.294925358547729` | 60,355 |
| Promoted Golden interpolation | `1.2595738817841562` | `0.29519411664101247` | 60,410 |
| Candidate minus reference | `-0.00032149648120638474` | `+0.02687580932834721` percentage points | +55 |

Both primary metrics use **204,645 positive-byte targets** in the same 100
frozen Phase574 development rows (971,222 bytes). There are 204,800 all-valid
positions, including 155 zero-byte targets; Golden's separately reported
all-valid accuracy is `0.29521484375`. Do not interchange denominators. The
experiment used A40, PyTorch `2.8.0+cu128`, FP32 logits and FP64 nats accumulation.
Same-device export/reload logits were exact; retrieved artifact hash matched.

## Subgroup regressions retained, not hidden

Flagged token-byte-length groups with either fewer correct predictions or
higher BPB are below. Deltas are candidate minus the unaveraged reference.

| Byte length | Targets | Delta correct | Delta BPB |
| ---: | ---: | ---: | ---: |
| 1 | 23,329 | -6 | +0.001194027718230295 |
| 2 | 19,895 | -9 | -0.00009661414081207909 |
| 4 | 33,553 | +25 | +0.0008463841409844797 |
| 6 | 17,759 | -11 | -0.0003782777597653997 |
| 7 | 13,329 | -7 | -0.0006177230039805526 |
| 11 | 4,007 | -1 | -0.0006821728516720471 |
| 13 | 1,203 | -1 | -0.0005643305287477141 |
| 15 | 258 | -1 | +0.0014771006337306325 |
| 17 | 9 | 0 | +0.003920731206364714 |
| 19 | 1 | 0 | +0.0024634070772110994 |

The user accepted these limitations for promotion. The aggregate reused-data
gate alone would not establish a generalization improvement. No early-stop,
chat-training, inference-speed or large-scale scaling-law claim is supported.

## Internal execution improvement, separate from weight interpolation

Shared-prefill replaces two native prefills with one per Golden update. A short
counterbalanced A40 trained-state benchmark (two rounds, 40 timed updates per
arm total) measured `0.534744858074555` to `0.4218560337249073` seconds/update:
`1.2676003549193338x` throughput, or `21.110782580710385%` lower update time.
Outputs, gain credit, native model/trainer, gain optimizer and RNG were exact in
the retained proof. Timing includes finite guards/synchronization and excludes
validation/checkpoint/setup. This is **not a NanoChat baseline speed comparison**,
not a new 50k run, and not an interpolation speed benefit.

## Storage and historical boundaries

Golden has 23,930,435 learned parameters, including one RLS scalar; baseline has
36,700,242. Dense lexical/other weights and FP32 training carriers remain.
Selective ternarity is not all-ternary deployment; the exported inference file
is 96,134,229 actual bytes. RLS adds work and 69,760 FP32 branch-state bytes per
batch lane, beyond native state, weights and temporaries.

The old Phase827 5k BPB `1.4867165192` / accuracy `0.2393999365` and its matched
GPT comparison are historical, not Golden scores. Current provenance is pinned
in [the manifest](../configs/reciprocalpc-release-manifest.json), without importing
old run archives or machine-dependent paths.
