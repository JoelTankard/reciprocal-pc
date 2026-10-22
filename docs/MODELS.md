# Golden Model Contract

## Identity and architecture

**Golden** is the public ReciprocalPC release. **baseline** means the
ordinary-backpropagation NanoChat control (36,700,242 parameters). Blue
exact-credit PC is a diagnostic oracle, not either public model.

Golden combines `NativeReciprocalPCLanguageModel` with `Golden` (the additive
RLS branch). The native source snapshot is
`ec993a3cdcbcf677b1d199a06695a63e779bbefa`; package integration does not change
the identity of the measured weights.

- Width `512`, two recurrent layers, SwiGLU FFN width `1,040`.
- Vocabulary `32,768`, tied learned embedding/decoder, training context `2,048`.
- `23,930,435` learned parameters: `23,930,434` native plus one branch gain.
- **Selectively ternary**: grouped ternary-owned matrices and fixed ternary
  features coexist with dense lexical weights, other dense parameters, and FP32
  training carriers. Not every recurrent or FFN matrix is a ternary matrix.
- The release is a **base language model with token APIs**, not an SFT/chat model.

## Forward and causal memory path

1. Token IDs select rows from the shared embedding.
2. Native layers use an affine recurrent scan and local SwiGLU features.
3. Fixed ternary Hadamard lexical features, temporal HGF precision, and projected
   self-bilinear token observation contribute to the native representation.
4. The final normalized representation uses the tied lexical decoder.
5. A causal predictive-RLS branch adds a learned-gain logit correction using only
   tokens already observed. Labels and target masks never enter forward execution.

The model uses persistent native recurrent state plus branch state, rather than
an attention KV cache growing with context. The FP32 **additional branch** state
(`S`, `P`, previous key) is `69,760` bytes per batch lane; this is not total model
memory. Branch state and its computation are additional costs. No lower-memory
or faster-inference conclusion follows from parameter count alone.

## Public inference API

```python
import torch
from pfc import Golden, RLSState, RLSBranchState, create_golden, load_golden_checkpoint

untrained = create_golden(device="cpu", seed=539)
model, payload = load_golden_checkpoint("golden-inference.pt", map_location="cpu")
first = model.prefill(torch.tensor([[32759, 1, 2]], dtype=torch.long))
continued = model(torch.tensor([[3, 4]], dtype=torch.long), first["state"])
one = model.step(torch.tensor([5], dtype=torch.long), continued["state"])
```

`create_golden` constructs the released architecture but **does not load released
weights**. `load_golden_checkpoint` loads the actual inference-only artifact
without constructing/restoring a trainer. `forward` and `prefill` take `[B,T]`
int64 tokens and return `logits`, `native_logits`, `branch_logits` and `state`;
`step` takes one token per lane (`[B]`) and returns `[B,V]` logits plus state.

State is caller-owned (`RLSBranchState`, with native state and RLS memory).
Pass it to continue the same sequence; omit it to reset independent contexts.
Do not accidentally carry it between unrelated examples. Full-sequence and
stateful execution are causal. `branch_state_dict()` /
`load_branch_state_dict()` serialize the slow branch gain and fixed projection,
not the native weights or transient per-sequence memory; `state_bytes` exposes
branch-memory accounting.

## Strict-local training remains unaveraged

`create_reciprocal_pc()` constructs the original native model/trainer pair.
`load_reciprocal_pc_checkpoint()` remains for native historical training
checkpoints; it is **not** the loader for the promoted inference artifact.
`NativePCTrainer` uses local prediction errors and learned reciprocal feedback,
with dense and grouped-ternary update paths. The RLS scalar's output-local
credit is assigned explicitly outside autograd; do not send all branch/native
parameters to the gain optimizer.

The executable optimized helper is:

```python
from pfc import Golden, create_reciprocal_pc, golden_local_update
import torch

native_model, trainer = create_reciprocal_pc()
branch = Golden(native_model)
gain_optimizer = torch.optim.Adam([branch.gain], lr=5e-5)
# tokens and targets are the caller's int64 training minibatch tensors.
metrics = golden_local_update(branch, trainer, gain_optimizer, tokens, targets)
```

`golden_local_update(branch, trainer, gain_optimizer, tokens, targets)` returns
`{"native": native_metrics, "gain": gain_objective}`. It runs branch forward,
output-local gain credit and native trainer update inside
`shared_prefill(native_model, trainer, tokens)`; the gain optimizer step follows
the scope. Sharing one native prefill replaces two without changing the intended
update. The low-level context manager is also exported for existing trainers.
The gain optimizer must own only `[branch.gain]`, never `branch.parameters()`.

Training invariants remain:

- `global_backprops == 0`;
- `forward_weight_transpose_reads == 0`;
- no exact VJP used as learned credit;
- no target-dependent feature in deployed forward logits;
- tied token embedding and decoder.

The promoted interpolation is a **separate inference transformation** of the
50k and 49k snapshots, not a changed training algorithm or training checkpoint.
Never resume it: optimizer moments and reciprocal feedback are unreconciled.
The full unaveraged model/trainer/optimizer/schedule/RNG/data-cursor checkpoint
is needed for a true resume; a branch slow-state export alone is insufficient.

## Measured scope

Current Golden BPB is `1.2595738817841562`; exact-token accuracy is
`60,410 / 204,645 = 0.29519411664101247` on reused development rows. See
[FINDINGS.md](FINDINGS.md) for the explicit promotion decision and subgroup
regressions. The retained 1.268x optimization result is internal training-update
throughput, not a comparison against baseline or deployed inference speed.
