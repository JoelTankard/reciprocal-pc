# Path to the Nanochat Chat Model

## Overall Goal

Scale ReciprocalPC into a local-device chat model that outperforms the existing
nanochat chat model while:

- learning from fewer unique pretraining and chat-training tokens;
- using no more total token exposures;
- using less inference memory, especially at long context;
- keeping prompt loading and generation practical;
- preserving strict local reciprocal credit with zero global backpropagation;
- remaining deployable on local hardware after training on rented GPUs.

This is the project's overall path. The filename remains `path-to-gpt-2.md`
because nanochat's reference model targets GPT-2-grade base capability, but the
actual end product and final comparison are chat models.

No UI work belongs on this path. Required interface: tokenized conversations in,
assistant tokens out.

## What Nanochat Actually Trains

Nanochat does not start from OpenAI's released GPT-2 weights. It follows the
same broad two-stage pattern:

1. Train its own GPT-style base model from scratch on ordinary text.
2. Continue training that base checkpoint on conversation data using supervised
   fine-tuning (SFT).

Current `runs/speedrun.sh` does:

```text
raw text
  -> tokenizer
  -> d24 base-model pretraining
  -> base evaluation
  -> conversation SFT
  -> chat evaluation
```

The current reference speedrun uses a 32,768-token vocabulary, a depth-24 model,
a target parameter-to-data ratio of 8, FP8 training on eight GPUs, then
`scripts.chat_sft` and `scripts.chat_eval`. It does not run chat RL.

Pretraining supplies:

- language;
- factual associations;
- completion ability;
- most reasoning capacity.

SFT supplies:

- user/assistant turn structure;
- response termination;
- instruction-following behavior;
- tool and multiple-choice formatting;
- conversational style.

Pure chat-data training from random initialization would waste scarce
conversation data teaching basic language. It would usually produce a worse
base model. Our path should therefore match nanochat's base-pretrain-then-SFT
structure, replacing only the GPT architecture and global backpropagation with
ReciprocalPC and strict local credit.

Primary reference:
[nanochat `runs/speedrun.sh`](https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh).

## Exact Target

Before final scaling, freeze one nanochat Git commit and reproduce its reference
speedrun. That pinned run becomes the control. Nanochat master changes too often
to serve as an unfrozen target.

Record:

- Git commit;
- model configuration and parameter count;
- tokenizer and dataset hashes;
- unique pretraining tokens;
- pretraining exposures;
- unique SFT tokens;
- SFT exposures;
- base BPB and CORE;
- final ChatCORE and component metrics;
- training FLOPs and wall time;
- prefill and decode throughput;
- weight, KV-cache, and peak inference memory.

The final target is the SFT checkpoint, not the base checkpoint and not a web
demo.

## Final Win Contract

ChatPC wins only if all required gates pass against the pinned nanochat SFT
control.

| Axis | Final requirement |
|---|---|
| Chat quality | Better preregistered aggregate chat score |
| Component quality | No major hidden regression on core chat tasks |
| Unique data | Fewer total unique pretraining plus SFT tokens |
| Exposures | Equal or fewer total token exposures |
| Inference memory | Lower peak memory at matched context and batch |
| Long-context memory | State grows slower than nanochat KV cache |
| Decode | Comparable or faster on matched hardware |
| Prefill | Practical time-to-first-token at matched prompt lengths |
| Strictness | Global backprops `0`; forward transpose reads `0` |

Three efficiency claims remain separate:

| Claim | Meaning |
|---|---|
| Data efficiency | Better chat quality from fewer unique tokens |
| Exposure efficiency | Better chat quality from fewer total presentations |
| Compute efficiency | Better chat quality from fewer FLOPs or less time |

Replay cannot masquerade as data efficiency.

## Evaluation Target

Use nanochat's own chat evaluator first. Freeze its task versions and scoring
code with the control commit.

Expected components include:

- ARC-Easy;
- ARC-Challenge;
- MMLU;
- GSM8K;
- HumanEval-style coding;
- spelling and instruction-format tasks;
- aggregate ChatCORE or its pinned equivalent.

Also add deterministic conversation checks:

- follows user instruction;
- stops correctly;
- maintains role boundaries;
- resists repetitive loops;
- preserves information across turns;
- handles long conversations;
- produces finite logits and valid tokens.

Base BPB still matters as a diagnostic. SFT cannot repair a base model that
never learned enough language or knowledge.

## Fair Comparison

ReciprocalPC and nanochat GPT control must use:

- the same frozen tokenizer and chat special tokens;
- the same pretraining documents and order;
- the same validation split;
- the same SFT conversations and order;
- the same maximum sequence length;
- measured unique-token counts;
- measured total exposures;
- the same chat evaluator;
- same-hardware deployment benchmarks.

Maintain separate accounting for:

| Stage | Required counters |
|---|---|
| Pretraining | Unique tokens, exposures, repeats |
| SFT | Unique tokens, exposures, repeats |
| Optional future RL | Prompt tokens, sampled tokens, reward-training tokens |

RL is excluded from the first final comparison because the current reference
speedrun target is SFT. Add RL only through a later separately matched contract.

## Memory at Final Scale

The current nanochat speedrun uses a roughly GPT-2-scale dense model. A
similarly sized ReciprocalPC has a similar weight-memory floor:

| Format | Approximate 1.5B weight storage |
|---|---:|
| FP32 | 6.2GB |
| FP16/BF16 | 3.1GB |
| INT8 | 1.6GB |
| INT4 | 0.8GB plus metadata |

ReciprocalPC's likely advantage is context memory, not dense weight memory.
Nanochat attention uses a KV cache that grows with context length.
ReciprocalPC must retain bounded recurrent and predictive-memory state.

At equal parameter count, substantially lower total memory is realistic for
long conversations when:

- recurrent state does not grow with context length;
- state does not scale quadratically with model width;
- weights are quantized to INT8 or INT4;
- temporary inference buffers remain bounded;
- quality survives state compression.

Training may use rented H100-class or newer GPU nodes. Deployment remains the
local-device constraint.

## Prefill Requirement

Current small-scale result:

| Model | Decode tok/s | Prefill tok/s | Context state |
|---|---:|---:|---:|
| Phase298 ReciprocalPC | 276.62 | 10,414 | 89.6KB |
| Matched GPT control | 255.18 | 110,784 | 3.15MB KV capacity |

ReciprocalPC already shows the intended bounded-memory behavior, but prompt
loading is about 10.6 times slower. Scaling this path unchanged would produce a
poor chat experience even if decode remained competitive.

Required architecture:

1. Compute token-local prompt features in parallel.
2. Fuse recurrent and predictive-memory kernels.
3. Use exact chunked or associative scans where recurrence permits.
4. Merge memory sufficient statistics by blocks where mathematically valid.
5. Limit unavoidable sequential work to compact state transitions.
6. Prove output equivalence against the sequential reference.
7. Benchmark `512`, `2K`, `8K`, and `32K` prompt lengths.

Initial gate: at least `50K` prefill tok/s at current scale without losing
quality, strictness, decode speed, or bounded state.

## Linear Execution Path

### 1. Freeze and Reproduce the Release

- Hash the source manifest, tokenizer, data manifests, configuration, and
  retained Phase827 checkpoint.
- Load the checkpoint through the public release adapter on CPU and CUDA.
- Re-evaluate the frozen validation rows and match the recorded 5k metrics.
- Resume for one update and prove model, trainer, optimizer, schedule, and
  strict counters remain continuous.

Exit gate: exact checkpoint reload, metric reproduction, and a resumable
`global_backprops=0`, `forward_weight_transpose_reads=0` update.

### 2. Make Exact Retraining Boring

- Extract one supported production trainer around `create_reciprocal_pc()`.
- Freeze the NanoChat tokenizer and BOS best-fit loader.
- Record unique tokens, total exposures, row identity, and batch order.
- Save atomic checkpoints containing model, trainer, optimizer, schedule, RNG,
  and data-cursor state.
- Add restart and interruption drills before any long run.

Exit gate: two independent short runs and one interrupted/resumed run produce
identical measured rows and checkpoint hashes.

### 3. Reproduce the 5k Training Curve

- Start from the exact released initialization and seed.
- Evaluate only at the five original 1k milestones.
- Compare every measured row with the authenticated Phase827 and matched-GPT
  curves.
- Treat any unexplained numerical drift as a release blocker.

Exit gate: the complete measured curve and final checkpoint reproduce within a
preregistered numerical tolerance.

### 4. Profile Before Optimizing

- Measure training tokens/s, peak GPU memory, checkpoint time, prefill, decode,
  and recurrent-state bytes.
- Separate model weights, optimizer state, training-only reciprocal state, and
  temporary buffers.
- Fuse or compile only operations that preserve logits, recurrent state, local
  update tensors, and strict counters.

Exit gate: a reproducible resource table plus at least one verified
semantics-preserving performance improvement.

### 5. Extend the Matched Base-Language Run

- Train ReciprocalPC and a matched GPT control at the same tokenizer, rows,
  exposures, validation set, and schedule horizon.
- Use measured checkpoints at 10k, 20k, and 50k; do not splice the 5k schedule.
- Track BPB and exact-token accuracy together.
- Stop if quality flattens or stability degrades before spending the full
  budget.

Exit gate: a trustworthy longer-horizon Pareto curve. Scaling is not justified
by the 5k endpoint alone.

### 6. Resolve Only Measured Bottlenecks

- Keep the released model frozen as the parent.
- Preregister one causal mechanism and one promotion gate per experiment.
- Prioritize representation and ranking only if the short-token accuracy gap
  remains operationally important at the longer horizon.
- Reject duplicate, target-leaking, exact-credit, replay-heavy, and
  parameter-only variants before GPU work.

Exit gate: a candidate beats the parent on the sealed BPB/accuracy frontier
without changing data, exposure, or strict-credit accounting.

### 7. Freeze the NanoChat Chat Control

- Pin one NanoChat commit and reproduce its base-pretraining and SFT pipeline.
- Save base and SFT checkpoints.
- Freeze tokenizer, chat rendering, data, evaluator, metrics, and hashes.
- Measure unique tokens, exposures, compute, speed, and memory.

Exit gate: one reproducible NanoChat base row and one reproducible SFT target.

### 8. Add Strict-Local Conversation SFT

- Reuse the frozen NanoChat tokenizer and chat special tokens.
- Render the exact same conversations and supervise the same assistant tokens.
- Add local SFT updates without global backpropagation.
- Support deterministic multi-turn generation and correct stop behavior.

Exit gate: ReciprocalPC completes matched SFT, emits valid assistant turns, and
preserves strict `0/0` counters.

### 9. Fix Prefill

- Compute token-local prompt features in parallel.
- fuse recurrent and predictive-memory kernels;
- use exact chunked or associative scans where recurrence permits;
- prove output and final-state equivalence against the sequential reference;
- benchmark `512`, `2K`, `8K`, and `32K` prompt lengths.

Exit gate: practical time-to-first-token with unchanged quality and state.

### 10. Scale in Measured Steps

Run `20M-30M`, `124M`, `355M`, and `774M` pilots before approaching the pinned
NanoChat parameter class. At each size:

- train a matched GPT control;
- measure base quality before SFT and chat quality after identical SFT;
- verify local credit remains active at every layer;
- measure training state, inference state, prefill, decode, and peak memory;
- stop when quality, stability, or efficiency fails to improve monotonically.

Final success:

```text
ReciprocalPC chat quality > pinned NanoChat SFT quality
AND ReciprocalPC unique tokens < NanoChat unique tokens
AND ReciprocalPC exposures <= NanoChat exposures
AND matched-context inference memory < NanoChat memory
AND strict counters = 0/0
```

## Training-Memory Work

One 24GB RTX 3090 can validate the released 24M-parameter model but is not the
assumed final-scale training target. Before large scaling:

- use BF16 or FP16 compute where numerical checks pass;
- fuse local updates and release intermediates layer by layer;
- test 8-bit optimizer moments;
- replace permanent full-model displacement copies with blockwise charging;
- shard checkpoints and support CPU optimizer offload;
- support distributed training without changing update semantics.

Inference, not final training, must fit the intended local device.

## Rules for Every Step

- Create a dedicated branch before experimental edits.
- Preregister architecture and promotion gates.
- Compare against a matched GPT or chat control.
- Record pretraining and SFT tokens separately.
- Never call replay a learning-efficiency win.
- Do not tune parameters after tiny or absent progress.
- Preserve measured rejections and decision receipts.
- Record checkpoint, dataset, tokenizer, evaluator, source, and result hashes.
- Keep base capability and chat behavior as separate measured stages.

## Immediate Priorities

1. Verify the retained Phase827 checkpoint through this release.
2. Reproduce one strict-local update and then the full 5k curve.
3. Profile training, checkpointing, prefill, decode, and memory.
4. Run a matched longer-horizon base-language comparison.
5. Freeze NanoChat chat semantics only after the base model is reproducible.
6. Scale only after a small matched base-plus-chat result passes every gate.

End goal: a quantized local ChatPC model that beats the pinned nanochat SFT
model, learns from less data, and uses substantially less long-context memory.
