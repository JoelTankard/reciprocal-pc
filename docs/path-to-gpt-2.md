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

### 1. Freeze the Nanochat SFT Control

- Pin nanochat commit.
- Reproduce `runs/speedrun.sh` on rented GPUs.
- Save base and SFT checkpoints.
- Freeze tokenizer, data, evaluator, metrics, and hashes.
- Measure data, exposure, compute, speed, and memory.

Exit gate: one reproducible target row and checkpoint.

### 2. Add Conversation Training to Current ReciprocalPC

- Reuse the frozen nanochat tokenizer and chat special tokens.
- Render the exact same conversations.
- Compute loss only on the same target tokens as nanochat SFT.
- Add strict local SFT updates without global backpropagation.
- Support deterministic multi-turn token generation.
- Do not build CLI or web UI.

Exit gate: current small model completes SFT, generates valid assistant turns,
and preserves strict `0/0` counters.

### 3. Close the Small-Scale Base and Chat Gaps

- Continue architectural credit-assignment research.
- Compare base quality before SFT.
- Compare chat quality after identical SFT.
- Reject replay-heavy or parameter-only explanations.
- Preserve Phase298-style bounded state.

Exit gate: ReciprocalPC beats its matched small GPT control on both base and
chat quality at equal or lower unique tokens and exposures.

### 4. Fix Prefill

- Build parallel or chunked prefill.
- Fuse the production kernels.
- Preserve exact logits and state.
- Measure time-to-first-token on realistic conversations.

Exit gate: at least `50K` prefill tok/s at current scale, with competitive
decode and unchanged quality.

### 5. Run a 20M-30M Scaling Pilot

- Parameterize width, depth, transition width, and memory dimensions.
- Verify local credit remains active at every layer.
- Measure how recurrent state and training-only state scale.
- Train matched GPT and ChatPC controls.

Exit gate: quality improves monotonically without memory, prefill, or credit
collapse.

### 6. Build a 124M ChatPC

- Train on real contiguous web text.
- Apply the exact frozen nanochat SFT stage.
- Evaluate base BPB/CORE and final chat metrics.
- Quantize to INT8 and INT4.
- Benchmark consumer GPU and CPU deployment.

Exit gate: outperform the matched 124M nanochat-style chat control while using
equal or fewer unique tokens and exposures.

### 7. Build a 355M ChatPC

- Confirm learning advantage survives scale.
- Improve long-conversation retention.
- Introduce optimizer offload or sharding.
- Keep recurrent state bounded.

Exit gate: monotonic improvement from `30M` to `124M` to `355M`.

### 8. Build a 774M ChatPC

- Validate production quantization.
- Measure memory, latency, and energy on local hardware.
- Test long prompts where KV-cache growth becomes expensive.
- Run the full frozen chat suite.

Exit gate: retain better data efficiency and clear long-context memory
advantage.

### 9. Challenge the Pinned Nanochat SFT Model

- Scale ReciprocalPC to the control's approximate parameter class.
- Rent the required multi-GPU training node.
- Train from scratch on the frozen pretraining corpus.
- Apply identical SFT.
- Run the frozen evaluator.
- Run same-hardware memory and speed benchmarks.

Final success:

```text
ChatPC chat quality > pinned nanochat SFT quality
AND ChatPC unique tokens < nanochat unique tokens
AND ChatPC exposures <= nanochat exposures
AND ChatPC matched-context inference memory < nanochat memory
AND strict counters = 0/0
```

## Training-Memory Work

One 24GB RTX 3090 is insufficient for efficient final-scale training with the
current full optimizer and held-displacement state. Final training can use
rented GPUs.

Before large scaling:

- use BF16 or FP16 compute;
- fuse local updates;
- release gradients layer by layer;
- test 8-bit optimizer moments;
- replace permanent full-model displacement copies with blockwise charging;
- shard checkpoints;
- support CPU optimizer offload;
- support distributed training without changing update semantics.

Inference, not training, must fit the intended local device.

## Rules for Every Step

- Create a dedicated phase branch before edits.
- Preregister architecture and promotion gates.
- Compare against a matched GPT/chat control.
- Record pretraining and SFT tokens separately.
- Never call replay a learning-efficiency win.
- Do not tune parameters after tiny or absent progress.
- Preserve rejected mechanisms and measured ledgers.
- End every phase with a clean committed branch.
- Record checkpoint, dataset, tokenizer, evaluator, and result hashes.
- Keep base capability and chat behavior as separate measured stages.

## Immediate Priorities

1. Continue current architectural learning research.
2. Freeze the nanochat speedrun SFT control.
3. Add exact nanochat conversation SFT semantics to ReciprocalPC.
4. Fix prefill before large scaling.
5. Prove a joint base-plus-chat win at small scale.
6. Start the `20M-30M` scaling pilot.

End goal: a quantized local ChatPC model that beats the pinned nanochat SFT
model, learns from less data, and uses substantially less long-context memory.
