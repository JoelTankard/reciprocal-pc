# ReciprocalPC Model Contract

## Identity

The public model is `ReciprocalPC`. Its checkpoint-compatible implementation is
`NativeReciprocalPCLanguageModel` with `NativePCTrainer`. Historical
`stage_reciprocal` serialization keys remain unchanged.

The release is the accepted Phase827 architecture at update 5,000:

- width `512`;
- two recurrent layers;
- SwiGLU FFN width `1,040`;
- vocabulary and shared embedding/decoder width `32,768`;
- context length `2,048`;
- `23,930,434` learned model parameters;
- grouped ternary recurrent and FFN weights.

## Forward path

1. Token IDs select rows from the shared token embedding.
2. Each layer applies an affine recurrent scan. The recurrence is expressed as
   associative decay/write composition, allowing exact parallel prefix-scan
   execution instead of an inherently serial nonlinear recurrence.
3. Layer-local SwiGLU features transform the recurrent representation.
4. A fixed ternary Hadamard lexical predictor contributes token-sensitive
   features without adding a second learned lexical endpoint.
5. Temporal HGF precision state regulates recurrent information over time.
6. Phase827 projects the learned token row through a centered self-bilinear
   observation before the tied decoder.
7. The final state is RMS-normalized and decoded with the same matrix used for
   token embedding.

The model has bounded recurrent state rather than an attention KV cache that
grows with context length. Practical long-prompt performance still depends on
compiled scan and prefill kernels.

## Local training path

There is no global loss graph through the model.

- Each layer receives a local prediction error.
- Learned reciprocal feedback maps local errors into local update directions.
- Predictive-RLS state and training-only correction signals improve the local
  credit estimate.
- Dense carrier parameters are updated by the configured optimizer; grouped
  ternary parameters use the local ternary update path.
- Training-only corrections do not enter ordinary deployed logits.

Release invariants:

- `global_backprops == 0`;
- `forward_weight_transpose_reads == 0`;
- `exact_vjp_credit == false`;
- no target-token feature enters the forward predictor;
- token embedding and lexical decoder remain tied.

## Measured result

At 5,000 updates of 2,048 tokens each, the retained checkpoint measured:

| Metric | ReciprocalPC | Matched GPT |
| --- | ---: | ---: |
| Validation BPB | `1.4867165192` | `1.5625470909` |
| Exact-token accuracy | `0.2393999365` | `0.2420239928` |

ReciprocalPC wins BPB and trails accuracy by `0.262406` percentage points. The
accuracy deficit is concentrated in ranking common short-token alternatives,
not a general inability to use context. This release preserves the measured
Pareto result rather than importing later unproven architecture changes.

Blue exact-credit PC remains a diagnostic oracle in the broader research
repository. GPT-50k remains the ordinary-backpropagation reference. Neither is
implemented in this focused release.
