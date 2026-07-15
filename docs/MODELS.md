# Model Contract

## ReciprocalPC

No-global-backprop target using learned local reciprocal credit. Internal mode:
`meta_pc_feedback_mode=stage_reciprocal`.

- Four recurrent message-passing blocks, width 256, four heads, FFN 1022.
- One predictive-coding inference step per update.
- Each stage learns a reciprocal feedback map from local perturbation and local
  response pairs.
- Token outcome enters through learned `kp_learned` output feedback.
- Weight updates use local settled/free signals and Muon.
- No exact block VJP and no end-to-end loss backpropagation.
- `global_backprops=0` through 50k.
- Training-only reciprocal state raises parameter slots to `26,757,117`; learned
  model parameters are `12,317,700`.

## Blue exact-credit PC

Same core forward family and data contract, but internal mode
`meta_pc_feedback_mode=exact_vjp` computes exact local block VJPs. This is the
credit oracle, not the biological result.

- `global_backprops=0`, but exact autograd VJPs are used inside local blocks.
- Learned model parameters: `9,176,068`.
- Parameter slots: `23,615,485`.

## GPT-50k

Official nanochat depth-4 GPT reference: width 256, full causal attention,
sequence length 1024, Muon plus AdamW, ordinary backpropagation, `7,340,162`
parameters.

Historical run tags remain unchanged inside configs and checkpoint files. Do
not rename serialization keys; public naming is presentation-only.
