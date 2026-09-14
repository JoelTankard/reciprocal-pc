# Experiment Rules

1. Preserve the released Phase827 model as the baseline. Test candidates on a
   branch or disposable worktree; do not silently change the release contract.
2. Freeze tokenizer, loader, shards, row order, sequence length, seed, validation
   rows, update count, and token exposures for every claimed comparison.
3. Keep the three broader research roles distinct: ReciprocalPC is the strict
   local model, Blue exact-credit PC is the exact-VJP oracle, and GPT-50k is the
   ordinary-backpropagation reference.
4. Reject any strict run where `global_backprops != 0`,
   `forward_weight_transpose_reads != 0`, or exact VJP enters learned credit.
5. Record BPB and exact-token accuracy together. An improvement in one metric
   does not erase a regression in the other.
6. Use only measured evaluation rows. Never interpolate, splice schedules, or
   compare unequal training horizons as if matched.
7. Hash source, configuration, tokenizer, data manifests, and checkpoint inputs
   before launch. Save resumable model, trainer, optimizer, and schedule state.
8. Profile training throughput, peak memory, prefill, decode, and recurrent-state
   size before scaling. Do not infer large-scale efficiency from parameter count.
9. Change one causal mechanism per experiment. Stop duplicate or
   contract-incompatible work before GPU allocation.
10. Document the change, measured result, comparison, verdict, checkpoint, and
    reason before beginning another experiment.
