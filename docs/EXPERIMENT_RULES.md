# Experiment Rules

1. Preserve current **Golden** and its immutable inference artifact identity.
   Use a branch or disposable worktree for candidates; do not silently change
   the public contract. The old Phase827 5k model is historical, not current Golden.
2. Call the ordinary-backpropagation NanoChat control **baseline**. Blue
   exact-credit PC remains a separate diagnostic oracle, never a local-credit run.
3. Freeze tokenizer, data rows/order, sequence length, seed, validation masks,
   update count, token exposures and schedule for each claimed comparison.
   Report parameter, arithmetic and hardware differences explicitly.
4. Reject strict training where `global_backprops != 0`,
   `forward_weight_transpose_reads != 0`, or exact VJP enters learned credit.
   Target labels/masks must never enter deployed forward predictors.
5. Report BPB and exact-token accuracy together with denominators and subgroup
   regressions. An aggregate improvement cannot erase a regression or turn
   reused development rows into a sealed test.
6. Plot measured evaluation rows only; never interpolate scores, splice
   schedules or substitute unequal horizons. Separately evaluated **weight**
   interpolation must be labeled as such, never represented as a raw checkpoint.
7. Hash source, configuration, tokenizer, data manifests and checkpoint inputs.
   A true resume needs complete unaveraged model, trainer, optimizer, schedule,
   RNG and cursor state. The promoted interpolated inference artifact is never
   resumable: its optimizer moments and reciprocal feedback are unreconciled.
8. Use the shared-prefill helper for the original strict-local update. Preserve
   output/state/update equivalence. Distinguish internal optimization timings
   from external baseline comparisons and full-run/inference speed claims.
9. Profile weights, optimizer/training carriers, native/branch recurrent state,
   temporaries, prefill and decode separately. Golden is selectively ternary,
   not all-ternary; RLS consumes extra compute and persistent memory.
10. Record measured result, verdict, artifact and decision before the next
    experiment. The user-authorized 2026-10-22 promotion explicitly accepted
    reused data and byte-length subgroup regressions; it is not a blanket waiver
    of future evaluation gates. Do not import archives or giant run logs.
