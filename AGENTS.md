# Repository focus

The public release is **Golden**, the ReciprocalPC native model plus its causal
additive RLS branch. Call the ordinary-backpropagation NanoChat comparison
**baseline**. Blue exact-credit PC is a diagnostic oracle, not Golden and not
baseline. Preserve internal serialization keys where actual native checkpoint
consumers still require them; do not add aliases merely for historical naming.

The user explicitly authorized the 2026-10-22 Golden promotion, overriding the
old public-name restriction. Current scores and artifact identity belong to
`configs/reciprocalpc-release-manifest.json`; the old Phase827 5k release is
historical only. Promotion does not erase reused-development-data limitations
or token-length subgroup regressions.

Golden's published interpolated weights are inference-only. Never resume them
as if optimizer moments, reciprocal feedback, schedule, RNG and cursor had been
reconciled. Preserve unaveraged strict-local training and its 0 global-backprop /
0 forward-weight-transpose-read contract. Keep the executable shared-prefill
optimization equivalent to the original update.

Do not import run archives, discarded launchers, giant logs/checkpoints or
machine-absolute path dependencies. Record compact current contracts, measured
outcomes and immutable provenance. Curves must use actual measured rows, never
interpolated scores or mixed schedule horizons. Weight interpolation is an
explicit separately evaluated transformation, not permission to fabricate curves.

Describe Golden as selectively ternary, not all-ternary. Report dense weights,
FP32 training carriers, RLS computation and persistent state separately. The
1.268x A40 result is an internal update-path comparison, not a speed claim against
baseline or evidence of faster deployed inference.
