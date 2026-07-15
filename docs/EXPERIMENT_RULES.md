# Experiment Rules

1. Focus changes on ReciprocalPC learning and credit transport.
2. Keep GPT and Blue frozen as canonical measured references.
3. Preserve tokenizer, loader, shards, batch order, sequence length, seed,
   validation batches, and parameter budget when claiming a fair comparison.
4. Log validation BPB and accuracy every 100 steps. Save resumable checkpoints
   every 1k steps on Windows `D:`.
5. Render complete measured curves. No interpolation, endpoint-only lines, or
   schedule splicing.
6. Reject any strict run where `global_backprops != 0` or an exact VJP enters
   learned credit transport.
7. Promote at 100, 1k, 2k, 5k, 10k, 20k, and 50k only when the result improves
   the hypothesis or provides a decisive diagnostic.
8. Document one concise finding per completed/stopped experiment: change,
   metrics, comparison, verdict, checkpoint, and reason.

Primary objective: match or beat GPT without global backpropagation. Use Blue
as a diagnostic oracle; do not optimize its already-small credit gap in
isolation.
