# Repository focus

Keep this repository limited to three canonical models:

1. `ReciprocalPC`: strict local reciprocal credit, zero global backpropagation.
2. `Blue exact-credit PC`: exact-VJP oracle used to measure the credit gap.
3. `GPT-50k`: ordinary backpropagation reference.

Use `ReciprocalPC` as the public name. Keep internal `stage_reciprocal` keys and
historical checkpoint tags unchanged for compatibility.

Do not import the archived experiment wall, discarded launchers, or long
chronological findings from `predictive-feature-correction`. Record only current
contracts, measured outcomes, and decisions. Curves must use measured rows; no
interpolation or mixed schedule horizons.
