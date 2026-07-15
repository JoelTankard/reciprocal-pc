# ReciprocalPC

Focused repository for improving strict predictive-coding language learning
without global backpropagation.

## Canonical models

| Model | Credit rule | 50k BPB | 50k accuracy |
| --- | --- | ---: | ---: |
| GPT-50k | Global backpropagation | `1.195398` | `0.436651` |
| Blue exact-credit PC | Exact block VJPs, oracle | `1.277254` | `0.406250` |
| ReciprocalPC | Learned local reciprocal paths | `1.283907` | `0.404540` |

ReciprocalPC records `global_backprops=0` at every measured step. Blue is not
the strict target: exact block VJPs make it an oracle for how much quality is
lost by learned local credit transport.

## Research target

Close the shared PC-to-GPT gap by improving representation and inference while
keeping ReciprocalPC credit local. Use Blue to distinguish credit-transport
failures from shared model failures. At 50k, ReciprocalPC retains `93.11%` of
GPT BPB quality and `92.65%` of GPT accuracy quality.

## Repository map

- `src/pfc/`: exact model and training implementation from the archived repo.
- `tools/`: canonical Windows runners and focused diagnostics.
- `data/curves/`: measured 100-step curves through 50k for all three models.
- `configs/`: full final-row configuration snapshots for both PC models.
- `docs/`: short model, baseline, checkpoint, findings, and experiment contracts.

Render the canonical comparison:

```bash
uv run python tools/render_comparison.py
```

Run tests:

```bash
uv sync --extra dev
uv run pytest -q
```

Full historical experiments remain in
`/Users/joel/ai/predictive-feature-correction`.
