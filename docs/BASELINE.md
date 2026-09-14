# Phase827 Baseline Contract

The release comparison uses the same frozen NanoChat tokenizer, training rows,
validation rows, batch order, seed, sequence length, token budget, and evaluation
code for ReciprocalPC and matched GPT.

- vocabulary: `32,768`;
- sequence length: `2,048`;
- tokens per update: `2,048`;
- measured updates: `1,000`, `2,000`, `3,000`, `4,000`, `5,000`;
- total release exposure: `10,240,000` tokens.

## Final measured row

| Model | BPB | Exact-token accuracy |
| --- | ---: | ---: |
| ReciprocalPC | `1.4867165192` | `0.2393999365` |
| Matched GPT | `1.5625470909` | `0.2420239928` |

ReciprocalPC has lower BPB; GPT has `0.262406` percentage points higher
accuracy. Do not mix these 5k rows with later 50k schedules or architectures.
Curves must use measured rows only: no interpolation or endpoint substitution.
