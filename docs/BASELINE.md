# Baseline Contract

All three curves use:

- nanochat BPE vocabulary 4096
- tokenizer SHA-256
  `b3f1cf411824a833cdf06becd54acc7debb72a7ebc9d9967960d66f6b47257fe`
- nanochat BOS best-fit ClimbMix loader
- eight mixed training shards
- batch size 2, sequence length 1024, 2048 tokens/update
- 102.4 million training tokens at step 50k
- measured validation rows every 100 steps

## Final and best values

| Model | Final BPB | Best BPB | Final accuracy | Best accuracy |
| --- | ---: | ---: | ---: | ---: |
| GPT-50k | `1.1953982591` | `1.1953982591` | `0.4366513847` | `0.4374334060` at 49.9k |
| Blue exact-credit PC | `1.2772543162` | `1.2767477977` | `0.4062503055` | `0.4069052483` |
| ReciprocalPC | `1.2839073546` | `1.2835324868` | `0.4045396338` | `0.4048719929` |

## Schedule rule

GPT schedule horizon is part of the baseline. Use the complete fresh 50k GPT
curve for 50k comparisons. Never splice in the completed 20k-schedule curve or
resume the 20k checkpoint under a 50k schedule; warmdown timing differs.

Canonical curves:

- `data/curves/gpt-50k.jsonl`
- `data/curves/blue-exact-credit-pc-50k.jsonl`
- `data/curves/reciprocalpc-50k.jsonl`
