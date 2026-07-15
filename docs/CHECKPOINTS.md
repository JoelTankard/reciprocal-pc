# Canonical Checkpoints

Protected Windows directory: `D:/pfc_models/canonical_50k/`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `reciprocalpc-50k.pt` | `214269299` | `331c6eae03773795443c4b7b308b6164cc04760e46325528ec094dc5e9257fc5` |
| `blue-exact-credit-pc-50k.pt` | `283631819` | `e7992f1f87fc0642983dd00bf255d6d91e5f21be20268569f44d5317e1ade6b1` |
| `gpt-50k-model.pt` | `23083271` | `9fadeb0a8198b5b34a767025b612f4ef2a75480f352cb1f09f189e08e6617404` |
| `gpt-50k-optimizer.pt` | `33613877` | `e1151d166567e5804f63ebc745591f5d41f6fc891e552ee81cf12c5d27c9bdcf` |
| `gpt-50k-meta.json` | `1431` | `8653c203fb29047f1a7965f47ffa0e976361177a6c875bad9a1f9b3a1745b3ca` |

Files are hard links on Windows `D:`. Deleting historical run paths does not
delete canonical checkpoint bytes while these links remain.
