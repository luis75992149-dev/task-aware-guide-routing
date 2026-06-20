# Data

## Files

| File | Description |
|------|-------------|
| `data/labels/xdataset_labels_v7.jsonl` | Per-sample GTT, stem, arm correctness (7,028 rows) |
| `data/labels/xdataset_features_v7.npz` | 452-d pre-routing features (52 structural + 400 TF-IDF stem) |
| `data/labels/xdataset_features_v7.meta.json` | Feature metadata |

## Label schema (per line)

| Field | Description |
|-------|-------------|
| `sample_id` | Global index 0…7027 |
| `dataset` | `IB`, `DAILY`, or `WS` |
| `gtt` | Global task type (13 classes) |
| `stem` | Question text |
| `arm_mask` | `[ng, e1, e2, e3]` availability |
| `correct_ng` / `correct_e1` / `correct_e2` / `correct_e3` | 0/1 correctness per arm |
| `duration_bucket`, `has_asr` | Pre-routing metadata |
| `source`, `focus`, `task_type`, `wh` | Benchmark metadata used for GTT assignment |

## Notes

- Labels are offline HumanOmniV2 four-arm results; videos and weights are not included.
- TF-IDF is fit on the full corpus for representation; router training is fold-separated.
- IB/Daily have no E3 arm (`arm_mask[3]=0`); WorldSense has all four arms.
