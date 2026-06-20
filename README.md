# Task-Aware Guide Routing (Anonymous Review Release)

Minimal code to reproduce **Setting A** main results from the paper
*Task-Aware Guide Routing for Frozen Video Question Answering*.

## Reproduce (CPU, ~3–4 min)

```bash
pip install -r requirements.txt
python algorithms/self_check.py
```

Expected **Paper Table V — Setting A** (mean ± std across 5 outer folds, %):

| Split | Mean | Std |
|-------|-----:|----:|
| IntentBench | 69.84 | 2.12 |
| Daily-Omni | 59.06 | 1.99 |
| WorldSense | 47.23 | 1.36 |
| Pooled | 57.90 | 1.14 |

`self_check.py` verifies both mean (±0.25 pp) and std (±0.05 pp).

## Contents

- `algorithms/` — Setting A router + E1/E2/E3 guide modules + self-check
- `data/labels/` — 7,028 cached per-arm labels and v7 (452-d) features
- `results/main_results.md` — full Table V reference numbers
- `docs/setting_a_router.md` — method summary

No GPU or model weights required. Routing uses offline HumanOmniV2 arm labels.

## License

To be released with the camera-ready version.
