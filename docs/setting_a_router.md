# Setting A Uplift-Stability Router

## Purpose

The router selects one arm from `{no-guide, E1, E2, E3}` before the frozen MLLM
runs. It is deployable because it uses only pre-routing features and keeps a
single MLLM forward per sample.

## Inputs

### Pre-routing feature vector (v7, 452-d)

The router consumes a sparse 452-dimensional vector built **before** any
guide-specific MLLM forward:

| Block | Dim | Description |
|-------|----:|-------------|
| Stem structure | 17 | WH-type one-hot, length buckets, quote/time/visual/audio/OCR hints, token count |
| Keyword groups | 12 | Regex indicators (temporal, counting, spatial, emotion, …) plus word-count proxies |
| GTT one-hot | 13 | Global task type from offline taxonomy rules |
| Duration bucket | 7 | Video length bucket |
| ASR flag | 1 | Binary `has_asr` metadata |
| TF-IDF (stem) | 400 | Unigram/bigram TF-IDF of the **question stem only** (not answer options) |

No dataset ID, arm predictions, or expert-side decision metadata are used.

### Training labels

- `correct_ng`, `correct_e1`, `correct_e2`, `correct_e3` — per-arm MCQ correctness
- `arm_mask` — which arms are available (IB/Daily: E3 unavailable)

## Output

A selected arm id:

```text
0 = no-guide
1 = E1
2 = E2
3 = E3
```

## Algorithm

### Stage 1: Uplift-gated routing

For each guided arm `Ei`, train two binary models (LR + HistGBDT ensemble):

```text
benefit_i = 1[Ei correct and no-guide wrong]
harm_i    = 1[Ei wrong and no-guide correct]
```

At inference time:

```text
score_i = P(benefit_i | x) - lambda * P(harm_i | x)
```

Choose a guide only when the best score exceeds a nested-CV threshold.
Otherwise fall back to no-guide.

### Stage 2: Stability-gated GTT overlay

For each GTT bucket, use **training-fold statistics only**:

1. Estimate each arm's accuracy inside the GTT bucket.
2. Require `best_non_ng - no_guide > min_gap`.
3. Recompute on inner sub-splits (`stability_folds`).
4. Overlay only if uplift remains positive on every sub-split.

## Evaluation protocol

- Pool: 7,028 samples (IB 2,689 + Daily 1,197 + WS 3,142)
- Outer: GTT-stratified 5-fold CV (`seed=42`)
- Inner: 3-fold nested CV for hyper-parameters (λ, τ, β, stability config)
- Report: mean ± sample std of per-fold held-out accuracy

## Expected result (Paper Table V)

| Split | Mean (%) | Std (pp) |
|-------|---------:|---------:|
| IntentBench | 69.84 | 2.12 |
| Daily-Omni | 59.06 | 1.99 |
| WorldSense | 47.23 | 1.36 |
| Pooled | 57.90 | 1.14 |

## Code

`algorithms/setting_a_router.py` — run `python algorithms/self_check.py` to verify.
