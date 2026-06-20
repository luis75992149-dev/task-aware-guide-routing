# Main Results (Paper Table V)

Setting A uplift-stability router under GTT-stratified 5-fold CV with nested 3-fold hyper-parameter selection (v7 pre-routing features, 452-d). Each outer fold trains on ~5,622 samples and evaluates on ~1,406 held-out samples; every sample is tested exactly once across the five folds.

## Per-dataset accuracy

| Method | IB (2689) | Daily (1197) | WS (3142) | Pooled (7028) |
|--------|----------:|-------------:|----------:|--------------:|
| Reproduced no-guide (∅) | 66.01 | 57.31 | 47.07 | 56.06 |
| Always E1 | 67.72 | 59.31 | 46.88 | 56.97 |
| Always E2 | 69.95 | 59.57 | 46.40 | 57.66 |
| Always E3 (NG fallback) | 66.01 | 57.31 | 46.05 | 55.61 |
| Best-single (peek) | 69.95 | 59.57 | 47.07 | 57.66 |
| **Setting A (uplift + stability)** | **69.84 ± 2.12** | **59.06 ± 1.99** | **47.23 ± 1.36** | **57.90 ± 1.14** |
| Sample oracle (4-way) | 76.31 | 66.58 | 57.80 | 66.38 |

± values are sample standard deviations across the five outer folds.

## Hyper-parameters selected per outer fold

| Fold | uplift (λ, τ, β) | stability (min_gap, min_bucket, folds) |
|------|-------------------|----------------------------------------|
| 0 | (1.0, 0.0, 0.0) | (0.01, 120, 2) |
| 1 | (0.8, 0.0, 0.06) | (0.015, 80, 3) |
| 2 | (0.8, −0.02, 0.0) | none |
| 3 | (1.0, −0.02, 0.06) | (0.01, 10, 2) |
| 4 | (1.0, 0.04, 0.06) | (0.015, 80, 3) |

## Comparison to HumanOmniV2 paper baseline

| Split | Setting A | Paper baseline | Δ (pp) |
|-------|----------:|---------------:|-------:|
| IB | 69.84 | 69.33 | +0.51 |
| Daily | 59.06 | 58.47 | +0.59 |
| WS | 47.23 | 47.10 | +0.13 |
