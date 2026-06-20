# Reproduction

## Primary command

```bash
pip install -r requirements.txt
python algorithms/self_check.py
```

## Optional: guide smoke test

```bash
python algorithms/self_check.py --guides
```

## Direct router script

```bash
python algorithms/setting_a_router.py
```

Writes predictions under `results/run/`.

## Protocol (summary)

- 7,028 samples: IntentBench (2,689) + Daily-Omni (1,197) + WorldSense (3,142)
- GTT-stratified 5-fold CV, nested 3-fold hyper-parameter selection
- Setting A: uplift-gated router + optional GTT stability overlay (452-d v7 features)
