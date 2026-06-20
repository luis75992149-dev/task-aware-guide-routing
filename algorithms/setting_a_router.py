"""Deployable Setting A router: uplift gate + stability-gated GTT overlay.

This is the public, self-contained implementation of the final Setting A
algorithm. It uses only pre-routing v7 features:

  - structural question / metadata features
  - GTT one-hot features
  - TF-IDF stem features

It does NOT use arm predicted letters, pairwise agreement, softmax margins, or
champion decision fields. The router therefore preserves the single-forward
deployment contract: choose one guide arm before running the frozen MLLM.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from .common import CORRECT_KEYS, accuracy, correct, paper_root, read_jsonl, write_jsonl
except ImportError:  # Allows `python setting_a_router.py` directly.
    from common import CORRECT_KEYS, accuracy, correct, paper_root, read_jsonl, write_jsonl


SEED = 42
LAMBDA_HARM = [0.8, 1.0, 1.25, 1.5, 2.0]
THRESHOLDS = [-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.12]
BUCKET_BONUS = [0.0, 0.02, 0.04, 0.06]
STABILITY_CONFIGS: List[Optional[Tuple[float, int, int]]] = [
    None,
    (0.010, 10, 2),
    (0.010, 120, 2),
    (0.015, 10, 2),
    (0.015, 10, 3),
    (0.015, 80, 3),
    (0.015, 120, 3),
    (0.020, 80, 3),
    (0.020, 120, 2),
    (0.025, 80, 2),
    (0.025, 120, 3),
]


def folds_by_gtt(rows: List[dict], k: int = 5, seed: int = SEED) -> list[list[int]]:
    rng = random.Random(seed)
    by_gtt = defaultdict(list)
    for i, row in enumerate(rows):
        by_gtt[row["gtt"]].append(i)
    folds = [[] for _ in range(k)]
    for ids in by_gtt.values():
        rng.shuffle(ids)
        for j, idx in enumerate(ids):
            folds[j % k].append(idx)
    return folds


def dataset_weights(rows: List[dict]) -> np.ndarray:
    counts = defaultdict(int)
    for row in rows:
        counts[row["dataset"]] += 1
    return np.asarray(
        [len(rows) / (len(counts) * max(1, counts[row["dataset"]])) for row in rows],
        dtype=np.float64,
    )


def _constant_model(value: float) -> dict:
    return {"kind": "constant", "value": float(value)}


def train_binary_ensemble(X_train, rows_train: List[dict], labels: np.ndarray, idx: List[int]) -> dict:
    """Train LR+HGB calibrated binary ensemble for benefit or harm."""
    if not idx:
        return _constant_model(0.0)
    y = labels[idx].astype(np.int64)
    if len(set(y.tolist())) < 2:
        return _constant_model(float(y[0]) if len(y) else 0.0)

    X = X_train[idx]
    X_dense = X.toarray() if sparse.issparse(X) else X
    sample_weight = dataset_weights([rows_train[i] for i in idx])

    lr = CalibratedClassifierCV(
        LogisticRegression(max_iter=3000, solver="liblinear", C=0.75, class_weight="balanced"),
        method="isotonic",
        cv=3,
    )
    try:
        lr.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        lr.fit(X, y)

    hgb = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=250,
        learning_rate=0.04,
        l2_regularization=0.01,
        random_state=SEED,
    )
    hgb.fit(X_dense, y, sample_weight=sample_weight)
    return {"kind": "ensemble", "LR": lr, "HGB": hgb}


def predict_binary_ensemble(model: dict, X_eval) -> np.ndarray:
    n = X_eval.shape[0]
    if model["kind"] == "constant":
        return np.full(n, model["value"], dtype=np.float64)
    X_dense = X_eval.toarray() if sparse.issparse(X_eval) else X_eval
    lr = model["LR"]
    hgb = model["HGB"]
    p_lr = lr.predict_proba(X_eval)[:, list(lr.classes_).index(1)]
    p_hgb = hgb.predict_proba(X_dense)[:, list(hgb.classes_).index(1)]
    return (p_lr + p_hgb) / 2.0


def train_uplift_models(X_train, rows_train: List[dict]) -> Dict[int, Tuple[dict, dict]]:
    """Train benefit/harm models for E1/E2/E3 relative to no-guide."""
    ng = np.asarray([correct(row, 0) for row in rows_train], dtype=np.int64)
    models: Dict[int, Tuple[dict, dict]] = {}
    for arm_id in (1, 2, 3):
        idx = [j for j, row in enumerate(rows_train) if row["arm_mask"][arm_id] == 1]
        arm = np.asarray([correct(row, arm_id) for row in rows_train], dtype=np.int64)
        benefit = ((arm == 1) & (ng == 0)).astype(np.int64)
        harm = ((arm == 0) & (ng == 1)).astype(np.int64)
        models[arm_id] = (
            train_binary_ensemble(X_train, rows_train, benefit, idx),
            train_binary_ensemble(X_train, rows_train, harm, idx),
        )
    return models


def predict_uplift(models: Dict[int, Tuple[dict, dict]], X_eval) -> Tuple[dict, dict]:
    benefit, harm = {}, {}
    for arm_id, (benefit_model, harm_model) in models.items():
        benefit[arm_id] = predict_binary_ensemble(benefit_model, X_eval)
        harm[arm_id] = predict_binary_ensemble(harm_model, X_eval)
    return benefit, harm


def uplift_bucket_policy(rows: List[dict], tau: float = 0.01) -> dict:
    """GTT prior: choose an expert only if its mean uplift over NG exceeds tau."""
    by_gtt = defaultdict(list)
    for row in rows:
        by_gtt[row["gtt"]].append(row)
    policy = {}
    for gtt, group in by_gtt.items():
        scores = {}
        for arm_id in (1, 2, 3):
            sub = [row for row in group if row["arm_mask"][arm_id] == 1]
            if sub:
                scores[arm_id] = np.mean([correct(row, arm_id) - correct(row, 0) for row in sub])
        if not scores:
            policy[gtt] = 0
            continue
        best = max(scores, key=scores.get)
        policy[gtt] = best if scores[best] > tau else 0
    return policy


def route_uplift(
    benefit: dict,
    harm: dict,
    masks: List[List[int]],
    bucket_picks: List[int],
    *,
    lambda_harm: float,
    threshold: float,
    bucket_bonus: float,
) -> list[int]:
    picks = []
    for i, mask in enumerate(masks):
        best_arm = 0
        best_score = threshold
        for arm_id in (1, 2, 3):
            if not mask[arm_id]:
                continue
            score = benefit[arm_id][i] - lambda_harm * harm[arm_id][i]
            if arm_id == bucket_picks[i]:
                score += bucket_bonus
            if score > best_score:
                best_score = score
                best_arm = arm_id
        picks.append(best_arm)
    return picks


def rates_per_bucket(rows: List[dict]) -> dict:
    by_gtt = defaultdict(list)
    for row in rows:
        by_gtt[row["gtt"]].append(row)
    out = {}
    for gtt, group in by_gtt.items():
        sums = [0] * 4
        ns = [0] * 4
        for row in group:
            for arm_id in range(4):
                if row["arm_mask"][arm_id]:
                    sums[arm_id] += correct(row, arm_id)
                    ns[arm_id] += 1
        out[gtt] = ([sums[i] / ns[i] if ns[i] else -1 for i in range(4)], len(group))
    return out


def stable_policy(rows_train: List[dict], min_gap: float, min_bucket: int, stability_folds: int) -> dict:
    """GTT overlay: cover only buckets with stable expert-vs-NG gain."""
    base = rates_per_bucket(rows_train)
    rng = random.Random(7)
    ids = list(range(len(rows_train)))
    rng.shuffle(ids)
    step = len(ids) // stability_folds
    splits = [ids[s * step:(s + 1) * step] for s in range(stability_folds)]
    splits[-1] = ids[(stability_folds - 1) * step:]

    mini_rates = []
    for split in splits:
        held = set(split)
        mini = [rows_train[i] for i in range(len(rows_train)) if i not in held]
        mini_rates.append(rates_per_bucket(mini))

    policy = {}
    for gtt, (rates, n) in base.items():
        ng = rates[0]
        if n < min_bucket or ng < 0:
            policy[gtt] = 0
            continue
        candidates = [(rates[i], i) for i in range(1, 4) if rates[i] >= 0]
        if not candidates:
            policy[gtt] = 0
            continue
        best_rate, best_arm = max(candidates)
        if best_rate - ng <= min_gap:
            policy[gtt] = 0
            continue
        ok = True
        for mini in mini_rates:
            info = mini.get(gtt)
            if not info:
                ok = False
                break
            mini_rates_for_gtt, _ = info
            if mini_rates_for_gtt[best_arm] - mini_rates_for_gtt[0] <= 0:
                ok = False
                break
        policy[gtt] = best_arm if ok else 0
    return policy


def apply_stability_overlay(rows: List[dict], base_picks: Iterable[int], policy: Optional[dict]) -> list[int]:
    if policy is None:
        return list(base_picks)
    out = []
    for row, pick in zip(rows, base_picks):
        overlay_pick = policy.get(row["gtt"], 0)
        out.append(overlay_pick if overlay_pick and row["arm_mask"][overlay_pick] else int(pick))
    return out


def group_score(rows: List[dict], picks: Iterable[int]) -> float:
    """Inner-CV objective: mean group accuracy with a worst-group penalty."""
    picks = list(picks)
    values = []
    for ds in ("IB", "DAILY", "WS"):
        ids = [i for i, row in enumerate(rows) if row["dataset"] == ds]
        if ids:
            values.append(accuracy([rows[i] for i in ids], [picks[i] for i in ids]))
    return 0.55 * (sum(values) / len(values)) + 0.45 * min(values)


def predict_uplift_for_rows(X_train, rows_train, X_eval, rows_eval, cfg: dict) -> list[int]:
    models = train_uplift_models(X_train, rows_train)
    benefit, harm = predict_uplift(models, X_eval)
    bucket = uplift_bucket_policy(rows_train, tau=0.01)
    bucket_picks = [bucket.get(row["gtt"], 0) for row in rows_eval]
    return route_uplift(
        benefit,
        harm,
        [row["arm_mask"] for row in rows_eval],
        bucket_picks,
        lambda_harm=cfg["lambda_harm"],
        threshold=cfg["threshold"],
        bucket_bonus=cfg["bucket_bonus"],
    )


def choose_uplift_config_nested(X_train, rows_train: List[dict], fold_id: int) -> dict:
    rng = random.Random(700 + fold_id)
    ids = list(range(len(rows_train)))
    rng.shuffle(ids)
    step = len(ids) // 3
    splits = [ids[s * step:(s + 1) * step] for s in range(3)]
    splits[-1] = ids[2 * step:]
    scores = defaultdict(list)
    for si in range(3):
        val_ids = splits[si]
        val_set = set(val_ids)
        tr_ids = [i for i in ids if i not in val_set]
        rows_a = [rows_train[i] for i in tr_ids]
        rows_b = [rows_train[i] for i in val_ids]
        Xa = X_train[tr_ids]
        Xb = X_train[val_ids]
        models = train_uplift_models(Xa, rows_a)
        benefit, harm = predict_uplift(models, Xb)
        bucket = uplift_bucket_policy(rows_a, tau=0.01)
        bucket_picks = [bucket.get(row["gtt"], 0) for row in rows_b]
        masks = [row["arm_mask"] for row in rows_b]
        for lam in LAMBDA_HARM:
            for threshold in THRESHOLDS:
                for bonus in BUCKET_BONUS:
                    picks = route_uplift(
                        benefit,
                        harm,
                        masks,
                        bucket_picks,
                        lambda_harm=lam,
                        threshold=threshold,
                        bucket_bonus=bonus,
                    )
                    scores[(lam, threshold, bonus)].append(group_score(rows_b, picks))
    best = max(scores, key=lambda k: sum(scores[k]) / len(scores[k]))
    return {"lambda_harm": best[0], "threshold": best[1], "bucket_bonus": best[2]}


def choose_stability_config_nested(X_train, rows_train: List[dict], uplift_cfg: dict, fold_id: int):
    rng = random.Random(1700 + fold_id)
    ids = list(range(len(rows_train)))
    rng.shuffle(ids)
    step = len(ids) // 3
    splits = [ids[s * step:(s + 1) * step] for s in range(3)]
    splits[-1] = ids[2 * step:]
    scores = defaultdict(list)
    for si in range(3):
        val_ids = splits[si]
        val_set = set(val_ids)
        tr_ids = [i for i in ids if i not in val_set]
        rows_a = [rows_train[i] for i in tr_ids]
        rows_b = [rows_train[i] for i in val_ids]
        base = predict_uplift_for_rows(X_train[tr_ids], rows_a, X_train[val_ids], rows_b, uplift_cfg)
        for cfg in STABILITY_CONFIGS:
            policy = None if cfg is None else stable_policy(rows_a, cfg[0], cfg[1], cfg[2])
            picks = apply_stability_overlay(rows_b, base, policy)
            scores[cfg].append(group_score(rows_b, picks))
    return max(scores, key=lambda c: sum(scores[c]) / len(scores[c]))


def summarize(rows: list[dict], picks: list[int], folds: list[list[int]], configs: list[dict]) -> dict:
    summary = {"setting": "public_setting_a_uplift_stability", "chosen_config_per_fold": configs}
    for ds in ("IB", "DAILY", "WS", "ALL"):
        ids = list(range(len(rows))) if ds == "ALL" else [i for i, row in enumerate(rows) if row["dataset"] == ds]
        subset = [rows[i] for i in ids]
        subpicks = [picks[i] for i in ids]
        fold_accs = []
        id_set = set(ids)
        for fold in folds:
            fold_ids = [i for i in fold if i in id_set]
            if fold_ids:
                fold_accs.append(accuracy([rows[i] for i in fold_ids], [picks[i] for i in fold_ids]))
        summary[ds] = {
            "n": len(subset),
            "always_ng": arm_accuracy(subset, 0),
            "always_e1": arm_accuracy(subset, 1),
            "always_e2": arm_accuracy(subset, 2),
            "always_e3_ng_fallback": arm_accuracy(subset, 3, fallback_to_ng=True),
            "best_single_acc": best_single_accuracy(subset),
            "sample_oracle": sample_oracle_accuracy(subset),
            "router_auto": accuracy(subset, subpicks),
            "router_auto_folds": fold_accs,
            "router_auto_std": float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0,
        }
    return summary


def arm_accuracy(rows: list[dict], arm_id: int, fallback_to_ng: bool = False) -> float:
    if fallback_to_ng:
        hits = sum(correct(row, arm_id) if row["arm_mask"][arm_id] else correct(row, 0) for row in rows)
        return 100.0 * hits / max(1, len(rows))
    sub = [row for row in rows if row["arm_mask"][arm_id] == 1]
    return 100.0 * sum(correct(row, arm_id) for row in sub) / max(1, len(sub))


def best_single_accuracy(rows: list[dict]) -> float:
    rates = []
    for arm_id in range(4):
        sub = [row for row in rows if row["arm_mask"][arm_id] == 1]
        rates.append(100.0 * sum(correct(row, arm_id) for row in sub) / max(1, len(sub)) if sub else -1)
    return max(rates)


def sample_oracle_accuracy(rows: list[dict]) -> float:
    return 100.0 * sum(any(correct(row, i) for i, m in enumerate(row["arm_mask"]) if m) for row in rows) / max(1, len(rows))


def run_cv(labels_path: Path, features_path: Path, output_dir: Path) -> dict:
    t0 = time.time()
    rows = read_jsonl(labels_path)
    X = sparse.load_npz(str(features_path))
    if X.shape[0] != len(rows):
        raise ValueError(f"feature/label row mismatch: {X.shape[0]} vs {len(rows)}")

    folds = folds_by_gtt(rows, k=5, seed=SEED)
    sample_fold = [-1] * len(rows)
    for fold_id, ids in enumerate(folds):
        for idx in ids:
            sample_fold[idx] = fold_id

    picks = [-1] * len(rows)
    configs = []
    for fold_id, test_ids in enumerate(folds):
        test_set = set(test_ids)
        train_ids = [i for i in range(len(rows)) if i not in test_set]
        rows_train = [rows[i] for i in train_ids]
        rows_test = [rows[i] for i in test_ids]
        X_train = X[train_ids]
        X_test = X[test_ids]

        uplift_cfg = choose_uplift_config_nested(X_train, rows_train, fold_id)
        stability_cfg = choose_stability_config_nested(X_train, rows_train, uplift_cfg, fold_id)
        base = predict_uplift_for_rows(X_train, rows_train, X_test, rows_test, uplift_cfg)
        policy = None if stability_cfg is None else stable_policy(rows_train, stability_cfg[0], stability_cfg[1], stability_cfg[2])
        fold_picks = apply_stability_overlay(rows_test, base, policy)
        for j, idx in enumerate(test_ids):
            picks[idx] = fold_picks[j]
        cfg = {"uplift": uplift_cfg, "stability": stability_cfg}
        configs.append(cfg)
        print(f"[fold {fold_id}] cfg={cfg} acc={accuracy(rows_test, fold_picks):.2f} elapsed={time.time() - t0:.1f}s")

    summary = summarize(rows, picks, folds, configs)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "setting_a_router_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(
        output_dir / "setting_a_router_predictions.jsonl",
        (
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "gtt": row["gtt"],
                "fold": sample_fold[i],
                "arm_mask": row["arm_mask"],
                "correct_ng": int(correct(row, 0)),
                "correct_e1": int(correct(row, 1)),
                "correct_e2": int(correct(row, 2)),
                "correct_e3": int(correct(row, 3)),
                "router_pick": int(picks[i]),
                "chosen_config": configs[sample_fold[i]],
            }
            for i, row in enumerate(rows)
        ),
    )
    return summary


def default_paths() -> tuple[Path, Path, Path]:
    paper = paper_root()
    return (
        paper / "data/labels/xdataset_labels_v7.jsonl",
        paper / "data/labels/xdataset_features_v7.npz",
        paper / "results/run",
    )


def main() -> None:
    labels, features, out_dir = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=labels)
    parser.add_argument("--features", type=Path, default=features)
    parser.add_argument("--out-dir", type=Path, default=out_dir)
    args = parser.parse_args()
    summary = run_cv(args.labels, args.features, args.out_dir)
    print(json.dumps({ds: round(summary[ds]["router_auto"], 2) for ds in ("IB", "DAILY", "WS", "ALL")}, indent=2))


if __name__ == "__main__":
    main()

