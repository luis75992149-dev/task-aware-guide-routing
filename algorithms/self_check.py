"""Self-check for Setting A router (paper main results)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .common import paper_root
    from .e1_evidence_preserving import generate_e1_guide
    from .e2_semantic_scaffold import TaskFrame, generate_e2_guide_from_frame
    from .e3_perceptual_scaffold import generate_e3_guide
    from .setting_a_router import default_paths, run_cv
except ImportError:
    from common import paper_root
    from e1_evidence_preserving import generate_e1_guide
    from e2_semantic_scaffold import TaskFrame, generate_e2_guide_from_frame
    from e3_perceptual_scaffold import generate_e3_guide
    from setting_a_router import default_paths, run_cv

# Paper Table V — Setting A (mean ± sample std across 5 outer folds)
EXPECTED_MEAN = {"IB": 69.84, "DAILY": 59.06, "WS": 47.23, "ALL": 57.90}
EXPECTED_STD = {"IB": 2.12, "DAILY": 1.99, "WS": 1.36, "ALL": 1.14}


def guide_smoke_tests() -> dict:
    e1 = generate_e1_guide(
        "What color is the object shown immediately after the speaker says the phrase?",
        ["A. red cube", "B. blue cone", "C. green ball", "D. yellow sign"],
        asr_segments=[{"start": 1.0, "end": 2.0, "text": "the speaker says the phrase"}],
        ocr_spans=[],
    )
    e2 = generate_e2_guide_from_frame(
        TaskFrame("dialogue, video", "frustration cue vs indifference cue"),
        "How does the man feel about the other person?",
        ["A. indifferent", "B. confused", "C. envious", "D. frustrated"],
    )
    e3 = generate_e3_guide("Anomaly Recognition", 120.0)
    checks = {
        "E1_used_guide": bool(e1["guide_text"]),
        "E2_used_guide": bool(e2["guide_text"]),
        "E3_used_guide": bool(e3["guide_text"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"guide smoke test failed: {checks}")
    return checks


def router_check(out_dir: Path, mean_tol_pp: float, std_tol_pp: float) -> dict:
    labels, features, _ = default_paths()
    summary = run_cv(labels, features, out_dir)

    observed_mean = {ds: float(summary[ds]["router_auto"]) for ds in EXPECTED_MEAN}
    observed_std = {ds: float(summary[ds]["router_auto_std"]) for ds in EXPECTED_STD}

    mean_delta = {ds: observed_mean[ds] - EXPECTED_MEAN[ds] for ds in EXPECTED_MEAN}
    std_delta = {ds: observed_std[ds] - EXPECTED_STD[ds] for ds in EXPECTED_STD}

    mean_fail = {ds: d for ds, d in mean_delta.items() if abs(d) > mean_tol_pp}
    std_fail = {ds: d for ds, d in std_delta.items() if abs(d) > std_tol_pp}
    if mean_fail:
        raise AssertionError(f"mean drift > {mean_tol_pp} pp: {mean_fail}")
    if std_fail:
        raise AssertionError(f"std drift > {std_tol_pp} pp: {std_fail}")

    return {
        "observed_mean": {k: round(v, 4) for k, v in observed_mean.items()},
        "observed_std": {k: round(v, 4) for k, v in observed_std.items()},
        "expected_mean": EXPECTED_MEAN,
        "expected_std": EXPECTED_STD,
        "delta_mean_pp": {k: round(v, 4) for k, v in mean_delta.items()},
        "delta_std_pp": {k: round(v, 4) for k, v in std_delta.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Setting A main results.")
    parser.add_argument("--guides", action="store_true", help="run E1/E2/E3 smoke test only")
    parser.add_argument("--mean-tol-pp", type=float, default=0.25, help="tolerance on accuracy mean")
    parser.add_argument("--std-tol-pp", type=float, default=0.05, help="tolerance on fold std")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (paper_root() / "results/run")
    report = {}
    if args.guides:
        report["guide_smoke"] = guide_smoke_tests()
    else:
        report["router"] = router_check(out_dir, args.mean_tol_pp, args.std_tol_pp)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
