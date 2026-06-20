"""Shared helpers for the public guide algorithms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


ARM_NAMES = ["NG", "E1", "E2", "E3"]
CORRECT_KEYS = ["correct_ng", "correct_e1", "correct_e2", "correct_e3"]


def repo_root() -> Path:
    """Repository root."""
    return Path(__file__).resolve().parents[1]


def paper_root() -> Path:
    return repo_root()


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def correct(row: dict, arm_id: int) -> int:
    value = row[CORRECT_KEYS[arm_id]]
    return 0 if value is None else int(value)


def allowed_arms(row: dict) -> list[int]:
    return [i for i, m in enumerate(row["arm_mask"]) if m]


def accuracy(rows: list[dict], picks: Iterable[int]) -> float:
    picks = list(picks)
    return 100.0 * sum(correct(row, pick) for row, pick in zip(rows, picks)) / max(1, len(rows))

