"""E2: semantic task-frame scaffold.

E2 uses a short task frame to inject semantic disambiguation cues. In the full
system, the task frame is produced offline by a text LLM (Qwen3-8B in the paper).
This public module exposes the parts that are independent of any proprietary
runtime: prompt construction, frame validation, risk-gated action selection, and
guide rendering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


TASK_FRAME_SYSTEM = """You output a two-field routing guide for a multiple-choice question (MC). You do NOT answer the question.

Output exactly:
Task Frame (minimal):
- required_modalities: ...
- key_disambiguation_axis: ...

Rules:
1. required_modalities: only the minimum modalities needed (e.g. audio, video, dialogue). One short phrase.
2. key_disambiguation_axis: describe the observable or wording difference that separates the closest options.
3. Do NOT output option letters.
4. Keep the axis short and concrete.
"""

ABSTRACT_AXIS_PATTERNS = (
    "underlying belief",
    "emotional state",
    "true intention",
    "inner feeling",
    "deep motivation",
    "hidden meaning",
)

FALSE_DICHOTOMY_PATTERNS = (
    r"\bwhether\b.{1,30}\bor\b",
    r"\brather\s+than\b",
    r"\bnot\s+\w+\s+but\b",
    r"\bvs\.?\b",
)

DECEPTION_KEYWORDS = {
    "pretend", "lie", "lying", "hide", "hiding", "trick", "deceive",
    "sarcasm", "sarcastic", "fake",
}


@dataclass
class TaskFrame:
    required_modalities: str
    key_disambiguation_axis: str
    raw_text: str = ""


@dataclass
class RiskDecision:
    risk_score: float
    risk_level: str
    risk_tags: List[str] = field(default_factory=list)
    guide_action: str = "full_guide"


def build_task_frame_prompt(question: str, options: Sequence[str]) -> list[dict]:
    """Return chat-style messages for the offline task-frame LLM."""
    option_block = "\n".join(options or [])
    user = f"Input (question + options):\n{question}\nOptions:\n{option_block}\n\nProduce the Task Frame (minimal) now."
    return [
        {"role": "system", "content": TASK_FRAME_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_task_frame(text: str) -> TaskFrame:
    """Parse the two public task-frame fields from bullet-style text."""
    def field(name: str) -> str:
        m = re.search(rf"^\s*[-*]?\s*{re.escape(name)}\s*:\s*(.+)$", text, re.I | re.M)
        return m.group(1).strip() if m else ""

    return TaskFrame(
        required_modalities=field("required_modalities"),
        key_disambiguation_axis=field("key_disambiguation_axis"),
        raw_text=text,
    )


def count_english_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or ""))


def validate_task_frame(frame: TaskFrame) -> tuple[bool, str]:
    """Return `(ok, reason)`. Failed frames should become no-guide."""
    mod = frame.required_modalities.strip()
    axis = frame.key_disambiguation_axis.strip()
    if not mod:
        return False, "modalities_empty"
    if not axis:
        return False, "axis_empty"
    if len(mod) > 140 or count_english_words(mod) > 18:
        return False, "modalities_too_long"
    if axis.lower() == "generic_mc":
        return False, "axis_generic"
    if len(axis) > 120 or count_english_words(axis) > 12:
        return False, "axis_too_long"
    axis_low = axis.lower()
    if any(p in axis_low for p in ABSTRACT_AXIS_PATTERNS):
        return False, "axis_abstract"
    return True, ""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", (text or "").lower())


def _axis_overlap(axis: str, question: str, options: Sequence[str]) -> float:
    stop = {"the", "a", "an", "is", "are", "of", "to", "in", "for", "and", "or", "vs", "with"}
    axis_tokens = [t for t in _tokens(axis) if t not in stop]
    if not axis_tokens:
        return 1.0
    ref = set(_tokens(question + " " + " ".join(options or [])))
    return sum(t in ref for t in axis_tokens) / len(axis_tokens)


def assess_axis_risk(
    frame: TaskFrame,
    question: str,
    options: Sequence[str],
    *,
    medium_threshold: float = 0.30,
    high_threshold: float = 0.60,
) -> RiskDecision:
    """Heuristic risk gate used before guide rendering."""
    axis = frame.key_disambiguation_axis or ""
    axis_low = axis.lower()
    context = (question + " " + " ".join(options or [])).lower()
    score = 0.0
    tags: list[str] = []

    if any(re.search(p, axis_low, re.I) for p in FALSE_DICHOTOMY_PATTERNS):
        score += 0.25
        tags.append("false_dichotomy")
    if any(kw in context for kw in DECEPTION_KEYWORDS):
        score += 0.20
        tags.append("deception_overread")
    overlap = _axis_overlap(axis, question, options)
    if overlap < 0.30:
        score += 0.25
        tags.append("axis_low_overlap")
    elif overlap < 0.50:
        score += 0.10
        tags.append("axis_moderate_overlap")
    if any(p in axis_low for p in ("underlying", "subconscious", "psychological", "hidden meaning")):
        score += 0.15
        tags.append("axis_abstract")

    score = min(score, 1.0)
    if score >= high_threshold:
        return RiskDecision(round(score, 4), "high", tags, "no_guide")
    if score >= medium_threshold:
        return RiskDecision(round(score, 4), "medium", tags, "modality_only")
    return RiskDecision(round(score, 4), "low", tags, "full_guide")


def render_e2_guide(frame: TaskFrame, decision: RiskDecision) -> str:
    """Render E2 guide text. Empty string means suppressed/no-guide."""
    if decision.guide_action == "no_guide":
        return ""
    mod = frame.required_modalities.strip()
    axis = frame.key_disambiguation_axis.strip()
    if decision.guide_action == "modality_only":
        return (
            "[GUIDE]\n"
            "Use these cues only as lightweight guidance.\n"
            f"Required modalities: {mod}\n"
            "[/GUIDE]\n"
        )
    return (
        "[GUIDE]\n"
        "Use these cues only as lightweight guidance.\n"
        f"Required modalities: {mod}\n"
        f"One local cue worth checking: {axis}\n"
        "[/GUIDE]\n"
    )


def generate_e2_guide_from_frame(
    frame: TaskFrame,
    question: str,
    options: Sequence[str],
) -> Dict[str, object]:
    """Public E2 API once the offline task frame is available."""
    ok, reason = validate_task_frame(frame)
    if not ok:
        return {
            "expert": "E2",
            "task_frame": frame.__dict__,
            "guide_text": "",
            "used_guide": False,
            "fallback_reason": reason,
        }
    decision = assess_axis_risk(frame, question, options)
    guide = render_e2_guide(frame, decision)
    return {
        "expert": "E2",
        "task_frame": frame.__dict__,
        "risk": decision.__dict__,
        "guide_text": guide,
        "used_guide": bool(guide),
    }

