"""E1: evidence-preserving deterministic guide.

E1 is a rule-first expert. It never asks an LLM to invent strategy text.
Instead, it builds a low-entropy evidence sketch from observable inputs, selects
a guide mode by priority rules, validates closed-set slots, and renders one of a
small number of fixed guide templates.

The implementation below is a compact public version of the ESG-lite pipeline:

  sketch -> selector -> validator -> renderer

It is suitable for method figures and pseudocode. The original full experiment
implementation remains under `experiments/esg_lite/`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


TIME_LEX = (
    "after", "before", "first", "last", "next", "then", "during", "while",
    "beginning", "end", "sequence", "order", "simultaneous", "coincide",
)
VISUAL_LEX = ("color", "text", "logo", "sign", "shirt", "dress", "scene", "object")
AUDIO_LEX = ("voice", "say", "said", "saying", "sound", "music", "tone", "quote")


@dataclass
class EvidenceSketch:
    """Closed sketch emitted by E1 before guide rendering."""

    audio_events: List[str] = field(default_factory=list)
    ocr_spans: List[str] = field(default_factory=list)
    temporal_markers: List[str] = field(default_factory=list)
    uncertainty_flags: List[str] = field(default_factory=list)
    question_hooks: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "audio_events": list(self.audio_events),
            "ocr_spans": list(self.ocr_spans),
            "temporal_markers": list(self.temporal_markers),
            "uncertainty_flags": list(self.uncertainty_flags),
            "question_hooks": list(self.question_hooks),
        }


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    low = (text or "").lower()
    return any(n in low for n in needles)


def _clip_ocr(spans: Sequence[str], max_items: int = 5, max_chars: int = 32) -> List[str]:
    out, seen = [], set()
    for raw in spans or []:
        span = str(raw or "").strip()[:max_chars]
        if span and span not in seen:
            out.append(span)
            seen.add(span)
        if len(out) >= max_items:
            break
    return out


def build_evidence_sketch(
    question: str,
    options: Sequence[str] | None = None,
    *,
    asr_segments: Sequence[Dict[str, Any]] | None = None,
    ocr_spans: Sequence[str] | None = None,
) -> EvidenceSketch:
    """Build a compact evidence sketch from observable inputs only."""
    options = list(options or [])
    asr_segments = list(asr_segments or [])
    question_text = question or ""
    option_text = " ".join(options)
    all_text = f"{question_text} {option_text}"

    sketch = EvidenceSketch()
    sketch.ocr_spans = _clip_ocr(ocr_spans or [])

    if asr_segments:
        sketch.audio_events.append("speech")
        if len(asr_segments) >= 3:
            sketch.audio_events.append("speech_long")
    else:
        sketch.audio_events.append("no_audio")
        sketch.uncertainty_flags.append("no_asr")

    if not sketch.ocr_spans:
        sketch.uncertainty_flags.append("no_ocr")

    if _contains_any(question_text, TIME_LEX):
        sketch.temporal_markers.append("time_or_order_lex_in_stem")
        sketch.question_hooks.append("hook_event_order")
    if _contains_any(question_text, VISUAL_LEX):
        sketch.question_hooks.append("hook_visual_only")
    if _contains_any(question_text, AUDIO_LEX):
        sketch.question_hooks.append("hook_audio_only")

    # A simple observable anchor: quoted or keyword-overlapping ASR segment.
    if asr_segments and _contains_any(question_text, ("quote", "said", "saying", "while")):
        sketch.temporal_markers.append("asr_anchor_mid")

    if not sketch.temporal_markers:
        sketch.uncertainty_flags.append("no_temporal_signal")
    if sketch.audio_events == ["no_audio"]:
        sketch.uncertainty_flags.append("no_audio_event")
    if not sketch.question_hooks:
        sketch.question_hooks.append("hook_generic")
    return sketch


def select_e1_mode(sketch: EvidenceSketch, question: str, options: Sequence[str] | None = None) -> Dict[str, Any]:
    """Select one of the fixed E1 guide modes by deterministic priority rules."""
    flags = set(sketch.uncertainty_flags)
    hooks = set(sketch.question_hooks)
    has_temporal = bool(sketch.temporal_markers)
    has_anchor = any(m.startswith("asr_anchor") for m in sketch.temporal_markers) or bool(sketch.ocr_spans)

    if {"no_asr", "no_ocr", "no_audio_event"}.issubset(flags):
        return {"guide_mode": "no_guide", "reason": "no_observable_evidence"}
    if has_temporal and has_anchor:
        return {
            "guide_mode": "temporal_anchor",
            "required_modalities": "video, audio",
            "temporal_relation": "local_order",
            "temporal_scope": "local temporal window",
            "reason": "temporal_signal_with_anchor",
        }
    if "hook_visual_only" in hooks:
        return {
            "guide_mode": "modality_only",
            "required_modalities": "video",
            "target_modality": "video",
            "reason": "visual_question",
        }
    if "hook_audio_only" in hooks and "speech" in sketch.audio_events:
        return {
            "guide_mode": "modality_only",
            "required_modalities": "video, audio",
            "target_modality": "audio",
            "reason": "audio_question",
        }

    axis = _distinct_option_axis(options or [])
    if axis:
        return {
            "guide_mode": "contrast_axis",
            "required_modalities": "video",
            "contrast_axis": axis,
            "reason": "distinct_option_token",
        }
    return {"guide_mode": "no_guide", "reason": "conservative_default"}


def _distinct_option_axis(options: Sequence[str]) -> str:
    token_sets = []
    for opt in options:
        token_sets.append(set(re.findall(r"[A-Za-z][A-Za-z'-]{3,}", opt.lower())))
    if not token_sets:
        return ""
    union = set().union(*token_sets)
    for token in sorted(union):
        if sum(token in s for s in token_sets) == 1:
            return f"option-specific cue: {token}"
    return ""


def render_e1_guide(selection: Dict[str, Any]) -> str:
    """Render the selected E1 mode into a fixed guide block."""
    mode = selection.get("guide_mode", "")
    if mode == "no_guide" or not mode:
        return ""
    lines = ["[GUIDE]", "Use these cues only as lightweight guidance."]
    if mode == "temporal_anchor":
        lines += [
            "Guide mode: temporal_anchor",
            f"Required modalities: {selection.get('required_modalities', 'video, audio')}",
            f"Temporal relation to verify: {selection.get('temporal_relation', 'local_order')}",
            f"Temporal scope to focus on: {selection.get('temporal_scope', 'local temporal window')}",
        ]
    elif mode == "modality_only":
        target = selection.get("target_modality", "video")
        lines += [
            f"Guide mode: modality_only ({target})",
            f"Required modalities: {selection.get('required_modalities', 'video')}",
            f"Focus on the {target} evidence before choosing an option.",
        ]
    elif mode == "contrast_axis":
        lines += [
            "Guide mode: contrast_axis",
            f"Required modalities: {selection.get('required_modalities', 'video')}",
            f"Main distinction to resolve: {selection.get('contrast_axis', '')}",
        ]
    else:
        return ""
    lines.append("[/GUIDE]")
    return "\n".join(lines) + "\n"


def generate_e1_guide(
    question: str,
    options: Sequence[str] | None = None,
    *,
    asr_segments: Sequence[Dict[str, Any]] | None = None,
    ocr_spans: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """End-to-end public E1 API."""
    sketch = build_evidence_sketch(question, options, asr_segments=asr_segments, ocr_spans=ocr_spans)
    selection = select_e1_mode(sketch, question, options)
    guide = render_e1_guide(selection)
    return {
        "expert": "E1",
        "sketch": sketch.as_dict(),
        "selection": selection,
        "guide_text": guide,
        "used_guide": bool(guide),
    }

