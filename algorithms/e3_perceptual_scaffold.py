"""E3: perceptual scaffold expert.

E3 is a deterministic task-type scaffold. It is intentionally simple: use
offline metadata (`task_type`, video duration) to decide whether a WorldSense
sample should receive a fixed how-to-think scaffold for counting, spatial,
temporal, anomaly, interaction, or long-video reasoning.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


COUNT_TYPES = {"Object Counting", "Action Counting", "Audio Counting"}
TEMPORAL_TYPES = {"Temporal Localization", "Temporal Prediction", "Event Sorting"}
INTERACTION_TYPES = {"Human Interaction", "Human-object Interaction", "Audio Source Localization"}
SUPPRESS_TYPES = {"Audio Recognition", "Text and Diagram Understanding"}


SCAFFOLD_TEXT = {
    "E3.count": (
        "[GUIDE]\n"
        "This question asks for a count. Before you answer:\n"
        "1) Enumerate each occurrence in chronological order.\n"
        "2) State your running count at each observation.\n"
        "3) Prefer the tightest integer that matches your enumeration, not a round estimate.\n"
        "[/GUIDE]\n"
    ),
    "E3.spatial": (
        "[GUIDE]\n"
        "This is a spatial-relation question. Before you answer:\n"
        "1) Anchor to the camera viewpoint.\n"
        "2) Describe what is to the left, right, in front of, and behind the reference object.\n"
        "3) Only after the anchor description is fixed, map it to the options.\n"
        "[/GUIDE]\n"
    ),
    "E3.temporal": (
        "[GUIDE]\n"
        "This question targets temporal ordering or localization. Before you answer:\n"
        "1) Sketch the video timeline as [start -> mid -> end].\n"
        "2) Mark the boundary of the target event on that sketch.\n"
        "3) Only then select the option whose timing matches the sketch.\n"
        "[/GUIDE]\n"
    ),
    "E3.anomaly": (
        "[GUIDE]\n"
        "This question asks what is unusual. Before you answer:\n"
        "1) State the normal expected pattern for this scene in one sentence.\n"
        "2) State the specific deviation you observed in one sentence.\n"
        "3) Pick the option matching the deviation, not the normal baseline.\n"
        "[/GUIDE]\n"
    ),
    "E3.interaction": (
        "[GUIDE]\n"
        "This question targets an interaction. Before you answer:\n"
        "1) List all actors or objects present.\n"
        "2) For each relevant pair, identify the active party and the passive party.\n"
        "3) Pick the option specifying the correct direction of interaction, not mere co-presence.\n"
        "[/GUIDE]\n"
    ),
    "E3.longvid": (
        "[GUIDE]\n"
        "The video is long. Before you answer:\n"
        "1) Identify at most 3 distinct segments by visible content shifts.\n"
        "2) Summarize each segment in one short phrase.\n"
        "3) Locate the question's subject within one of those segments before selecting.\n"
        "[/GUIDE]\n"
    ),
    "suppress": "",
}

REQUIRED_MODALITIES = {
    "E3.count": "video",
    "E3.spatial": "video",
    "E3.temporal": "video, audio",
    "E3.anomaly": "video, audio",
    "E3.interaction": "video, audio",
    "E3.longvid": "video, audio",
    "suppress": "",
}


def decide_e3_scaffold(task_type: Optional[str], duration_s: Optional[float]) -> Tuple[str, str]:
    """Return `(scaffold_name, reason)` for a WorldSense-style sample."""
    if task_type in SUPPRESS_TYPES:
        return "suppress", "low_ceiling_task_type"
    if task_type in COUNT_TYPES:
        return "E3.count", "count_routing"
    if task_type == "Spatial Relation":
        return "E3.spatial", "spatial_routing"
    if task_type in TEMPORAL_TYPES:
        return "E3.temporal", "temporal_routing"
    if task_type == "Anomaly Recognition":
        return "E3.anomaly", "anomaly_routing"
    if task_type in INTERACTION_TYPES:
        return "E3.interaction", "interaction_routing"
    if duration_s is not None and duration_s >= 360:
        return "E3.longvid", "duration_fallback"
    return "suppress", "default_suppress"


def generate_e3_guide(task_type: Optional[str], duration_s: Optional[float]) -> Dict[str, object]:
    scaffold, reason = decide_e3_scaffold(task_type, duration_s)
    guide = SCAFFOLD_TEXT[scaffold]
    return {
        "expert": "E3",
        "scaffold": scaffold,
        "reason": reason,
        "required_modalities": REQUIRED_MODALITIES[scaffold],
        "guide_text": guide,
        "used_guide": bool(guide),
    }

