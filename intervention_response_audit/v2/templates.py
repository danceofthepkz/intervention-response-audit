"""Versioned v2-only authority/hedge templates.

The v1 ``FRAMED_TEMPLATE_BANK`` is historical provenance and must not be
mutated.  Revision 1 repairs the low-tier authority manipulation after the
pre-live blind review failed all five low-tier pairs.
"""

from __future__ import annotations

from typing import Any

from intervention_response_audit.network import CREDIBILITY_TIERS, FRAMED_TEMPLATE_BANK, FRAMING_TYPES, TOPICS


V2_TEMPLATE_REVISION = "v2_low_tier_revision1_20260717"

_TOPIC_OBJECTS = {
    "course_material": "the course-material update",
    "club_event": "the club-event update",
    "gossip": "the campus story",
    "internship": "the internship lead",
    "policy_change": "the policy-change update",
}

_LOW_AUTHORITY_SOURCES = {
    "course_material": "A course staff member",
    "club_event": "A club officer",
    "gossip": "A residence hall staff member",
    "internship": "A Career Center staff member",
    "policy_change": "A campus office staff member",
}


def _v2_text(topic: str, tier: str, framing_type: str) -> str:
    if tier != "low" or framing_type not in {"authority", "hedge"}:
        return str(FRAMED_TEMPLATE_BANK[topic][tier][framing_type]["text"])
    obj = _TOPIC_OBJECTS[topic]
    if framing_type == "authority":
        return (
            f"{_LOW_AUTHORITY_SOURCES[topic]} says {obj} was confirmed internally, "
            "although no public notice has been posted."
        )
    return f"A student post says {obj} might be happening, but no official source has confirmed it."


V2_TEMPLATE_BANK: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    topic: {
        tier: {
            framing_type: {
                **FRAMED_TEMPLATE_BANK[topic][tier][framing_type],
                "text": _v2_text(topic, tier, framing_type),
            }
            for framing_type in FRAMING_TYPES
        }
        for tier in CREDIBILITY_TIERS
    }
    for topic in TOPICS
}
