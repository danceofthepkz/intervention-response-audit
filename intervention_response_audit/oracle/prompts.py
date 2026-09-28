"""Feed-based natural-language prompt builders for the oracle."""

from __future__ import annotations

import numpy as np

from intervention_response_audit.network import CampusNetwork, Condition, EdgeFeatures, Persona

STANCES = ("endorse", "neutral", "doubt")

NAME_POOL = (
    "Maya", "Jordan", "Alex", "Sam", "Taylor", "Riley", "Casey", "Morgan", "Avery", "Jamie",
    "Quinn", "Cameron", "Drew", "Skyler", "Reese", "Parker", "Rowan", "Elliot", "Harper", "Emerson",
    "Nina", "Leo", "Priya", "Mateo", "Sofia", "Noah", "Aisha", "Ethan", "Zoe", "Liam",
    "Mina", "Owen", "Ivy", "Caleb", "Nora", "Elena", "Arjun", "Grace", "Miles", "Chloe",
    "Luca", "Amara", "Ben", "Isla", "Diego", "Tara", "Jules", "Naomi", "Felix", "Anika",
    "Kai", "Lena", "Omar", "Mira", "Theo", "Aya", "Jonah", "Selin", "Victor", "Yara",
    "Hana", "Andre", "Mei", "Dylan", "Sasha", "Iris",
)

TOPIC_PHRASES = {
    "course_material": "course-material chatter",
    "club_event": "club events",
    "gossip": "campus gossip",
    "internship": "internship leads",
    "policy_change": "campus policy changes",
}

COMMENT_BANK: dict[str, dict[str, list[tuple[str, str]]]] = {
    topic: {
        "endorse": [
            (f"{topic}:endorse:0", "I checked, this is real."),
            (f"{topic}:endorse:1", "A professor also posted this."),
            (f"{topic}:endorse:2", "This matches what staff confirmed."),
            (f"{topic}:endorse:3", "Worth paying attention to this one."),
        ],
        "neutral": [
            (f"{topic}:neutral:0", "Passing this along in case it matters."),
            (f"{topic}:neutral:1", "No strong take, but people are sharing it."),
            (f"{topic}:neutral:2", "This just showed up in my feed."),
            (f"{topic}:neutral:3", "Might be useful for someone here."),
        ],
        "doubt": [
            (f"{topic}:doubt:0", "No idea if this is true."),
            (f"{topic}:doubt:1", "This is not verified yet."),
            (f"{topic}:doubt:2", "Looks like promo spam to me."),
            (f"{topic}:doubt:3", "I would wait for a better source."),
        ],
    }
    for topic in TOPIC_PHRASES
}

SCAFFOLDS = (
    (
        "Think like this student in this campus situation. Return only the strict JSON object, "
        "with probabilities between 0 and 1."
    ),
    (
        "Take the student's point of view and judge the message naturally. Answer with only valid JSON "
        "using probabilities from 0 to 1."
    ),
    (
        "Imagine you are the student described below reading your social feed. Provide just the JSON "
        "probability object."
    ),
    (
        "Use the context as a social vignette, not as numeric data. Respond only with the required JSON."
    ),
    (
        "Estimate how the student would react to this message from their peers. Output only strict JSON."
    ),
)


def _socialness_phrase(value: float) -> str:
    if value < 0.33:
        return "you keep a small circle and rarely jump into campus chatter"
    if value < 0.67:
        return "you are friendly but selective about what you pass along"
    return "you are highly social and often hear what is moving through campus"


def _trust_phrase(value: float) -> str:
    if value < 0.33:
        return "you are quite skeptical of unverified information"
    if value < 0.67:
        return "you like some evidence before fully trusting a claim"
    return "you tend to trust campus information when it comes from peers you know"


def student_name(net: CampusNetwork, agent_id: int) -> str:
    """Return a deterministic first name for a node within a generated network."""

    seed = net.network_seed or 0
    return NAME_POOL[(seed * 37 + agent_id * 17) % len(NAME_POOL)]


def tie_strength_phrase(ef: EdgeFeatures) -> str:
    """Bucket tie strength into one natural relationship phrase."""

    if ef.tie_strength < 0.3:
        return "a loose acquaintance"
    if ef.tie_strength < 0.65:
        return "a familiar friend"
    return "a close friend"


def _join_phrases(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _context_phrase(net: CampusNetwork, agent_i: int, neighbor: int, ef: EdgeFeatures) -> str:
    pi = net.personas[agent_i]
    pj = net.personas[neighbor]
    parts = []
    if "classmate" in ef.contexts or ef.same_major:
        parts.append("class")
    if "clubmate" in ef.contexts or ef.same_club:
        shared = sorted(set(pi.clubs) & set(pj.clubs))
        parts.append(f"{shared[0]} club" if shared else "a club")
    if "dorm" in ef.contexts or ef.same_year:
        parts.append("living in the same residence hall")
    if "online" in ef.contexts:
        parts.append("online group chats")
    if not parts:
        parts.append("campus")
    return f"you know them from {_join_phrases(parts)}"


def _channel_phrase(ef: EdgeFeatures) -> str:
    if ef.chat_frequency == "high" and ef.tie_strength >= 0.55:
        return "a direct message"
    if "clubmate" in ef.contexts:
        return "a club group chat"
    return "a big campus group chat"


def _topic_interest_phrase(persona: Persona, cond: Condition) -> str:
    interest = persona.topic_interests.get(cond.topic, 0.5)
    phrase = TOPIC_PHRASES[cond.topic]
    if interest < 0.33:
        return f"You don't usually pay attention to {phrase}."
    if interest < 0.67:
        return f"You occasionally follow {phrase}."
    return f"You actively keep up with {phrase}."


def _persona_sketch(persona: Persona, cond: Condition) -> str:
    clubs = ", ".join(persona.clubs) if persona.clubs else "no regular clubs"
    return (
        f"You are a year {persona.year} {persona.major} student. "
        f"Your clubs: {clubs}. {_socialness_phrase(persona.socialness)}. "
        f"{_trust_phrase(persona.trust)}. {_topic_interest_phrase(persona, cond)}"
    )


def sample_share_metadata(topic: str, rng: np.random.Generator, stance_probs: tuple[float, float, float]) -> dict[str, str]:
    """Sample stance/comment metadata for one reshare event."""

    stance = str(rng.choice(STANCES, p=np.array(stance_probs) / sum(stance_probs)))
    options = COMMENT_BANK[topic][stance]
    comment_id, comment = options[int(rng.integers(0, len(options)))]
    return {"stance": stance, "comment_id": comment_id, "comment": comment}


def build_decision_prompt(
    net: CampusNetwork,
    agent_i: int,
    exposures: list[int] | set[int] | tuple[int, ...],
    cond: Condition,
    scaffold_id: int = 0,
    exposure_details: dict[int, dict[str, str]] | None = None,
) -> str:
    """Build a feed-style decision prompt for one agent and exposure set."""

    if scaffold_id < 0 or scaffold_id >= len(SCAFFOLDS):
        raise ValueError("scaffold_id must be in 0..4")
    persona = net.personas[agent_i]
    exposure_lines = []
    for idx, neighbor in enumerate(sorted(exposures)):
        ef = net.edge_features[net.edge_key(agent_i, neighbor)]
        name = student_name(net, neighbor)
        quote = f' They quote the message: "{cond.text}"' if idx == 0 else ""
        detail = exposure_details.get(neighbor, {}) if exposure_details else {}
        comment = detail.get("comment")
        comment_text = f' {name} added: "{comment}"' if comment else ""
        exposure_lines.append(
            f"- {name}, {tie_strength_phrase(ef)}; {_context_phrase(net, agent_i, neighbor, ef)}. "
            f"They sent it through {_channel_phrase(ef)}.{quote}{comment_text}"
        )
    if not exposure_lines:
        exposure_lines.append("- No one has directly sent you this yet.")

    return "\n".join(
        [
            SCAFFOLDS[scaffold_id],
            "",
            _persona_sketch(persona, cond),
            "",
            "What you see:",
            *exposure_lines,
            "",
            "Return exactly this JSON shape:",
            '{"believe": 0.0, "reshare": 0.0, "attend": 0.0, "reason_code": "short_snake_case"}',
        ]
    )


def build_requery_prompt(
    net: CampusNetwork,
    agent_i: int,
    previous_exposures: list[int] | set[int] | tuple[int, ...],
    new_exposures: list[int] | set[int] | tuple[int, ...],
    cond: Condition,
    scaffold_id: int = 0,
    exposure_details: dict[int, dict[str, str]] | None = None,
) -> str:
    """Build a prompt for a later query after the exposure set has grown."""

    previous = ", ".join(student_name(net, i) for i in sorted(previous_exposures))
    new = ", ".join(student_name(net, i) for i in sorted(new_exposures))
    base = build_decision_prompt(net, agent_i, new_exposures, cond, scaffold_id, exposure_details)
    return "\n".join(
        [
            SCAFFOLDS[scaffold_id],
            "",
            _persona_sketch(net.personas[agent_i], cond),
            "",
            f"You already saw this earlier when {previous} shared it.",
            f"Since then, new shares came from: {new}.",
            "",
            "Updated feed:",
            *base.split("What you see:", 1)[1].splitlines()[1:],
        ]
    )
