"""Campus social network generation, personas, and message conditions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

MAJORS = ("CS", "Bio", "Econ", "Psych", "Math", "English", "Engineering", "Art")
CLUBS = (
    "Robotics",
    "Debate",
    "Theater",
    "Newspaper",
    "Photography",
    "Outdoors",
    "Music",
    "Volunteering",
    "Entrepreneurship",
    "Gaming",
    "Dance",
    "Sustainability",
)
TOPICS = ("course_material", "club_event", "gossip", "internship", "policy_change")
CONTEXTS = ("classmate", "clubmate", "dorm", "online")
CHAT_FREQUENCIES = ("low", "med", "high")


@dataclass(frozen=True)
class Persona:
    """Student attributes used by cascade models and prompt rendering."""

    year: int
    major: str
    clubs: tuple[str, ...]
    socialness: float
    trust: float
    topic_interests: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        return cls(
            year=int(data["year"]),
            major=str(data["major"]),
            clubs=tuple(data["clubs"]),
            socialness=float(data["socialness"]),
            trust=float(data["trust"]),
            topic_interests={str(k): float(v) for k, v in data["topic_interests"].items()},
        )


@dataclass(frozen=True)
class EdgeFeatures:
    """Dyad attributes for one undirected edge."""

    tie_strength: float
    same_major: bool
    same_club: bool
    same_year: bool
    contexts: frozenset[str]
    chat_frequency: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeFeatures":
        return cls(
            tie_strength=float(data["tie_strength"]),
            same_major=bool(data["same_major"]),
            same_club=bool(data["same_club"]),
            same_year=bool(data["same_year"]),
            contexts=frozenset(data["contexts"]),
            chat_frequency=str(data["chat_frequency"]),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["contexts"] = sorted(self.contexts)
        return data


@dataclass(frozen=True)
class Condition:
    """Message condition with rendered text and fixed semantic parameters."""

    topic: str
    urgency: float
    credibility: float
    controversy: float
    text: str
    wording_id: str
    framing_type: str = "unknown"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Condition":
        return cls(
            topic=str(data["topic"]),
            urgency=float(data["urgency"]),
            credibility=float(data["credibility"]),
            controversy=float(data["controversy"]),
            text=str(data["text"]),
            wording_id=str(data["wording_id"]),
            framing_type=str(data.get("framing_type", "unknown")),
        )


@dataclass
class CampusNetwork:
    """Graph plus aligned node and edge metadata.

    `G` has nodes 0..N-1. `personas` maps node id to persona. `edge_features`
    uses sorted `(u, v)` tuple keys.
    """

    G: nx.Graph
    personas: dict[int, Persona]
    edge_features: dict[tuple[int, int], EdgeFeatures]
    network_seed: int | None = None

    def edge_key(self, u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": sorted(self.G.nodes()),
            "network_seed": self.network_seed,
            "edges": [list(e) for e in sorted(self.G.edges())],
            "personas": {str(k): asdict(v) for k, v in sorted(self.personas.items())},
            "edge_features": {
                f"{u}-{v}": ef.to_dict()
                for (u, v), ef in sorted(self.edge_features.items())
            },
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    @classmethod
    def from_json(cls, source: str | Path) -> "CampusNetwork":
        source_text = str(source)
        if source_text.lstrip().startswith("{"):
            text = source_text
        else:
            text = Path(source).read_text(encoding="utf-8")
        data = json.loads(text)
        G = nx.Graph()
        G.add_nodes_from(int(n) for n in data["nodes"])
        G.add_edges_from((int(u), int(v)) for u, v in data["edges"])
        personas = {int(k): Persona.from_dict(v) for k, v in data["personas"].items()}
        edge_features = {
            tuple(int(x) for x in k.split("-")): EdgeFeatures.from_dict(v)
            for k, v in data["edge_features"].items()
        }
        return cls(G=G, personas=personas, edge_features=edge_features, network_seed=data.get("network_seed"))


TEMPLATE_BANK: dict[str, list[dict[str, Any]]] = {
    "course_material": [
        {
            "urgency": 0.75,
            "credibility": 0.82,
            "controversy": 0.15,
            "variants": [
                "Professor Lin also shared a review sheet for tomorrow's exam; it is in the class drive.",
                "The TA said Professor Lin's review sheet for tomorrow's exam is now in the class drive.",
                "A professor-backed review sheet for tomorrow's exam just showed up in the class drive.",
            ],
        },
        {
            "urgency": 0.50,
            "credibility": 0.65,
            "controversy": 0.20,
            "variants": [
                "Someone posted corrected lecture notes for the midterm topics in the course chat.",
                "The course chat has a set of corrected notes that people say match the midterm topics.",
                "A corrected notes packet for the midterm topics is circulating in the course chat.",
            ],
        },
        {
            "urgency": 0.35,
            "credibility": 0.25,
            "controversy": 0.30,
            "variants": [
                "A student claims there is a shortcut summary for the problem set, but no staff confirmed it.",
                "People are passing around an unofficial problem-set summary that staff have not verified.",
                "An unconfirmed shortcut guide for the problem set is making the rounds.",
            ],
        },
    ],
    "club_event": [
        {
            "urgency": 0.70,
            "credibility": 0.80,
            "controversy": 0.10,
            "variants": [
                "The Photography club moved tonight's gallery walk to the student center at 7.",
                "Club officers confirmed the gallery walk is at the student center tonight at 7.",
                "Tonight's Photography gallery walk has been relocated to the student center for 7.",
            ],
        },
        {
            "urgency": 0.55,
            "credibility": 0.62,
            "controversy": 0.18,
            "variants": [
                "The Robotics demo may need extra volunteers this weekend, according to the club chat.",
                "The club chat says Robotics is looking for more weekend demo volunteers.",
                "Robotics might be short on volunteers for the weekend demo, based on the group thread.",
            ],
        },
        {
            "urgency": 0.40,
            "credibility": 0.28,
            "controversy": 0.22,
            "variants": [
                "Someone said the Music open mic has free pizza, but the event page does not mention it.",
                "Free pizza at the Music open mic is being rumored, though the event page is silent.",
                "The Music open mic may have free pizza, if the group-chat rumor is right.",
            ],
        },
    ],
    "gossip": [
        {
            "urgency": 0.65,
            "credibility": 0.35,
            "controversy": 0.80,
            "variants": [
                "Everyone in the group chat is saying a popular senior got reported after the party.",
                "A rumor is spreading that a well-known senior was reported after the party.",
                "The big group chat is buzzing about a senior supposedly getting reported after the party.",
            ],
        },
        {
            "urgency": 0.35,
            "credibility": 0.28,
            "controversy": 0.70,
            "variants": [
                "Someone claims two club leaders had a huge argument after rehearsal.",
                "People are whispering that two club leaders argued after rehearsal.",
                "There is unverified chatter about two club leaders having a blowup after rehearsal.",
            ],
        },
        {
            "urgency": 0.50,
            "credibility": 0.42,
            "controversy": 0.65,
            "variants": [
                "A dorm thread says an RA may be leaving mid-semester, but nobody official confirmed it.",
                "People in the dorm chat think an RA might leave mid-semester, without confirmation.",
                "An unconfirmed RA departure rumor is circulating in the dorm thread.",
            ],
        },
        {
            "urgency": 0.55,
            "credibility": 0.72,
            "controversy": 0.75,
            "variants": [
                "The RA confirmed that a resident was formally written up after the party.",
                "A hall staff member posted that the party write-up actually happened.",
                "Someone from hall staff verified the disciplinary write-up after the party.",
            ],
        },
    ],
    "internship": [
        {
            "urgency": 0.85,
            "credibility": 0.75,
            "controversy": 0.12,
            "variants": [
                "An alum posted that internship referrals close tonight and they can forward resumes.",
                "A recent alum says they can refer students before the internship window closes tonight.",
                "Internship referrals from an alum are open until tonight, and resumes can be forwarded.",
            ],
        },
        {
            "urgency": 0.60,
            "credibility": 0.58,
            "controversy": 0.20,
            "variants": [
                "Someone shared a summer research opening that may fit students outside the major too.",
                "A summer research position is circulating, and it might be open across majors.",
                "There may be a cross-major summer research opening, based on a shared posting.",
            ],
        },
        {
            "urgency": 0.42,
            "credibility": 0.27,
            "controversy": 0.28,
            "variants": [
                "A startup recruiter link is going around, though someone said it might be promo spam.",
                "People are sharing a startup recruiter form that may just be promo spam.",
                "A possible startup internship form is circulating, but its legitimacy is unclear.",
            ],
        },
    ],
    "policy_change": [
        {
            "urgency": 0.78,
            "credibility": 0.82,
            "controversy": 0.45,
            "variants": [
                "Residential Life emailed that dorm access rules change starting Monday.",
                "Dorm access rules are changing Monday, according to a Residential Life email.",
                "Residential Life says the new dorm access policy starts this Monday.",
            ],
        },
        {
            "urgency": 0.58,
            "credibility": 0.60,
            "controversy": 0.55,
            "variants": [
                "Student government says the meal-plan petition will be discussed at Friday's meeting.",
                "Friday's student government meeting will include the meal-plan petition.",
                "The meal-plan petition is on the agenda for student government's Friday meeting.",
            ],
        },
        {
            "urgency": 0.45,
            "credibility": 0.29,
            "controversy": 0.60,
            "variants": [
                "A dorm chat claims quiet-hours enforcement is changing, but the policy page is unchanged.",
                "People say quiet-hours rules may be enforced differently, though the policy page has not changed.",
                "There is dorm-chat chatter about stricter quiet-hours enforcement with no official update.",
            ],
        },
    ],
}

FRAMING_TYPES = ("authority", "social_proof", "firsthand", "hedge")
CREDIBILITY_TIERS = ("low", "mid", "high")

_TIER_SCALARS = {
    "low": {"urgency": 0.45, "credibility": 0.25, "controversy": 0.45},
    "mid": {"urgency": 0.58, "credibility": 0.55, "controversy": 0.40},
    "high": {"urgency": 0.76, "credibility": 0.82, "controversy": 0.35},
}

_TOPIC_OBJECTS = {
    "course_material": "the course-material update",
    "club_event": "the club-event update",
    "gossip": "the campus story",
    "internship": "the internship lead",
    "policy_change": "the policy-change update",
}


def _framed_text(topic: str, tier: str, framing_type: str) -> str:
    obj = _TOPIC_OBJECTS[topic]
    if tier == "high":
        source = {
            "authority": f"An official campus source confirmed {obj} and posted the details.",
            "social_proof": f"Half the relevant group chats are sharing {obj} after a source-backed official post.",
            "firsthand": f"Someone says they checked the original notice and verified {obj}.",
            "hedge": f"A cautious post says {obj} is not verified yet and could still be rumor.",
        }
    elif tier == "mid":
        source = {
            "authority": f"A student leader says an official source confirmed {obj}.",
            "social_proof": f"Several students are passing around {obj}, but the source trail is a little mixed.",
            "firsthand": f"Someone says they checked {obj} firsthand and it mostly checks out.",
            "hedge": f"A cautious thread says {obj} is not verified and might just be rumor.",
        }
    else:
        source = {
            "authority": f"Someone claims an unnamed official confirmed {obj}, but there is no posted confirmation.",
            "social_proof": f"People are repeating {obj} in chats, though nobody can point to a solid source.",
            "firsthand": f"Someone says they personally heard {obj}, but it has not been verified.",
            "hedge": f"A vague post says {obj} might be happening, but it could be rumor or promo spam.",
        }
    return source[framing_type]


FRAMED_TEMPLATE_BANK: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    topic: {
        tier: {
            framing_type: {
                **_TIER_SCALARS[tier],
                "text": _framed_text(topic, tier, framing_type),
                "framing_type": framing_type,
            }
            for framing_type in FRAMING_TYPES
        }
        for tier in CREDIBILITY_TIERS
    }
    for topic in TOPICS
}


def _sample_persona(rng: np.random.Generator) -> Persona:
    club_count = int(rng.choice([0, 1, 2], p=[0.25, 0.50, 0.25]))
    clubs = tuple(sorted(rng.choice(CLUBS, size=club_count, replace=False).tolist()))
    interests = {topic: float(rng.beta(2.0, 2.0)) for topic in TOPICS}
    return Persona(
        year=int(rng.integers(1, 5)),
        major=str(rng.choice(MAJORS)),
        clubs=clubs,
        socialness=float(rng.beta(2.0, 2.0)),
        trust=float(rng.beta(2.0, 2.0)),
        topic_interests=interests,
    )


def _edge_features(pi: Persona, pj: Persona, rng: np.random.Generator) -> EdgeFeatures:
    same_major = pi.major == pj.major
    same_club = bool(set(pi.clubs) & set(pj.clubs))
    same_year = pi.year == pj.year
    contexts: set[str] = set()
    if same_major and rng.random() < 0.75:
        contexts.add("classmate")
    if same_club and rng.random() < 0.85:
        contexts.add("clubmate")
    if same_year and rng.random() < 0.35:
        contexts.add("dorm")
    if rng.random() < 0.45:
        contexts.add("online")
    if not contexts:
        contexts.add(str(rng.choice(CONTEXTS)))
    tie = float(rng.beta(2.0, 5.0))
    tie += 0.18 * same_major + 0.22 * same_club + 0.12 * same_year + 0.05 * ("online" in contexts)
    tie = float(np.clip(tie, 0.0, 1.0))
    if tie < 0.3:
        chat = "low"
    elif tie < 0.65:
        chat = str(rng.choice(["low", "med"], p=[0.35, 0.65]))
    else:
        chat = str(rng.choice(["med", "high"], p=[0.35, 0.65]))
    return EdgeFeatures(tie, same_major, same_club, same_year, frozenset(contexts), chat)


def generate_campus_network(n: int = 50, seed: int = 42) -> CampusNetwork:
    """Generate a connected campus graph with about 6-10 mean degree at n=50."""

    rng = np.random.default_rng(seed)
    personas = {i: _sample_persona(rng) for i in range(n)}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edge_features: dict[tuple[int, int], EdgeFeatures] = {}

    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = personas[i], personas[j]
            shared_club = bool(set(pi.clubs) & set(pj.clubs))
            score = 0.055
            score += 0.085 * (pi.major == pj.major)
            score += 0.105 * shared_club
            score += 0.050 * (pi.year == pj.year)
            score += 0.025 * (pi.socialness + pj.socialness)
            if rng.random() < min(score, 0.75):
                G.add_edge(i, j)
                edge_features[(i, j)] = _edge_features(pi, pj, rng)

    # Connect components by linking representative nodes to a similar peer in the largest component.
    while not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        main = list(components[0])
        for comp in components[1:]:
            u = int(sorted(comp)[0])
            same_major = [v for v in main if personas[v].major == personas[u].major]
            candidates = same_major or main
            v = int(rng.choice(candidates))
            key = (u, v) if u < v else (v, u)
            G.add_edge(*key)
            edge_features[key] = _edge_features(personas[key[0]], personas[key[1]], rng)

    return CampusNetwork(G=G, personas=personas, edge_features=edge_features, network_seed=seed)


def sample_condition(rng: np.random.Generator, topic: str | None = None) -> Condition:
    """Sample a message condition and one framed wording from the template bank."""

    chosen_topic = topic or str(rng.choice(TOPICS))
    tier = str(rng.choice(CREDIBILITY_TIERS))
    framing_type = str(rng.choice(FRAMING_TYPES))
    variant = FRAMED_TEMPLATE_BANK[chosen_topic][tier][framing_type]
    return Condition(
        topic=chosen_topic,
        urgency=float(variant["urgency"]),
        credibility=float(variant["credibility"]),
        controversy=float(variant["controversy"]),
        text=str(variant["text"]),
        wording_id=f"{chosen_topic}:{tier}:{framing_type}",
        framing_type=framing_type,
    )


def render_semantic_twins(
    condition: Condition, rng: np.random.Generator, k: int = 2
) -> list[Condition]:
    """Return `k` conditions with identical semantics and different wording."""

    topic, tier, current_framing = condition.wording_id.split(":")
    opposite = {
        "authority": "hedge",
        "firsthand": "hedge",
        "social_proof": "hedge",
        "hedge": "authority",
    }
    ordered_choices = [opposite[current_framing]] + [
        frame for frame in FRAMING_TYPES if frame not in {current_framing, opposite[current_framing]}
    ]
    choices = ordered_choices
    if k > len(choices):
        raise ValueError(f"Only {len(choices)} alternate framings available for {condition.wording_id}")
    selected = rng.choice(choices, size=k, replace=False)
    twins = []
    for framing_type in selected:
        variant = FRAMED_TEMPLATE_BANK[topic][tier][str(framing_type)]
        twins.append(Condition(
            topic=condition.topic,
            urgency=condition.urgency,
            credibility=condition.credibility,
            controversy=condition.controversy,
            text=str(variant["text"]),
            wording_id=f"{topic}:{tier}:{str(framing_type)}",
            framing_type=str(framing_type),
        ))
    return twins
