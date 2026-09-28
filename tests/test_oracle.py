import json

import networkx as nx
import numpy as np
import pytest

from intervention_response_audit.network import CampusNetwork, Condition, EdgeFeatures, Persona, generate_campus_network, sample_condition
from intervention_response_audit.oracle.client import BudgetExceeded, BudgetGuard, MockLLMClient, complete_json_logged, parse_oracle_json
from intervention_response_audit.oracle.prompts import build_decision_prompt, build_requery_prompt, student_name
from intervention_response_audit.oracle import runner as runner_module
from intervention_response_audit.oracle.runner import run_oracle_scenario


class ConstantClient:
    model = "constant"
    temperature = 0.0
    max_tokens = 300

    def __init__(self, believe: float = 0.8, reshare: float = 1.0, attend: float = 0.2) -> None:
        self.believe = believe
        self.reshare = reshare
        self.attend = attend

    def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "believe": self.believe,
                "reshare": self.reshare,
                "attend": self.attend,
                "reason_code": "constant",
            }
        )


def _persona() -> Persona:
    return Persona(
        year=1,
        major="CS",
        clubs=("Robotics",),
        socialness=0.7,
        trust=0.6,
        topic_interests={
            "course_material": 0.8,
            "club_event": 0.6,
            "gossip": 0.3,
            "internship": 0.7,
            "policy_change": 0.5,
        },
    )


def _condition() -> Condition:
    return Condition("course_material", 0.7, 0.8, 0.1, "Official review sheet posted.", "course_material:0:0")


def _graph_net(edges: list[tuple[int, int]]) -> CampusNetwork:
    G = nx.Graph()
    G.add_edges_from(edges)
    personas = {i: _persona() for i in G.nodes()}
    ef = EdgeFeatures(0.8, True, True, True, frozenset({"classmate", "clubmate"}), "high")
    return CampusNetwork(G, personas, {tuple(sorted(edge)): ef for edge in G.edges()})


def test_prompt_is_feed_based_without_numeric_feature_leaks() -> None:
    net = generate_campus_network(20, seed=1)
    cond = sample_condition(np.random.default_rng(2), topic="gossip")
    agent = next(i for i in net.G.nodes() if net.G.degree(i) > 0)
    exposures = [next(iter(net.G.neighbors(agent)))]

    prompt = build_decision_prompt(
        net,
        agent,
        exposures,
        cond,
        scaffold_id=0,
        exposure_details={
            exposures[0]: {
                "stance": "endorse",
                "comment_id": "gossip:endorse:0",
                "comment": "I checked, this is real.",
            }
        },
    )

    assert cond.text in prompt
    assert "I checked, this is real." in prompt
    assert "campus gossip" in prompt
    assert "you know them from" in prompt
    assert "Student " not in prompt
    assert "tie_strength" not in prompt
    assert "active neighbors" not in prompt
    assert '{"believe": 0.0' in prompt
    assert '"hear"' not in prompt


def test_prompt_does_not_directly_render_condition_scalar_adjectives() -> None:
    net = generate_campus_network(20, seed=12)
    cond = Condition(
        topic="policy_change",
        urgency=0.95,
        credibility=0.95,
        controversy=0.95,
        text="Residential Life officially posted the new access policy.",
        wording_id="policy_change:test:0",
    )
    agent = next(i for i in net.G.nodes() if net.G.degree(i) > 0)
    exposer = next(iter(net.G.neighbors(agent)))
    prompt = build_decision_prompt(
        net,
        agent,
        [exposer],
        cond,
        exposure_details={
            exposer: {
                "stance": "neutral",
                "comment_id": "policy_change:neutral:0",
                "comment": "Passing this along in case it matters.",
            }
        },
    )
    banned = [
        "strongly sourced",
        "somewhat sourced",
        "weakly sourced",
        "highly controversial",
        "somewhat controversial",
        "unlikely to stir controversy",
        "urgency",
        "credibility",
        "controversy",
    ]

    assert all(phrase not in prompt.lower() for phrase in banned)


def test_oracle_json_schema_is_four_keys() -> None:
    parsed = parse_oracle_json(MockLLMClient().complete("Professor confirmed this."))

    assert set(parsed) == {"believe", "reshare", "attend", "reason_code"}
    with pytest.raises(ValueError):
        parse_oracle_json('{"hear": 1, "believe": 1, "reshare": 1, "attend": 0, "reason_code": "x"}')


def test_mock_distribution_has_dynamic_range_and_reason_codes() -> None:
    client = MockLLMClient(semantic_tweak=True, tweak_strength=1.5)
    prompts = []
    for i in range(80):
        social = "you are highly social" if i % 4 == 0 else "you are quite skeptical"
        interest = "You actively keep up with internship leads." if i % 5 == 0 else "You don't usually pay attention to campus gossip."
        comments = [
            "I checked, this is real.",
            "This matches what staff confirmed.",
            "Passing this along in case it matters.",
            "No idea if this is true.",
            "Looks like promo spam to me.",
        ]
        selected_comments = [comments[(i + j) % len(comments)] for j in range(1 + (i % 4))]
        exposures = "\n".join(
            f'- Student {j}, a close friend, sent it through a direct message. Student {j} added: "{comment}"'
            for j, comment in enumerate(selected_comments)
        )
        prompts.append(f"{social}. {interest}\nWhat you see:\n{exposures}")

    outputs = [parse_oracle_json(client.complete(prompt)) for prompt in prompts]
    believes = np.array([item["believe"] for item in outputs])
    reasons = {item["reason_code"] for item in outputs}

    assert believes.std() > 0.15
    assert believes.min() < 0.15
    assert believes.max() > 0.85
    assert len(reasons) >= 3


def test_mock_decouples_believe_and_reshare_heads_on_batch() -> None:
    client = MockLLMClient(semantic_tweak=True, tweak_strength=1.5)
    prompts = []
    social_phrases = [
        "you are highly social and often hear what is moving through campus",
        "you are friendly but selective about what you pass along",
        "you keep a small circle and rarely jump into campus chatter",
    ]
    trust_phrases = [
        "you are quite skeptical of unverified information",
        "you like some evidence before fully trusting a claim",
        "you tend to trust campus information when it comes from peers you know",
    ]
    interest_phrases = [
        "You actively keep up with internship leads.",
        "You occasionally follow campus policy changes.",
        "You don't usually pay attention to campus gossip.",
    ]
    controversy_phrases = [
        "The message itself feels strongly sourced and unlikely to stir controversy.",
        "The message itself feels somewhat sourced and somewhat controversial.",
        "The message itself feels weakly sourced and highly controversial.",
    ]
    comments = [
        "I checked, this is real.",
        "A professor also posted this.",
        "No strong take, but people are sharing it.",
        "No idea if this is true.",
        "Looks like promo spam to me.",
    ]
    for i in range(240):
        exposures = "\n".join(
            f'- Maya, a close friend; you know them from class. They sent it through a direct message. Maya added: "{comments[(i + j) % len(comments)]}"'
            for j in range(1 + (i % 4))
        )
        prompts.append(
            "\n".join(
                [
                    social_phrases[i % 3],
                    trust_phrases[(i // 3) % 3],
                    interest_phrases[(i // 9) % 3],
                    controversy_phrases[(i // 27) % 3],
                    "What you see:",
                    exposures,
                ]
            )
        )

    outputs = [parse_oracle_json(client.complete(prompt)) for prompt in prompts]
    believe = np.array([item["believe"] for item in outputs])
    reshare = np.array([item["reshare"] for item in outputs])

    assert np.corrcoef(believe, reshare)[0, 1] < 0.95
    assert np.mean((believe - reshare) > 0.15) >= 0.10
    assert reshare.std() > 0.12
    assert reshare.min() < 0.15
    assert reshare.max() > 0.70


def test_mock_direct_twin_authority_cue_depends_on_tweak_flag() -> None:
    base = (
        "you are friendly but selective. You occasionally follow course-material chatter.\n"
        'What you see:\n- Student 1, a familiar friend, sent it through a big campus group chat. '
        'They quote the message: "The review sheet is posted." Student 1 added: "Passing this along in case it matters."'
    )
    authority = base.replace("The review sheet is posted.", "The professor posted the official review sheet.")
    on = MockLLMClient(semantic_tweak=True, tweak_strength=1.5)
    off = MockLLMClient(semantic_tweak=False, tweak_strength=1.5)

    on_delta = abs(parse_oracle_json(on.complete(authority))["believe"] - parse_oracle_json(on.complete(base))["believe"])
    off_delta = abs(parse_oracle_json(off.complete(authority))["believe"] - parse_oracle_json(off.complete(base))["believe"])

    assert on_delta >= 0.15
    assert off_delta <= 0.02


def test_mock_outputs_are_identical_across_scaffold_paraphrases() -> None:
    net = generate_campus_network(20, seed=44)
    cond = sample_condition(np.random.default_rng(45), topic="internship")
    agent = next(i for i in net.G.nodes() if net.G.degree(i) > 0)
    exposer = next(iter(net.G.neighbors(agent)))
    details = {
        exposer: {
            "stance": "endorse",
            "comment_id": "internship:endorse:0",
            "comment": "I checked, this is real.",
        }
    }
    client = MockLLMClient(semantic_tweak=True)
    outputs = [
        client.complete(build_decision_prompt(net, agent, [exposer], cond, scaffold_id=scaffold, exposure_details=details))
        for scaffold in range(5)
    ]

    assert len(set(outputs)) == 1


def test_full_mock_scenario_runs_and_is_monotone(tmp_path) -> None:
    net = generate_campus_network(50, seed=3)
    cond = sample_condition(np.random.default_rng(4), topic="club_event")
    result = run_oracle_scenario(
        net,
        cond,
        seeds={0, 1, 2},
        client=MockLLMClient(),
        T=8,
        rng=np.random.default_rng(5),
        scenario_id="full",
        run_id="full",
        log_dir=tmp_path,
    )
    frame = result.to_frame()

    assert not frame.empty
    assert "hear" not in frame.columns
    assert "sampled_heard" not in frame.columns
    assert "sampled_believed" not in frame.columns
    assert {"n_endorse", "n_neutral", "n_doubt", "exposer_comment_ids", "exposure_metadata"} <= set(frame.columns)
    assert frame.groupby("agent_id")["query_index"].max().max() <= 2
    assert [len(x) for x in result.heard_history] == sorted(len(x) for x in result.heard_history)
    assert [len(x) for x in result.reshared_history] == sorted(len(x) for x in result.reshared_history)
    assert not set([0, 1, 2]) & set(frame["agent_id"])
    parquet_path = tmp_path / "result.parquet"
    result.to_parquet(parquet_path)
    assert parquet_path.exists()
    record = json.loads((tmp_path / "full.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "exposure_metadata" in record
    assert {"stance", "comment_id", "comment"} <= set(record["exposure_metadata"][0])


def test_line_graph_exposure_front_advances_one_hop_per_step(tmp_path) -> None:
    net = _graph_net([(0, 1), (1, 2), (2, 3)])
    result = run_oracle_scenario(
        net,
        _condition(),
        seeds={0},
        client=ConstantClient(reshare=1.0),
        T=3,
        rng=np.random.default_rng(1),
        scenario_id="line",
        run_id="line",
        log_dir=tmp_path,
    )

    assert result.heard_history == [{0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}]
    assert result.reshared_history == [{0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}]
    first_queries = result.to_frame()[lambda df: df["query_index"] == 1]
    assert first_queries["step"].tolist() == [1, 2, 3]


def test_requery_only_when_exposure_set_grows_and_budget_remains(tmp_path) -> None:
    net = _graph_net([(0, 2), (2, 1)])
    result = run_oracle_scenario(
        net,
        _condition(),
        seeds={0},
        client=ConstantClient(reshare=1.0),
        T=4,
        rng=np.random.default_rng(2),
        scenario_id="requery",
        run_id="requery",
        log_dir=tmp_path,
        max_queries_per_agent=2,
    )
    frame = result.to_frame()
    records = [
        json.loads(line)
        for line in (tmp_path / "requery.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert frame[frame["agent_id"] == 2]["query_index"].tolist() == [1, 2]
    assert any("already saw this earlier" in record["full_prompt"] for record in records)


def test_requery_prompt_recaps_prior_and_lists_only_new_exposures() -> None:
    net = _graph_net([(0, 3), (1, 3), (2, 3)])
    cond = _condition()
    details = {
        1: {"stance": "endorse", "comment_id": "course_material:endorse:0", "comment": "I checked, this is real."},
        2: {"stance": "neutral", "comment_id": "course_material:neutral:1", "comment": "No strong take, but people are sharing it."},
    }
    prompt = build_requery_prompt(
        net,
        agent_i=3,
        previous_exposures={0},
        new_exposures={1, 2},
        cond=cond,
        scaffold_id=0,
        exposure_details=details,
    )
    prior_name = student_name(net, 0)
    new_names = [student_name(net, 1), student_name(net, 2)]

    assert f"You already saw this earlier when {prior_name} shared it." in prompt
    assert prompt.count(prior_name) == 1
    for name in new_names:
        assert name in prompt
    assert "I checked, this is real." in prompt
    assert "No strong take, but people are sharing it." in prompt
    assert prompt.count(cond.text) == 1
    assert prompt.rfind('{"believe": 0.0') > prompt.find("Updated feed:")


def test_semantic_twin_replay_keeps_exposers_stances_and_comments_identical(tmp_path) -> None:
    net = _graph_net([(0, 1), (1, 2), (2, 3)])
    cond_a = _condition()
    cond_b = Condition(
        cond_a.topic,
        cond_a.urgency,
        cond_a.credibility,
        cond_a.controversy,
        "A professor also posted the official review sheet.",
        "course_material:0:1",
    )
    first = run_oracle_scenario(
        net,
        cond_a,
        seeds={0},
        client=ConstantClient(reshare=1.0),
        T=3,
        rng=np.random.default_rng(30),
        scenario_id="twin_a",
        run_id="twin_a",
        log_dir=tmp_path,
    )
    second = run_oracle_scenario(
        net,
        cond_b,
        seeds={0},
        client=ConstantClient(reshare=1.0),
        T=3,
        rng=np.random.default_rng(999),
        scenario_id="twin_b",
        run_id="twin_b",
        log_dir=tmp_path,
        exposure_script=first.exposure_script,
    )
    cols = ["step", "agent_id", "exposer_ids", "exposer_stances", "exposer_comment_ids"]

    assert first.to_frame()[cols].to_json() == second.to_frame()[cols].to_json()
    assert first.to_frame()["condition_text"].tolist() != second.to_frame()["condition_text"].tolist()


def test_feed_exposure_comments_are_distinct_within_stance(monkeypatch, tmp_path) -> None:
    def duplicate_comment(topic: str, rng: np.random.Generator, stance_probs: tuple[float, float, float]) -> dict[str, str]:
        return {"stance": "endorse", "comment_id": f"{topic}:endorse:0", "comment": "I checked, this is real."}

    monkeypatch.setattr(runner_module, "sample_share_metadata", duplicate_comment)
    net = _graph_net([(0, 3), (1, 3), (2, 3)])
    result = run_oracle_scenario(
        net,
        _condition(),
        seeds={0, 1, 2},
        client=ConstantClient(reshare=0.0),
        T=1,
        rng=np.random.default_rng(4),
        scenario_id="distinct_comments",
        run_id="distinct_comments",
        log_dir=tmp_path,
    )
    metadata = json.loads(result.to_frame().iloc[0]["exposure_metadata"])
    comment_ids = [item["comment_id"] for item in metadata]
    comments = [item["comment"] for item in metadata]

    assert len(comment_ids) == len(set(comment_ids))
    assert len(comments) == len(set(comments))


def test_jsonl_replay_is_identical_and_no_duplicate_calls(tmp_path) -> None:
    net = generate_campus_network(30, seed=6)
    cond = sample_condition(np.random.default_rng(7), topic="internship")
    kwargs = dict(
        net=net,
        cond=cond,
        seeds={0, 1},
        client=MockLLMClient(),
        T=5,
        scenario_id="replay",
        run_id="replay",
        log_dir=tmp_path,
    )
    first = run_oracle_scenario(**kwargs, rng=np.random.default_rng(8)).to_frame()
    log_path = tmp_path / "replay.jsonl"
    first_log_lines = log_path.read_text(encoding="utf-8").splitlines()
    second = run_oracle_scenario(**kwargs, rng=np.random.default_rng(8)).to_frame()
    second_log_lines = log_path.read_text(encoding="utf-8").splitlines()

    assert len(first_log_lines) == len(second_log_lines)
    assert first.to_json() == second.to_json()


def test_resumability_after_budget_stop(tmp_path) -> None:
    net = generate_campus_network(35, seed=9)
    cond = sample_condition(np.random.default_rng(10), topic="policy_change")

    with pytest.raises(BudgetExceeded):
        run_oracle_scenario(
            net,
            cond,
            seeds={0, 1, 2},
            client=MockLLMClient(),
            T=4,
            rng=np.random.default_rng(11),
            scenario_id="resume",
            run_id="resume",
            log_dir=tmp_path,
            budget=BudgetGuard(max_api_calls=1),
        )
    partial_lines = (tmp_path / "resume.jsonl").read_text(encoding="utf-8").splitlines()

    run_oracle_scenario(
        net,
        cond,
        seeds={0, 1, 2},
        client=MockLLMClient(),
        T=4,
        rng=np.random.default_rng(11),
        scenario_id="resume",
        run_id="resume",
        log_dir=tmp_path,
        budget=BudgetGuard(max_api_calls=500),
    )
    final_records = [
        json.loads(line)
        for line in (tmp_path / "resume.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    keys = [(r["scenario_id"], r["step"], r["agent_id"], r["scaffold_id"]) for r in final_records]

    assert len(partial_lines) == 1
    assert len(keys) == len(set(keys))


def test_budget_guard_trips_on_tiny_cap(tmp_path) -> None:
    client = MockLLMClient()
    with pytest.raises(BudgetExceeded):
        complete_json_logged(
            client=client,
            prompt="Return JSON",
            log_path=tmp_path / "budget.jsonl",
            budget=BudgetGuard(max_api_calls=0),
            scenario_id="budget",
            step=1,
            agent_id=1,
            scaffold_id=0,
        )


def test_api_usage_events_drive_logged_tokens_and_cost(tmp_path) -> None:
    class UsageClient(ConstantClient):
        usage_events: list[dict[str, object]]

        def __init__(self) -> None:
            super().__init__()
            self.usage_events = []

        def complete(self, prompt: str) -> str:
            self.usage_events.append(
                {
                    "response_id": "response_test",
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "total_tokens": 1100,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                }
            )
            return super().complete(prompt)

    log_path = tmp_path / "usage.jsonl"
    budget = BudgetGuard(
        max_usd=1.0,
        input_usd_per_1m=0.75,
        output_usd_per_1m=4.50,
    )
    complete_json_logged(
        client=UsageClient(),
        prompt="Return JSON",
        log_path=log_path,
        budget=budget,
        scenario_id="usage",
        step=1,
        agent_id=1,
        scaffold_id=0,
    )
    record = json.loads(log_path.read_text(encoding="utf-8"))

    assert record["usage_source"] == "api_response"
    assert record["token_counts"]["prompt"] == 1000
    assert record["token_counts"]["completion"] == 100
    assert record["first_attempt_valid"] is True
    assert record["repaired"] is False
    assert record["call_cost_usd"] == pytest.approx(0.0012)
