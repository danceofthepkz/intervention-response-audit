"""Export only the data displayed by the interactive article. No API calls.

Run from the repository root: python docs/build_data.py
Uses the original feature builder; verifies joins, means, collision, and KL.
"""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from intervention_response_audit.audit import build_feature_frame


def read_json(path):
    return json.loads((ROOT / path).read_text())


def read_rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line]


def kl(p, q):
    return sum(a * math.log(a / b) for a, b in ((p, q), (1-p, 1-q)) if a)


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-10), (a, b)


paths = {
    "states": "prereg/v2/stage0/state_allocation.jsonl",
    "prompts": "prereg/v2/stage0/prompt_corpus.jsonl",
    "calls": "results/v2/confirmatory/attempts.jsonl",
    "primary": "results/v2/confirmatory/primary_result.json",
    "predictions": "results/v2/secondary/structured_anchor_predictions.json",
    "secondary": "results/v2/secondary/secondary_result.json",
    "config": "prereg/v2/stage0/frozen_config.json",
    "comparison": "docs/representation-summary.json",
}
states = sorted((s for s in read_rows(paths["states"]) if s["allocation"] == "confirmatory"), key=lambda s:s["scenario_id"])
prompts = {(p["state_id"], p["condition"]):p for p in read_rows(paths["prompts"])}
calls = read_rows(paths["calls"])
predictions = read_json(paths["predictions"])
primary = read_json(paths["primary"])["primary"]
secondary = read_json(paths["secondary"])
assert len(states) == 60 and len(calls) == 360 and all(c["accepted"] for c in calls)
frame = build_feature_frame(pd.DataFrame([s["structured_features"] for s in states]))
assert frame.shape == (60, 27)
export = []
for i, s in enumerate(states):
    sid = s["state_id"]
    a, h = [prompts[sid, arm] for arm in ("authority", "hedge")]
    outside = []
    for p in (a, h):
        full = p["full_prompt"]
        assert full[p["message_start"]:p["message_end"]] == p["message_text"]
        outside.append((full[:p["message_start"]], full[p["message_end"]:]))
    assert outside[0] == outside[1]
    arm_data = {}
    for arm, p in (("authority", a), ("hedge", h)):
        selected = sorted((c for c in calls if c["state_id"] == sid and c["condition"] == arm), key=lambda c:c["replicate_index"])
        assert len(selected) == 3 and len({c["replicate_index"] for c in selected}) == 3
        assert all(c["prompt_sha256"] == p["prompt_sha256"] for c in selected)
        samples = [c["parsed_output"]["reshare"] for c in selected]
        arm_data[arm] = {"text":p["message_text"], "samples":samples, "mean":statistics.mean(samples)}
    pa, ph = [arm_data[arm]["mean"] for arm in ("authority", "hedge")]
    m = (pa + ph) / 2
    floor = (kl(pa, m) + kl(ph, m))/2 if 0 < m < 1 else 0
    q = predictions[sid]
    total = (kl(pa,q) + kl(ph,q))/2
    fit = kl(m,q)
    close(total, fit+floor)
    close(pa-ph, primary["state_contrasts"][sid])
    export.append({"id":sid, "scenario":s["scenario_id"], "topic":s["topic"], "credibility":s["credibility_tier"], "agent":s["agent_id"], "step":s["step"], "features":frame.iloc[i].to_dict(), "context":outside[0][0].split("\n\nReturn exactly")[0], "promptBefore":outside[0][0], "promptAfter":outside[0][1], "arms":arm_data, "delta":pa-ph, "midpoint":m, "floor":floor, "q":q, "fit":fit, "total":total})
mean = lambda key:statistics.mean(s[key] for s in export)
close(mean("delta"), primary["estimate"])
close(mean("floor"), secondary["reshare"]["r_text_raw"])
close(mean("total"), secondary["structured"]["r_total_cf_anchor"])
close(mean("fit"), secondary["structured"]["r_struct_cf_anchor"])
# Choose the state nearest the median signed response, rather than an extreme.
median = statistics.median(s["delta"] for s in export)
default = min(range(60), key=lambda i:abs(export[i]["delta"]-median))
data = {"states":export, "defaultIndex":default, "summary":{"delta":mean("delta"), "floor":mean("floor"), "total":mean("total"), "fit":mean("fit"), "ci":primary["fixed_template_ci"], "clusterCi":primary["template_cluster_ci"], "k":3, "n":60, "model":read_json(paths["config"])["model_snapshot"]}, "comparison":read_json(paths["comparison"]), "provenance":[{"path":p, "sha256":hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in paths.values()]}
report = {"status":"PASS", "states":60, "accepted_calls":360, "features":27, "verified":["Paired prompts differ only inside the message span", "Three unique repetitions per arm, matched to frozen prompts", "All 60 contrasts match the primary result", "Features derived with the original 27-feature builder", "Exact KL decomposition on every state", "Aggregate contrast and loss components match frozen results"], "not_recomputed":"R1–R3 are transcribed manuscript summaries, not refitted predictions.", "summary":data["summary"]}
out = ROOT/"docs"
encoded = json.dumps(data, ensure_ascii=False, separators=(",",":"))
(out/"data.json").write_text(encoded+"\n")
(out/"data.js").write_text("window.AUDIT_DATA = "+encoded+";\n")
(out/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
