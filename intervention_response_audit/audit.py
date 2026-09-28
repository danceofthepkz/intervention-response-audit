"""Logistic hazard audit, noise floor, semantic twins, and go/no-go report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from intervention_response_audit.config import settings
from intervention_response_audit.network import TOPICS

TARGETS = ("believe", "reshare", "attend")
EPS = 1e-6


@dataclass
class AuditResult:
    metrics: pd.DataFrame
    decision: str
    report_path: Path
    nll_plot_path: Path
    twin_plot_path: Path


def build_feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Build numeric mechanistic features from oracle decision rows."""

    frame = pd.DataFrame(index=rows.index)
    for col in (
        "n_exposures",
        "sum_tie_strength",
        "max_tie_strength",
        "same_major_exposers",
        "same_club_exposers",
        "source_diversity",
        "n_endorse",
        "n_doubt",
        "n_neutral",
        "step",
        "urgency",
        "credibility",
        "controversy",
        "socialness",
        "trust",
        "topic_interest",
        "year",
    ):
        frame[col] = rows[col].astype(float)
    for topic in TOPICS:
        frame[f"topic_{topic}"] = (rows["topic"] == topic).astype(float)
    frame["trust_x_credibility"] = frame["trust"] * frame["credibility"]
    frame["socialness_x_club_event"] = frame["socialness"] * frame["topic_club_event"]
    frame["interest_x_urgency"] = frame["topic_interest"] * frame["urgency"]
    frame["trust_x_n_doubt"] = frame["trust"] * frame["n_doubt"]
    frame["interest_x_n_endorse"] = frame["topic_interest"] * frame["n_endorse"]
    return frame


def _bernoulli_training_rows(
    X: pd.DataFrame, probs: pd.Series, seed: int, draws_per_row: int = 10
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    repeated = pd.concat([X] * draws_per_row, ignore_index=True)
    p = np.tile(probs.to_numpy(dtype=float), draws_per_row)
    y = (rng.random(len(p)) < p).astype(int)
    if len(np.unique(y)) == 1:
        y[0] = 1 - y[0]
    return repeated, y


def _fit_logistic(X: pd.DataFrame, probs: pd.Series, seed: int) -> Any:
    Cs = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    cv = min(5, max(2, len(X) // 8), len(X))
    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    scores: list[tuple[float, float]] = []
    for C in Cs:
        fold_scores = []
        for train_idx, valid_idx in splitter.split(X):
            X_train = X.iloc[train_idx]
            p_train = probs.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            p_valid = probs.iloc[valid_idx].to_numpy(dtype=float)
            X2, y = _bernoulli_training_rows(X_train, p_train, seed + int(C * 1000))
            model = _make_logistic(C, seed)
            model.fit(X2, y)
            fold_scores.append(_nll(p_valid, _predict(model, X_valid)))
        scores.append((float(np.mean(fold_scores)), float(C)))
    best_C = min(scores, key=lambda item: item[0])[1]
    X2, y = _bernoulli_training_rows(X, probs, seed)
    model = _make_logistic(best_C, seed)
    model.fit(X2, y)
    return model


def _make_logistic(C: float, seed: int) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        ),
    )
    return model


def _fit_gbm(X: pd.DataFrame, probs: pd.Series, seed: int) -> Any:
    X2, y = _bernoulli_training_rows(X, probs, seed)
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X2, y)
    return model


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        probs = model.predict_proba(X)[:, 1]
    return np.clip(probs, EPS, 1.0 - EPS)


def _nll(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    q = np.clip(q, EPS, 1.0 - EPS)
    return float(np.mean(-(p * np.log(q) + (1.0 - p) * np.log(1.0 - q))))


def _brier(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean((p - q) ** 2))


def _scenario_split(rows: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    scenarios = np.array(sorted(rows["scenario_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios)
    test_n = max(1, int(round(len(scenarios) * 0.25)))
    test_scenarios = set(scenarios[:test_n])
    test = rows["scenario_id"].astype(str).isin(test_scenarios).to_numpy()
    return ~test, test


def noise_floor(rows: pd.DataFrame, seed: int = 42, bootstrap_R: int = 1000) -> dict[str, dict[str, float]]:
    """Compute scaffold-repeat cross entropy against item means."""

    group_cols = ["scenario_id", "step", "agent_id"]
    repeated = rows.groupby(group_cols).filter(lambda g: g["scaffold_id"].nunique() >= 2)
    if repeated.empty:
        return {target: {"nll_floor": float("nan"), "bootstrap_std": float("nan"), "mean_item_std": 0.0} for target in TARGETS}

    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    grouped = list(repeated.groupby(group_cols))
    for target in TARGETS:
        item_losses = []
        item_stds = []
        for _, group in grouped:
            vals = group[target].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            losses = -(vals * np.log(np.clip(mean, EPS, 1 - EPS)) + (1 - vals) * np.log(np.clip(1 - mean, EPS, 1 - EPS)))
            item_losses.append(float(np.mean(losses)))
            item_stds.append(float(np.std(vals)))
        samples = [
            float(np.mean(rng.choice(item_losses, size=len(item_losses), replace=True)))
            for _ in range(bootstrap_R)
        ]
        out[target] = {
            "nll_floor": float(np.mean(item_losses)),
            "bootstrap_std": float(np.std(samples)),
            "mean_item_std": float(np.mean(item_stds)),
        }
    return out


def semantic_twins(rows: pd.DataFrame, models: dict[str, Any], features: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Measure oracle and mechanistic deltas for paired semantic twins."""

    work = rows.copy()
    if "twin_group" not in work.columns:
        parsed = work["scenario_id"].astype(str).str.extract(r"^(?P<twin_group>.+)_(?P<twin_side>[ab])$")
        work["twin_group"] = parsed["twin_group"]
        work["twin_side"] = parsed["twin_side"]
    if "twin_side" not in work.columns:
        return {target: {"oracle_abs_delta": 0.0, "mech_abs_delta": 0.0} for target in TARGETS}

    out: dict[str, dict[str, float]] = {}
    work["_row_index"] = np.arange(len(work))
    key_cols = ["twin_group", "step", "agent_id", "query_index"]
    left = work[work["twin_side"] == "a"]
    right = work[work["twin_side"] == "b"]
    merged = left.merge(right, on=key_cols, suffixes=("_a", "_b"))
    for target in TARGETS:
        if merged.empty:
            out[target] = {"oracle_abs_delta": 0.0, "mech_abs_delta": 0.0}
            continue
        oracle_delta = np.abs(merged[f"{target}_a"] - merged[f"{target}_b"]).to_numpy(dtype=float)
        idx_a = merged["_row_index_a"].to_numpy(dtype=int)
        idx_b = merged["_row_index_b"].to_numpy(dtype=int)
        pred_a = _predict(models[target], features.iloc[idx_a])
        pred_b = _predict(models[target], features.iloc[idx_b])
        out[target] = {
            "oracle_abs_delta": float(np.mean(oracle_delta)),
            "mech_abs_delta": float(np.mean(np.abs(pred_a - pred_b))),
        }
    return out


def run_audit(
    rows: pd.DataFrame,
    output_dir: str | Path = settings.data_dir / "audit",
    seed: int = 42,
    fit_gbm: bool = False,
) -> AuditResult:
    """Run the full audit and write report plus plots."""

    if rows.empty:
        raise ValueError("audit requires at least one oracle decision row")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = rows.reset_index(drop=True)
    features = build_feature_frame(rows)
    train_mask, test_mask = _scenario_split(rows, seed)
    models: dict[str, Any] = {}
    metrics: list[dict[str, float | str]] = []
    floors = noise_floor(rows, seed=seed, bootstrap_R=250)

    for target in TARGETS:
        model = _fit_logistic(features[train_mask], rows.loc[train_mask, target], seed)
        models[target] = model
        pred = _predict(model, features[test_mask])
        truth = rows.loc[test_mask, target].to_numpy(dtype=float)
        row: dict[str, float | str] = {
            "target": target,
            "NLL_mech": _nll(truth, pred),
            "Brier_mech": _brier(truth, pred),
            "NLL_floor": floors[target]["nll_floor"],
            "floor_bootstrap_std": floors[target]["bootstrap_std"],
        }
        row["Gap"] = float(row["NLL_mech"]) - float(row["NLL_floor"])
        if fit_gbm:
            gbm = _fit_gbm(features[train_mask], rows.loc[train_mask, target], seed)
            row["NLL_gbm"] = _nll(truth, _predict(gbm, features[test_mask]))
        metrics.append(row)

    twins = semantic_twins(rows, models, features)
    for row in metrics:
        target = str(row["target"])
        row["twins_oracle_abs_delta"] = twins[target]["oracle_abs_delta"]
        row["twins_mech_abs_delta"] = twins[target]["mech_abs_delta"]
    metrics_df = pd.DataFrame(metrics)

    pass_targets = []
    for target in ("believe", "reshare"):
        row = metrics_df[metrics_df["target"] == target].iloc[0]
        threshold = max(settings.pass_gap_nats, 3.0 * float(row["floor_bootstrap_std"]))
        pass_targets.append(
            float(row["Gap"]) > threshold
            and float(row["twins_oracle_abs_delta"]) >= settings.pass_twin_delta
            and float(row["twins_mech_abs_delta"]) < 0.02
        )
    decision = "PASS" if all(pass_targets) else "FAIL"

    nll_plot = output / "nll_gap.png"
    twin_plot = output / "twins_scatter.png"
    _plot_nll(metrics_df, nll_plot)
    _plot_twins(metrics_df, twin_plot)
    report = output / "audit_report.md"
    _write_report(metrics_df, decision, report, nll_plot, twin_plot)
    return AuditResult(metrics_df, decision, report, nll_plot, twin_plot)


def _plot_nll(metrics: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, metrics["NLL_mech"], width, label="Mechanistic")
    ax.bar(
        x + width / 2,
        metrics["NLL_floor"],
        width,
        yerr=metrics["floor_bootstrap_std"],
        label="Noise floor",
    )
    ax.set_xticks(x, metrics["target"])
    ax.set_ylabel("NLL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_twins(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(metrics["twins_mech_abs_delta"], metrics["twins_oracle_abs_delta"])
    lim = max(0.2, float(metrics[["twins_mech_abs_delta", "twins_oracle_abs_delta"]].max().max()) + 0.02)
    ax.plot([0, lim], [0, lim], linestyle="--", color="gray")
    for _, row in metrics.iterrows():
        ax.annotate(str(row["target"]), (row["twins_mech_abs_delta"], row["twins_oracle_abs_delta"]))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mechanistic |delta p|")
    ax.set_ylabel("Oracle |delta p|")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report(metrics: pd.DataFrame, decision: str, report: Path, nll_plot: Path, twin_plot: Path) -> None:
    lines = [
        "# Intervention Response Audit Phase 1 Audit Report",
        "",
        f"Decision: **{decision}**",
        "",
        _markdown_table(metrics),
        "",
        f"![NLL gap]({nll_plot.name})",
        "",
        f"![Semantic twins]({twin_plot.name})",
        "",
    ]
    if decision == "FAIL":
        lines.append(
            "Oracle is mechanistically explainable at this granularity; do not proceed to Phase 2 without redesigning the oracle."
        )
    report.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
