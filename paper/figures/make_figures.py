#!/usr/bin/env python3
"""Generate the SEA 2026 paper figures from frozen v2 artifacts.

All final outputs are vector PDFs. The script fails loudly if the result files
no longer match the numerical invariants used by the manuscript.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output" / "pdf"

PRIMARY_PATH = ROOT / "results" / "v2" / "confirmatory" / "primary_result.json"
SECONDARY_PATH = ROOT / "results" / "v2" / "secondary" / "secondary_result.json"
PILOT_PATH = ROOT / "results" / "v2" / "pilot" / "attempts.jsonl"
PILOT_DECISION_PATH = ROOT / "results" / "v2" / "pilot" / "pilot_decision.json"
ALLOCATION_PATH = ROOT / "prereg" / "v2" / "stage0" / "state_allocation.csv"
CONFIG_PATH = ROOT / "prereg" / "v2" / "stage0" / "frozen_config.json"


# Okabe-Ito-inspired, colorblind-safe palette.
BLUE = "#0072B2"
SKY = "#8CCBE7"
ORANGE = "#D55E00"
PALE_ORANGE = "#F3C6A5"
DARK = "#24292F"
MID_GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
ROW_GRAY = "#E5E7EB"
WHITE = "#FFFFFF"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": "none",
        }
    )


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close_enough(actual: float, expected: float, tolerance: float = 5e-10) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ValueError(f"Unexpected value: got {actual}, expected {expected}")


def load_data() -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    primary = load_json(PRIMARY_PATH)["primary"]
    secondary = load_json(SECONDARY_PATH)
    config = load_json(CONFIG_PATH)
    allocation = pd.read_csv(ALLOCATION_PATH)

    state_effects = (
        pd.Series(primary["state_contrasts"], name="contrast")
        .rename_axis("state_id")
        .reset_index()
        .merge(
            allocation[
                [
                    "state_id",
                    "scenario_id",
                    "allocation",
                    "topic",
                    "credibility_tier",
                    "template_cell",
                ]
            ],
            on="state_id",
            how="left",
            validate="one_to_one",
        )
    )
    if len(state_effects) != 60 or state_effects.isna().any().any():
        raise ValueError("The 60 confirmatory state effects did not map cleanly to metadata")
    if not (state_effects["allocation"] == "confirmatory").all():
        raise ValueError("A non-confirmatory state entered the state-effect figure")

    pilot = pd.DataFrame(load_jsonl(PILOT_PATH))
    if len(pilot) != 200 or not pilot["accepted"].all():
        raise ValueError("Expected exactly 200 accepted pilot calls")
    pilot["reshare"] = pilot["parsed_output"].map(lambda output: output["reshare"])
    pilot = pilot.merge(
        allocation[
            ["state_id", "scenario_id", "allocation", "topic", "credibility_tier"]
        ],
        on=["state_id", "scenario_id"],
        how="left",
        validate="many_to_one",
    )
    if pilot[["topic", "credibility_tier"]].isna().any().any():
        raise ValueError("Pilot calls did not map cleanly to state metadata")
    if not (pilot["allocation"] == "pilot").all():
        raise ValueError("A non-pilot state entered the repeatability figure")

    # Manuscript-level invariants.
    close_enough(primary["estimate"], 0.0866111111111111)
    close_enough(primary["fixed_template_ci"][0], 0.0670)
    close_enough(primary["fixed_template_ci"][1], 0.10627777777777778)
    close_enough(primary["template_cluster_ci"][0], 0.059277314814814824)
    close_enough(primary["template_cluster_ci"][1], 0.11477037037037037)
    close_enough(config["practical_delta"], 0.03)
    close_enough(secondary["reshare"]["r_text_raw"], 0.011211142235726477)
    close_enough(secondary["reshare"]["r_text_adjusted"], 0.008715964522750178)
    close_enough(
        secondary["reshare"]["r_text_permutation_null"], 0.002495177712976299
    )
    close_enough(secondary["structured"]["r_struct_cf_anchor"], 0.02685248546001513)
    close_enough(secondary["structured"]["r_total_cf_anchor"], 0.03806362769574163)
    close_enough(
        secondary["structured"]["r_struct_cf_anchor"]
        + secondary["reshare"]["r_text_raw"],
        secondary["structured"]["r_total_cf_anchor"],
    )
    return primary, secondary, config, state_effects, pilot


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=DARK, length=3, width=0.6)
    ax.tick_params(axis="y", length=0)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return path


def probability_tick(value: float, _position: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:.2f}"


def make_figure_1(primary: dict, secondary: dict, threshold: float) -> Path:
    estimate = primary["estimate"]
    fixed_low, fixed_high = primary["fixed_template_ci"]
    cluster_low, cluster_high = primary["template_cluster_ci"]

    r_text = secondary["reshare"]["r_text_raw"]
    r_noise = secondary["reshare"]["r_text_permutation_null"]
    r_adjusted = secondary["reshare"]["r_text_adjusted"]
    r_struct = secondary["structured"]["r_struct_cf_anchor"]
    r_total = secondary["structured"]["r_total_cf_anchor"]

    # Native width targets a NeurIPS-style 5.5 inch text block, keeping the
    # smallest labels at or above 7 pt after inclusion at \linewidth.
    fig = plt.figure(figsize=(5.5, 2.30))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(3.0, 2.0),
        left=0.075,
        right=0.985,
        bottom=0.24,
        top=0.78,
        wspace=0.32,
    )
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    # A shared top rule makes the common 60-anchor population explicit.
    fig.text(
        0.525,
        0.935,
        "Same 60 frozen anchor states",
        ha="center",
        va="center",
        color=DARK,
        fontsize=9.2,
        fontweight="bold",
    )
    fig.add_artist(
        Line2D(
            [0.15, 0.90],
            [0.885, 0.885],
            transform=fig.transFigure,
            color=BLUE,
            linewidth=1.2,
            solid_capstyle="round",
        )
    )

    # Panel a: behavioral contrast with nested confidence intervals.
    y_effect = 0.50
    ax_left.axvline(0.0, color=MID_GRAY, linewidth=0.85, linestyle=(0, (2, 2)), zorder=0)
    ax_left.axvline(
        threshold, color=ORANGE, linewidth=1.0, linestyle=(0, (4, 2)), zorder=0
    )
    ax_left.hlines(
        y_effect,
        cluster_low,
        cluster_high,
        color=BLUE,
        linewidth=1.1,
        zorder=3,
    )
    ax_left.vlines(
        [cluster_low, cluster_high],
        y_effect - 0.075,
        y_effect + 0.075,
        color=BLUE,
        linewidth=1.0,
        zorder=3,
    )
    ax_left.hlines(
        y_effect,
        fixed_low,
        fixed_high,
        color=BLUE,
        linewidth=5.0,
        zorder=4,
    )
    ax_left.vlines(
        [fixed_low, fixed_high],
        y_effect - 0.05,
        y_effect + 0.05,
        color=BLUE,
        linewidth=2.0,
        zorder=4,
    )
    ax_left.scatter(
        [estimate],
        [y_effect],
        s=42,
        marker="D",
        facecolor=ORANGE,
        edgecolor=WHITE,
        linewidth=0.7,
        zorder=5,
    )
    ax_left.text(
        estimate,
        0.69,
        "8.66 pp",
        ha="center",
        va="bottom",
        color=DARK,
        fontsize=8.5,
        fontweight="bold",
    )
    ax_left.text(
        estimate,
        0.29,
        "~3 additional expected resharers per scenario\n(first-order)",
        ha="center",
        va="top",
        color=MID_GRAY,
        fontsize=7.2,
        linespacing=1.15,
    )
    ax_left.text(
        cluster_high,
        0.59,
        "template-cluster CI",
        ha="right",
        va="bottom",
        color=BLUE,
        fontsize=7.0,
    )
    ax_left.text(
        fixed_high,
        0.42,
        "fixed-template CI",
        ha="right",
        va="top",
        color=BLUE,
        fontsize=7.0,
        fontweight="bold",
    )
    ax_left.text(
        0.0015,
        0.89,
        "zero",
        color=MID_GRAY,
        ha="left",
        va="top",
        fontsize=7.0,
    )
    ax_left.text(
        threshold + 0.002,
        0.89,
        "0.03 threshold",
        color=ORANGE,
        ha="left",
        va="top",
        fontsize=7.0,
    )
    ax_left.set_xlim(-0.015, 0.135)
    ax_left.set_ylim(0.12, 0.94)
    ax_left.set_yticks([])
    ax_left.xaxis.set_major_locator(MultipleLocator(0.03))
    ax_left.xaxis.set_major_formatter(FuncFormatter(probability_tick))
    ax_left.set_xlabel("Authority - hedge reshare probability")
    ax_left.set_title("a   Behavioral scale", loc="left", pad=5, fontweight="bold")
    style_axis(ax_left)

    # Panel b: exact same-anchor regret decomposition.
    x_bar = 0.40
    width = 0.44
    ax_right.bar(
        x_bar,
        r_struct,
        width=width,
        color=SKY,
        edgecolor=BLUE,
        linewidth=0.8,
        zorder=2,
    )
    ax_right.bar(
        x_bar,
        r_adjusted,
        bottom=r_struct,
        width=width,
        color=ORANGE,
        edgecolor=ORANGE,
        linewidth=0.8,
        zorder=2,
    )
    ax_right.bar(
        x_bar,
        r_noise,
        bottom=r_struct + r_adjusted,
        width=width,
        color=PALE_ORANGE,
        edgecolor=ORANGE,
        linewidth=0.8,
        hatch="////",
        zorder=2,
    )
    ax_right.text(
        x_bar,
        r_struct / 2,
        "$R_{struct}$\n0.0269",
        ha="center",
        va="center",
        color=DARK,
        fontsize=8.0,
        linespacing=1.05,
    )
    ax_right.text(
        x_bar,
        r_struct + r_adjusted / 2,
        "adjusted\n0.0087",
        ha="center",
        va="center",
        color=WHITE,
        fontsize=7.0,
        linespacing=1.0,
    )
    ax_right.annotate(
        "noise 0.0025",
        xy=(x_bar + width / 2, r_struct + r_adjusted + r_noise / 2),
        xytext=(0.73, r_total - 0.0002),
        ha="left",
        va="center",
        fontsize=7.0,
        color=DARK,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.7},
    )
    bracket_x = x_bar + width / 2 + 0.09
    ax_right.annotate(
        "",
        xy=(bracket_x, r_struct),
        xytext=(bracket_x, r_total),
        arrowprops={
            "arrowstyle": "|-|",
            "color": ORANGE,
            "linewidth": 0.85,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax_right.text(
        bracket_x + 0.055,
        r_struct + r_text / 2 - 0.0007,
        "$R_{text}$ = 0.0112\n29% of total",
        ha="left",
        va="center",
        color=DARK,
        fontsize=7.4,
        linespacing=1.1,
    )
    ax_right.text(
        x_bar,
        r_total + 0.0012,
        "total 0.0381",
        ha="center",
        va="bottom",
        color=DARK,
        fontsize=8.0,
        fontweight="bold",
    )
    ax_right.set_xlim(0.0, 1.18)
    ax_right.set_ylim(0.0, 0.043)
    ax_right.set_xticks([])
    ax_right.yaxis.set_major_locator(MultipleLocator(0.01))
    ax_right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax_right.set_ylabel("Regret (nats)", labelpad=2)
    ax_right.yaxis.label.set_bbox({"facecolor": WHITE, "edgecolor": "none", "pad": 0.5})
    ax_right.set_title("b   Log-loss scale", loc="left", pad=5, fontweight="bold")
    style_axis(ax_right)

    connector = ConnectionPatch(
        xyA=(x_bar - width / 2, r_struct + r_text / 2),
        coordsA=ax_right.transData,
        xyB=(estimate, y_effect),
        coordsB=ax_left.transData,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.12",
        color=ORANGE,
        linewidth=0.85,
        alpha=0.9,
        zorder=1,
    )
    fig.add_artist(connector)
    fig.text(
        0.61,
        0.69,
        "same A/H responses",
        ha="center",
        va="center",
        fontsize=7.0,
        color=ORANGE,
        rotation=0,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.4},
    )

    return save_figure(fig, "fig1_dual_scale_dissociation.pdf")


def state_jitter(state_id: str) -> float:
    """Stable vertical jitter in [-0.16, 0.16], derived from the frozen ID."""
    return ((int(state_id[:8], 16) % 1001) / 1000.0 - 0.5) * 0.32


def make_figure_2(primary: dict, state_effects: pd.DataFrame, threshold: float) -> Path:
    topic_order = [
        "internship",
        "club_event",
        "course_material",
        "gossip",
        "policy_change",
    ]
    topic_labels = {
        "internship": "Internship",
        "club_event": "Club event",
        "course_material": "Course material",
        "gossip": "Gossip",
        "policy_change": "Policy change",
    }
    tier_style = {
        "low": ("o", "#8A8F98", "Low credibility"),
        "mid": ("s", BLUE, "Mid credibility"),
        "high": ("^", ORANGE, "High credibility"),
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.00))
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.16, top=0.76)

    y_positions = {topic: 4 - index for index, topic in enumerate(topic_order)}
    for y in y_positions.values():
        ax.axhline(y, color=ROW_GRAY, linewidth=0.6, zorder=0)

    for topic in topic_order:
        topic_rows = state_effects[state_effects["topic"] == topic]
        if len(topic_rows) != 12:
            raise ValueError(f"Expected 12 state effects for {topic}")
        y_base = y_positions[topic]
        for tier, (marker, color, _label) in tier_style.items():
            rows = topic_rows[topic_rows["credibility_tier"] == tier]
            y_values = [y_base + state_jitter(state_id) for state_id in rows["state_id"]]
            ax.scatter(
                rows["contrast"],
                y_values,
                s=28,
                marker=marker,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.55,
                alpha=0.92,
                zorder=3,
            )
        topic_mean = topic_rows["contrast"].mean()
        ax.scatter(
            [topic_mean],
            [y_base],
            s=34,
            marker="D",
            facecolor=WHITE,
            edgecolor=DARK,
            linewidth=1.0,
            zorder=5,
        )

    overall_y = 5.15
    estimate = primary["estimate"]
    fixed_low, fixed_high = primary["fixed_template_ci"]
    cluster_low, cluster_high = primary["template_cluster_ci"]
    ax.axhline(4.65, color=LIGHT_GRAY, linewidth=0.75, zorder=0)
    ax.hlines(overall_y, cluster_low, cluster_high, color=BLUE, linewidth=1.1, zorder=3)
    ax.vlines(
        [cluster_low, cluster_high],
        overall_y - 0.10,
        overall_y + 0.10,
        color=BLUE,
        linewidth=0.9,
        zorder=3,
    )
    ax.hlines(overall_y, fixed_low, fixed_high, color=BLUE, linewidth=4.6, zorder=4)
    ax.vlines(
        [fixed_low, fixed_high],
        overall_y - 0.065,
        overall_y + 0.065,
        color=BLUE,
        linewidth=1.7,
        zorder=4,
    )
    ax.scatter(
        [estimate],
        [overall_y],
        s=42,
        marker="D",
        facecolor=ORANGE,
        edgecolor=WHITE,
        linewidth=0.65,
        zorder=5,
    )
    ax.text(
        estimate + 0.012,
        overall_y,
        "0.0866",
        ha="left",
        va="center",
        fontsize=8.0,
        color=DARK,
        fontweight="bold",
    )

    ax.axvline(0.0, color=MID_GRAY, linewidth=0.85, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(
        threshold, color=ORANGE, linewidth=0.95, linestyle=(0, (4, 2)), zorder=1
    )
    ax.text(0.002, 5.55, "zero", color=MID_GRAY, ha="left", va="top", fontsize=7.5)
    ax.text(
        threshold + 0.002,
        5.55,
        "0.03 threshold",
        color=ORANGE,
        ha="left",
        va="top",
        fontsize=7.5,
    )

    positive = int((state_effects["contrast"] > 0).sum())
    above = int((state_effects["contrast"] > threshold).sum())
    ax.text(
        0.375,
        -0.43,
        f"{positive}/60 > 0; {above}/60 > 0.03",
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=MID_GRAY,
    )

    ax.set_xlim(-0.12, 0.39)
    ax.set_ylim(-0.55, 5.68)
    ax.set_yticks(
        [overall_y] + [y_positions[topic] for topic in topic_order],
        ["Overall"] + [topic_labels[topic] for topic in topic_order],
    )
    ax.xaxis.set_major_locator(MultipleLocator(0.10))
    ax.xaxis.set_major_formatter(FuncFormatter(probability_tick))
    ax.set_xlabel(r"State-level contrast $d_j = \bar{p}_{A,j} - \bar{p}_{H,j}$")
    style_axis(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=tier_style[tier][0],
            linestyle="None",
            markerfacecolor=tier_style[tier][1],
            markeredgecolor=WHITE,
            markeredgewidth=0.5,
            markersize=5.8,
            label=tier_style[tier][2],
        )
        for tier in ("low", "mid", "high")
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor=WHITE,
            markeredgecolor=DARK,
            markeredgewidth=0.9,
            markersize=5.3,
            label="Topic mean",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.69, 0.90),
        ncol=4,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.text(
        0.03,
        0.955,
        "State-level effects by topic",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=DARK,
    )

    return save_figure(fig, "fig2_state_effects_by_topic.pdf")


def make_figure_a1(pilot: pd.DataFrame, pilot_decision: dict) -> Path:
    topic_order = [
        "club_event",
        "course_material",
        "gossip",
        "internship",
        "policy_change",
    ]
    prefix = {
        "club_event": "CLB",
        "course_material": "CRS",
        "gossip": "GOS",
        "internship": "INT",
        "policy_change": "POL",
    }
    state_rows = (
        pilot[["state_id", "scenario_id", "topic"]]
        .drop_duplicates()
        .assign(
            topic_order=lambda frame: frame["topic"].map(
                {topic: index for index, topic in enumerate(topic_order)}
            ),
            scenario_number=lambda frame: frame["scenario_id"].str.extract(r"(\d+)")[0].astype(int),
        )
        .sort_values(["topic_order", "scenario_number"])
        .reset_index(drop=True)
    )
    if len(state_rows) != 10:
        raise ValueError("Expected 10 pilot states")
    state_rows["x"] = np.arange(len(state_rows), dtype=float)
    state_rows["label"] = state_rows.apply(
        lambda row: f"{prefix[row['topic']]}-{int(row['scenario_number']):02d}", axis=1
    )
    pilot = pilot.merge(
        state_rows[["state_id", "x", "label"]], on="state_id", how="left", validate="many_to_one"
    )

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.25, top=0.80)

    condition_style = {
        "authority": (-0.16, BLUE, "Authority"),
        "hedge": (0.16, ORANGE, "Hedge"),
    }
    jitter_values = np.linspace(-0.045, 0.045, 10)
    cell_stats = []

    for _, state in state_rows.iterrows():
        for condition, (offset, color, _label) in condition_style.items():
            rows = pilot[
                (pilot["state_id"] == state["state_id"]) & (pilot["condition"] == condition)
            ].sort_values("replicate_index")
            if len(rows) != 10:
                raise ValueError("Each pilot state-condition cell must contain 10 calls")
            x_center = state["x"] + offset
            x_values = x_center + jitter_values
            y_values = rows["reshare"].to_numpy()
            ax.vlines(
                x_center,
                y_values.min(),
                y_values.max(),
                color=color,
                linewidth=0.7,
                alpha=0.45,
                zorder=1,
            )
            ax.scatter(
                x_values,
                y_values,
                s=17,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.35,
                alpha=0.82,
                zorder=3,
            )
            mean_value = y_values.mean()
            ax.hlines(
                mean_value,
                x_center - 0.075,
                x_center + 0.075,
                color=DARK,
                linewidth=1.15,
                zorder=4,
            )
            # Use the observed range rather than floating-point sample variance
            # to classify exact repeatability (e.g., ten copies of 0.08).
            cell_stats.append(float(np.ptp(y_values)))

    for separator in (1.5, 3.5, 5.5, 7.5):
        ax.axvline(separator, color=LIGHT_GRAY, linewidth=0.65, zorder=0)

    varying_cells = sum(observed_range > 1e-12 for observed_range in cell_stats)
    decision = pilot_decision["decision"]
    if varying_cells != 15 or decision["selected_k"] != 3:
        raise ValueError("Pilot variance summary no longer matches the manuscript")

    ax.text(
        0.0,
        1.105,
        f"Temperature = 0; 10 calls per cell; {varying_cells}/20 cells vary",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=DARK,
    )
    ax.text(
        1.0,
        1.105,
        f"Q90(v) = {decision['q90_v']:.5f} -> K = {decision['selected_k']}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=MID_GRAY,
    )
    ax.set_xlim(-0.55, 9.55)
    ax.set_ylim(0.05, 0.62)
    ax.set_xticks(state_rows["x"], state_rows["label"], rotation=32, ha="right")
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_formatter(FuncFormatter(probability_tick))
    ax.set_xlabel("Pilot state (topic prefix - scenario number)")
    ax.set_ylabel("Reshare probability")
    ax.grid(axis="y", color=ROW_GRAY, linewidth=0.55, zorder=0)
    style_axis(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=WHITE,
            markeredgewidth=0.4,
            markersize=5.5,
            label=label,
        )
        for _condition, (_offset, color, label) in condition_style.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        frameon=False,
        ncol=2,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    return save_figure(fig, "figA1_pilot_repeatability.pdf")


def main() -> None:
    configure_matplotlib()
    primary, secondary, config, state_effects, pilot = load_data()
    pilot_decision = load_json(PILOT_DECISION_PATH)
    outputs = [
        make_figure_1(primary, secondary, config["practical_delta"]),
        make_figure_2(primary, state_effects, config["practical_delta"]),
        make_figure_a1(pilot, pilot_decision),
    ]
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
