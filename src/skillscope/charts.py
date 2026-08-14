"""Reproducible Phase 5 charts built from verified Phase 4 results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "skillscope-phase5"

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from skillscope.analysis import QUERY_FILES, query_name, validate_results
from skillscope.source import sha256_file


ROLE_ORDER = (
    "Data Analyst / BI",
    "Data Scientist",
    "Data Engineer",
    "ML / AI Engineer",
)
ROLE_COLORS = {
    "Data Analyst / BI": "#D97706",
    "Data Scientist": "#059669",
    "Data Engineer": "#2563EB",
    "ML / AI Engineer": "#7C3AED",
}
ACCENT = "#2563EB"
TEXT = "#172033"
MUTED = "#5B6475"
GRID = "#D9DEE8"
EXPECTED_CHARTS = (
    "role_distribution",
    "top_skills_by_role",
    "entry_level_skills",
    "city_role_mix",
    "skill_cooccurrence",
)


def load_verified_results(
    analysis_dir: Path,
    summary_path: Path,
    jobs_path: Path,
    skills_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load chart inputs only when CSVs, summary, and Parquet hashes agree."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_inputs = {
        "jobs_sha256": sha256_file(jobs_path),
        "job_skills_sha256": sha256_file(skills_path),
    }
    if summary.get("inputs") != actual_inputs:
        raise ValueError("Analysis summary is stale for the current processed tables")

    results: dict[str, pd.DataFrame] = {}
    for filename in QUERY_FILES:
        name = query_name(Path(filename))
        csv_path = analysis_dir / f"{name}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        results[name] = pd.read_csv(csv_path)

    errors = validate_results(results)
    if errors:
        raise ValueError("Invalid analysis inputs: " + "; ".join(errors))

    summary_results = summary.get("results", {})
    if set(summary_results) != set(results):
        raise ValueError("Analysis summary result set does not match the CSV files")
    for name, frame in results.items():
        expected = pd.DataFrame(summary_results[name], columns=frame.columns)
        try:
            pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
        except AssertionError as error:
            raise ValueError(f"{name}.csv does not match analysis_summary.json") from error
    return results


def create_charts(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, list[Path]]:
    """Create five focused charts as both PNG and SVG files."""
    errors = validate_results(results)
    if errors:
        raise ValueError("Invalid chart inputs: " + "; ".join(errors))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    figures = {
        "role_distribution": _role_distribution(results["role_distribution"]),
        "top_skills_by_role": _top_skills_by_role(results["top_skills_by_role"]),
        "entry_level_skills": _entry_level_skills(results["entry_level_skills"]),
        "city_role_mix": _city_role_mix(results["city_role_distribution"]),
        "skill_cooccurrence": _skill_cooccurrence(results["skill_cooccurrence"]),
    }
    outputs: dict[str, list[Path]] = {}
    for name, figure in figures.items():
        png_path = output_dir / f"{name}.png"
        svg_path = output_dir / f"{name}.svg"
        figure.savefig(
            png_path,
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "SkillScope"},
        )
        figure.savefig(
            svg_path,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Date": None, "Creator": "SkillScope"},
        )
        plt.close(figure)
        outputs[name] = [png_path, svg_path]
    return outputs


def _footer(figure: plt.Figure) -> None:
    figure.text(
        0.01,
        0.01,
        "Source: SkillScope India — postings represented in the verified dataset (n=2,158)",
        fontsize=8,
        color=MUTED,
    )


def _clean_axis(axis: plt.Axes, grid_axis: str = "x") -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)


def _role_distribution(frame: pd.DataFrame) -> plt.Figure:
    data = frame.sort_values("share_pct")
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.barh(
        data["role_category"],
        data["share_pct"],
        color=[ROLE_COLORS[role] for role in data["role_category"]],
        height=0.62,
    )
    labels = [
        f"{share:.2f}%  ({count:,})"
        for share, count in zip(data["share_pct"], data["posting_count"])
    ]
    axis.bar_label(bars, labels=labels, padding=6, fontsize=10, color=TEXT)
    axis.set_xlim(0, data["share_pct"].max() * 1.28)
    axis.set_xlabel("Share of retained postings (%)")
    axis.set_ylabel("Role family")
    axis.set_title(
        "Data engineering is the largest role family",
        loc="left",
        fontweight="bold",
        pad=28,
    )
    axis.text(
        0,
        1.01,
        "Distribution across four reviewed role categories",
        transform=axis.transAxes,
        color=MUTED,
    )
    _clean_axis(axis)
    _footer(figure)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    return figure


def _top_skills_by_role(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    max_value = float(frame.loc[frame["skill_rank"] <= 5, "penetration_pct"].max())
    for axis, role in zip(axes.flat, ROLE_ORDER):
        data = frame[(frame["role_category"] == role) & (frame["skill_rank"] <= 5)]
        data = data.sort_values("penetration_pct")
        bars = axis.barh(
            data["skill"],
            data["penetration_pct"],
            color=ROLE_COLORS[role],
            height=0.62,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in data["penetration_pct"]],
            padding=4,
            fontsize=9,
        )
        axis.set_xlim(0, max_value * 1.16)
        axis.set_title(role, loc="left", fontsize=12, fontweight="bold")
        axis.set_xlabel("Postings mentioning skill (%)")
        axis.set_ylabel("")
        _clean_axis(axis)
    figure.suptitle(
        "Skill demand changes meaningfully by role",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    _footer(figure)
    figure.tight_layout(rect=(0, 0.04, 1, 0.94), h_pad=2.2, w_pad=2.5)
    return figure


def _entry_level_skills(frame: pd.DataFrame) -> plt.Figure:
    data = frame.head(10).sort_values("penetration_pct")
    denominator = int(frame["denominator"].iloc[0])
    figure, axis = plt.subplots(figsize=(9, 6.3))
    bars = axis.barh(data["skill"], data["penetration_pct"], color=ACCENT, height=0.6)
    axis.bar_label(
        bars,
        labels=[f"{value:.1f}%" for value in data["penetration_pct"]],
        padding=5,
        fontsize=9,
    )
    axis.set_xlim(0, data["penetration_pct"].max() * 1.18)
    axis.set_xlabel("Entry-level postings mentioning skill (%)")
    axis.set_ylabel("Skill")
    axis.set_title(
        "Python leads entry-level skill demand",
        loc="left",
        fontweight="bold",
        pad=28,
    )
    axis.text(
        0,
        1.01,
        f"Minimum stated experience of 0–2 years; denominator = {denominator:,} postings",
        transform=axis.transAxes,
        color=MUTED,
    )
    _clean_axis(axis)
    _footer(figure)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    return figure


def _city_role_mix(frame: pd.DataFrame) -> plt.Figure:
    city_order = (
        frame[["primary_city", "city_postings"]]
        .drop_duplicates()
        .sort_values("city_postings", ascending=False)
        .head(7)["primary_city"]
        .tolist()
    )
    data = frame[frame["primary_city"].isin(city_order)]
    pivot = (
        data.pivot(index="primary_city", columns="role_category", values="within_city_pct")
        .reindex(city_order)
        .fillna(0)
    )
    figure, axis = plt.subplots(figsize=(11, 6.2))
    bottom = pd.Series(0.0, index=pivot.index)
    for role in ROLE_ORDER:
        values = pivot[role] if role in pivot else pd.Series(0.0, index=pivot.index)
        bars = axis.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=role,
            color=ROLE_COLORS[role],
            width=0.7,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 12:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
        bottom = bottom + values
    axis.set_ylim(0, 100)
    axis.set_xlabel("Primary city")
    axis.set_ylabel("Within-city role share (%)")
    axis.set_title("Data engineering leads in the three largest cities", loc="left", fontweight="bold")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    _footer(figure)
    figure.tight_layout(rect=(0, 0.05, 1, 0.9))
    return figure


def _skill_cooccurrence(frame: pd.DataFrame) -> plt.Figure:
    data = frame.head(10).copy()
    data["pair"] = data["skill_a"] + " + " + data["skill_b"]
    data = data.sort_values("pair_penetration_pct")
    figure, axis = plt.subplots(figsize=(10.5, 6.8))
    bars = axis.barh(data["pair"], data["pair_penetration_pct"], color="#0F766E", height=0.6)
    labels = [
        f"{penetration:.1f}%  |  lift {lift:.2f}×"
        for penetration, lift in zip(data["pair_penetration_pct"], data["lift"])
    ]
    axis.bar_label(bars, labels=labels, padding=5, fontsize=9)
    axis.set_xlim(0, data["pair_penetration_pct"].max() * 1.52)
    axis.set_xlabel("All retained postings containing both skills (%)")
    axis.set_ylabel("Skill pair")
    axis.set_title(
        "Python + SQL is the most frequent skill pair",
        loc="left",
        fontweight="bold",
        pad=28,
    )
    axis.text(
        0,
        1.01,
        "Lift above 1 indicates association, not causation",
        transform=axis.transAxes,
        color=MUTED,
    )
    _clean_axis(axis)
    _footer(figure)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    return figure
