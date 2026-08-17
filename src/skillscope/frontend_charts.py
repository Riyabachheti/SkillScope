"""Plotly figures for the SkillScope editorial frontend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from skillscope.taxonomy import ROLE_LABELS


ROLE_COLORS = {
    "Data Analyst / BI": "#A7B7AF",
    "Data Scientist": "#789487",
    "Data Engineer": "#496F5F",
    "ML / AI Engineer": "#244C3D",
}
INK = "#13271F"
MUTED = "#58655F"
GRID = "#C9D0CB"
PAPER = "#F3F1EB"
ACCENT = "#C45132"


def role_distribution_figure(frame: pd.DataFrame) -> go.Figure:
    data = frame.sort_values("share_pct", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=data["share_pct"],
            y=data["role_category"],
            orientation="h",
            marker_color=[ROLE_COLORS[role] for role in data["role_category"]],
            customdata=data[["posting_count"]],
            text=[f"{value:.1f}%" for value in data["share_pct"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{customdata[0]:,} postings<br>%{x:.2f}%<extra></extra>",
        )
    )
    return _style(figure, height=330, x_title="Share of retained postings (%)")


def skill_ranking_figure(
    frame: pd.DataFrame, role: str, limit: int, shared_max: float | None = None
) -> go.Figure:
    data = (
        frame.loc[frame["role_category"].eq(role)]
        .sort_values("skill_rank")
        .head(limit)
        .sort_values("penetration_pct")
    )
    figure = go.Figure(
        go.Bar(
            x=data["penetration_pct"],
            y=data["skill"],
            orientation="h",
            marker_color=ROLE_COLORS[role],
            customdata=data[["posting_count", "denominator"]],
            text=[f"{value:.1f}%" for value in data["penetration_pct"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>%{customdata[0]:,} of %{customdata[1]:,} postings"
                "<br>%{x:.2f}%<extra></extra>"
            ),
        )
    )
    figure = _style(
        figure,
        height=max(300, 52 * len(data)),
        x_title="Postings mentioning skill (%)",
    )
    if shared_max is not None:
        figure.update_xaxes(range=[0, shared_max])
    return figure


def simple_skill_figure(
    frame: pd.DataFrame, limit: int, color: str = "#496F5F"
) -> go.Figure:
    data = frame.sort_values("skill_rank").head(limit).sort_values("penetration_pct")
    figure = go.Figure(
        go.Bar(
            x=data["penetration_pct"],
            y=data["skill"],
            orientation="h",
            marker_color=color,
            customdata=data[["posting_count", "denominator"]],
            text=[f"{value:.1f}%" for value in data["penetration_pct"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>%{customdata[0]:,} of %{customdata[1]:,} postings"
                "<br>%{x:.2f}%<extra></extra>"
            ),
        )
    )
    return _style(
        figure,
        height=max(350, 48 * len(data)),
        x_title="Entry-level postings mentioning skill (%)",
    )


def skill_pair_figure(frame: pd.DataFrame, limit: int = 15) -> go.Figure:
    data = frame.sort_values("pair_rank").head(limit).copy().sort_values("pair_penetration_pct")
    data["pair"] = data["skill_a"] + " + " + data["skill_b"]
    short_skill = {
        "Machine Learning": "ML",
        "Artificial Intelligence": "AI",
        "Natural Language Processing": "NLP",
        "Data Engineering": "Data Eng.",
        "Apache Airflow": "Airflow",
        "Apache Hive": "Hive",
        "Apache Spark": "Spark",
    }
    data["pair_display"] = [
        f"{short_skill.get(skill_a, skill_a)} + {short_skill.get(skill_b, skill_b)}"
        for skill_a, skill_b in zip(data["skill_a"], data["skill_b"])
    ]
    figure = go.Figure(
        go.Scatter(
            x=data["pair_penetration_pct"],
            y=data["pair_display"],
            mode="markers+text",
            marker={
                "color": data["pair_posting_count"],
                "colorscale": [[0, "#C9D7D0"], [1, "#244C3D"]],
                "size": 12,
                "line": {"color": PAPER, "width": 1},
                "showscale": False,
            },
            text=[f"{rate:.1f}% · {lift:.2f}×" for rate, lift in zip(data["pair_penetration_pct"], data["lift"])],
            textposition="middle right",
            textfont={"size": 10, "color": INK},
            customdata=data[["pair", "pair_posting_count", "lift"]],
            hovertemplate=(
                "%{customdata[0]}<br>%{customdata[1]:,} postings"
                "<br>%{x:.2f}% of postings<br>Lift %{customdata[2]:.2f}<extra></extra>"
            ),
            cliponaxis=False,
        )
    )
    figure = _style(
        figure,
        height=max(430, 34 * len(data)),
        x_title="Postings containing the pair (%)",
    )
    figure.update_layout(margin={"l": 130, "r": 50, "t": 16, "b": 55})
    figure.update_xaxes(range=[0, float(data["pair_penetration_pct"].max()) * 1.45])
    return figure


def city_mix_figure(frame: pd.DataFrame, city: str) -> go.Figure:
    data = frame.loc[frame["primary_city"].eq(city)].copy()
    data["role_category"] = pd.Categorical(
        data["role_category"], categories=ROLE_LABELS, ordered=True
    )
    data = data.sort_values("role_category")
    figure = go.Figure()
    for row in data.itertuples(index=False):
        figure.add_trace(
            go.Bar(
                x=[row.within_city_pct],
                y=[city],
                orientation="h",
                name=row.role_category,
                marker_color=ROLE_COLORS[row.role_category],
                text=[f"{row.within_city_pct:.0f}%" if row.within_city_pct >= 10 else ""],
                textposition="inside",
                customdata=[[row.role_postings, row.city_postings]],
                hovertemplate=(
                    f"{row.role_category}<br>"
                    "%{customdata[0]:,} of %{customdata[1]:,} postings"
                    "<br>%{x:.2f}%<extra></extra>"
                ),
            )
        )
    figure.update_layout(barmode="stack")
    figure = _style(figure, height=260, x_title="Share within selected city (%)")
    figure.update_xaxes(range=[0, 100])
    figure.update_layout(
        legend={"orientation": "h", "y": -0.45, "x": 0, "title": None},
        margin={"l": 18, "r": 18, "t": 18, "b": 95},
    )
    return figure


def model_comparison_figure(models: Mapping[str, Mapping[str, float]]) -> go.Figure:
    order = [
        "majority_baseline",
        "skills_linear_svm",
        "skills_logistic_regression",
        "combined_linear_svm",
        "combined_logistic_regression",
    ]
    labels = {
        "majority_baseline": "Majority baseline",
        "skills_linear_svm": "Skills · SVM",
        "skills_logistic_regression": "Skills · Logistic",
        "combined_linear_svm": "Combined · SVM",
        "combined_logistic_regression": "Combined · Logistic",
    }
    values = [float(models[name]["macro_f1"]) for name in order]
    colors = ["#A9B1AC", "#8EA399", "#6F8C7F", "#496F5F", ACCENT]
    figure = go.Figure()
    for index, (name, value, color) in enumerate(zip(order, values, colors)):
        figure.add_shape(
            type="line",
            x0=0,
            x1=value,
            y0=index,
            y1=index,
            line={"color": GRID, "width": 2},
        )
        figure.add_trace(
            go.Scatter(
                x=[value],
                y=[labels[name]],
                mode="markers+text",
                marker={"size": 13, "color": color},
                text=[f"{value:.3f}"],
                textposition="middle right",
                hovertemplate=f"{labels[name]}<br>Macro F1 {value:.6f}<extra></extra>",
                showlegend=False,
                cliponaxis=False,
            )
        )
    figure = _style(figure, height=390, x_title="Balanced role score (macro F1)")
    figure.update_xaxes(range=[0, max(values) * 1.14])
    return figure


def confusion_matrix_figure(
    matrix: Sequence[Sequence[int]], labels: Sequence[str]
) -> go.Figure:
    short = {
        "Data Analyst / BI": "Analyst",
        "Data Scientist": "Scientist",
        "Data Engineer": "Engineer",
        "ML / AI Engineer": "ML / AI",
    }
    display_labels = [short[label] for label in labels]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=display_labels,
            y=display_labels,
            colorscale=[[0, "#E8ECE9"], [1, "#244C3D"]],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} postings<extra></extra>",
            showscale=False,
        )
    )
    figure = _style(
        figure,
        height=430,
        x_title="Predicted role",
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def _style(
    figure: go.Figure,
    *,
    height: int,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Avenir Next, Avenir, Helvetica Neue, Arial, sans-serif", "color": INK},
        hoverlabel={"bgcolor": INK, "font_color": "white", "bordercolor": INK},
        margin={"l": 105, "r": 50, "t": 16, "b": 55},
        showlegend=False,
    )
    figure.update_xaxes(
        title=x_title,
        automargin=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
    )
    figure.update_yaxes(
        title=y_title,
        automargin=True,
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        tickfont={"color": INK},
        title_font={"color": MUTED},
    )
    return figure
