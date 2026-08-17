"""Local single-page presentation layer for verified SkillScope findings."""

from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from skillscope.frontend_charts import (
    city_mix_figure,
    confusion_matrix_figure,
    model_comparison_figure,
    role_distribution_figure,
    simple_skill_figure,
    skill_pair_figure,
    skill_ranking_figure,
)
from skillscope.frontend_data import FrontendData, FrontendDataError, load_frontend_data
from skillscope.taxonomy import ROLE_LABELS


PROJECT_ROOT = Path(os.environ.get("SKILLSCOPE_PROJECT_ROOT", Path(__file__).parent))
PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
}


st.set_page_config(
    page_title="SkillScope India | Data career evidence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)


@st.cache_data(show_spinner=False)
def _load(root: str) -> FrontendData:
    return load_frontend_data(Path(root))


def _inject_styles() -> None:
    st.html(
        """
        <style>
        :root {
          --paper: #F3F1EB;
          --ink: #13271F;
          --muted: #58655F;
          --rule: #C9D0CB;
          --soft: #E3E8E3;
          --verified: #245F4B;
          --accent: #C45132;
          --mono: "SFMono-Regular", "Roboto Mono", Consolas, monospace;
          --sans: "Avenir Next", Avenir, "Helvetica Neue", Arial, sans-serif;
        }
        html, body, [data-testid="stAppViewContainer"] { font-family: var(--sans); }
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] { scroll-behavior: smooth !important; }
        .stMainBlockContainer { width: 94vw; max-width: 1800px; padding: 1.1rem 1.5rem 5rem; }
        .stElementContainer:has(iframe[title="st.iframe"]) {
          position: sticky;
          top: 0;
          z-index: 1000001;
          margin-bottom: 4.5rem;
        }
        iframe[title="st.iframe"] { display: block; background: var(--paper); }
        .section-anchor, .content-anchor { scroll-margin-top: 4.5rem; }
        .hero-intro { max-width: 920px; margin: 0 0 2.4rem; }
        .hero-kicker { color: var(--verified); font-family: var(--mono); font-size: .76rem; font-weight: 700; margin: 0 0 1rem; }
        .hero-title {
          max-width: 760px;
          color: var(--ink);
          font-family: var(--sans);
          font-size: clamp(3.6rem, 6vw, 5.6rem) !important;
          font-weight: 750;
          letter-spacing: -0.065em;
          line-height: 0.94 !important;
          margin: 0 0 1.65rem;
        }
        .hero-copy {
          max-width: 600px;
          color: var(--muted);
          font-size: 1.08rem;
          line-height: 1.7;
          margin: 0;
        }
        .verification-row {
          display: grid;
          grid-template-columns: auto minmax(180px, .7fr) minmax(320px, 1.3fr);
          gap: 1rem 2rem;
          align-items: center;
          border-top: 1px solid var(--rule);
          border-bottom: 1px solid var(--rule);
          margin: 1.2rem 0 .8rem;
          padding: 1rem 0;
        }
        .verified-stamp {
          display: inline-flex;
          align-items: center;
          gap: .45rem;
          border: 1px solid var(--verified);
          background: var(--verified);
          color: #F6F4EE;
          font-family: var(--mono);
          font-size: .7rem;
          font-weight: 800;
          letter-spacing: .025em;
          padding: .42rem .62rem;
        }
        .fingerprint { color: var(--muted); font-family: var(--mono); font-size: .72rem; overflow-wrap: anywhere; }
        .verification-row p { color: var(--muted); font-size: .9rem; line-height: 1.5; margin: 0; }
        .detail-copy { color: var(--muted); font-size: .88rem; line-height: 1.65; }
        .detail-copy code { font-family: var(--mono); font-size: .72rem; overflow-wrap: anywhere; }
        .scope-note { max-width: 780px; color: var(--muted); line-height: 1.65; margin: 1rem 0 0; font-size: .92rem; }
        .roles-intro {
          display: grid;
          grid-template-columns: minmax(0, 1.4fr) minmax(250px, .6fr);
          gap: 3rem;
          align-items: end;
          margin: 6rem 0 2rem;
          scroll-margin-top: 4.5rem;
        }
        .roles-intro h2, .places-intro h2, .method-head h2 { color: var(--ink); font-family: var(--sans); font-size: clamp(2.2rem, 4.5vw, 4rem) !important; font-weight: 750; letter-spacing: -.055em; line-height: 1 !important; margin: 0; }
        .roles-intro aside { border-left: 3px solid var(--accent); color: var(--muted); line-height: 1.55; padding-left: 1rem; }
        .roles-intro aside b { color: var(--ink); font-family: var(--sans); font-size: 1.05rem; }
        .compare-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin: 4.5rem 0 1rem; }
        .compare-head h3 { color: var(--ink); font-size: 1.55rem; letter-spacing: -.025em; margin: 0; }
        .compare-head span { color: var(--muted); font-family: var(--sans); font-size: .8rem; }
        .skills-banner {
          background: var(--ink);
          color: #F6F4EE;
          margin: 7.5rem 0 2rem;
          padding: clamp(1.5rem, 4vw, 3rem);
          scroll-margin-top: 4.5rem;
          display: grid;
          grid-template-columns: 1fr .75fr;
          gap: 3rem;
          align-items: end;
        }
        .skills-banner h2 { color: #F6F4EE; font-size: clamp(2.4rem, 5vw, 4.5rem) !important; font-weight: 720; letter-spacing: -.06em; line-height: .98 !important; margin: 0; }
        .skills-banner p { color: #B9C4BE; line-height: 1.65; margin: 0; }
        .places-intro { margin: 7.5rem 0 2rem; scroll-margin-top: 4.5rem; }
        .places-intro p { color: var(--muted); max-width: 650px; line-height: 1.65; }
        .model-proof {
          background: #DFE6E1;
          border-top: 8px solid var(--verified);
          margin: 7.5rem 0 2rem;
          padding: clamp(1.5rem, 4vw, 2.6rem);
          scroll-margin-top: 4.5rem;
        }
        .model-proof h2 { color: var(--ink); font-size: clamp(2.2rem, 4.4vw, 3.8rem) !important; font-weight: 750; letter-spacing: -.055em; line-height: 1 !important; margin: 1.2rem 0 .8rem; }
        .model-proof > p { color: var(--muted); max-width: 750px; line-height: 1.65; }
        .figure-label {
          color: var(--ink);
          font-size: 1.08rem;
          font-weight: 750;
          letter-spacing: -.02em;
          margin: 2.2rem 0 .2rem;
        }
        .comparison-note { background: var(--soft); color: var(--muted); line-height: 1.65; margin-top: 1rem; padding: 1rem 1.2rem; }
        .comparison-note strong { color: var(--ink); }
        .model-index {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 1.25rem;
          margin: 1.5rem 0 0;
        }
        .model-index div { border-top: 1px solid #A9B8B0; padding-top: .75rem; }
        .model-index small { color: var(--muted); display: block; font-family: var(--mono); font-size: .65rem; }
        .model-index strong { color: var(--ink); font-family: var(--mono); font-size: 1.65rem; font-variant-numeric: tabular-nums; }
        .model-index em { color: var(--muted); display: block; font-size: .72rem; font-style: normal; line-height: 1.45; margin-top: .35rem; }
        .method-head { margin: 8rem 0 2rem; scroll-margin-top: 4.5rem; }
        .method-head p { color: var(--muted); max-width: 720px; line-height: 1.65; }
        .pipeline {
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          margin: 2rem 0 3.5rem;
        }
        .pipeline div { min-height: 118px; background: var(--soft); border-top: 4px solid var(--verified); padding: 1rem .85rem; }
        .pipeline div + div { border-left: 4px solid var(--paper); }
        .pipeline span { color: var(--accent); display: block; font-family: var(--mono); font-size: .72rem; font-weight: 800; margin-bottom: .65rem; }
        .pipeline strong { color: var(--ink); display: block; font-size: .96rem; line-height: 1.35; }
        .footer-rule { border-top: 2px solid var(--ink); margin-top: 5rem; padding-top: 1rem; color: var(--muted); font-family: var(--mono); font-size: .75rem; line-height: 1.6; }
        div[data-baseweb="select"] > div {
          background: #FAF9F5 !important;
          border: 1.5px solid var(--ink) !important;
          border-radius: 0 !important;
          box-shadow: 4px 4px 0 #D7DDD8;
          min-height: 48px;
        }
        [data-testid="stSelectbox"] label p { color: var(--muted); font-family: var(--mono); font-size: .7rem; }
        div[data-testid="stExpander"] { border: 1.5px solid var(--ink); border-radius: 0; background: #FAF9F5; }
        [data-testid="stCaptionContainer"] { font-family: var(--sans); color: var(--muted); }
        @media (max-width: 760px) {
          .stMainBlockContainer { padding-left: 1rem; padding-right: 1rem; }
          .stElementContainer:has(iframe[title="st.iframe"]) { margin-bottom: 2.4rem; }
          .verification-row { grid-template-columns: 1fr; gap: .7rem; }
          .roles-intro, .skills-banner { grid-template-columns: 1fr; gap: 1.5rem; }
          .hero-title { font-size: clamp(3.2rem, 14vw, 4.3rem) !important; }
          .roles-intro, .skills-banner, .places-intro, .model-proof { margin-top: 5.5rem; }
          .model-index { grid-template-columns: repeat(2, 1fr); }
          .pipeline { grid-template-columns: repeat(2, 1fr); }
          .pipeline div:nth-child(odd) { border-left: 0; }
          .pipeline div:nth-child(n+3) { border-top-color: var(--paper); }
        }
        </style>
        """
    )


def _navigation() -> None:
    st.iframe(
        """
        <style>
          * { box-sizing: border-box; }
          html, body { margin: 0; background: #F3F1EB; font-family: "Avenir Next", Avenir, "Helvetica Neue", Arial, sans-serif; }
          nav { display: flex; align-items: center; gap: 1.25rem; height: 58px; overflow-x: auto; border-bottom: 2px solid #13271F; white-space: nowrap; }
          .brand { color: #13271F; font-family: "SFMono-Regular", "Roboto Mono", Consolas, monospace; font-size: .84rem; font-weight: 750; margin-right: auto; letter-spacing: -.025em; }
          a { color: #58655F; font-size: .8rem; font-weight: 650; text-decoration: none; }
          a:hover, a:focus-visible { color: #C45132; outline: 2px solid #C45132; outline-offset: 4px; }
          @media (max-width: 760px) { nav { gap: .65rem; } .brand { display: none; } a { font-size: .69rem; } }
        </style>
        <nav aria-label="Page sections">
          <span class="brand">SkillScope India</span>
          <a href="#snapshot">Snapshot</a><a href="#roles">Roles</a>
          <a href="#skills">Skills</a><a href="#places">Places</a>
          <a href="#model">Model</a><a href="#method">Method</a>
        </nav>
        <script>
          const host = window.parent;
          document.querySelectorAll('a[href^="#"]').forEach((link) => {
            link.addEventListener("click", (event) => {
              event.preventDefault();
              const scroller = host.document.querySelector('[data-testid="stMain"]');
              const target = host.document.querySelector(link.getAttribute("href"));
              if (!scroller || !target) return;
              const start = scroller.scrollTop;
              const offset = target.getBoundingClientRect().top
                - scroller.getBoundingClientRect().top + start - 72;
              const distance = offset - start;
              const startedAt = performance.now();
              const animate = (now) => {
                const elapsed = Math.min((now - startedAt) / 520, 1);
                scroller.scrollTop = start + distance * (1 - Math.pow(1 - elapsed, 3));
                if (elapsed < 1) {
                  requestAnimationFrame(animate);
                } else {
                  host.history.replaceState(null, "", link.getAttribute("href"));
                }
              };
              requestAnimationFrame(animate);
            });
          });
        </script>
        """,
        height=58,
        width="stretch",
    )


def _anchor(anchor: str) -> None:
    st.markdown(
        f'<div id="{html.escape(anchor)}" class="content-anchor"></div>',
        unsafe_allow_html=True,
    )


def _figure_label(text: str) -> None:
    st.markdown(
        f'<p class="figure-label">{html.escape(text)}</p>', unsafe_allow_html=True
    )


def _plot(figure, key: str) -> None:
    st.plotly_chart(
        figure,
        key=key,
        width="stretch",
        theme=None,
        config=PLOT_CONFIG,
    )


def _comparison_copy(frame: pd.DataFrame, role_a: str, role_b: str) -> str:
    top = {
        role: frame.loc[frame["role_category"].eq(role)]
        .sort_values("skill_rank")
        .head(5)["skill"]
        .tolist()
        for role in (role_a, role_b)
    }
    overlap = [skill for skill in top[role_a] if skill in top[role_b]]
    overlap_text = ", ".join(overlap) if overlap else "no shared top-five skills"
    return (
        f"<strong>{html.escape(role_a)}</strong> is led by {html.escape(top[role_a][0])}; "
        f"<strong>{html.escape(role_b)}</strong> is led by {html.escape(top[role_b][0])}. "
        f"Their shared top-five signals are {html.escape(overlap_text)}. "
        "These percentages describe mentions in retained postings, not prerequisites for every job."
    )


def _render_snapshot(data: FrontendData) -> None:
    jobs_hash = str(data.transformation_metrics["jobs_sha256"])
    skills_hash = str(data.transformation_metrics["job_skills_sha256"])
    short_jobs_hash = f"{jobs_hash[:8]}…{jobs_hash[-6:]}"
    st.markdown(
        f"""
        <div id="snapshot" class="section-anchor"></div>
        <header class="hero-intro" aria-labelledby="snapshot-title">
          <p class="hero-kicker">SKILLSCOPE / CHECKED CAREER EVIDENCE</p>
          <h1 class="hero-title" id="snapshot-title">Skill demand,<br>with receipts.</h1>
          <p class="hero-copy">
            Explore what {data.posting_count:,} Indian data-job postings ask for across four
            career paths. Every result is checked against the same prepared source file.
          </p>
        </header>
        """,
        unsafe_allow_html=True,
    )
    _figure_label(f"How the {data.posting_count:,} retained postings divide by role")
    _plot(role_distribution_figure(data.results["role_distribution"]), "snapshot-role-mix")
    st.caption(
        "Data Engineer is the largest group in this dataset. The chart describes these retained postings, not the whole Indian job market."
    )
    st.markdown(
        f"""
        <div class="verification-row" aria-label="Verified source fingerprint">
          <span class="verified-stamp">✓ SOURCE FILE MATCHED</span>
          <span class="fingerprint" title="{html.escape(jobs_hash)}">jobs · {html.escape(short_jobs_hash)}</span>
          <p>This exact prepared file, unchanged, is what every number on this page comes from.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("For the curious: what was checked"):
        st.markdown(
            f"""
            <div class="detail-copy">
              SkillScope matched the prepared jobs and job-skill files, {len(data.results)} saved
              analyses, and the selected model before showing the page. It found
              <strong>{data.posting_count:,} postings</strong> and
              <strong>{data.job_skill_pairs:,} posting-skill relationships</strong>.<br><br>
              Jobs fingerprint: <code>{html.escape(jobs_hash)}</code><br>
              Job-skills fingerprint: <code>{html.escape(skills_hash)}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_roles(data: FrontendData) -> None:
    _anchor("roles")
    st.markdown(
        """
        <section class="roles-intro" aria-labelledby="roles-title">
          <h2 id="roles-title">Compare roles side by side.</h2>
          <aside><b>What is a role group?</b><br>A reviewed umbrella label that brings similar job titles together for comparison.</aside>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="compare-head"><h3>See which skills separate them</h3><span>share of each role\'s postings that mention the skill</span></div>',
        unsafe_allow_html=True,
    )
    control_a, control_b, control_limit = st.columns([1, 1, 0.65])
    with control_a:
        role_a = st.selectbox("First role", ROLE_LABELS, index=0, key="compare-role-a")
    with control_b:
        role_b = st.selectbox("Second role", ROLE_LABELS, index=2, key="compare-role-b")
    with control_limit:
        limit = st.selectbox("Skills shown", (5, 8, 10), index=0, key="compare-limit")

    role_skills = data.results["top_skills_by_role"]
    shared_max = float(
        role_skills.loc[
            role_skills["role_category"].isin((role_a, role_b))
            & role_skills["skill_rank"].le(limit),
            "penetration_pct",
        ].max()
        * 1.2
    )
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown(f"#### {role_a}")
        _plot(
            skill_ranking_figure(role_skills, role_a, limit, shared_max),
            "role-a-skills",
        )
    with chart_b:
        st.markdown(f"#### {role_b}")
        _plot(
            skill_ranking_figure(role_skills, role_b, limit, shared_max),
            "role-b-skills",
        )
    st.markdown(
        '<p class="comparison-note">'
        + _comparison_copy(role_skills, role_a, role_b)
        + "</p>",
        unsafe_allow_html=True,
    )


def _render_skills(data: FrontendData) -> None:
    _anchor("skills")
    st.markdown(
        """
        <section class="skills-banner" aria-labelledby="skills-title">
          <h2 id="skills-title">Which skills appear most often?</h2>
          <p>Choose a role, then compare its most-mentioned skills with entry-level postings and skills that commonly appear together.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.72, 0.28])
    with left:
        selected_role = st.selectbox(
            "Role to inspect", ROLE_LABELS, index=2, key="skills-role"
        )
    with right:
        selected_limit = st.selectbox(
            "Ranking depth", (5, 8, 10), index=1, key="skills-limit"
        )
    _figure_label(f"{selected_role}: most-mentioned skills")
    _plot(
        skill_ranking_figure(
            data.results["top_skills_by_role"], selected_role, selected_limit
        ),
        "selected-role-skills",
    )

    entry = data.results["entry_level_skills"]
    denominator = int(entry["denominator"].iloc[0])
    entry_col, pair_col = st.columns([0.48, 0.52])
    with entry_col:
        _figure_label("Entry-level evidence")
        st.markdown(f"Minimum stated experience: **0–2 years** · **n={denominator:,}**")
        _plot(simple_skill_figure(entry, 10), "entry-level-skills")
        st.caption("Entry level means the posting states a minimum of 0–2 years of experience. It does not guarantee that every applicant qualifies.")
    with pair_col:
        _figure_label("Skills that travel together")
        _plot(skill_pair_figure(data.results["skill_cooccurrence"], 15), "skill-pairs")
        st.caption("Lift compares how often two skills appear together with how often we would expect by chance. A value above 1 means the pair appears together more often than expected; it does not prove that one skill causes the other.")


def _render_places(data: FrontendData) -> None:
    _anchor("places")
    st.markdown(
        """
        <section class="places-intro" aria-labelledby="places-title">
          <h2 id="places-title">How does the role mix change by city?</h2>
          <p>Choose a city to see how its represented postings divide across the four reviewed role groups.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    frame = data.results["city_role_distribution"]
    cities = (
        frame[["primary_city", "city_postings"]]
        .drop_duplicates()
        .sort_values(["city_postings", "primary_city"], ascending=[False, True])
    )
    city_control, city_chart = st.columns([0.32, 0.68])
    with city_control:
        city = st.selectbox("Choose city", cities["primary_city"].tolist(), key="city")
        city_total = int(cities.loc[cities["primary_city"].eq(city), "city_postings"].iloc[0])
        st.markdown(f"### `{city_total:,}` postings")
        st.markdown(f"list **{city}** as their first physical city after names were standardized.")
    with city_chart:
        _figure_label(f"Role composition inside {city}")
        _plot(city_mix_figure(frame, city), "city-mix")
    st.caption(
        "A posting with several locations is counted under its first listed physical city, so it is not duplicated across every city."
    )


def _render_model(data: FrontendData) -> None:
    _anchor("model")
    selected = data.selected_model
    st.markdown(
        f"""
        <section class="model-proof" aria-labelledby="model-title">
          <span class="verified-stamp">✓ MODEL CHECK PASSED</span>
          <h2 id="model-title">Can the four role groups be told apart?</h2>
          <p>The model uses skills and cleaned responsibility text to recover the reviewed role group. Each test uses postings from companies the model did not train on.</p>
          <div class="model-index" aria-label="Selected model results">
            <div><small>BALANCED ROLE SCORE</small><strong>{float(selected['macro_f1']):.3f}</strong><em>0–1 score that gives all four roles equal weight on held-out postings.</em></div>
            <div><small>OVERALL ACCURACY</small><strong>{float(selected['accuracy']):.3f}</strong><em>Share of held-out postings assigned to the reviewed role group.</em></div>
            <div><small>TEST ROUNDS</small><strong>{int(data.ml_metrics['experiment']['folds'])}</strong><em>Five company-separated train-and-test rounds.</em></div>
            <div><small>COMPANIES</small><strong>{int(data.ml_metrics['experiment']['companies']):,}</strong><em>Distinct employers represented in the experiment.</em></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    model_left, model_right = st.columns([1.05, 0.95])
    with model_left:
        _figure_label("How well each approach balanced all four roles")
        _plot(model_comparison_figure(data.ml_metrics["models"]), "model-comparison")
    with model_right:
        _figure_label("Where the selected model still confuses roles")
        _plot(
            confusion_matrix_figure(selected["confusion_matrix"], ROLE_LABELS),
            "confusion-matrix",
        )
    st.caption("Confusion matrix rows are actual reviewed roles; columns are model predictions.")
    st.markdown(
        "The largest remaining confusion is between **Data Scientist** and **ML / AI Engineer**. "
        "That overlap is evidence about the represented posting language, not proof that the roles are universally interchangeable."
    )
    st.markdown(
        """
        <p class="scope-note">
          Job titles created the reviewed labels, so the model never sees titles as clues. To avoid
          an unrealistically easy test, every posting from the same company—and every repeated
          description template—stays on one side of each train/test split. This prevents leakage:
          familiar employer wording cannot appear in both training and testing. The experiment is
          evidence about these role groups, not a tool for recruitment or personal career assignment.
        </p>
        """,
        unsafe_allow_html=True,
    )


def _render_method(data: FrontendData) -> None:
    _anchor("method")
    method_hash = str(data.ml_metrics["inputs"]["jobs_sha256"])
    short_method_hash = f"{method_hash[:8]}…{method_hash[-6:]}"
    st.markdown(
        f"""
        <section class="method-head" aria-labelledby="method-title">
          <span class="verified-stamp">✓ TRACEABLE END TO END</span>
          <h2 id="method-title">How every result was checked.</h2>
          <p>This is the only sequence that needs numbers: each stage must pass before the next one can use its output.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    steps = (
        ("01", "Source audit"),
        ("02", "Reviewed taxonomy"),
        ("03", "Clean tables"),
        ("04", "Verified SQL"),
        ("05", "Static charts"),
        ("06", "Grouped ML"),
    )
    pipeline = "".join(
        f"<div><span>{number}</span><strong>{html.escape(label)}</strong></div>"
        for number, label in steps
    )
    st.markdown(
        f'<div class="pipeline" aria-label="Project evidence pipeline">{pipeline}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="verification-row" aria-label="Matching input fingerprint">
          <span class="verified-stamp">✓ SAME INPUT AT BOTH ENDS</span>
          <span class="fingerprint" title="{html.escape(method_hash)}">jobs · {html.escape(short_method_hash)}</span>
          <p>Same file. Same fingerprint. It still matches after the full source-to-interface trail.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    terms_left, terms_right = st.columns(2)
    with terms_left:
        st.markdown(
            """
            #### Quick definitions

            **Skill penetration**  
            Postings mentioning a skill divided by every posting in the relevant role or cohort.

            **Entry level**  
            Minimum stated experience of 0–2 years; it is not a guarantee that a student qualifies.

            **Co-occurrence**  
            Two normalized skills appear in the same posting, counted once per posting.
            """
        )
    with terms_right:
        st.markdown(
            """
            #### Project terms

            **Lift**  
            Observed pair rate divided by its expected rate under independence.

            **Derived role family**  
            A reviewed umbrella label created from messy employer titles, not a source-provided field.

            **Primary city**  
            The first listed physical location after deterministic normalization.
            """
        )

    with st.expander("Source credit and project scope"):
        st.markdown(
            """
            Data source: Shivam Shrivastava's
            [Indian Job Market Dataset 2025](https://www.kaggle.com/datasets/shivamshrivastava21/indian-job-market-dataset-2025-2026),
            accessed through Kaggle. SkillScope applies its own reviewed role labels, cleaning,
            deduplication, summaries, charts, and model evaluation. This personal project is
            presented locally and does not redistribute individual source records.
            """
        )

    st.markdown(
        f"""
        <footer class="footer-rule">
          SkillScope India · Verified local presentation · {data.posting_count:,} retained postings<br>
          Evidence for comparison, not a universal curriculum or labour-market census.
        </footer>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    _navigation()
    try:
        data = _load(str(PROJECT_ROOT.resolve()))
    except FrontendDataError as error:
        st.error("SkillScope cannot verify its local presentation inputs.")
        st.code(str(error), language=None)
        st.markdown(
            "Run the existing Phase 3–6 build and validation commands from the project root, then refresh this page. No demonstration data has been substituted."
        )
        st.stop()

    _render_snapshot(data)
    _render_roles(data)
    _render_skills(data)
    _render_places(data)
    _render_model(data)
    _render_method(data)


if __name__ == "__main__":
    main()
