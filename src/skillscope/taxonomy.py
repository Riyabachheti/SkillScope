"""Auditable title rules for SkillScope role categories."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


DATA_ANALYST_BI = "Data Analyst / BI"
DATA_SCIENTIST = "Data Scientist"
DATA_ENGINEER = "Data Engineer"
ML_AI_ENGINEER = "ML / AI Engineer"

ROLE_LABELS = (
    DATA_ANALYST_BI,
    DATA_SCIENTIST,
    DATA_ENGINEER,
    ML_AI_ENGINEER,
)

# These six exact titles were changed by the completed Phase 2 human review.
# Exact keys avoid making the general regex rules broader for one-off wording.
REVIEWED_TITLE_OVERRIDES = {
    "ai and data scientist engineer": ML_AI_ENGINEER,
    "sfdc data cloud developer": DATA_ENGINEER,
    "vis bi strategy practitioner": DATA_ANALYST_BI,
    "python machine learning pan india": ML_AI_ENGINEER,
    "sr ai ml gen ai agentic ai professional": ML_AI_ENGINEER,
    "bi reports developer senior": DATA_ANALYST_BI,
}

# Patterns operate on normalized lowercase titles. They intentionally favor
# precision over recall because these labels later become the ML target.
ROLE_PATTERNS = {
    DATA_ANALYST_BI: (
        r"\bdata analysts?\b",
        r"\bbi analysts?\b",
        r"\bbusiness intelligence analysts?\b",
        r"\breporting analysts?\b",
        r"\banalytics analysts?\b",
        r"\banalytics interns?\b",
        r"\bbi engineers?\b",
        r"\banalytics technical specialists?\b",
        r"\bdata ai(?: [a-z0-9+#]+){0,3} analysts?\b",
    ),
    DATA_SCIENTIST: (
        r"\bdata scientists?\b",
        r"\bdecision scientists?\b",
        r"\bapplied scientists?\b",
        r"\b(?:lead|head|manager|director)\s+(?:of\s+)?data science\b",
        r"\bdata science\s+(?:lead|head|manager|director)\b",
    ),
    DATA_ENGINEER: (
        r"\bdata engineers?\b",
        r"\bbig data engineers?\b",
        r"\bdata platform engineers?\b",
        r"\betl developers?\b",
    ),
    ML_AI_ENGINEER: (
        r"\bmachine learning engineers?\b",
        r"\bml engineers?\b",
        r"\bai engineers?\b",
        r"\bartificial intelligence engineers?\b",
        r"\bai solutions engineers?\b",
        r"\bdata science engineers?\b",
    ),
}

ROLE_EXCLUSION_PATTERNS = {
    DATA_ANALYST_BI: (
        r"\baccounting\s+(?:and\s+)?reporting analysts?\b",
        r"\bcapital reporting analysts?\b",
        r"\bregulatory reporting analysts?\b",
    ),
    DATA_SCIENTIST: (r"\binstructors?\b",),
    DATA_ENGINEER: (),
    ML_AI_ENGINEER: (),
}

NEAR_MISS_PATTERN = re.compile(
    r"\b(?:data|analytics|business intelligence|machine learning|artificial intelligence|ai|ml|bi|etl)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RoleMatch:
    """Result of applying all title rules without hiding collisions."""

    normalized_title: str
    matched_roles: tuple[str, ...]
    predicted_role: str | None
    stratum: str


def normalize_title(title: object) -> str:
    """Normalize case, Unicode, punctuation, and whitespace for rule matching."""
    if title is None:
        return ""
    text = unicodedata.normalize("NFKC", str(title)).casefold()
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_title(title: object) -> RoleMatch:
    """Return every matching role and a prediction only when it is unambiguous."""
    normalized = normalize_title(title)
    matches = tuple(
        role
        for role in ROLE_LABELS
        if any(re.search(pattern, normalized) for pattern in ROLE_PATTERNS[role])
        and not any(re.search(pattern, normalized) for pattern in ROLE_EXCLUSION_PATTERNS[role])
    )
    if len(matches) == 1:
        return RoleMatch(normalized, matches, matches[0], "candidate")
    if len(matches) > 1:
        return RoleMatch(normalized, matches, None, "ambiguous")
    if NEAR_MISS_PATTERN.search(normalized):
        return RoleMatch(normalized, matches, None, "near_miss")
    return RoleMatch(normalized, matches, None, "unmatched")


def apply_reviewed_title_override(base: RoleMatch) -> RoleMatch:
    """Apply an exact reviewed correction to an existing broad-rule result."""
    reviewed_role = REVIEWED_TITLE_OVERRIDES.get(base.normalized_title)
    if reviewed_role is None or base.stratum == "ambiguous":
        return base
    return RoleMatch(
        normalized_title=base.normalized_title,
        matched_roles=(reviewed_role,),
        predicted_role=reviewed_role,
        stratum="reviewed_override",
    )


def match_title_with_review(title: object) -> RoleMatch:
    """Run broad title rules, then apply an exact reviewed correction."""
    return apply_reviewed_title_override(match_title(title))
