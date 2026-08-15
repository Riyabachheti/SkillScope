"""Leakage-aware role-classification experiment for Phase 6."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping

import pandas as pd

from skillscope.taxonomy import ROLE_LABELS


RANDOM_STATE = 42
N_SPLITS = 5
FEATURE_SETS = ("skills", "combined")
LEARNED_CONFIGURATIONS = (
    ("skills_logistic_regression", "skills", "logistic_regression"),
    ("skills_linear_svm", "skills", "linear_svm"),
    ("combined_logistic_regression", "combined", "logistic_regression"),
    ("combined_linear_svm", "combined", "linear_svm"),
)
MODEL_NAMES = ("majority_baseline",) + tuple(name for name, _, _ in LEARNED_CONFIGURATIONS)
PREDICTION_COLUMNS = {
    name: f"predicted_{name}" for name in MODEL_NAMES
}
FEATURE_SOURCE_COLUMNS = ("skill_key", "description_clean")
EXCLUDED_LEAKAGE_COLUMNS = (
    "title_original",
    "title_normalized",
    "job_id",
    "company_id",
    "company_name",
    "company_key",
    "template_group_id",
    "primary_city",
    "location_original",
    "experience_band",
    "minimum_experience",
    "maximum_experience",
    "salary_original",
    "minimum_salary",
    "maximum_salary",
    "aggregate_rating",
    "reviews_count",
)

# These compound phrases mirror the reviewed title taxonomy. They are removed
# globally from model-only descriptions so the secondary experiment cannot win
# merely by reading an explicit role name. Broad words such as "engineer" are
# deliberately retained because they also occur in legitimate general context.
ROLE_PROXY_PATTERNS = (
    r"\bbusiness intelligence analysts?\b",
    r"\bartificial intelligence engineers?\b",
    r"\bmachine learning engineers?\b",
    r"\bdata platform engineers?\b",
    r"\bbig data engineers?\b",
    r"\bdata science engineers?\b",
    r"\bdecision scientists?\b",
    r"\bapplied scientists?\b",
    r"\bdata scientists?\b",
    r"\bdata engineers?\b",
    r"\bdata analysts?\b",
    r"\betl developers?\b",
    r"\bbi analysts?\b",
    r"\bml engineers?\b",
    r"\bai engineers?\b",
)
ROLE_PROXY_PATTERN = re.compile(
    "|".join(f"(?:{pattern})" for pattern in ROLE_PROXY_PATTERNS),
    flags=re.IGNORECASE,
)

SKILL_VECTORIZER_CONFIG = {"sparse": True, "binary": True, "sort": True}
DESCRIPTION_VECTORIZER_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 30_000,
    "strip_accents": "unicode",
    "sublinear_tf": True,
}


def sanitize_description(value: object) -> str:
    """Remove audited compound role phrases without modifying processed data."""
    text = "" if value is None else str(value)
    text = ROLE_PROXY_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def role_proxy_match_count(value: object) -> int:
    """Count explicit role-name occurrences removed from one description."""
    text = "" if value is None else str(value)
    return len(ROLE_PROXY_PATTERN.findall(text))


def build_model_frame(jobs: pd.DataFrame, job_skills: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs and evaluation groups without exposing title fields."""
    required_jobs = {
        "job_id",
        "role_category",
        "company_key",
        "template_group_id",
        "description_clean",
    }
    required_skills = {"job_id", "skill_key"}
    missing_jobs = sorted(required_jobs - set(jobs.columns))
    missing_skills = sorted(required_skills - set(job_skills.columns))
    if missing_jobs or missing_skills:
        raise ValueError(
            f"missing ML inputs: jobs={missing_jobs}, job_skills={missing_skills}"
        )
    if jobs["job_id"].isna().any() or jobs["job_id"].duplicated().any():
        raise ValueError("ML input requires one non-null jobs row per job_id")
    if job_skills[["job_id", "skill_key"]].duplicated().any():
        raise ValueError("ML input contains duplicate job_id/skill_key pairs")
    if jobs["company_key"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("company_key must be non-empty for grouped evaluation")
    if jobs["template_group_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("template_group_id must be non-empty for grouped evaluation")
    unknown_roles = set(jobs["role_category"].dropna()) - set(ROLE_LABELS)
    if unknown_roles or jobs["role_category"].isna().any():
        raise ValueError(f"unknown or missing role labels: {sorted(unknown_roles)}")
    orphan_skills = set(job_skills["job_id"]) - set(jobs["job_id"])
    if orphan_skills:
        raise ValueError(f"ML input contains {len(orphan_skills)} orphan skill job IDs")

    skills_by_job = (
        job_skills.assign(skill_key=job_skills["skill_key"].astype(str).str.strip())
        .loc[lambda frame: frame["skill_key"].ne("")]
        .groupby("job_id", sort=False)["skill_key"]
        .agg(lambda values: tuple(sorted(set(values))))
    )
    frame = jobs[
        ["job_id", "role_category", "company_key", "template_group_id", "description_clean"]
    ].copy()
    frame["skills"] = frame["job_id"].map(skills_by_job).map(
        lambda value: value if isinstance(value, tuple) else ()
    )
    frame["role_proxy_matches"] = frame["description_clean"].map(role_proxy_match_count)
    frame["description_model"] = frame["description_clean"].map(sanitize_description)
    frame = frame.rename(
        columns={
            "role_category": "target",
            "company_key": "group_company",
            "template_group_id": "group_template",
        }
    )
    return frame[
        [
            "job_id",
            "target",
            "group_company",
            "group_template",
            "skills",
            "description_model",
            "role_proxy_matches",
        ]
    ].sort_values("job_id", kind="stable").reset_index(drop=True)


def make_company_grouped_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[object, object]]:
    """Create deterministic stratified folds with no company or template overlap."""
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as error:  # pragma: no cover - exercised by command boundary
        raise RuntimeError(
            "Phase 6 requires scikit-learn; install the pinned requirements first"
        ) from error

    template_company_counts = frame.groupby("group_template")["group_company"].nunique()
    if (template_company_counts > 1).any():
        raise ValueError("a template group maps to more than one company group")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = list(
        splitter.split(
            frame["job_id"],
            frame["target"],
            groups=frame["group_company"],
        )
    )
    expected_roles = set(ROLE_LABELS)
    for fold_number, (train_index, test_index) in enumerate(folds, start=1):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        if set(train["group_company"]) & set(test["group_company"]):
            raise ValueError(f"company leakage detected in fold {fold_number}")
        if set(train["group_template"]) & set(test["group_template"]):
            raise ValueError(f"template leakage detected in fold {fold_number}")
        if set(train["target"]) != expected_roles or set(test["target"]) != expected_roles:
            raise ValueError(f"fold {fold_number} does not contain all four roles")
    return folds


def _skill_records(frame: pd.DataFrame) -> list[dict[str, float]]:
    return [{skill: 1.0 for skill in skills} for skills in frame["skills"]]


def fit_feature_transformers(
    train: pd.DataFrame,
    feature_set: str,
) -> dict[str, object]:
    """Fit sparse feature transformers on training rows only."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set}")
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as error:  # pragma: no cover - exercised by command boundary
        raise RuntimeError(
            "Phase 6 requires scikit-learn; install the pinned requirements first"
        ) from error

    skill_vectorizer = DictVectorizer(sparse=True, sort=True)
    skill_vectorizer.fit(_skill_records(train))
    description_vectorizer = None
    if feature_set == "combined":
        description_vectorizer = TfidfVectorizer(**DESCRIPTION_VECTORIZER_CONFIG)
        description_vectorizer.fit(train["description_model"])
    return {
        "feature_set": feature_set,
        "skill_vectorizer": skill_vectorizer,
        "description_vectorizer": description_vectorizer,
    }


def transform_features(frame: pd.DataFrame, transformers: Mapping[str, object]) -> object:
    """Apply already-fitted sparse transformers without learning from these rows."""
    from scipy import sparse

    skill_matrix = transformers["skill_vectorizer"].transform(_skill_records(frame))
    if transformers["feature_set"] == "skills":
        return skill_matrix.tocsr()
    description_matrix = transformers["description_vectorizer"].transform(
        frame["description_model"]
    )
    return sparse.hstack([skill_matrix, description_matrix], format="csr")


def _make_classifier(classifier_name: str, random_state: int) -> object:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    if classifier_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=2_000,
            random_state=random_state,
        )
    if classifier_name == "linear_svm":
        return LinearSVC(
            class_weight="balanced",
            C=1.0,
            random_state=random_state,
        )
    raise ValueError(f"unknown classifier: {classifier_name}")


def _majority_label(target: pd.Series) -> str:
    counts = target.value_counts().rename_axis("label").reset_index(name="count")
    return str(
        counts.sort_values(["count", "label"], ascending=[False, True]).iloc[0]["label"]
    )


def metric_summary(actual: pd.Series, predicted: pd.Series) -> dict[str, object]:
    """Return deterministic aggregate and per-role classification metrics."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    report = classification_report(
        actual,
        predicted,
        labels=list(ROLE_LABELS),
        output_dict=True,
        zero_division=0,
    )
    per_role = {
        role: {
            "precision": round(float(report[role]["precision"]), 6),
            "recall": round(float(report[role]["recall"]), 6),
            "f1": round(float(report[role]["f1-score"]), 6),
            "support": int(report[role]["support"]),
        }
        for role in ROLE_LABELS
    }
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 6),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro")), 6),
        "per_role": per_role,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(ROLE_LABELS)
        ).astype(int).tolist(),
    }


def _fold_summary(fold_number: int, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    return {
        "fold": fold_number,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_companies": int(train["group_company"].nunique()),
        "test_companies": int(test["group_company"].nunique()),
        "train_templates": int(train["group_template"].nunique()),
        "test_templates": int(test["group_template"].nunique()),
        "train_role_support": {
            role: int((train["target"] == role).sum()) for role in ROLE_LABELS
        },
        "test_role_support": {
            role: int((test["target"] == role).sum()) for role in ROLE_LABELS
        },
    }


def select_model(model_metrics: Mapping[str, object]) -> str:
    """Select by macro F1, then prefer skills-only and Logistic Regression."""
    preference = {
        "skills_logistic_regression": (0, 0),
        "skills_linear_svm": (0, 1),
        "combined_logistic_regression": (1, 0),
        "combined_linear_svm": (1, 1),
    }
    candidates = [name for name, _, _ in LEARNED_CONFIGURATIONS]
    return sorted(
        candidates,
        key=lambda name: (
            -round(float(model_metrics[name]["macro_f1"]), 6),
            preference[name][0],
            preference[name][1],
        ),
    )[0]


def predict_with_model(model_bundle: Mapping[str, object], frame: pd.DataFrame) -> list[str]:
    """Predict from a fitted, serializable Phase 6 model bundle."""
    matrix = transform_features(frame, model_bundle["transformers"])
    return [str(value) for value in model_bundle["classifier"].predict(matrix)]


def _finite_numbers(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def validate_experiment_result(
    metrics: Mapping[str, object],
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
) -> list[str]:
    """Validate leakage boundaries and artifact-shaped in-memory results."""
    errors: list[str] = []
    expected_columns = {
        "job_id",
        "actual_role",
        "fold",
        *PREDICTION_COLUMNS.values(),
    }
    missing = sorted(expected_columns - set(predictions.columns))
    if missing:
        return [f"ML predictions are missing columns: {missing}"]
    if len(predictions) != len(frame) or predictions["job_id"].duplicated().any():
        errors.append("ML predictions must contain exactly one row per model row")
    if set(predictions["job_id"]) != set(frame["job_id"]):
        errors.append("ML prediction job IDs do not match the model frame")
    if set(predictions["fold"]) != set(range(1, N_SPLITS + 1)):
        errors.append("ML predictions must use every declared fold")
    if predictions[list(PREDICTION_COLUMNS.values())].isna().any(axis=None):
        errors.append("every configuration must predict every model row")
    for column in ["actual_role", *PREDICTION_COLUMNS.values()]:
        if not set(predictions[column]).issubset(set(ROLE_LABELS)):
            errors.append(f"{column} contains an unknown role")

    assignments = frame[["job_id", "target", "group_company", "group_template"]].merge(
        predictions[["job_id", "fold", "actual_role"]], on="job_id", validate="one_to_one"
    )
    if not assignments["target"].equals(assignments["actual_role"]):
        errors.append("prediction actual roles do not match the model targets")
    for fold_number in range(1, N_SPLITS + 1):
        test = assignments[assignments["fold"] == fold_number]
        train = assignments[assignments["fold"] != fold_number]
        if set(test["group_company"]) & set(train["group_company"]):
            errors.append(f"company leakage detected in saved fold {fold_number}")
        if set(test["group_template"]) & set(train["group_template"]):
            errors.append(f"template leakage detected in saved fold {fold_number}")
        if set(test["target"]) != set(ROLE_LABELS) or set(train["target"]) != set(ROLE_LABELS):
            errors.append(f"saved fold {fold_number} does not contain all four roles")

    model_metrics = metrics.get("models", {})
    if set(model_metrics) != set(MODEL_NAMES):
        errors.append("ML metrics do not contain the five approved configurations")
    elif metrics.get("selected_model") != select_model(model_metrics):
        errors.append("selected model does not follow the approved selection rule")
    experiment = metrics.get("experiment", {})
    excluded = set(experiment.get("excluded_leakage_columns", []))
    if not {"title_original", "title_normalized"}.issubset(excluded):
        errors.append("both title fields must be declared as excluded leakage columns")
    if experiment.get("feature_sources") != list(FEATURE_SOURCE_COLUMNS):
        errors.append("ML feature sources do not match the approved contract")
    configured_names = {
        item.get("name") for item in experiment.get("configurations", [])
    }
    expected_learned = {name for name, _, _ in LEARNED_CONFIGURATIONS}
    if configured_names != expected_learned:
        errors.append("ML feature/model configurations do not match the approved contract")
    if not _finite_numbers(metrics):
        errors.append("ML metrics contain non-finite values")
    return errors


def run_grouped_experiment(
    jobs: pd.DataFrame,
    job_skills: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    """Evaluate the approved five configurations and fit the best learned model."""
    if n_splits != N_SPLITS:
        raise ValueError(f"the approved experiment requires exactly {N_SPLITS} folds")
    frame = build_model_frame(jobs, job_skills)
    folds = make_company_grouped_folds(
        frame,
        n_splits=n_splits,
        random_state=random_state,
    )
    predictions = pd.DataFrame(
        {"job_id": frame["job_id"], "actual_role": frame["target"], "fold": 0}
    )
    predictions_by_model = {
        name: pd.Series(index=frame.index, dtype="object") for name in MODEL_NAMES
    }
    fold_metrics: dict[str, list[dict[str, object]]] = {name: [] for name in MODEL_NAMES}
    fold_summaries: list[dict[str, object]] = []
    preprocessing_by_fold: dict[str, list[dict[str, int]]] = {
        feature_set: [] for feature_set in FEATURE_SETS
    }

    for fold_number, (train_index, test_index) in enumerate(folds, start=1):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        predictions.loc[test_index, "fold"] = fold_number
        fold_summaries.append(_fold_summary(fold_number, train, test))

        majority = _majority_label(train["target"])
        baseline_prediction = pd.Series(majority, index=test_index)
        predictions_by_model["majority_baseline"].loc[test_index] = baseline_prediction
        baseline_metrics = metric_summary(test["target"], baseline_prediction)
        fold_metrics["majority_baseline"].append(
            {"fold": fold_number, "test_rows": len(test), **baseline_metrics}
        )

        for feature_set in FEATURE_SETS:
            transformers = fit_feature_transformers(train, feature_set)
            train_matrix = transform_features(train, transformers)
            test_matrix = transform_features(test, transformers)
            description_vectorizer = transformers["description_vectorizer"]
            preprocessing_by_fold[feature_set].append(
                {
                    "fold": fold_number,
                    "skill_features": len(
                        transformers["skill_vectorizer"].feature_names_
                    ),
                    "description_features": (
                        0
                        if description_vectorizer is None
                        else len(description_vectorizer.get_feature_names_out())
                    ),
                }
            )
            for name, configured_feature_set, classifier_name in LEARNED_CONFIGURATIONS:
                if configured_feature_set != feature_set:
                    continue
                classifier = _make_classifier(classifier_name, random_state)
                classifier.fit(train_matrix, train["target"])
                fold_prediction = pd.Series(
                    classifier.predict(test_matrix), index=test_index, dtype="object"
                )
                predictions_by_model[name].loc[test_index] = fold_prediction
                result = metric_summary(test["target"], fold_prediction)
                fold_metrics[name].append(
                    {"fold": fold_number, "test_rows": len(test), **result}
                )

    model_metrics: dict[str, object] = {}
    for name in MODEL_NAMES:
        prediction = predictions_by_model[name]
        if prediction.isna().any():
            raise ValueError(f"{name} did not predict every model row")
        predictions[PREDICTION_COLUMNS[name]] = prediction
        macro_scores = [float(result["macro_f1"]) for result in fold_metrics[name]]
        model_metrics[name] = {
            **metric_summary(frame["target"], prediction),
            "fold_macro_f1_mean": round(statistics.fmean(macro_scores), 6),
            "fold_macro_f1_std": round(statistics.pstdev(macro_scores), 6),
            "folds": fold_metrics[name],
        }

    selected_name = select_model(model_metrics)
    selected_spec = next(
        spec for spec in LEARNED_CONFIGURATIONS if spec[0] == selected_name
    )
    _, selected_feature_set, selected_classifier_name = selected_spec
    final_transformers = fit_feature_transformers(frame, selected_feature_set)
    final_matrix = transform_features(frame, final_transformers)
    final_classifier = _make_classifier(selected_classifier_name, random_state)
    final_classifier.fit(final_matrix, frame["target"])
    model_bundle: dict[str, object] = {
        "selected_model": selected_name,
        "feature_set": selected_feature_set,
        "classifier_name": selected_classifier_name,
        "transformers": final_transformers,
        "classifier": final_classifier,
        "role_proxy_patterns": list(ROLE_PROXY_PATTERNS),
        "role_labels": list(ROLE_LABELS),
    }

    skill_counts = frame["skills"].map(len)
    template_sizes = frame["group_template"].value_counts()
    baseline_macro_f1 = float(model_metrics["majority_baseline"]["macro_f1"])
    selected_macro_f1 = float(model_metrics[selected_name]["macro_f1"])
    metrics: dict[str, object] = {
        "experiment": {
            "rows": len(frame),
            "jobs_with_skills": int((skill_counts > 0).sum()),
            "jobs_without_skills": int((skill_counts == 0).sum()),
            "companies": int(frame["group_company"].nunique()),
            "template_groups": int(frame["group_template"].nunique()),
            "repeated_template_groups": int((template_sizes > 1).sum()),
            "jobs_in_repeated_template_groups": int(template_sizes[template_sizes > 1].sum()),
            "empty_model_descriptions": int(frame["description_model"].eq("").sum()),
            "description_role_proxy_matches_removed": int(
                frame["role_proxy_matches"].sum()
            ),
            "descriptions_with_role_proxy_matches": int(
                frame["role_proxy_matches"].gt(0).sum()
            ),
            "role_counts": {
                role: int((frame["target"] == role).sum()) for role in ROLE_LABELS
            },
            "folds": n_splits,
            "random_state": random_state,
            "splitter": "StratifiedGroupKFold",
            "split_group": "company_key",
            "template_groups_kept_intact": True,
            "headline_metric": "pooled_out_of_fold_macro_f1",
            "feature_sources": list(FEATURE_SOURCE_COLUMNS),
            "excluded_leakage_columns": list(EXCLUDED_LEAKAGE_COLUMNS),
            "skill_vectorizer": dict(SKILL_VECTORIZER_CONFIG),
            "description_vectorizer": {
                **DESCRIPTION_VECTORIZER_CONFIG,
                "ngram_range": list(DESCRIPTION_VECTORIZER_CONFIG["ngram_range"]),
            },
            "role_proxy_patterns": list(ROLE_PROXY_PATTERNS),
            "configurations": [
                {
                    "name": name,
                    "feature_set": feature_set,
                    "classifier": classifier,
                }
                for name, feature_set, classifier in LEARNED_CONFIGURATIONS
            ],
            "fold_summaries": fold_summaries,
            "preprocessing_by_fold": preprocessing_by_fold,
        },
        "models": model_metrics,
        "selected_model": selected_name,
        "selected_model_beats_baseline": selected_macro_f1 > baseline_macro_f1,
    }
    predictions = predictions.sort_values("job_id", kind="stable").reset_index(drop=True)
    errors = validate_experiment_result(metrics, predictions, frame)
    if errors:
        raise ValueError("; ".join(errors))
    return metrics, predictions, model_bundle
