"""Core utilities for the credit-card fraud decision-support demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_NAMES = ("Class", "Fraud", "Is_Fraud", "is_fraud", "target", "label")
MODEL_NAMES = ("Logistic Regression", "Random Forest", "Gradient Boosting")


@dataclass
class TrainingResult:
    model: Pipeline
    models: dict[str, Pipeline]
    feature_columns: list[str]
    target_column: str
    metrics: dict[str, Any]
    model_metrics: dict[str, dict[str, float]]
    holdout_labels: np.ndarray
    holdout_probabilities: dict[str, np.ndarray]
    holdout_features: pd.DataFrame
    feature_defaults: dict[str, float]


def find_target_column(columns: Any) -> str | None:
    """Find a common fraud-label column, case-insensitively."""
    by_lower = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in TARGET_NAMES:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def validate_training_data(
    data: pd.DataFrame, target_column: str | None = None
) -> tuple[str, list[str]]:
    """Validate labeled tabular data and return its target and numeric features."""
    if data.empty or len(data) < 8:
        raise ValueError("Upload at least 8 transaction rows to train and evaluate a model.")

    target = target_column or find_target_column(data.columns)
    if not target or target not in data.columns:
        raise ValueError(
            "A fraud label column is required. Use a column named Class, Fraud, "
            "Is_Fraud, or select the target column in the app."
        )

    labels = pd.to_numeric(data[target], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"The target column '{target}' must contain only 0 and 1 values.")
    unique_labels = set(labels.unique().tolist())
    if not unique_labels.issubset({0, 1}) or unique_labels != {0, 1}:
        raise ValueError(f"The target column '{target}' must include both 0 (legitimate) and 1 (fraud).")

    features = [
        str(column)
        for column in data.columns
        if column != target and pd.api.types.is_numeric_dtype(data[column])
    ]
    features = [column for column in features if not data[column].isna().all()]
    if not features:
        raise ValueError("No numeric transaction features with usable values found besides the target column.")

    counts = labels.value_counts()
    if counts.min() < 2:
        raise ValueError("At least 2 examples of each class are required for a stratified evaluation split.")
    return target, features


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def train_detector(data: pd.DataFrame, target_column: str | None = None) -> TrainingResult:
    """Train three imbalanced-data-aware classifiers and evaluate a shared holdout."""
    target, features = validate_training_data(data, target_column)
    frame = data[features].apply(pd.to_numeric, errors="coerce")
    labels = pd.to_numeric(data[target]).astype(int).to_numpy()
    if frame.isna().all(axis=None):
        raise ValueError("The selected numeric features contain no usable values.")

    x_train, x_test, y_train, y_test = train_test_split(
        frame,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    model_specs: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    n_estimators=180,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                )),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.06,
                    max_depth=2,
                    random_state=42,
                )),
            ]
        ),
    }

    probabilities_by_model: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, float]] = {}
    sample_weights = np.where(y_train == 1, len(y_train) / (2 * max(np.sum(y_train == 1), 1)),
                              len(y_train) / (2 * max(np.sum(y_train == 0), 1)))
    for name, pipeline in model_specs.items():
        fit_params = {"classifier__sample_weight": sample_weights} if name == "Gradient Boosting" else {}
        pipeline.fit(x_train, y_train, **fit_params)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        probabilities_by_model[name] = probabilities
        model_metrics[name] = _classification_metrics(y_test, probabilities)

    selected = "Logistic Regression"
    selected_metrics = model_metrics[selected]
    matrix = confusion_matrix(y_test, probabilities_by_model[selected] >= 0.5, labels=[0, 1])
    metrics: dict[str, Any] = {
        **selected_metrics,
        "transactions": int(len(data)),
        "fraud_count": int(labels.sum()),
        "fraud_rate": float(labels.mean()),
        "holdout_count": int(len(y_test)),
    }
    defaults = frame.median().fillna(0).astype(float).to_dict()
    return TrainingResult(
        model=model_specs[selected],
        models=model_specs,
        feature_columns=features,
        target_column=target,
        metrics=metrics,
        model_metrics=model_metrics,
        holdout_labels=y_test,
        holdout_probabilities=probabilities_by_model,
        holdout_features=x_test,
        feature_defaults=defaults,
    )


def threshold_analysis(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
    average_fraud_loss: float = 500.0,
    false_alert_cost: float = 5.0,
) -> pd.DataFrame:
    """Compute review and estimated financial outcomes across thresholds."""
    if average_fraud_loss < 0 or false_alert_cost < 0:
        raise ValueError("Cost assumptions must be zero or greater.")
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    fraud_total = int(labels.sum())
    rows = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = fraud_total - tp
        rows.append({
            "Threshold": float(threshold),
            "Precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "Recall": float(tp / fraud_total) if fraud_total else 0.0,
            "False positives": fp,
            "Fraud caught": tp,
            "Fraud missed": fn,
            "Estimated loss": float(fn * average_fraud_loss),
            "Estimated operating cost": float(fn * average_fraud_loss + fp * false_alert_cost),
        })
    return pd.DataFrame(rows)


def explain_transaction(
    model: Pipeline,
    row: pd.DataFrame,
    reference: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Estimate per-feature score influence without claiming causal explanations.

    Logistic regression uses signed standardized log-odds contributions. Other
    estimators use one-feature-at-a-time probability changes from a median row;
    those changes are local sensitivity indicators and are not additive.
    """
    values = row[feature_columns].apply(pd.to_numeric, errors="coerce")
    reference_row = reference[feature_columns].median().fillna(0).to_frame().T
    classifier = model.named_steps["classifier"]
    if isinstance(classifier, LogisticRegression):
        imputer = model.named_steps["imputer"]
        scaler = model.named_steps["scaler"]
        transformed = scaler.transform(imputer.transform(values))[0]
        contributions = classifier.coef_[0] * transformed
        method = "Standardized logistic-regression log-odds contribution"
    else:
        baseline_probability = float(model.predict_proba(reference_row)[0, 1])
        base_row = reference_row.copy()
        contributions = []
        for feature in feature_columns:
            changed = base_row.copy()
            changed.loc[:, feature] = values.iloc[0][feature]
            contributions.append(float(model.predict_proba(changed)[0, 1] - baseline_probability))
        contributions = np.asarray(contributions)
        method = "One-feature-at-a-time probability change from median baseline"

    return pd.DataFrame({
        "Feature": feature_columns,
        "Value": [float(values.iloc[0][column]) for column in feature_columns],
        "Influence": contributions,
        "Method": method,
    }).assign(Absolute=lambda table: table["Influence"].abs()).sort_values(
        "Absolute", ascending=False
    ).drop(columns="Absolute")


def score_transactions(
    model: Pipeline,
    data: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score a batch while preserving original columns."""
    missing = [column for column in feature_columns if column not in data.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    features = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().all(axis=None):
        raise ValueError("The uploaded file has no usable numeric feature values.")
    output = data.copy()
    probabilities = model.predict_proba(features)[:, 1]
    output["Fraud probability"] = probabilities
    output["Fraud prediction"] = (probabilities >= threshold).astype(int)
    return output


def make_demo_data(rows: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """Create explicitly artificial transactions for a UI walkthrough only."""
    if rows < 100:
        raise ValueError("Demo data requires at least 100 rows.")
    rng = np.random.default_rng(random_state)
    fraud = rng.choice([0, 1], size=rows, p=[0.975, 0.025])
    amount = rng.lognormal(mean=3.1, sigma=1.0, size=rows)
    amount += fraud * rng.lognormal(mean=4.0, sigma=0.8, size=rows)
    elapsed = rng.uniform(0, 172800, size=rows)

    data: dict[str, np.ndarray] = {"Time": elapsed, "Amount": amount}
    for index in range(1, 9):
        normal_mean = rng.normal(0, 1, size=rows)
        fraud_shift = rng.normal(0, 1.5, size=rows) * fraud
        data[f"V{index}"] = normal_mean + fraud_shift
    data["Class"] = fraud
    return pd.DataFrame(data)
