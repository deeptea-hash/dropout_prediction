"""
Predictive Analytics for E-Learning Dropout Rates
=================================================
Core pipeline implementing the functional requirements in the project report:

    FR1 Import            -> load_dataset()
    FR2 Validate          -> validate_dataset()
    FR3 Data-quality      -> data_quality_summary()
    FR4 Visualize         -> plot_eda()
    FR5 Preprocess        -> build_preprocessor()
    FR6 Train             -> train_models()
    FR7 Evaluate/compare  -> evaluate_models()
    FR8 Predict           -> predict_risk()
    FR9 Report / export   -> export_report()

Design rules taken from the report:
  * Imputation / scaling / encoding are fitted on TRAINING data only
    (scikit-learn Pipeline + ColumnTransformer) to avoid leakage.
  * Positive class = dropout (1), so recall = share of true dropouts found.
  * Single-class targets, missing columns and invalid labels raise a clear
    DatasetError instead of producing misleading metrics.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # works without a display (servers, Colab, tests)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_DEFAULT = "dropout_status"

# Features from Table 3.1 / Appendix A of the report.
DEFAULT_FEATURES = [
    "login_frequency",
    "days_since_last_activity",
    "assessment_average",
    "assignment_submission_rate",
    "resource_interactions",
    "discussion_participation",
]


class DatasetError(ValueError):
    """Raised when the dataset cannot be used (TC02, TC05 and similar)."""


# --------------------------------------------------------------------------
# FR1  Import
# --------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV (or Excel) learner dataset."""
    if not os.path.exists(path):
        raise DatasetError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# FR2  Validate
# --------------------------------------------------------------------------
_LABEL_MAP = {
    "1": 1, "0": 0,
    "yes": 1, "no": 0,
    "true": 1, "false": 0,
    "dropout": 1, "dropped": 1, "withdrawn": 1, "discontinued": 1, "fail": 1,
    "completed": 0, "complete": 0, "graduate": 0, "pass": 0, "enrolled": 0,
    "distinction": 0, "retained": 0,
}


def normalise_target(series: pd.Series) -> pd.Series:
    """Map common label spellings to 0/1 (1 = dropout). Raises on unknown labels."""
    if series.isna().any():
        raise DatasetError(
            f"Target column has {int(series.isna().sum())} missing label(s); "
            "remove or label these rows first."
        )
    mapped = series.astype(str).str.strip().str.lower().map(_LABEL_MAP)
    bad = series[mapped.isna()].unique()
    if len(bad):
        raise DatasetError(
            f"Invalid target labels found: {list(bad)[:5]}. "
            "Expected 0/1, yes/no, true/false, dropout/completed, withdrawn/pass."
        )
    return mapped.astype(int)


def validate_dataset(
    df: pd.DataFrame,
    features: Sequence[str] = DEFAULT_FEATURES,
    target: str = TARGET_DEFAULT,
) -> pd.DataFrame:
    """Check columns and labels; return a cleaned copy with a 0/1 target."""
    if df is None or df.empty:
        raise DatasetError("Dataset is empty.")
    if target not in df.columns:
        raise DatasetError(f"Required target column '{target}' is missing.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise DatasetError(f"Missing required feature column(s): {missing}")
    out = df.copy()
    out[target] = normalise_target(out[target])
    if out[target].nunique() < 2:
        raise DatasetError(
            "Target has a single class; a classifier cannot be trained or "
            "evaluated. Provide data containing both completers and dropouts."
        )
    return out


# --------------------------------------------------------------------------
# FR3  Data-quality summary
# --------------------------------------------------------------------------
def data_quality_summary(df: pd.DataFrame, target: str = TARGET_DEFAULT) -> Dict:
    """Missing values, duplicates, dtypes and class balance."""
    miss = df.isna().sum()
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in miss.items() if v > 0},
        "missing_total": int(miss.sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }
    if target in df.columns:
        counts = df[target].value_counts()
        summary["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        summary["dropout_rate"] = float((df[target] == 1).mean()) if set(
            df[target].unique()
        ) <= {0, 1} else None
    return summary


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Documented rule (TC04): drop exact duplicate rows, keep first."""
    n = int(df.duplicated().sum())
    return df.drop_duplicates().reset_index(drop=True), n


# --------------------------------------------------------------------------
# FR4  Visualize
# --------------------------------------------------------------------------
def plot_eda(
    df: pd.DataFrame,
    features: Sequence[str] = DEFAULT_FEATURES,
    target: str = TARGET_DEFAULT,
    out_dir: str = "outputs/figures",
) -> List[str]:
    """Save class balance, histograms, box plots and a correlation heat map."""
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []
    sns.set_theme(style="whitegrid")
    num = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]

    # 1. class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[target].map({0: "Completed", 1: "Dropout"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, legend=False)
    ax.set_title("Target class balance")
    ax.set_ylabel("Learners")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    saved.append(_save(fig, out_dir, "01_class_balance.png"))

    # 2. histograms
    cols = 3
    rows = int(np.ceil(len(num) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows))
    for ax, f in zip(np.ravel(axes), num):
        sns.histplot(df[f].dropna(), kde=True, ax=ax)
        ax.set_title(f)
    for ax in np.ravel(axes)[len(num):]:
        ax.axis("off")
    fig.suptitle("Feature distributions", y=1.0)
    saved.append(_save(fig, out_dir, "02_feature_histograms.png"))

    # 3. box plots by outcome
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows))
    plot_df = df.assign(
        outcome=df[target].map({0: "Completed", 1: "Dropout"})
    )
    for ax, f in zip(np.ravel(axes), num):
        sns.boxplot(data=plot_df, x="outcome", y=f, ax=ax, hue="outcome", legend=False)
        ax.set_title(f)
        ax.set_xlabel("")
    for ax in np.ravel(axes)[len(num):]:
        ax.axis("off")
    fig.suptitle("Features by outcome", y=1.0)
    saved.append(_save(fig, out_dir, "03_boxplots_by_outcome.png"))

    # 4. correlation heat map
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = df[num + [target]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heat map (association, not causation)")
    saved.append(_save(fig, out_dir, "04_correlation_heatmap.png"))
    return saved


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# FR5  Preprocess
# --------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Optional engineered features (Section 3.5). Uses only pre-outcome data."""
    out = df.copy()
    if {"login_frequency", "resource_interactions"} <= set(out.columns):
        out["interactions_per_login"] = out["resource_interactions"] / (
            out["login_frequency"].clip(lower=1)
        )
    if {"assessment_average", "assignment_submission_rate"} <= set(out.columns):
        out["engagement_score"] = (
            out["assessment_average"] / 100.0 + out["assignment_submission_rate"]
        ) / 2
    return out


def build_preprocessor(
    X: pd.DataFrame, scale: bool = True
) -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + one-hot categoricals."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num_cols)]
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers)


# --------------------------------------------------------------------------
# FR6  Train
# --------------------------------------------------------------------------
def candidate_models() -> Dict[str, object]:
    """Table 3.2 candidates + a majority-class baseline (Section 5.2)."""
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_data(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str = TARGET_DEFAULT,
    test_size: float = 0.2,
) -> Split:
    """Stratified train/test split; test set reserved for final evaluation."""
    X, y = df[list(features)], df[target]
    minority = y.value_counts().min()
    if minority < 2:
        raise DatasetError(
            "Fewer than 2 examples in the minority class; cannot stratify/split."
        )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return Split(Xtr, Xte, ytr, yte)


def train_models(
    split: Split,
    models: Optional[Dict[str, object]] = None,
    cv_folds: int = 5,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """Fit each model inside a Pipeline; also report CV F1 on the training set."""
    models = models or candidate_models()
    fitted: Dict[str, Pipeline] = {}
    cv_rows = []
    min_class = int(split.y_train.value_counts().min())
    folds = max(2, min(cv_folds, min_class))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for name, est in models.items():
        pipe = Pipeline(
            [("prep", build_preprocessor(split.X_train)), ("model", est)]
        )
        scores = cross_val_score(pipe, split.X_train, split.y_train, cv=cv, scoring="f1")
        cv_rows.append(
            {"Model": name, "CV F1 (mean)": scores.mean(), "CV F1 (std)": scores.std()}
        )
        pipe.fit(split.X_train, split.y_train)
        fitted[name] = pipe
    return fitted, pd.DataFrame(cv_rows).set_index("Model")


# --------------------------------------------------------------------------
# FR7  Evaluate and compare
# --------------------------------------------------------------------------
def evaluate_models(
    fitted: Dict[str, Pipeline], split: Split
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Held-out metrics. Positive class = dropout (1)."""
    rows, cms = [], {}
    for name, pipe in fitted.items():
        pred = pipe.predict(split.X_test)
        cms[name] = confusion_matrix(split.y_test, pred, labels=[0, 1])
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(split.y_test, pred),
                "Precision": precision_score(split.y_test, pred, zero_division=0),
                "Recall": recall_score(split.y_test, pred, zero_division=0),
                "F1-score": f1_score(split.y_test, pred, zero_division=0),
            }
        )
    table = pd.DataFrame(rows).set_index("Model").round(4)
    return table.sort_values("F1-score", ascending=False), cms


def plot_confusion_matrix(cm: np.ndarray, title: str, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Pred: Completed", "Pred: Dropout"],
        yticklabels=["Actual: Completed", "Actual: Dropout"],
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(table: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    table.plot(kind="bar", ax=ax, rot=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Held-out test metrics by model")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def feature_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Permutation importance on the held-out set (works for any model)."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        pipe, X, y, scoring="f1", n_repeats=10, random_state=RANDOM_STATE
    )
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


# --------------------------------------------------------------------------
# FR8  Predict
# --------------------------------------------------------------------------
def risk_category(p: float) -> str:
    if p >= 0.66:
        return "High"
    if p >= 0.33:
        return "Medium"
    return "Low"


def predict_risk(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Predicted class + dropout probability + Low/Medium/High support band."""
    proba = pipe.predict_proba(X)[:, 1]
    out = X.copy()
    out["predicted_status"] = np.where(proba >= 0.5, "Dropout", "Completed")
    out["dropout_probability"] = proba.round(4)
    out["risk_category"] = [risk_category(p) for p in proba]
    return out


# --------------------------------------------------------------------------
# FR9  Report / export
# --------------------------------------------------------------------------
def environment_info() -> Dict[str, str]:
    """NFR1 - record versions for reproducibility."""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "random_state": str(RANDOM_STATE),
    }


PLAIN_LANGUAGE = """\
How to read these numbers (dropout = positive class)
- Accuracy : share of all learners classified correctly. Misleading if dropouts are rare.
- Precision: of learners flagged as at-risk, the share who really dropped out.
              High precision = few unnecessary alerts.
- Recall   : of learners who really dropped out, the share we caught.
              High recall = few learners missed.
- F1-score : balance of precision and recall.
A prediction is an estimate, not a certainty. Use it to offer support, never to penalise.
"""


def export_report(
    out_dir: str,
    dataset_name: str,
    quality: Dict,
    cv_table: pd.DataFrame,
    metrics: pd.DataFrame,
    cms: Dict[str, np.ndarray],
    best_model: str,
    importance: Optional[pd.Series] = None,
    synthetic: bool = False,
) -> str:
    """Write metrics CSV/JSON and a short Markdown report; return its path."""
    os.makedirs(out_dir, exist_ok=True)
    metrics.to_csv(os.path.join(out_dir, "model_metrics.csv"))
    cv_table.to_csv(os.path.join(out_dir, "cross_validation.csv"))
    with open(os.path.join(out_dir, "run_summary.json"), "w") as fh:
        json.dump(
            {
                "dataset": dataset_name,
                "synthetic_demo_data": synthetic,
                "data_quality": quality,
                "environment": environment_info(),
                "best_model_by_f1": best_model,
                "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
            },
            fh,
            indent=2,
        )
    lines = [f"# Dropout Prediction Report", ""]
    if synthetic:
        lines += [
            "> **WARNING: these results come from SYNTHETIC demonstration data.**",
            "> They show that the code works. They are NOT findings about real learners",
            "> and must not be reported as experimental results.",
            "",
        ]
    lines += [
        f"**Dataset:** {dataset_name}  ",
        f"**Rows:** {quality['rows']}  |  **Duplicates:** {quality['duplicate_rows']}  "
        f"|  **Missing cells:** {quality['missing_total']}  ",
        f"**Class counts:** {quality.get('class_counts')}  ",
        "",
        "## Held-out test metrics",
        "",
        metrics.to_markdown(),
        "",
        "## 5-fold cross-validation (training set, F1)",
        "",
        cv_table.round(4).to_markdown(),
        "",
        f"**Best model by F1:** {best_model}",
        "",
    ]
    if importance is not None:
        lines += [
            "## Permutation importance (best model, test set)",
            "",
            importance.round(4).to_frame("importance").to_markdown(),
            "",
        ]
    lines += ["## Interpretation guide", "", "```", PLAIN_LANGUAGE, "```", ""]
    lines += ["## Environment", "", "```", json.dumps(environment_info(), indent=2), "```"]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


# --------------------------------------------------------------------------
# End-to-end convenience wrapper
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    metrics: pd.DataFrame
    cv_table: pd.DataFrame
    best_model: str
    fitted: Dict[str, Pipeline]
    split: Split
    quality: Dict
    report_path: str
    figures: List[str] = field(default_factory=list)


def run_pipeline(
    df: pd.DataFrame,
    out_dir: str = "outputs",
    features: Sequence[str] = DEFAULT_FEATURES,
    target: str = TARGET_DEFAULT,
    dataset_name: str = "user-supplied dataset",
    synthetic: bool = False,
    engineer: bool = True,
    make_plots: bool = True,
) -> RunResult:
    """Validate -> clean -> EDA -> split -> train -> evaluate -> predict -> report."""
    df = validate_dataset(df, features, target)
    quality = data_quality_summary(df, target)
    df, _ = remove_duplicates(df)
    if engineer:
        df = add_engineered_features(df)
        features = list(features) + [
            c for c in ("interactions_per_login", "engagement_score") if c in df.columns
        ]
    # Re-check after de-duplication
    if df[target].nunique() < 2:
        raise DatasetError("Target has a single class after cleaning.")

    figs: List[str] = []
    fig_dir = os.path.join(out_dir, "figures")
    if make_plots:
        figs += plot_eda(df, features, target, fig_dir)

    split = split_data(df, features, target)
    fitted, cv_table = train_models(split)
    metrics, cms = evaluate_models(fitted, split)

    # best *real* model (exclude the baseline from "best")
    real = metrics.drop(index="Baseline (majority class)", errors="ignore")
    best = real["F1-score"].idxmax()

    importance = feature_importance(fitted[best], split.X_test, split.y_test)
    if make_plots:
        figs.append(plot_confusion_matrix(
            cms[best], f"Confusion matrix - {best}",
            os.path.join(fig_dir, "05_confusion_matrix_best.png")))
        figs.append(plot_metric_comparison(
            metrics, os.path.join(fig_dir, "06_metric_comparison.png")))

    preds = predict_risk(fitted[best], split.X_test)
    preds.insert(0, "actual_status", split.y_test.map({0: "Completed", 1: "Dropout"}))
    os.makedirs(out_dir, exist_ok=True)
    preds.to_csv(os.path.join(out_dir, "predictions_test_set.csv"), index=False)

    report = export_report(
        out_dir, dataset_name, quality, cv_table, metrics, cms, best,
        importance, synthetic,
    )
    return RunResult(metrics, cv_table, best, fitted, split, quality, report, figs)
