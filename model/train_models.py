"""
BITS Pilani M.Tech (DSE) - Machine Learning Assignment 2
Training script: Adult Income classification (UCI).

Run from the project root:
    python model/train_models.py

Outputs:
    data/dataset.csv            - cleaned full dataset
    test_data.csv               - held-out test split (used in the Streamlit app)
    model/*.pkl                 - 5 trained pipelines (preprocessing + classifier)
    metrics/model_metrics.csv   - evaluation metrics for all 5 models
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "dataset.csv"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
MODEL_DIR = PROJECT_DIR / "model"
METRICS_PATH = PROJECT_DIR / "metrics" / "model_metrics.csv"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

TARGET = "income"
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", TARGET,
]
NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

# File name for each saved model (also used by app.py)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """Load the cleaned dataset from disk, downloading it from UCI on first run."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    print(f"Downloading dataset from {DATA_URL} ...")
    raw_df = pd.read_csv(
        DATA_URL,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,   # values in the raw file are prefixed by a space
        na_values="?",           # UCI marks missing values with '?'
    )
    clean_df = raw_df.drop_duplicates().reset_index(drop=True)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(DATA_PATH, index=False)
    return clean_df


def encode_target(labels: pd.Series) -> pd.Series:
    """Map income labels to binary: '>50K' -> 1, '<=50K' -> 0 (handles trailing '.')."""
    cleaned = labels.astype(str).str.strip().str.rstrip(".")
    return (cleaned == ">50K").astype(int)


# ---------------------------------------------------------------------------
# Preprocessing and models
# ---------------------------------------------------------------------------
def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Impute missing values, one-hot encode categoricals and optionally scale numerics."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict:
    """Create the five classification pipelines required by the assignment."""
    classifiers = {
        "Logistic Regression": (LogisticRegression(max_iter=2000), True),
        # Depth/leaf limits keep the tree from memorising the training data
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=RANDOM_STATE),
            False,  # trees are scale-invariant
        ),
        "kNN": (KNeighborsClassifier(n_neighbors=15), True),
        "Naive Bayes": (GaussianNB(), True),
        "Support Vector Machine": (SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE), True),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor(scale)), ("classifier", clf)])
        for name, (clf, scale) in classifiers.items()
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def get_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return continuous scores for AUC (probabilities, or SVM decision values)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def evaluate_model(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict:
    """Compute the six evaluation metrics required by the assignment."""
    predictions = model.predict(features)
    scores = get_scores(model, features)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": roc_auc_score(labels, scores),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def summarise_results(metrics_df: pd.DataFrame, train_accuracy: dict) -> None:
    """Print best models and short observations for each model."""
    best_accuracy = metrics_df["Accuracy"].idxmax()
    best_auc = metrics_df["AUC"].idxmax()
    # Overall winner = best average rank across all six metrics
    overall_best = metrics_df.rank(ascending=False).mean(axis=1).idxmin()

    print("\n" + "=" * 70)
    print(f"Best Accuracy model : {best_accuracy} ({metrics_df.loc[best_accuracy, 'Accuracy']:.4f})")
    print(f"Best AUC model      : {best_auc} ({metrics_df.loc[best_auc, 'AUC']:.4f})")
    print(f"Overall best model  : {overall_best} (best average rank over all 6 metrics)")
    print("=" * 70)

    print("\nObservations (train vs test accuracy gap indicates over-fitting):")
    for name, row in metrics_df.iterrows():
        gap = train_accuracy[name] - row["Accuracy"]
        print(
            f"- {name}: test acc={row['Accuracy']:.4f}, train acc={train_accuracy[name]:.4f} "
            f"(gap {gap:+.4f}), precision={row['Precision']:.4f}, recall={row['Recall']:.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    dataset = load_dataset()
    features = dataset.drop(columns=[TARGET])
    labels = encode_target(dataset[TARGET])

    print(f"Dataset shape: {dataset.shape[0]} rows x {features.shape[1]} features")
    print(f"Missing values per column:\n{features.isna().sum()[features.isna().sum() > 0]}")
    print(f"Class balance (>50K share): {labels.mean():.3f}")

    # One stratified split shared by all models
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )

    # Save held-out test split (raw features + original label) for the Streamlit app
    test_export = X_test.copy()
    test_export[TARGET] = dataset.loc[X_test.index, TARGET]
    test_export.to_csv(TEST_DATA_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    train_accuracy = {}
    for name, pipeline in build_models().items():
        print(f"Training {name} ...")
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        train_accuracy[name] = accuracy_score(y_train, pipeline.predict(X_train))
        joblib.dump(pipeline, MODEL_DIR / MODEL_FILES[name], compress=3)

    metrics_df = pd.DataFrame(results).T.round(4)
    metrics_df.index.name = "ML Model Name"
    metrics_df.to_csv(METRICS_PATH)

    print("\nModel comparison (test set):")
    print(metrics_df.to_string())
    summarise_results(metrics_df, train_accuracy)


if __name__ == "__main__":
    main()
