"""
Streamlit app - Adult Income classification (BITS ML Assignment 2).

Run locally:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.train_models import (
    CATEGORICAL_FEATURES,
    MODEL_FILES,
    NUMERIC_FEATURES,
    TARGET,
    encode_target,
    evaluate_model,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
METRICS_PATH = PROJECT_DIR / "metrics" / "model_metrics.csv"
SAMPLE_TEST_PATH = PROJECT_DIR / "test_data.csv"
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLASS_NAMES = ["<=50K", ">50K"]

st.set_page_config(page_title="Adult Income Classifier", page_icon="💼", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(model_name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


@st.cache_data
def load_benchmark_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH, index_col=0)


@st.cache_data
def compare_all_models(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Evaluate every saved model on the uploaded data."""
    rows = {name: evaluate_model(load_model(name), features, labels) for name in MODEL_FILES}
    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "ML Model Name"
    return table


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrix(labels, predictions, model_name: str):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted income")
    ax.set_ylabel("Actual income")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    return fig


def show_metric_cards(metrics: dict) -> None:
    columns = st.columns(3)
    for index, (metric_name, value) in enumerate(metrics.items()):
        columns[index % 3].metric(metric_name, f"{value:.4f}", border=True)


# ---------------------------------------------------------------------------
# Sidebar: data upload + model selection
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload test data")
uploaded_file = st.sidebar.file_uploader("CSV file (Adult Income format)", type=["csv"])
st.sidebar.download_button(
    "Download sample test_data.csv",
    data=SAMPLE_TEST_PATH.read_bytes(),
    file_name="test_data.csv",
    mime="text/csv",
)

st.sidebar.header("2. Choose a model")
selected_model = st.sidebar.selectbox("Classification model", list(MODEL_FILES))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: UCI Adult (Census Income). Target: whether annual income exceeds $50K."
)

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("💼 Adult Income Prediction - Model Evaluation Dashboard")
st.write(
    "Upload the held-out **test data**, pick one of the five trained classifiers and "
    "inspect its evaluation metrics, confusion matrix and classification report."
)

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin (you can download the sample test_data.csv there).")
    st.subheader("Benchmark results on the held-out test split")
    st.dataframe(load_benchmark_metrics(), width="stretch")
    st.stop()

uploaded_df = pd.read_csv(uploaded_file, skipinitialspace=True, na_values="?")
missing_columns = [col for col in REQUIRED_COLUMNS if col not in uploaded_df.columns]
if missing_columns:
    st.error(f"The uploaded file is missing required columns: {', '.join(missing_columns)}")
    st.stop()

features = uploaded_df[REQUIRED_COLUMNS]
has_labels = TARGET in uploaded_df.columns
model = load_model(selected_model)
predictions = model.predict(features)

with st.expander(f"Preview of uploaded data ({uploaded_df.shape[0]} rows x {uploaded_df.shape[1]} columns)"):
    st.dataframe(uploaded_df.head(20), width="stretch")

if not has_labels:
    st.warning(f"No '{TARGET}' column found - showing predictions only (metrics need true labels).")

tab_model, tab_compare, tab_predictions = st.tabs(
    ["📊 Selected model", "🏆 Model comparison", "📄 Predictions"]
)

with tab_model:
    st.subheader(f"Evaluation metrics - {selected_model}")
    if has_labels:
        labels = encode_target(uploaded_df[TARGET])
        show_metric_cards(evaluate_model(model, features, labels))

        left, right = st.columns(2)
        with left:
            st.pyplot(plot_confusion_matrix(labels, predictions, selected_model))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                labels, predictions, labels=[0, 1], target_names=CLASS_NAMES,
                output_dict=True, zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
    else:
        st.write("Upload data with the income column to view metrics.")

with tab_compare:
    if has_labels:
        st.subheader("All models on the uploaded data")
        comparison = compare_all_models(features, encode_target(uploaded_df[TARGET]))
        st.dataframe(
            comparison.style.highlight_max(
                axis=0, props="background-color:#2e7d32;color:white;font-weight:bold"
            ).format("{:.4f}"),
            width="stretch",
        )
        overall_best = comparison.rank(ascending=False).mean(axis=1).idxmin()
        st.success(
            f"Best accuracy: **{comparison['Accuracy'].idxmax()}** | "
            f"Best AUC: **{comparison['AUC'].idxmax()}** | "
            f"Overall best (average rank): **{overall_best}**"
        )
        st.bar_chart(comparison[["Accuracy", "AUC", "F1", "MCC"]], stack=False)

    st.subheader("Benchmark results saved during training")
    st.dataframe(load_benchmark_metrics(), width="stretch")

with tab_predictions:
    result_df = uploaded_df.copy()
    result_df["predicted_income"] = [CLASS_NAMES[p] for p in predictions]
    st.write(f"Predicted >50K for **{int(predictions.sum())}** of {len(predictions)} rows.")
    st.dataframe(result_df.head(100), width="stretch")
    st.download_button(
        "Download predictions as CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{MODEL_FILES[selected_model].replace('.pkl', '')}.csv",
        mime="text/csv",
    )
