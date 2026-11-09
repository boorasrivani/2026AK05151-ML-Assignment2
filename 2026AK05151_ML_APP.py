
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/gaussian_naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]
LABELS = {0: "malignant", 1: "benign"}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="AIML Assignment 2 | Classifier Lab", layout="wide")
st.title("Machine Learning Classification Model Explorer")
st.caption("UCI Wisconsin Diagnostic Breast Cancer dataset | Five saved classifiers")


@st.cache_resource
def load_model(path, modified_ns):
    # Modification time invalidates cached models when saved artifacts change.
    return joblib.load(path)


def evaluate(model_name, X_test, y_test):
    path = BASE_DIR / MODEL_FILES[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Required model file is missing: {MODEL_FILES[model_name]}")
    model = load_model(str(path), path.stat().st_mtime_ns)
    prediction = model.predict(X_test)
    positive_column = list(model.classes_).index(1)
    probability = model.predict_proba(X_test)[:, positive_column]
    metrics = {
        "Accuracy": accuracy_score(y_test, prediction),
        "AUC": roc_auc_score(y_test, probability) if y_test.nunique() == 2 else np.nan,
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0),
        "MCC": matthews_corrcoef(y_test, prediction),
    }
    return metrics, prediction, probability


def read_test_data(csv_file):
    frame = pd.read_csv(csv_file)
    missing = [name for name in FEATURES + ["target"] if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The uploaded CSV has no rows.")
    X_test = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    y_test = pd.to_numeric(frame["target"], errors="raise")
    if not y_test.isin([0, 1]).all():
        raise ValueError("Target values must be 0 (malignant) or 1 (benign).")
    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Feature values must be numeric and cannot be missing or infinite.")
    return frame, X_test, y_test.astype(int)


st.sidebar.header("Experiment settings")
model_name = st.sidebar.selectbox("Choose classifier", list(MODEL_FILES))
data_source = st.sidebar.radio("Test data", ["Built-in test data", "Upload CSV"])
st.sidebar.caption("Upload test data only: 30 feature columns plus target.")
builtin_path = BASE_DIR / "test_data.csv"
if builtin_path.is_file():
    builtin_csv = builtin_path.read_bytes()
    st.sidebar.download_button(
        "Download test CSV", builtin_csv,
        file_name="test_data.csv", mime="text/csv",
    )
else:
    builtin_csv = None

if data_source == "Built-in test data":
    csv_file = BytesIO(builtin_csv) if builtin_csv is not None else None
    if builtin_csv is None:
        st.error("The built-in test_data.csv is missing. Select Upload CSV to supply test data.")
else:
    csv_file = st.file_uploader(
        "Upload test CSV (30 feature columns and target)", type=["csv"]
    )

st.info("The app loads pre-trained models. Uploaded test data is used for evaluation only; models are never retrained here.")
st.caption("Labels: 0 = malignant, 1 = benign. Precision, recall and F1 use benign (1) as the positive class; AUC uses its predicted probability.")
with st.expander("Required CSV columns"):
    st.write(FEATURES + ["target"])

if csv_file is None:
    if data_source == "Upload CSV":
        st.write("Upload the provided test_data.csv or another held-out CSV with the same columns.")
    st.stop()

try:
    df, X_test, y_test = read_test_data(csv_file)
except (ValueError, TypeError, pd.errors.ParserError) as error:
    st.error(f"Could not process test CSV: {error}")
    st.stop()

st.caption(f"Evaluating {len(df)} test rows with {len(FEATURES)} input features.")
if y_test.nunique() < 2:
    st.warning("AUC is unavailable because the test CSV contains only one target class. Other metrics are still shown.")

# Every classifier is evaluated on these same rows. There are no fit calls in this app.
evaluations = {}
comparison_rows = []
for name in MODEL_FILES:
    try:
        evaluations[name] = evaluate(name, X_test, y_test)
        comparison_rows.append({"Model": name, **evaluations[name][0]})
    except Exception as error:
        st.error(f"Could not evaluate {name}: {error}")

if model_name not in evaluations:
    st.stop()

metrics, prediction, probability = evaluations[model_name]
st.subheader(f"Results: {model_name}")
for column, label in zip(st.columns(6), METRIC_NAMES):
    value = metrics[label]
    column.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

st.subheader("Comparison of all five classifiers")
comparison = pd.DataFrame(comparison_rows).set_index("Model")
st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), width="stretch")
st.download_button(
    "Download comparison CSV", comparison.to_csv().encode("utf-8"),
    file_name="model_comparison.csv", mime="text/csv",
)

st.subheader("Confusion matrix")
cm = confusion_matrix(y_test, prediction, labels=[0, 1])
figure, axis = plt.subplots(figsize=(5.5, 4.0))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
    xticklabels=["malignant (0)", "benign (1)"],
    yticklabels=["malignant (0)", "benign (1)"],
)
axis.set_xlabel("Predicted label")
axis.set_ylabel("Actual label")
axis.set_title(model_name)
figure.tight_layout()
st.pyplot(figure)
plt.close(figure)
with st.expander("View confusion matrix values"):
    st.dataframe(pd.DataFrame(
        cm, index=["Actual malignant", "Actual benign"],
        columns=["Predicted malignant", "Predicted benign"],
    ), width="stretch")

st.subheader("Classification report")
st.code(classification_report(
    y_test, prediction, labels=[0, 1],
    target_names=["malignant", "benign"], zero_division=0,
))
result = df.copy()
result["predicted_target"] = prediction
result["predicted_label"] = [LABELS[int(value)] for value in prediction]
result["probability_benign"] = probability
st.subheader("Input rows and predictions")
st.dataframe(result, width="stretch")
st.download_button(
    "Download predictions CSV", result.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv", mime="text/csv",
)
