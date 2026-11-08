import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="AIML Assignment 2 | Classifier Lab", layout="wide")
st.title("Machine Learning Classification Model Explorer")
st.caption("UCI Wisconsin Diagnostic Breast Cancer dataset | Five trained classifiers")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/gaussian_naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
FEATURES = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry",
    "mean fractal dimension","radius error","texture error","perimeter error",
    "area error","smoothness error","compactness error","concavity error",
    "concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area",
    "worst smoothness","worst compactness","worst concavity",
    "worst concave points","worst symmetry","worst fractal dimension"
]
LABELS = {0: "malignant", 1: "benign"}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.sidebar.header("Experiment settings")
model_name = st.sidebar.selectbox("Choose classifier", list(MODEL_FILES))
uploaded = st.file_uploader("Upload test CSV (must include 30 feature columns and target)", type=["csv"])

st.info("The app loads pre-trained models. It does not retrain on uploaded test data.")
if uploaded is None:
    st.write("Upload the provided `test_data.csv` to calculate test metrics.")
    st.write("Expected target column: `target` (0 = malignant, 1 = benign).")
    st.write("Expected input columns are listed below.")
    with st.expander("View required input features"):
        st.write(FEATURES)
else:
    try:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURES + ["target"] if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        elif df.empty:
            st.error("The uploaded CSV has no rows.")
        else:
            X_test = df[FEATURES].apply(pd.to_numeric, errors="raise")
            y_test = pd.to_numeric(df["target"], errors="raise").astype(int)
            if not set(y_test.unique()).issubset({0, 1}):
                st.error("Target values must be 0 or 1.")
            else:
                model = load_model(MODEL_FILES[model_name])
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]
                metrics = {
                    "Accuracy": accuracy_score(y_test, pred),
                    "AUC": roc_auc_score(y_test, prob) if y_test.nunique() == 2 else float("nan"),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1 Score": f1_score(y_test, pred, zero_division=0),
                    "MCC": matthews_corrcoef(y_test, pred),
                }
                st.subheader(f"Results: {model_name}")
                cols = st.columns(6)
                for col, (label, value) in zip(cols, metrics.items()):
                    col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")
                st.subheader("Confusion matrix")
                cm = confusion_matrix(y_test, pred, labels=[0, 1])
                st.dataframe(pd.DataFrame(cm, index=["Actual malignant", "Actual benign"],
                    columns=["Predicted malignant", "Predicted benign"]), use_container_width=True)
                st.subheader("Classification report")
                st.code(classification_report(y_test, pred, target_names=["malignant", "benign"], zero_division=0))
                result = df.copy()
                result["predicted_target"] = pred
                result["predicted_label"] = [LABELS[int(v)] for v in pred]
                st.subheader("Input rows and predictions")
                st.dataframe(result, use_container_width=True)
                st.download_button("Download predictions CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Could not process the uploaded CSV: {e}")
