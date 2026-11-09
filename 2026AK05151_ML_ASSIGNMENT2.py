import io
import base64
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, Markdown, clear_output
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report)

# 1. Load built-in Breast Cancer Dataset & Train Models
dataset = load_breast_cancer(as_frame=True)
FEATURES = list(dataset.feature_names)
LABELS = {0: "malignant", 1: "benign"}

X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2, random_state=42,
    stratify=dataset.target
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}

for name, model in models.items():
    model.fit(X_train, y_train)

# 2. Store built-in test dataset
builtin_test = X_test.copy()
builtin_test["target"] = y_test
current_test = builtin_test.copy()

def format_comparison_table(df_metrics):
    rows_html = ""
    for _, row in df_metrics.iterrows():
        rows_html += (
            f"<tr style='background-color: #ffffff; color: #000000;'>"
            f"<td style='border: 1px solid #000000; padding: 10px 14px; font-weight: bold; text-align: left; color: #000000;'>{row['ML Model Name']}</td>"
            f"<td style='border: 1px solid #000000; padding: 10px 14px; text-align: center; color: #000000;'>{row['Accuracy']:.4f}</td>"
            f"<td style='border: 1px solid #000000; padding: 10px 14px; text-align: center; color: #000000;'>{row['AUC']:.4f}</td>"
            f"<td style='border: 1px solid #000000; padding: 10px 14px; text-align: center; color: #000000;'>{row['Precision']:.4f}</td>"
            f"<td style='border: 1px solid #000000; padding: 10px 14px; text-align: center; color: #000000;'>{row['Recall']:.4f}</td>"
            f"<td style='border: 1px solid #000000; padding: 10px 14px; text-align: center; color: #000000;'>{row['F1']:.4f}</td>"
            f"<td style='border: 1px solid #000000; padding: 10px 14px; text-align: center; color: #000000;'>{row['MCC']:.4f}</td>"
            f"</tr>"
        )
    headers = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    header_html = "".join(
        f"<th style='border: 1px solid #000000; padding: 10px 14px; font-weight: bold; text-align: left; background-color: #ffffff; color: #000000;'>{h}</th>"
        for h in headers
    )
    return HTML(
        f"<table style='border-collapse: collapse; width: 100%; max-width: 850px; font-family: sans-serif; font-size: 15px; margin: 12px 0; background-color: #ffffff; color: #000000; border: 1px solid #000000;'>"
        f"<thead><tr style='background-color: #ffffff; color: #000000;'>{header_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
    )

# 3. Create Interactive Widgets
chooser = widgets.Dropdown(
    options=list(models.keys()), 
    description="Classifier:",
    style={"description_width": "initial"}, 
    layout=widgets.Layout(width="380px")
)

upload = widgets.FileUpload(
    accept=".csv", 
    multiple=False, 
    description="Upload test CSV"
)

reset_btn = widgets.Button(
    description="Reset to Built-in Test Data", 
    button_style="info",
    layout=widgets.Layout(width="220px")
)

output = widgets.Output()

def download_link(df, filename, link_text):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return HTML(f'<a download="{filename}" href="data:text/csv;base64,{b64}" target="_blank">📥 {link_text}</a>')

def render(change=None):
    with output:
        clear_output(wait=True)
        try:
            df = current_test.copy()

            # Check for required columns
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing[:3]}...")

            X = df[FEATURES].apply(pd.to_numeric, errors="raise")
            has_target = "target" in df.columns

            model = models[chooser.value]
            pred = model.predict(X)
            prob = model.predict_proba(X)[:, list(model.classes_).index(1)]

            display(Markdown(f"## Selected: **{chooser.value}**"))
            display(Markdown(f"**Rows evaluated:** {len(df)}"))

            # If target column is present, show metrics & confusion matrix
            if has_target:
                y = df["target"].astype(int)
                metrics = {
                    "ML Model Name": chooser.value,
                    "Accuracy": accuracy_score(y, pred),
                    "AUC": roc_auc_score(y, prob) if y.nunique() == 2 else np.nan,
                    "Precision": precision_score(y, pred, zero_division=0),
                    "Recall": recall_score(y, pred, zero_division=0),
                    "F1": f1_score(y, pred, zero_division=0),
                    "MCC": matthews_corrcoef(y, pred),
                }
                display(Markdown("### Evaluation Metrics"))
                display(format_comparison_table(pd.DataFrame([metrics])))

                display(Markdown("### Confusion Matrix"))
                display(pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1]),
                    index=["Actual malignant", "Actual benign"],
                    columns=["Predicted malignant", "Predicted benign"]))

                display(Markdown("### Classification Report"))
                print(classification_report(y, pred, labels=[0, 1],
                    target_names=["malignant", "benign"], zero_division=0))
            else:
                display(Markdown("*(Uploaded CSV does not have a `target` column, skipping ground-truth evaluation)*"))

            # Predictions table
            result = df.copy()
            result.insert(0, "predicted_label", [LABELS[int(v)] for v in pred])
            result.insert(0, "predicted_target", pred)

            display(Markdown(f"### Prediction Results (All {len(result)} rows)"))
            with pd.option_context("display.max_rows", None):
                display(result)

            display(download_link(result, "predictions.csv", "Download Predictions as CSV"))

        except Exception as error:
            display(Markdown(f"⚠️ **Could not evaluate CSV:** {error}"))

def on_upload(change):
    global current_test
    if not upload.value:
        return
    try:
        # Extract uploaded bytes (supports both ipywidgets 7 and 8)
        if isinstance(upload.value, dict):
            entry = next(iter(upload.value.values()))
            content = bytes(entry["content"])
        elif isinstance(upload.value, (list, tuple)) and len(upload.value) > 0:
            item = upload.value[0]
            content = bytes(item["content"]) if isinstance(item, dict) else bytes(item.content)
        else:
            return

        current_test = pd.read_csv(io.BytesIO(content))
        render()
    except Exception as error:
        with output:
            clear_output(wait=True)
            display(Markdown(f"⚠️ **Error reading CSV:** {error}"))

def on_reset(btn):
    global current_test
    current_test = builtin_test.copy()
    upload.value.clear() if hasattr(upload.value, 'clear') else None
    render()

# Attach events
chooser.observe(render, names="value")
upload.observe(on_upload, names="value")
reset_btn.on_click(on_reset)

# Display UI
display(Markdown("# 🔬 Machine Learning Classification Model Explorer"))
display(Markdown("Select a classifier, upload your custom test CSV, or use the built-in test data."))
display(chooser)
display(widgets.HBox([upload, reset_btn]))
display(output)

# Initial render
render()


# In[ ]:




