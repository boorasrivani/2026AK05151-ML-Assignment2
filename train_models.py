"""
Train five classifiers on the UCI Wisconsin Diagnostic Breast Cancer dataset.
Run: python train_models.py
"""
import os, json, joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=42))]),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, score),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib")
pd.DataFrame(results).to_csv("outputs/metrics.csv", index=False)
test = X_test.copy()
test["target"] = y_test
test.to_csv("test_data.csv", index=False)
print(pd.DataFrame(results).to_string(index=False))
print(f"Saved {len(models)} models and {len(test)} test rows.")
