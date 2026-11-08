# 2026AK05151-ML-Assignment2
2026AK05151-ML-Assignment-2 Repository
AIML Machine Learning Assignment 2 

## a. Problem statement
Build and evaluate five classification algorithms on one common dataset, then demonstrate the saved models through a Streamlit application that accepts a labeled test CSV and reports evaluation metrics and predictions.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** UCI Machine Learning Repository (distributed in scikit-learn's `load_breast_cancer` dataset loader).
- **Instances:** 569
- **Input features:** 30 numeric features
- **Target:** `target` (0 = malignant, 1 = benign)
- **Split:** 80% training / 20% test, stratified, `random_state=42`.
- **Training rows:** 455; **test rows:** 114.
- The uploaded `test_data.csv` contains all 30 input features and the ground-truth target column.

## c. GitHub Repository Link
Add your own repository URL here after creating and pushing this project:
`[PASTE YOUR GITHUB REPOSITORY URL]`

## d. Models used and evaluation

All models use the same stratified train/test split. StandardScaler is included in pipelines for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes are used without scaling.

### Comparison table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression.   | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree          | 0.9211   | 0.9163 | 0.9565    | 0.9167 | 0.9362 | 0.8341 |
| KNN                    | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes.  | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825   | 0.9950 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |

### Observations
- **Logistic Regression:** Strong performance on this split, with 0.9825 accuracy and 0.9861 F1. Scaling is included in its pipeline.
- **Decision Tree:** Lowest accuracy and F1 among these five runs (0.9211 and 0.9362). Its single-tree structure is easier to interpret, but this configuration performed less strongly on the held-out rows.
- **KNN:** Good performance (0.9561 accuracy, 0.9655 F1). Feature scaling is important because KNN relies on distances.
- **Gaussian Naive Bayes:** AUC is high (0.9878), while accuracy is 0.9386. This shows why it is useful to inspect several metrics rather than accuracy alone.
- **Support Vector Machine:** Tied with Logistic Regression on accuracy, precision, recall, F1 and MCC; its AUC is 0.9950.
