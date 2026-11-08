import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_and_export_wdbc():
    # 1. Load dataset as a pandas DataFrame
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame  # Contains all 30 features + 'target' column

    X = dataset.data
    y = dataset.target

    # 2. Perform 80% train / 20% test stratified split
    df_train, df_test = train_test_split(
        df, 
        test_size=0.20, 
        stratify=y, 
        random_state=42
    )

    # 3. Export the test dataset (unscaled features + target) to CSV
    output_filename = "test_dataset.csv"
    df_test.to_csv(output_filename, index=False)
    print(f"Successfully generated '{output_filename}' ({len(df_test)} rows, {df_test.shape[1]} columns).")

    # 4. Standardize input features for model training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train.drop(columns=["target"]))
    X_test_scaled = scaler.transform(df_test.drop(columns=["target"]))

    print("\n=== Dataset Summary ===")
    print(f"Total Instances:     {len(df)}")
    print(f"Training Instances:  {len(df_train)}")
    print(f"Test Instances:      {len(df_test)}")

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": df_train["target"],
        "y_test": df_test["target"],
        "test_df": df_test
    }

if __name__ == "__main__":
    data_dict = process_and_export_wdbc()