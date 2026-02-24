import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing, common in data science
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Model inputs, target value, states applied to
target_column = 'Damage_Status'
states = ["ME", "NH", "RI", "MS"]

features = [
    "Distance_to_Coast_m",
    "Inundation_Duration_Min"
]

# Directory structure
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "../models")
results_dir = os.path.join(script_dir, "../results")
data_dir = os.path.join(script_dir, "../data/processed")

os.makedirs(results_dir, exist_ok=True)

# Load trained TX linear regression model
model_path = os.path.join(models_dir, "linear_regression_TX.pkl")
model = joblib.load(model_path)

print("\n Applying TX Linear Regression Model")

# Loop through states
for state in states:
    print(f"\nSTATE: {state} Linear Regression Performance Metrics")

    # Load state processed data
    data_file = os.path.join(data_dir, state, "processed.csv")
    df = pd.read_csv(data_file)

    # Damage status to binary
    df["Damage_Binary"] = df[target_column].map({
        "No Damage": 0,
        "Damage": 1
    })

    df = df.dropna(subset=["Damage_Binary"])
    
    # Features
    X = df[features].apply(pd.to_numeric, errors="coerce").values
    y = df["Damage_Binary"].values

    # Predictions
    y_score = model.predict(X)

    # Classification (threshold = 0.5)
    y_hat = (y_score >= 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y, y_hat)

    # Performance metrics
    metrics = {
        "Accuracy": accuracy_score(y, y_hat),
        "Balanced_Accuracy": balanced_accuracy_score(y, y_hat),
        "Cohens_Kappa": cohen_kappa_score(y, y_hat),
        "AUC": roc_auc_score(y, y_score),

        "Precision_NoDamage": precision_score(y, y_hat, pos_label=0, zero_division=0),
        "Recall_NoDamage": recall_score(y, y_hat, pos_label=0, zero_division=0),
        "F1_NoDamage": f1_score(y, y_hat, pos_label=0, zero_division=0),

        "Precision_Damage": precision_score(y, y_hat, pos_label=1, zero_division=0),
        "Recall_Damage": recall_score(y, y_hat, pos_label=1, zero_division=0),
        "F1_Damage": f1_score(y, y_hat, pos_label=1, zero_division=0),
    }

    print("\nConfusion Matrix:")
    print(cm)

    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")