import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import matplotlib.pyplot as plt # For plotting
import joblib # Used for parallel computing, common in data science
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Directory structure (Uses data provided by Dr. Webb)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "../data/processed/TX/webb_data.csv")

models_dir = os.path.join(script_dir, "../models")
figures_dir = os.path.join(script_dir, "../figures")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Load TX processed data
df = pd.read_csv(data_file)

# Binary damage indicator
df["Damage_Binary"] = df["Damage_Status"].map({
    "No Damage": 0,
    "Damage": 1
})

df = df.dropna(subset=["Damage_Binary"])

# Input features
features = [
    "Distance_to_Coast_m",
    "Inundation_Duration_Min"
]

X = df[features].apply(pd.to_numeric, errors="coerce").values
y = df["Damage_Binary"].values

# 2. Initialize the scaler
scaler = StandardScaler()

# 3. Fit to training data AND transform it
X = scaler.fit_transform(X)

# Linear regression model
model = LinearRegression()
model.fit(X, y)

# Save trained model
model_path = os.path.join(models_dir, "linear_regression_TX.pkl")
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")

# Continuous predictions
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

print("\n Linear Regression Model")

print("\nConfusion Matrix:")
print(cm)

for k, v in metrics.items():
    print(f"{k}: {v:.3f}")

# ROC curve (saves to /figures)
fpr, tpr, _ = roc_curve(y, y_score)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Linear Fragility Model (TX - All Points)")
plt.legend()
plt.grid(True)
plt.tight_layout()

roc_path = os.path.join(figures_dir, "ROC_TX_Linear_Regression.png")
plt.savefig(roc_path, dpi=300)
plt.close()

print(f"\nROC curve saved to: {roc_path}")

# Model coefficients
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\nLinear Regression Coefficients")
print(coef_df)

print("\nIntercept:", model.intercept_)