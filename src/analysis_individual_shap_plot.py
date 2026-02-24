import os # interact with operating system
import joblib # Used for parallel computing
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import shap # For SHAP plots (SHapley Additive exPlanations)
import matplotlib.pyplot as plt # For plotting
# This script will create a SHAP plot for any specified model run

# Directory,  creates paths to models, splits, and figures folder
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "../models")
splits_dir = os.path.join(script_dir, "../splits")
figures_dir = os.path.join(script_dir, "../figures")
os.makedirs(figures_dir, exist_ok=True)

# Specifies which training run to use
model_run = 0
model_file = os.path.join(models_dir, f"full_rf_model_run_{model_run}.pkl")
X_test_file = os.path.join(splits_dir, f"X_test_run_{model_run}.csv")

# Loads the data and trained pipeline
pipeline = joblib.load(model_file)
X_test = pd.read_csv(X_test_file)

# Selects a subset of rows for SHAP computations, changing this will affect computation speed
X_test_subset = X_test.sample(min(150, len(X_test)), random_state=0)

# Returns probabilities for the damage class and ensures data compatability
def predict_damage_proba(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=X_test.columns)
    return pipeline.predict_proba(X)[:, 1]

# Background sample or baseline data set for comparisons, selects 50 rows to compare data to
background = X_test_subset.sample(min(50, len(X_test_subset)), random_state=1)

# Computes the shap values, nsamples controls the number of model evaluations to approximate SHAP values
explainer = shap.KernelExplainer(predict_damage_proba, background)
shap_values = explainer.shap_values(X_test_subset, nsamples=100)

# Creates the SHAP plot
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_test_subset,
    plot_type="dot",
    show=False
)

# Saves the plot
plot_file = os.path.join(figures_dir, f"shap_summary_full_rf_run{model_run}.png")
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.close()

# Completion message
print(f"SHAP summary plot saved to {plot_file}")
