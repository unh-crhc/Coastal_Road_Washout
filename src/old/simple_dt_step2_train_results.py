import os # Interact with operating system
import joblib # Used for parallel computing
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
from sklearn.metrics import ( # Imports for performance analysis
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, precision_recall_fscore_support,
    roc_auc_score  # NEW
)

# Directory with paths to models and the train/test stplits used by the full RF
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "../models")
splits_dir = os.path.join(script_dir, "../splits")
n_runs = 100 # Number of runs
model_type = "simple_dt" # Model name
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min'] # Input features
target_column = 'Damage_Status'

# Creates containers for te results
results = {
    "conf_matrices": [],
    "accuracies": [],
    "balanced_accuracies": [],
    "cohen_kappas": [],
    "precisions": [],
    "recalls": [],
    "f1s": [],
    "aucs": []  # NEW
}

# Iterates through each training run
for run in range(n_runs):
    # Load X_test and y_test for each run, training and testing splits
    X_test_path = os.path.join(splits_dir, f"X_test_run_{run}.csv")
    y_test_path = os.path.join(splits_dir, f"y_test_run_{run}.csv")

    # If no splits found for this run it prints this message
    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print(f"Warning: Missing test split for run {run}. Skipping. Train full RF first")
        continue

    # Reads the csvs and selects predictors and target
    X_test = pd.read_csv(X_test_path)[input_features]
    y_test = pd.read_csv(y_test_path).squeeze()

    # Ensures all columns are numeric
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Loads the trained DT pipeline
    model_file = os.path.join(models_dir, f"{model_type}_model_run_{run}.pkl")
    if not os.path.exists(model_file):
        print(f"Warning: Model for run {run} missing. Skipping.")
        continue

    # Loads the pipeline
    pipeline = joblib.load(model_file)

    # Uses the model to predict on the X_test subset
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # NEW

    # Computes confusion matrix and performance metrics for each run
    results["conf_matrices"].append(confusion_matrix(y_test, y_pred, labels=[0,1]))
    results["accuracies"].append(accuracy_score(y_test, y_pred))
    results["balanced_accuracies"].append(balanced_accuracy_score(y_test, y_pred))
    results["cohen_kappas"].append(cohen_kappa_score(y_test, y_pred))
    results["aucs"].append(roc_auc_score(y_test, y_prob))  # NEW

    # Computes class specific precision, recall, and F1 for each run
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0,1], zero_division=0
    )
    results["precisions"].append(p)
    results["recalls"].append(r)
    results["f1s"].append(f1)

    # Shows progress
    print(f"Completed run {run+1}/{n_runs}")

# Computes averaged results
print(f"\nAveraged Confusion Matrix of Simple DT Training and Testing")

# Calculates mean and standard deviation for confusion matrices across runs
if len(results["conf_matrices"]) == 0:
    print("No results collected.")
else:
    # Confusion matrices
    cms = np.array(results["conf_matrices"])
    avg_cm = cms.mean(axis=0)
    std_cm = cms.std(axis=0)
    print("Average Confusion Matrix:")
    print(avg_cm)
    print("Std of Confusion Matrix:")
    print(std_cm)

    # Computes mean and standard deviation of performance metrics across runs
    acc = np.mean(results["accuracies"])
    acc_std = np.std(results["accuracies"])
    bal = np.mean(results["balanced_accuracies"])
    bal_std = np.std(results["balanced_accuracies"])
    kap = np.mean(results["cohen_kappas"])
    kap_std = np.std(results["cohen_kappas"])
    auc = np.mean(results["aucs"])        # NEW
    auc_std = np.std(results["aucs"])     # NEW

    # Prints performance metrics with standard deviation
    print(f"Accuracy: {acc:.3f} ± {acc_std:.3f}")
    print(f"Balanced Accuracy: {bal:.3f} ± {bal_std:.3f}")
    print(f"Cohen Kappa: {kap:.3f} ± {kap_std:.3f}")
    print(f"AUC: {auc:.3f} ± {auc_std:.3f}")  # NEW

    # Computes mean and standard deviation of class level metrics across runs
    precision = np.mean(results["precisions"], axis=0)
    precision_std = np.std(results["precisions"], axis=0)
    recall = np.mean(results["recalls"], axis=0)
    recall_std = np.std(results["recalls"], axis=0)
    f1 = np.mean(results["f1s"], axis=0)
    f1_std = np.std(results["f1s"], axis=0)

    # Prints metrics
    print("\nClass-level metrics (0 = No Damage, 1 = Damage)")
    for i, cls in enumerate([0, 1]):
        print(
            f"Class {cls}: "
            f"Precision {precision[i]:.3f} ± {precision_std[i]:.3f}, "
            f"Recall {recall[i]:.3f} ± {recall_std[i]:.3f}, "
            f"F1 {f1[i]:.3f} ± {f1_std[i]:.3f}"
        )

# Completion note
print(f"\nCompleted evaluation of {n_runs} runs for {model_type}.")
