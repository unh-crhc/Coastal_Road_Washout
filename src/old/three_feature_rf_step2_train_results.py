import os # Interact with operating system
import joblib # Used for parallel computing
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
from sklearn.metrics import ( # Imports for performance analysis
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, precision_recall_fscore_support
)

# Directory, provides paths to the models and splits
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(script_dir, "../models"))
splits_dir = os.path.abspath(os.path.join(script_dir, "../splits"))
n_runs = 100 # Number of runs should match the number of trained models
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min'] # Input features should match

# Naming for reference
model_type = "three_feature_rf"

# Creates containers for the results
results = {
    "conf_matrices": [],
    "accuracies": [],
    "balanced_accuracies": [],
    "cohen_kappas": [],
    "precisions": [],
    "recalls": [],
    "f1s": []
}

# Evaluates each of the 100 trained models
for run in range(n_runs):
    # Load pre-saved test splits
    X_test_path = os.path.join(splits_dir, f"X_test_run_{run}.csv")
    y_test_path = os.path.join(splits_dir, f"y_test_run_{run}.csv")

    # If files are missing, skips run and gives error message
    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print(f"Warning: Missing test split for run {run}. Skipping. Train full RF to establish splits")
        continue

    # Loads test features and keeps only the 3 input features, also loads the test labels (damage or no damage)
    X_test = pd.read_csv(X_test_path)[input_features].apply(pd.to_numeric, errors='coerce')
    y_test = pd.read_csv(y_test_path).squeeze()

    # Loads the trained models
    model_file = os.path.join(models_dir, f"{model_type}_model_run_{run}.pkl")
    if not os.path.exists(model_file):
        print(f"Warning: {model_type} model for run {run} missing. Skipping.") # Error message if no models are found, skips if no model found
        continue

    # Loads model pipeline
    pipeline = joblib.load(model_file)

    # Runs models and gets outputs
    y_pred = pipeline.predict(X_test)

    # Compute performance metrics
    results["conf_matrices"].append(confusion_matrix(y_test, y_pred, labels=[0,1]))
    results["accuracies"].append(accuracy_score(y_test, y_pred))
    results["balanced_accuracies"].append(balanced_accuracy_score(y_test, y_pred))
    results["cohen_kappas"].append(cohen_kappa_score(y_test, y_pred))

    # Computes class level performance metrics, precision, recall, and F1
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0,1], zero_division=0
    )

    # Stores class level performance metrics
    results["precisions"].append(p)
    results["recalls"].append(r)
    results["f1s"].append(f1)

    # Prints a completion note for each run, progress tracking
    print(f"Completed run {run+1}/{n_runs}")

# Aggregates results
print(f"\n{model_type.upper()}")

# Computes mean and standard deviation of the confusion matrices across runs
if len(results["conf_matrices"]) == 0:
    print("No results collected.") # Error message
else:
    cms = np.array(results["conf_matrices"])
    avg_cm = cms.mean(axis=0)
    std_cm = cms.std(axis=0)

    # Prints average confusion matrices and standard deviations
    print("Average Confusion Matrix:")
    print(avg_cm)
    print("Std of Confusion Matrix:")
    print(std_cm)

    # Computes summary metrics with standard deviation
    acc = np.mean(results["accuracies"])
    acc_std = np.std(results["accuracies"])
    bal = np.mean(results["balanced_accuracies"])
    bal_std = np.std(results["balanced_accuracies"])
    kap = np.mean(results["cohen_kappas"])
    kap_std = np.std(results["cohen_kappas"])

    # Prints summary statistics 
    print(f"Accuracy: {acc:.3f} ± {acc_std:.3f}")
    print(f"Balanced Accuracy: {bal:.3f} ± {bal_std:.3f}")
    print(f"Cohen Kappa: {kap:.3f} ± {kap_std:.3f}")

    # Computes class-level metrics
    precision = np.mean(results["precisions"], axis=0)
    precision_std = np.std(results["precisions"], axis=0)
    recall = np.mean(results["recalls"], axis=0)
    recall_std = np.std(results["recalls"], axis=0)
    f1 = np.mean(results["f1s"], axis=0)
    f1_std = np.std(results["f1s"], axis=0)

    # Prints class level metrics
    print("\nClass-level metrics (0 = No Damage, 1 = Damage)")
    for i, cls in enumerate([0, 1]):
        print(
            f"Class {cls}: "
            f"Precision {precision[i]:.3f} ± {precision_std[i]:.3f}, "
            f"Recall {recall[i]:.3f} ± {recall_std[i]:.3f}, "
            f"F1 {f1[i]:.3f} ± {f1_std[i]:.3f}"
        )

# Completion note
print(f"\nCompleted evaluation of {n_runs} runs for {model_type}")
