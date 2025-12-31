import os # interact with operating system
import numpy as np # Used for numerical and scientific computing
import pandas as pd # Package for csvs and data
from sklearn.metrics import ( # Imports for performance analysis
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, precision_recall_fscore_support
)

# Establishes paths for input data and prediction results
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, "../results")
data_dir = os.path.join(script_dir, "../data/processed")
states = ["ME", "NH", "RI"]
threshold = 0.5  # Threshold applied to probabilities

# Evaluate performance for each state
for state in states:
    pred_file = os.path.join(results_dir, f"{state}_full_rf_predictions.csv")
    all_probs_file = os.path.join(results_dir, f"{state}_full_rf_all_probs.npy")
    orig_file = os.path.join(data_dir, state, "processed.csv")

    # Gives a message if expected files do not exist
    if not os.path.exists(pred_file) or not os.path.exists(all_probs_file) or not os.path.exists(orig_file):
        print(f"Missing files for {state}, skipping")
        continue

    # Loads original data to get true targets
    orig_df = pd.read_csv(orig_file)
    if 'Damage_Status' not in orig_df.columns:
        print(f"No Damage_Status column in original {state} data, skipping") # Shows if can't find
        continue

    # Converts to binary, 0 for no damage and 1 for damage
    y_true = orig_df['Damage_Status'].map({'No Damage': 0, 'Damage': 1}).astype(int).values

    # Loads per-model probabilities
    all_probs = np.load(all_probs_file)

    # Ensure the matrix is the correct shape
    if all_probs.shape[1] != len(y_true):
        all_probs = all_probs.T

    # Number of trained models 
    n_models = all_probs.shape[0]

    # Prepares lists to store metrics
    cms, accs, bals, kaps, precs, recs, f1s = [], [], [], [], [], [], []

    # Evaluates each model
    for i in range(n_models):
        y_pred = (all_probs[i] >= threshold).astype(int)

        # Computes confusion matrix and performance metrics
        cms.append(confusion_matrix(y_true, y_pred))
        accs.append(accuracy_score(y_true, y_pred))
        bals.append(balanced_accuracy_score(y_true, y_pred))
        kaps.append(cohen_kappa_score(y_true, y_pred))

        # These are the class level performance metrics (precision, recall, F1)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )
        # Stores the class level performance metrics 
        precs.append(p)
        recs.append(r)
        f1s.append(f1)

    # Converts lists to arrays
    cms_array = np.stack(cms)
    accs = np.array(accs)
    bals = np.array(bals)
    kaps = np.array(kaps)
    precs = np.array(precs)
    recs = np.array(recs)
    f1s = np.array(f1s)

    # Print performance metrics
    print(f"\n{state} Full RF Performance")
    print("\nMean Confusion Matrix:\n", cms_array.mean(axis=0).round(2))
    print("Std Confusion Matrix:\n", cms_array.std(axis=0).round(2))
    print(f"\nAccuracy: {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"Balanced Accuracy: {bals.mean():.3f} ± {bals.std():.3f}")
    print(f"Cohen's Kappa: {kaps.mean():.3f} ± {kaps.std():.3f}")

    # Goes through each class and shows class specific performance
    for i, cls in enumerate(["No Damage", "Damage"]):
        print(
            f"{cls} → Precision {precs[:, i].mean():.3f} ± {precs[:, i].std():.3f}, "
            f"Recall {recs[:, i].mean():.3f} ± {recs[:, i].std():.3f}, "
            f"F1 {f1s[:, i].mean():.3f} ± {f1s[:, i].std():.3f}"
        )
