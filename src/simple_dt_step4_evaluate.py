import os # Interact with operating system
import numpy as np # Used for numerical and scientific computing
import pandas as pd # Package for csvs and data
from sklearn.metrics import ( # Imports for performance analysis
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)

# Directory, provides paths to data and folders
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, "../results")
states = ["NH", "RI", "TX", "MS"] # States to evaluate


# Loops through each state
for state in states:

    df_file = os.path.join(results_dir, f"{state}_simple_dt_predictions.csv")
    all_probs_file = os.path.join(results_dir, f"{state}_all_probs_simple_dt.npy")

    # Error message if prediction files are missing
    if not os.path.exists(df_file) or not os.path.exists(all_probs_file):
        print(f"No predictions for {state}, skipping")
        continue

    # Loads true damage labels
    df = pd.read_csv(df_file)
    y_true = df['Damage_Status'].map({'No Damage': 0, 'Damage': 1}).values

    # Loads per-model probabilities
    all_probs = np.load(all_probs_file)
    n_models = all_probs.shape[0]

    # Creates containers to hold performance metrics
    cms, accs, bals, kaps, precs, recs, f1s, aucs = [], [], [], [], [], [], [], []


    # Evaluates performance for each model
    for i in range(n_models):

        y_pred = (all_probs[i] > 0.5).astype(int)

        # Computes matrix and performance metrics
        cms.append(confusion_matrix(y_true, y_pred))

        accs.append(accuracy_score(y_true, y_pred))
        bals.append(balanced_accuracy_score(y_true, y_pred))
        kaps.append(cohen_kappa_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, all_probs[i]))

        # Class level performance metrics
        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0,1],
            zero_division=0
        )

        precs.append(p)
        recs.append(r)
        f1s.append(f1)


    # Converts metrics into arrays for aggregation
    cms_array = np.stack(cms)

    accs = np.array(accs)
    bals = np.array(bals)
    kaps = np.array(kaps)

    precs = np.array(precs)
    recs = np.array(recs)
    f1s = np.array(f1s)

    aucs = np.array(aucs)


    # Print metrics (IDENTICAL FORMAT)
    print(f"\n{state} Simple Decision Tree")

    print("\nMean Confusion Matrix:\n",
          cms_array.mean(axis=0).round(2))

    print("Std Confusion Matrix:\n",
          cms_array.std(axis=0).round(2))


    print(f"\nAccuracy: {accs.mean():.3f} ± {accs.std():.3f}")

    print(f"Balanced Accuracy: {bals.mean():.3f} ± {bals.std():.3f}")

    print(f"Cohen's Kappa: {kaps.mean():.3f} ± {kaps.std():.3f}")

    print(f"AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")


    # Prints class level performance
    for i, cls in enumerate(["No Damage", "Damage"]):

        print(
            f"{cls} → Precision {precs[:,i].mean():.3f} ± {precs[:,i].std():.3f}, "
            f"Recall {recs[:,i].mean():.3f} ± {recs[:,i].std():.3f}, "
            f"F1 {f1s[:,i].mean():.3f} ± {f1s[:,i].std():.3f}"
        )