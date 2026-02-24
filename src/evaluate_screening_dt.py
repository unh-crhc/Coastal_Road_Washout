import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing

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

# Directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data/processed")

# Screening decision tree structure
def screening_tree_predict(df):

    y_pred = []

    for _, row in df.iterrows():
        if row["Distance_to_Coast_m"] <= 48.06:
            if row["Rel_Elev_Min"] <= 1.71:
                if row["Z_Min"] <= 3.03:
                    y_pred.append(1)
                else:
                    y_pred.append(1)
            else:
                if row["Z_Min"] <= 3.03:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
        else:
            if row["Rel_Elev_Min"] <= 1.71:
                if row["Z_Min"] <= 3.03:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            else:
                y_pred.append(0)

    y_pred = np.array(y_pred)
    y_prob = y_pred.astype(float)

    return y_pred, y_prob

# States to apply to
states = ["ME", "NH", "RI", "TX", "MS"]

# Storage for overall metrics
all_y_true = []
all_y_pred = []
all_y_prob = []

# Evaluate each state

for state in states:

    print(f"\n{state} Screening Tree Performance\n")

    data_file = os.path.join(data_dir, state, "processed.csv")
    df = pd.read_csv(data_file)

    required_cols = [
        "Distance_to_Coast_m",
        "Rel_Elev_Min",
        "Z_Min",
        "Damage_Status"
    ]

    df = df.dropna(subset=required_cols)
    df = df[df["Damage_Status"].isin(["Damage", "No Damage"])]

    y_true = df["Damage_Status"].map({"No Damage": 0, "Damage": 1}).values
    y_pred, y_prob = screening_tree_predict(df)

    # Store for overall performance
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    all_y_prob.extend(y_prob)

    # Compute metrics
    cm = confusion_matrix(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    precision_no = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_no = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_no = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    precision_d = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_d = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_d = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Print performance metrics and confusion matrices
    print("Confusion Matrix:")
    print(cm)

    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    print(f"Cohen's Kappa: {kappa:.3f}")
    print(f"AUC: {auc:.3f}")

    print(
        f"\nNo Damage → Precision {precision_no:.3f}, "
        f"Recall {recall_no:.3f}, "
        f"F1 {f1_no:.3f}"
    )

    print(
        f"Damage → Precision {precision_d:.3f}, "
        f"Recall {recall_d:.3f}, "
        f"F1 {f1_d:.3f}"
    )

# Overall Combined Performance
print("\n Overall Performance\n")
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_prob = np.array(all_y_prob)

cm = confusion_matrix(all_y_true, all_y_pred)

accuracy = accuracy_score(all_y_true, all_y_pred)
balanced_acc = balanced_accuracy_score(all_y_true, all_y_pred)
kappa = cohen_kappa_score(all_y_true, all_y_pred)
auc = roc_auc_score(all_y_true, all_y_prob)

precision_no = precision_score(all_y_true, all_y_pred, pos_label=0, zero_division=0)
recall_no = recall_score(all_y_true, all_y_pred, pos_label=0, zero_division=0)
f1_no = f1_score(all_y_true, all_y_pred, pos_label=0, zero_division=0)

precision_d = precision_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)
recall_d = recall_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)
f1_d = f1_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)

print("Confusion Matrix:")
print(cm)

print(f"\nAccuracy: {accuracy:.3f}")
print(f"Balanced Accuracy: {balanced_acc:.3f}")
print(f"Cohen's Kappa: {kappa:.3f}")
print(f"AUC: {auc:.3f}")

print(
    f"\nNo Damage → Precision {precision_no:.3f}, "
    f"Recall {recall_no:.3f}, "
    f"F1 {f1_no:.3f}"
)

print(
    f"Damage → Precision {precision_d:.3f}, "
    f"Recall {recall_d:.3f}, "
    f"F1 {f1_d:.3f}"
)