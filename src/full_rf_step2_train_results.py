import os # interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing,
from sklearn.model_selection import train_test_split # Splits data sets into training and testing groups
from sklearn.metrics import ( # Imports for performance analysis
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, precision_recall_fscore_support
)

# Directory, for scripts and paths to models
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "../models")
data_file = os.path.join(script_dir, "../data/processed/ME/processed.csv")

# Load the data
df = pd.read_csv(data_file)

# Lists the input features and the target variable (Damage_Status)
input_features = [
    'aadt_type','priority','fedfunccls','capacity','jurisdictn','num_lanes',
    'st_urbrur','surfc_type','Shape_Leng','Z_Min','Z_Max','Z_Mean',
    'Min_Slope','Max_Slope','Avg_Slope','Max_Height',
    'Return_Period','Distance_to_Coast_m','Max_WVHT','Max_Dir',
    'Avg_WVHT','Avg_Dir','Max_WSPD','Max_WSPD_Dir','Avg_WSPD','Avg_WSPD_Dir',
    'Precipitation','Rel_Elev_Min','Rel_Elev_Mean','Rel_Elev_Max'
]
target_column = 'Damage_Status'

# Converts the target to binary, 0 for no damage and 1 for damage
df[target_column] = df[target_column].map({'No Damage': 0, 'Damage': 1})
df = df.dropna(subset=[target_column])

# Separates the data into damage and no damage subsets, this is done for controlled sampling of training and testing subsets
damage_df = df[df[target_column] == 1]
nodamage_df = df[df[target_column] == 0]

# Number of runs
n_runs = 100  # Must match the number of trained models

# Creates containers for results
results = {
    "conf_matrices": [],
    "accuracies": [],
    "balanced_accuracies": [],
    "cohen_kappas": [],
    "precisions": [],
    "recalls": [],
    "f1s": []
}

# Evaluates each trained model
for run in range(n_runs):
    
    # Load models
    model_path = os.path.join(models_dir, f"full_rf_model_run_{run}.pkl")
    model = joblib.load(model_path)

    # Recreates the exact balanced dataset used during training for this run
    nodamage_sample = nodamage_df.sample(n=len(damage_df), random_state=run)
    balanced_df = pd.concat([damage_df, nodamage_sample], axis=0)

    # Separates predictors and the target
    X = balanced_df[input_features]
    y = balanced_df[target_column]

    # Recreates the same train and test split used during model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=run
    )

    # Predictions
    y_pred = model.predict(X_test)

   # Store metrics
    results["conf_matrices"].append(confusion_matrix(y_test, y_pred, labels=[0,1]))
    results["accuracies"].append(accuracy_score(y_test, y_pred))
    results["balanced_accuracies"].append(balanced_accuracy_score(y_test, y_pred))
    results["cohen_kappas"].append(cohen_kappa_score(y_test, y_pred))

    # Class level metrics
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0,1], zero_division=0
    )

    # Saves class level metrics
    results["precisions"].append(p)
    results["recalls"].append(r)
    results["f1s"].append(f1)

# Calculates mean and standard deviation of performance metrics across all runs
print("\nFull RF Training and Testing Performance")

# Computes mean and standard deviation of confusion matrices
cms = np.array(results["conf_matrices"])
avg_cm = cms.mean(axis=0)
std_cm = cms.std(axis=0)

# Prints average confusion matrix
print("\nAverage Confusion Matrix:")
print(pd.DataFrame(avg_cm, index=["True No-Damage","True Damage"],
                           columns=["Pred No-Damage","Pred Damage"]))

# Prints standard deviation of average confusion matrix
print("\nConfusion Matrix Standard Deviation:")
print(pd.DataFrame(std_cm, index=["True No-Damage","True Damage"],
                           columns=["Pred No-Damage","Pred Damage"]))

# Aggregates summary metrics
acc = np.mean(results["accuracies"])
acc_std = np.std(results["accuracies"])
bal = np.mean(results["balanced_accuracies"])
bal_std = np.std(results["balanced_accuracies"])
kap = np.mean(results["cohen_kappas"])
kap_std = np.std(results["cohen_kappas"])

# Prints summary metrics with standard deviation
print(f"\nAccuracy: {acc:.3f} ± {acc_std:.3f}")
print(f"Balanced Accuracy: {bal:.3f} ± {bal_std:.3f}")
print(f"Cohen Kappa: {kap:.3f} ± {kap_std:.3f}")

# Class level metrics
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
print(f"\nCompleted evaluation of {n_runs} runs for Full RF")