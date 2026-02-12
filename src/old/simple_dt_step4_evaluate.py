import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
from sklearn.metrics import ( # Imports for performance analysis
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

# Directory, provides paths to data and folders
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, "../results")
states = ["ME", "NH", "RI"] # States to evaluate

# Loops through each state to evaluate predictions
for state in states:
    print(f"\nEvaluating state: {state}") # Progress note

    # Loads predictions and calibrated probabilities
    df_file = os.path.join(results_dir, f"{state}_simple_dt_predictions.csv")
    calib_file = os.path.join(results_dir, f"{state}_p_mean_calib_simple_dt.npy")
    all_probs_file = os.path.join(results_dir, f"{state}_all_probs_simple_dt.npy")

    # Loads files
    df = pd.read_csv(df_file)
    all_probs = np.load(all_probs_file)
    p_mean_calib = np.load(calib_file)
    n_models = all_probs.shape[0]

    # Converts damage status to binary, 0 for no damage and 1 for damage
    df['Damage_Status_num'] = df['Damage_Status'].map({'No Damage': 0, 'Damage': 1})

    # Creates containers for peformance metrics and fonsuion matrices
    metrics_list = []
    cms = []

    # Evaluates each model
    for i in range(n_models):
        y_pred = (all_probs[i] > 0.5).astype(int) # Converts raw probabilities to binary based on the damage threshold (0.5)
        y_true = df['Damage_Status_num'].values
        metrics_list.append({ # list of performance metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'precision_no_damage': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            'recall_no_damage': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'f1_no_damage': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
            'precision_damage': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            'recall_damage': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'f1_damage': f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        })
        cms.append(confusion_matrix(y_true, y_pred))

    # Aggregates metrics across the models
    metrics_df = pd.DataFrame(metrics_list)
    metrics_summary = metrics_df.agg(['mean', 'std']).T

    # Prints performance metrics
    print(f"\n{state} Simple Decision Tree Performance Metrics")
    print(metrics_summary)

    # Aggregates confusion matrices
    cms_array = np.stack(cms)
    cm_mean = cms_array.mean(axis=0)
    cm_std = cms_array.std(axis=0)

    # Prints mean confusion matrix and standard deviation
    print(f"\n{state} Mean Confusion Matrix:\n", cm_mean.round(2))
    print(f"{state} Std Confusion Matrix:\n", cm_std.round(2))
