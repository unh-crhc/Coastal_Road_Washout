import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing
from sklearn.linear_model import LogisticRegression # Used for platt scaling and probability calibration
from scipy.special import logit, expit # Convert between probabilities and log-odds
from sklearn.model_selection import train_test_split # Split data into training and testing sets
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    f1_score,
    roc_auc_score,
)

def align_schema(df, required_features):
    """Add missing columns from training, filled with np.nan."""
    for col in required_features:
        if col not in df.columns:
            df[col] = np.nan
    return df[required_features]

# Directory, paths to folders containg models, results, and data
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "../data/processed")
model_dir = os.path.join(base_dir, "../models")
results_dir = os.path.join(base_dir, "../results")
os.makedirs(results_dir, exist_ok=True)

# Input features, target (damage), states evaluated, and the number of models
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min']
target_column = 'Damage_Status'
states = ["NH", "RI", "TX", "MS"]
threshold = 0.5
n_models = 100 # Must match training
pi_train = 0.5
pi_pop = 1/6

# Loads each model 
models = []
for run in range(n_models):
    path = os.path.join(model_dir, f"simple_dt_model_run_{run}.pkl")
    if os.path.exists(path):
        models.append(joblib.load(path))

# Error message if no models are found
if not models:
    raise RuntimeError("No trained simple decision tree models were found, train models")

# Load the pre-saved calibration holdout that was never seen during model training
calib_path = os.path.join(
    base_dir,
    "../splits/full_rf_calibration_holdout.csv"
)

calib_df = pd.read_csv(calib_path)
calib_df = calib_df.dropna(subset=[target_column])

X_calib = calib_df[input_features]
y_calib = calib_df[target_column]

# Ensemble calibration probabilities
probs_calib = np.array([
    m.predict_proba(X_calib)[:,1]
    for m in models
]).mean(axis=0)

# Platt scaling
platt = LogisticRegression()
platt.fit(
    probs_calib.reshape(-1,1),
    y_calib
)

# Applies the models to each state
for state in states:
    print(f"Processing {state}...")

    df_file = os.path.join(
        data_dir,
        state,
        "processed.csv")
    df = pd.read_csv(df_file)
    df[target_column] = df[target_column].map(
        {'No Damage': 0, 'Damage': 1} )
    df = df.dropna(subset=[target_column])

    y_true = df[target_column].values
    X = align_schema(
        df,
        input_features )

    # Per-model probabilities
    all_probs = np.array([
        m.predict_proba(X)[:,1]
        for m in models
    ])

    # Ensemble stats
    df['p_mean_raw'] = all_probs.mean(axis=0)
    df['p_std'] = all_probs.std(axis=0)
    df['p_vote'] = (all_probs > threshold).mean(axis=0)

    # Calibrated probabilities
    df['p_mean_calib'] = platt.predict_proba(
        df['p_mean_raw'].values.reshape(-1,1)
    )[:,1]

    # Prevalence adjustment
    if state == "TX":
        pi_pop_state = 59 / (59 + 16)
    else:
        pi_pop_state = pi_pop

    logit_p = logit(
        df['p_mean_calib'])

    df['p_mean_adj'] = expit(
        logit_p +
        (logit(pi_pop_state) - logit(pi_train)))

    # Save predictions
    df.to_csv(
        os.path.join(
            results_dir,
            f"{state}_simple_dt_predictions.csv"
        ),
        index=False
    )

    np.save(
        os.path.join(
            results_dir,
            f"{state}_all_probs_simple_dt.npy"
        ),
        all_probs
    )

    print(f"{state} predictions saved")

    # Performance Metrics
    metrics_list = []
    cms = []

    for i in range(len(models)):

        y_pred = (
            all_probs[i] >= threshold
        ).astype(int)

        metrics_list.append({

            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, all_probs[i]),
            'precision_no_damage': precision_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0),
            'recall_no_damage': recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0),
            'f1_no_damage': f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0),
            'precision_damage': precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0),
            'recall_damage': recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0),
            'f1_damage':f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0)
        })

        cms.append(
            confusion_matrix(y_true, y_pred)
        )

    # Aggregate metrics
    metrics_df = pd.DataFrame(
        metrics_list
    )

    metrics_summary = metrics_df.agg(
        ['mean','std']
    ).T

    print(f"\n{state} Simple Decision Tree Performance Metrics")

    print(metrics_summary)

    # Confusion matrices
    cms_array = np.stack(cms)

    cm_mean = cms_array.mean(axis=0)
    cm_std = cms_array.std(axis=0)

    print(
        f"\n{state} Mean Confusion Matrix:\n",
        cm_mean.round(2)
    )

    print(
        f"{state} Std Confusion Matrix:\n",
        cm_std.round(2)
    )

print("\nApplication completed successfully")


