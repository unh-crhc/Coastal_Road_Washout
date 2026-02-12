''' 
predict the damage status for each state using the trained models, and
evaluates performance on Maine, New Hampshire and Rhode Island data.  
'''
import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing
from sklearn.linear_model import LogisticRegression # Used for platt scaling and probability calibration
from scipy.special import logit, expit # Convert between probabilities and log-odds
from sklearn.model_selection import train_test_split # Split data into training and testing sets
from sklearn.metrics import ( # Imports for performance analysis
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, confusion_matrix
)


def align_schema(df, required_features):
    """Add missing columns from training, filled with np.nan."""
    for col in required_features:
        if col not in df.columns:
            df[col] = np.nan
    return df[required_features]

# Directory, paths to folders containg models, results, and data
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "../models")
results_dir = os.path.join(script_dir, "../results")
data_dir = os.path.join(script_dir, "../data/processed")
os.makedirs(results_dir, exist_ok=True)

# Input features, target (damage), states evaluated, and the number of models
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min']
target_column = 'Damage_Status'
states = ["ME", "NH", "RI"]
n_models = 100 # Must match training
pi_train = 0.5
pi_pop = 1/6

# Loads each model 
models = []
for run in range(n_models):
    path = os.path.join(models_dir, f"simple_dt_model_run_{run}.pkl")
    if os.path.exists(path):
        models.append(joblib.load(path))

# Error message if no models are found
if not models:
    raise RuntimeError("No trained simple decision tree models were found, train models")

# Load the pre-saved calibration holdout that was never seen during model training
calib_path = os.path.join(script_dir, "../splits/full_rf_calibration_holdout.csv")
calib_df = pd.read_csv(calib_path)
# calib_df['Damage_Status'] = calib_df['Damage_Status'].map({'No Damage': 0, 'Damage': 1})
calib_df = calib_df.dropna(subset=['Damage_Status'])

# Aligns the data, prepares features and target (damage)
X_calib = align_schema(calib_df, input_features)
y_calib = calib_df['Damage_Status']

# Computes probabilities on calibration set, mean probability across models
probs_calib = np.array([m.predict_proba(X_calib)[:,1] for m in models]).mean(axis=0)

# Platt scaling, calibrates probabilities to better match observed outcomes, fits a logistic regression on ensemble mean probabilities versus observed labels
platt = LogisticRegression()
platt.fit(probs_calib.reshape(-1,1), y_calib)

# Applies the models to each state
for state in states:
    df_file = os.path.join(data_dir, state, "processed.csv") # Loops through each state
    df = pd.read_csv(df_file)
    df[target_column] = df[target_column].map({'No Damage': 0, 'Damage': 1}) # Turns Damage_Status to binary
    df = df.dropna(subset=[target_column])
    X = df[input_features].copy()  # ensure clean copy

    # Computes the ensemble predictions
    all_probs = np.array([m.predict_proba(X)[:,1] for m in models])
    df['p_mean_raw'] = all_probs.mean(axis=0) # Mean probability across runs
    df['p_std'] = all_probs.std(axis=0) # Standard deviation across runs
    df['p_vote'] = (all_probs > 0.5).mean(axis=0) # Fraction of models predicting > 0.5 (damage threshold)

    # Calibrated probabilities using Platt scaling
    df['p_mean_calib'] = platt.predict_proba(df['p_mean_raw'].values.reshape(-1,1))[:,1]

    # Prevalence adjustment, Adjusts probabilities to account for difference in damage prevalence between training (1/2) and the data set population (1/6)
    logit_p = logit(df['p_mean_calib'])
    df['p_mean_adj'] = expit(logit_p + (logit(pi_pop) - logit(pi_train))) # Uses log odd adjustments

    # Saves the results
    df.to_csv(os.path.join(results_dir, f"{state}_simple_dt_predictions.csv"), index=False)
    np.save(os.path.join(results_dir, f"{state}_all_probs_simple_dt.npy"), all_probs)
    np.save(os.path.join(results_dir, f"{state}_p_mean_calib_simple_dt.npy"), df['p_mean_calib'].values)

    # Completion note
    print(f"{state} predictions saved")

    # Creates containers for peformance metrics and fonsuion matrices
    metrics_list = []
    cms = []

    # Evaluates each model
    for i in range(n_models):
        y_pred = (all_probs[i] > 0.5).astype(int) # Converts raw probabilities to binary based on the damage threshold (0.5)
        y_true = df['Damage_Status'].values
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


