import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing
from sklearn.linear_model import LogisticRegression # Used for platt scaling and probability calibration
from scipy.special import logit, expit # Convert between probabilities and log-odds
from sklearn.model_selection import train_test_split # Split data into training and testing sets

# Directory, paths to folders containg models, results, and data
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "../models")
results_dir = os.path.join(script_dir, "../results")
data_dir = os.path.join(script_dir, "../data/processed")
os.makedirs(results_dir, exist_ok=True)

# Input features, target (damage), states evaluated, and the number of models
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min']
target_column = 'Damage_Status'
states = ["NH", "RI", "TX", "MS"]
n_models = 100 # Must match training

# Loads each model 
models = []
for run in range(n_models):
    path = os.path.join(models_dir, f"simple_dt_model_run_{run}.pkl")
    if os.path.exists(path):
        models.append(joblib.load(path))

# Error message if no models are found
if not models:
    raise RuntimeError("No trained simple decision tree models were found, train models")

# Loads Maine data for calibration
me_file = os.path.join(data_dir, "ME", "processed.csv")
me_df = pd.read_csv(me_file)
me_df[target_column] = me_df[target_column].map({'No Damage': 0, 'Damage': 1}) # Turns Damage_Status to binary
me_df = me_df.dropna(subset=[target_column]) # Drops rows where damage status is missing, shouldn't be an issue for this dataset
X_me, y_me = me_df[input_features], me_df[target_column] # Splits data into features and target 

# Separates data for platt calibration
# X_train and y_train are not used at all, they exist only to define the split
X_train, X_calib, y_train, y_calib = train_test_split(
    X_me, y_me, test_size=0.3, stratify=y_me, random_state=42
)

# Computes probabilities on calibration set, mean probability across models
probs_calib = np.array([m.predict_proba(X_calib)[:,1] for m in models]).mean(axis=0)

# Platt scaling, calibrates probabilities to better match observed outcomes, fits a logistic regression on ensemble mean probabilities versus observed labels
platt = LogisticRegression()
platt.fit(probs_calib.reshape(-1,1), y_calib)

# Applies the models to each state
for state in states:
    df_file = os.path.join(data_dir, state, "processed.csv") # Loops through each state
    df = pd.read_csv(df_file)
    X = df[input_features].copy()  # ensure clean copy

    # Computes the ensemble predictions
    all_probs = np.array([m.predict_proba(X)[:,1] for m in models])
    df['p_mean_raw'] = all_probs.mean(axis=0) # Mean probability across runs
    df['p_std'] = all_probs.std(axis=0) # Standard deviation across runs
    df['p_vote'] = (all_probs > 0.5).mean(axis=0) # Fraction of models predicting > 0.5 (damage threshold)

    # Calibrated probabilities using Platt scaling
    df['p_mean_calib'] = platt.predict_proba(df['p_mean_raw'].values.reshape(-1,1))[:,1]

    # Prevalence adjustment, Adjusts probabilities to account for difference in damage prevalence between training (1/2) and the data set population (1/6)
    pi_train = 0.5
    pi_pop = 1/6
    logit_p = logit(df['p_mean_calib'])
    df['p_mean_adj'] = expit(logit_p + (logit(pi_pop) - logit(pi_train))) # Uses log odd adjustments

    # Saves the results
    df.to_csv(os.path.join(results_dir, f"{state}_simple_dt_predictions.csv"), index=False)
    np.save(os.path.join(results_dir, f"{state}_all_probs_simple_dt.npy"), all_probs)
    np.save(os.path.join(results_dir, f"{state}_p_mean_calib_simple_dt.npy"), df['p_mean_calib'].values)

    # Completion note
    print(f"{state} predictions saved")
