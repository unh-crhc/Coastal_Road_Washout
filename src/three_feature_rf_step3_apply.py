import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing, common in data science
from sklearn.linear_model import LogisticRegression # Logistic regression for platt scaling
from scipy.special import logit, expit # Convert between probabilities and log-odds
from sklearn.model_selection import train_test_split # Split data into training and testing sets

# Directory, shows where models, results, and data are stored
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "../models")
results_dir = os.path.join(script_dir, "../results")
data_dir = os.path.join(script_dir, "../data/processed")
os.makedirs(results_dir, exist_ok=True)

# Model inputs, target (damage), states applied to, and number of model runs
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min']
target_column = 'Damage_Status'
states = ["NH", "RI", "TX", "MS"]
n_models = 100 # Should match the number of trained models

# Loads the trained models
models = []
for run in range(n_models):
    path = os.path.join(models_dir, f"three_feature_rf_model_run_{run}.pkl")
    if os.path.exists(path):
        models.append(joblib.load(path))

# Error message if no trained models found
if not models:
    raise RuntimeError("No three-feature RF models found, train models")

# Loads the Maine data for calibration
me_df = pd.read_csv(os.path.join(data_dir, "ME", "processed.csv"))
me_df[target_column] = me_df[target_column].map({'No Damage': 0, 'Damage': 1})
me_df = me_df.dropna(subset=[target_column])
X_me, y_me = me_df[input_features], me_df[target_column] # Separates predictors and the target

# Separates Maine data for platt calibration 
# X_train and y_train are not used, they exist only to define the split
X_train, X_calib, y_train, y_calib = train_test_split(
    X_me, y_me, test_size=0.3, stratify=y_me, random_state=42
)

# Each model predicts probabilities on the calibration set, those probabilities are averaged across the ensemble to form the ensemble’s raw prediction
probs_calib = np.array([m.predict_proba(X_calib)[:,1] for m in models]).mean(axis=0)

# Fits a logistic regression that maps the ensemble’s raw probabilities to better-calibrated probabilities
platt = LogisticRegression()
platt.fit(probs_calib.reshape(-1,1), y_calib)

# Applies the calibrated models to each state
for state in states:

    # Loads each states data and extracts the predictors
    df_file = os.path.join(data_dir, state, "processed.csv")
    df = pd.read_csv(df_file)
    X = df[input_features]

    # Get per-model probabilities
    all_probs = np.array([m.predict_proba(X)[:,1] for m in models])

    # Aggregate ensemble predictions
    df['p_mean_raw'] = all_probs.mean(axis=0) # Mean probability 
    df['p_std'] = all_probs.std(axis=0) # Standard deviation
    df['p_vote'] = (all_probs > 0.5).mean(axis=0) # Fraction of models that vote damage (above probability of damage threhold of 0.5)

    # Puts raw ensemble probabilities through the Platt model to produce calibrated probabilities
    df['p_mean_calib'] = platt.predict_proba(df['p_mean_raw'].values.reshape(-1,1))[:,1]

    # Prevalence adjustment
    pi_train = 0.5 # Prevalanece in the training data 1:1
    pi_pop = 1/6 # Prevalence in the whole data set 1:5
    logit_p = logit(df['p_mean_calib'])
    df['p_mean_adj'] = expit(logit_p + (logit(pi_pop) - logit(pi_train))) # Adjusts probabilities using log-odds

    # Saves the results
    df.to_csv(os.path.join(results_dir, f"{state}_three_feature_predictions.csv"), index=False)
    np.save(os.path.join(results_dir, f"{state}_three_feature_all_probs.npy"), all_probs)

    # Completion note
    print(f"{state} predictions saved")
