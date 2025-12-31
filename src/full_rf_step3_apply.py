import os  # Interact with operating system
import pandas as pd  # Package for csvs and data
import numpy as np  # Used for numerical and scientific computing
import joblib  # Used for parallel computing
from sklearn.linear_model import LogisticRegression # Logistic regression for platt scaling
from scipy.special import logit, expit # logit converts probabilities to log-odds abd expit converts log odds to probabilities
from sklearn.model_selection import train_test_split # Creates training and testing splits

# Directory, establish paath to data, models, and results
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "../data/processed")
model_dir = os.path.join(base_dir, "../models")
results_dir = os.path.join(base_dir, "../results")
os.makedirs(results_dir, exist_ok=True) # creates a results directory if one doesn't exist

# Defines which data sets will be used
states = ["ME", "NH", "RI"]

# Loads the trained model runs, goes through each one, creates an empty list to hold pipelines
models = []
for f in os.listdir(model_dir):
    if f.endswith(".pkl") and f.startswith("full_rf_model_run_"):
        models.append(joblib.load(os.path.join(model_dir, f)))

# Prints a message if no models have been trianed, or they can not be found in the directory
if not models:
    raise RuntimeError("No trained full RF models found.")

# Selects trained pipelines and extracts the preprocessing component of the pipeline
pipeline0 = models[0]
preprocessor = pipeline0.named_steps["preprocessor"]

# Retrieves the list of numerical and categorical feature names used during training
num_features = preprocessor.transformers_[0][2]
cat_features = preprocessor.transformers_[1][2]
input_features = list(num_features) + list(cat_features)

# Ensures that new data sets match the models training feauture structure
# Adds missing columns and fills them with NaNs, which are then handeled by the preprocessing pipeline
# Just makes sure that data looks the way the models expect
def align_schema(df, required_features):
    """Add missing columns from training, filled with np.nan."""
    for col in required_features:
        if col not in df.columns:
            df[col] = np.nan
    return df[required_features]

# Loads the ME dataset, converts damage statust to binary, 
me_df = pd.read_csv(os.path.join(data_dir, "ME", "processed.csv"))
me_df['Damage_Status'] = me_df['Damage_Status'].map({'No Damage': 0, 'Damage': 1})
me_df = me_df.dropna(subset=['Damage_Status'])

# Aligns the data, prepares features and targe (damage)
X_me = align_schema(me_df, input_features)
y_me = me_df['Damage_Status']

# Split Maine for platt calibration, this is different than the training and testing splits
# X_train and y_train are not used at all, they exist only to define the split
# Right now this only uses 30% of the data for platt scaling
X_train, X_calib, y_train, y_calib = train_test_split(
    X_me, y_me, test_size=0.3, stratify=y_me, random_state=42
)

# Gets per-model probabilities and averages them to form ensemble probabilities
probs_calib = np.array([m.predict_proba(X_calib)[:,1] for m in models]).mean(axis=0)

# Platt scaling, learns a calibration curve mapping raw ensemble probabilities to calibrated probabilities
platt = LogisticRegression()
platt.fit(probs_calib.reshape(-1,1), y_calib)

# Processess and applies to each state
for state in states:
    print(f"Processing {state}...") # Shows which state is being worked on

    # Loads the data
    df_file = os.path.join(data_dir, state, "processed.csv")
    df = pd.read_csv(df_file)

    # Align columns to training data
    X = align_schema(df, input_features)

    # Collects the model probabilities
    all_probs = np.array([m.predict_proba(X)[:,1] for m in models])

    # Computes the ensemble statistics
    df["p_mean_raw"] = all_probs.mean(axis=0)
    df["p_std"] = all_probs.std(axis=0)
    df["p_vote"] = (all_probs > 0.5).mean(axis=0)

    # Applies platt scaling to probabilities
    df["p_mean_calib"] = platt.predict_proba(df["p_mean_raw"].values.reshape(-1,1))[:,1]

    # Prevalence adjustment, for these data sets there are 5 no damage locations for every 1 damage location
    pi_train = 0.5
    pi_pop = 1/6
    logit_p = logit(df["p_mean_calib"])
    df["p_mean_adj"] = expit(logit_p + (logit(pi_pop) - logit(pi_train)))

    # Save results
    df.to_csv(os.path.join(results_dir, f"{state}_full_rf_predictions.csv"), index=False)
    np.save(os.path.join(results_dir, f"{state}_full_rf_all_probs.npy"), all_probs)

    # Shows when each state is done
    print(f"{state} predictions saved")

# Shows when entire script is done
print("\nApplication completed successfully")
