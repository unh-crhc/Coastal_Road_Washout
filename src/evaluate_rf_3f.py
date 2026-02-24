import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing, common in data science
from sklearn.linear_model import LogisticRegression # Logistic regression for platt scaling
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

# Directory, shows where models, results, and data are stored
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "../data/processed")
model_dir = os.path.join(base_dir, "../models")
results_dir = os.path.join(base_dir, "../results")
os.makedirs(results_dir, exist_ok=True)

# Model inputs, target (damage), states applied to, and number of model runs
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min']
target_column = 'Damage_Status'
states = ["NH", "RI", "TX", "MS"]
threshold = 0.5
n_models = 100 # Should match the number of trained models
pi_train = 0.5 # Prevalanece in the training data 1:1
pi_pop = 1/6 # Prevalence in the whole data set 1:5

# Loads the trained models
models = []
for run in range(n_models):
    path = os.path.join(model_dir, f"three_feature_rf_model_run_{run}.pkl")
    if os.path.exists(path):
        models.append(joblib.load(path))

# Error message if no trained models found
if not models:
    raise RuntimeError("No three-feature RF models found, train models")

# Load the pre-saved calibration holdout that was never seen during model training
calib_path = os.path.join(
    base_dir,
    "../splits/full_rf_calibration_holdout.csv"
)
calib_df = pd.read_csv(calib_path)
calib_df = calib_df.dropna(subset=[target_column])

X_calib = calib_df[input_features]
y_calib = calib_df[target_column]

# Ensemble Probabilities for Calibration
probs_calib = np.array([
    m.predict_proba(X_calib)[:,1]
    for m in models
]).mean(axis=0)

# Platt scaling, learns a calibration curve mapping raw ensemble probabilities to calibrated probabilities
platt = LogisticRegression()
platt.fit(probs_calib.reshape(-1,1), y_calib)

# Evaluation Function
def eval_binary_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    kap = cohen_kappa_score(y_true, y_pred)
    return cm, acc, bal, kap

# Applies the calibrated models to each state
for state in states:

    print(f"Processing {state}...")

    # Loads each states data and extracts the predictors
    df_file = os.path.join(data_dir, state, "processed.csv")
    df = pd.read_csv(df_file)
    X = df[input_features]

    if "Damage_Status" not in df.columns:
        print(f"No Damage_Status column in {state} data, skipping evaluation.")
        continue

    # Convert ground truth: allow either already {0,1} or strings
    if df["Damage_Status"].dtype == object:
        y_true = df["Damage_Status"].map({"No Damage": 0, "Damage": 1}).astype(int).values
    else:
        y_true = df["Damage_Status"].astype(int).values

     # Predictor matrix
    X = df[input_features]

    # Per-model probabilities
    all_probs = np.array([  m.predict_proba(X)[:,1]
        for m in models
    ])

    n_models = all_probs.shape[0]

    # Aggregate ensemble predictions
    df['p_mean_raw'] = all_probs.mean(axis=0) # Mean probability 
    df['p_std'] = all_probs.std(axis=0) # Standard deviation
    df['p_vote'] = (all_probs > 0.5).mean(axis=0) # Fraction of models that vote damage (above probability of damage threhold of 0.5)

    # Puts raw ensemble probabilities through the Platt model to produce calibrated probabilities
    df["p_mean_calib"] = platt.predict_proba(
        df["p_mean_raw"].values.reshape(-1,1)
    )[:,1]

    # Prevalence adjustment
    if state == "TX":
        pi_pop_state = 59 / (59 + 16)
    else:
        pi_pop_state = pi_pop

    logit_p = logit(df["p_mean_calib"])

    df["p_mean_adj"] = expit(
        logit_p +
        (logit(pi_pop_state) - logit(pi_train))
    )

    # Performance metrics
    cms, accs, bals, kaps, aucs = [], [], [], [], []
    precs, recs, f1s = [], [], []

    for i in range(n_models):
        y_pred = (all_probs[i] >= threshold).astype(int)
        cm, acc, bal, kap = eval_binary_metrics(
            y_true,
            y_pred
        )
        auc = roc_auc_score(
            y_true,
            all_probs[i]
        )
        cms.append(cm)
        accs.append(acc)
        bals.append(bal)
        kaps.append(kap)
        aucs.append(auc)
        p,r,f1,_ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0,1],
            zero_division=0
        )
        precs.append(p)
        recs.append(r)
        f1s.append(f1)

    cms_array = np.stack(cms)
    accs = np.array(accs)
    bals = np.array(bals)
    kaps = np.array(kaps)
    aucs = np.array(aucs)
    precs = np.array(precs)
    recs = np.array(recs)
    f1s = np.array(f1s)

    # Print Results
    print(f"\n{state} Three-Feature RF Performance")
    print("\nMean Confusion Matrix:\n",
          cms_array.mean(axis=0).round(2))
    print("Std Confusion Matrix:\n",
          cms_array.std(axis=0).round(2))

    print(f"\nAccuracy: {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"Balanced Accuracy: {bals.mean():.3f} ± {bals.std():.3f}")
    print(f"Cohen's Kappa: {kaps.mean():.3f} ± {kaps.std():.3f}")
    print(f"AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")

    print(
        f"No Damage(0) Precision: "
        f"{precs[:,0].mean():.3f} ± {precs[:,0].std():.3f}")
    print(
        f"No Damage(0) Recall: "
        f"{recs[:,0].mean():.3f} ± {recs[:,0].std():.3f}" )
    print(
        f"No Damage(0) F1: "
        f"{f1s[:,0].mean():.3f} ± {f1s[:,0].std():.3f}")

    print(
        f"Damage(1) Precision: "
        f"{precs[:,1].mean():.3f} ± {precs[:,1].std():.3f}")

    print(
        f"Damage(1) Recall: "
        f"{recs[:,1].mean():.3f} ± {recs[:,1].std():.3f}")

    print(
        f"Damage(1) F1: "
        f"{f1s[:,1].mean():.3f} ± {f1s[:,1].std():.3f}")

    # Save Results
    out_csv = os.path.join(
        results_dir,
        f"{state}_three_feature_predictions.csv")

    df.to_csv(out_csv, index=False)

    print(f"\nSaved predictions to: {out_csv}")

print("\nApplication completed successfully")