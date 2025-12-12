import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Parallel computing, loads the models created in the previous cell
from sklearn.linear_model import LogisticRegression # Used for binary classification problems
from scipy.special import logit, expit  # Used to ge probabilities, one converts probabilities into log-odds, the other from log-odds to probabilities
from sklearn.model_selection import train_test_split # Splits data sets into training and testing groups

# Load Maine training data
me_df = pd.read_csv("../data/ME_roads_all_info_with_structure.csv")

input_features = [
    'aadt_type', 'priority', 'capacity', 'Z_Min', 'Distance_to_Coast_m',
    'Max_WVHT', 'Max_Dir', 'Avg_WVHT', 'Avg_Dir',
    'Max_WSPD', 'Max_WSPD_Dir', 'Avg_WSPD', 'Avg_WSPD_Dir',
    'Precipitation', 'Rel_Elev_Min', 'Max_Height', 'Return_Period'
]
target_column = 'Damage_Status'

me_df[target_column] = me_df[target_column].map({'No Damage': 0, 'Damage': 1})
me_df = me_df.dropna(subset=[target_column])

X_me = me_df[input_features]
y_me = me_df[target_column]

# Load NH sample (7 damage and 35 no damage)
nh_df = pd.read_csv("../data/NH_roads_damage_sample.csv")

# Align features, NH must use same input_features
nh_features = input_features
X_nh = nh_df[nh_features].copy()

# Load ensemble models (100 models)
n_models = 100
models = []
for run in range(n_models):
    try:
        model = joblib.load(f"../models/washout_model_run_{run}.pkl")
        models.append(model)
    except:
        print(f"Warning: model washout_model_run_{run}.pkl not found, skipping")

if len(models) == 0:
    raise RuntimeError("No ensemble models loaded. Make sure the washout_model_run_*.pkl files exist.")

# Apply ensemble models to NH sample
all_probs = np.array([m.predict_proba(X_nh)[:,1] for m in models])  # shape (n_models, n_roads)

# Aggregate ensemble predictions
nh_df['p_mean_raw'] = all_probs.mean(axis=0)
nh_df['p_std'] = all_probs.std(axis=0)
nh_df['p_vote'] = (all_probs > 0.5).mean(axis=0)

# Scale predictions (Platt scaling using Maine data)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_me, y_me, test_size=0.2, stratify=y_me, random_state=42
)

probs_calib = np.array([m.predict_proba(X_calib)[:,1] for m in models]).mean(axis=0)

platt = LogisticRegression()
platt.fit(probs_calib.reshape(-1,1), y_calib)

nh_df['p_mean_calib'] = platt.predict_proba(nh_df['p_mean_raw'].values.reshape(-1,1))[:,1]

# Prevalence adjustment
pi_train = 0.5      # training prevalence (1:1 damage to no damage)
pi_pop = 1/6   # 1 damage location and 5 no damage locations

logit_p = logit(nh_df['p_mean_calib'])
logit_p_adj = logit_p + (logit(pi_pop) - logit(pi_train))
nh_df['p_mean_adj'] = expit(logit_p_adj)

# Save results
nh_df.to_csv("../data/NH_roads_sample_predictions.csv", index=False)
print("Saved to NH_roads_sample_predictions.csv (sample only)")
