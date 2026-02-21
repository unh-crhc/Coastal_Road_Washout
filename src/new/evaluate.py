import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 

# Load NH sample results
nh_df = pd.read_csv("../data/NH_roads_sample_predictions.csv")

# Change Damage_Status to numeric
nh_df['Damage_Status_num'] = nh_df['Damage_Status'].map({'No Damage': 0, 'Damage': 1})

input_features = [
    'aadt_type', 'priority', 'capacity', 'Z_Min', 'Distance_to_Coast_m',
    'Max_WVHT', 'Max_Dir', 'Avg_WVHT', 'Avg_Dir',
    'Max_WSPD', 'Max_WSPD_Dir', 'Avg_WSPD', 'Avg_WSPD_Dir',
    'Precipitation', 'Rel_Elev_Min', 'Max_Height', 'Return_Period'
]
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


# all_probs must match from Box 1 (stored in memory during same run)
# If running separately, you’d need to re-generate all_probs on the NH sample.
n_models = all_probs.shape[0]

# Loop through models to evaluate metrics
metrics_list = []
cms = []

for i in range(n_models):
    y_pred = (all_probs[i] > 0.5).astype(int)
    y_true = nh_df['Damage_Status_num'].values

    metrics_list.append({
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

metrics_df = pd.DataFrame(metrics_list)
metrics_summary = metrics_df.agg(['mean', 'std']).T

print("NH Ensemble Model Performance")
print(metrics_summary)

# Confusion matrix summary
cms_array = np.stack(cms)
cm_mean = cms_array.mean(axis=0)
cm_std = cms_array.std(axis=0)

print("\nMean Confusion Matrix:\n", cm_mean.round(2))
print("\nStd Confusion Matrix:\n", cm_std.round(2))
