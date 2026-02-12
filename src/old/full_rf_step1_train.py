import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing, common in data science
from sklearn.compose import ColumnTransformer # Scikit-learn (sklearn) is a machine learning library, ColumnTransformer lets you apply different preprocessing steps to different types of columns
from sklearn.pipeline import Pipeline # Chains model steps into one workflow, described as an assembly line
from sklearn.preprocessing import OneHotEncoder # Converts categorical variables into binary indicators
from sklearn.impute import SimpleImputer # Handles missing values by replacing them with a specified strategy (mean, median, or most frequent)
from sklearn.ensemble import RandomForestClassifier # Imports the random forest modeling package
from sklearn.model_selection import train_test_split # Splits data sets into training and testing groups
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)

# Directory guide, defines directories for storing trained models and dataset splits, creates them if they do not already exist
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "../models")
splits_dir = os.path.join(script_dir, "../splits")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

# Data file path
data_file = os.path.join(script_dir, "../data/processed/ME/processed.csv")

# Loads the data
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

# Hold out a fixed calibration set up front so it is never seen during model training
train_pool, calib_holdout = train_test_split(
    df, test_size=0.1, stratify=df[target_column], random_state=42
)

# Persist the calibration holdout for downstream probability calibration
calib_path = os.path.join(splits_dir, "full_rf_calibration_holdout.csv")
calib_holdout.to_csv(calib_path, index=False)

# Subsequent model training draws only from the training pool to avoid leakage
damage_df = train_pool[train_pool[target_column] == 1]
nodamage_df = train_pool[train_pool[target_column] == 0]

# Identifies the numerical and categorical columns for preprocessing
numerical_cols = df[input_features].select_dtypes(include=np.number).columns
categorical_cols = df[input_features].select_dtypes(include='object').columns

results = {
    "conf_matrices": [],
    "accuracies": [],
    "balanced_accuracies": [],
    "cohen_kappas": [],
    "precisions": [],
    "recalls": [],
    "f1s": []
}
# Preprocessing pipeline
# Preprocessor: defines how to handle missing data and categorical encoding:
# Missing numbers ('num') replaced with mean of the column
# Missing categorical ('cat') values replaced with most frequent in column
# Onehot encoding is employed to turn categorical variables into binary columns
# This data set is not missing any data, this is to cover bases so that things don't break if data with missing columns is introduced
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ],
    remainder='passthrough'
)

# Training and testing set up, this sets the number of runs
n_runs = 100

for run in range(n_runs):
    # Creates a balanced dataset of damage and no damage, a 1:1 ratio
    nodamage_sample = nodamage_df.sample(n=len(damage_df), random_state=run)
    balanced_df = pd.concat([damage_df, nodamage_sample], axis=0)

    # Separates the input/predictor features and the target
    X = balanced_df[input_features]
    y = balanced_df[target_column]

    # Training and testing split, separated into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=run
    )
    
    # Save splits for SHAP / reproducibility
    X_train.to_csv(os.path.join(splits_dir, f"X_train_run_{run}.csv"), index=False)
    X_test.to_csv(os.path.join(splits_dir, f"X_test_run_{run}.csv"), index=False)
    y_train.to_csv(os.path.join(splits_dir, f"y_train_run_{run}.csv"), index=False)
    y_test.to_csv(os.path.join(splits_dir, f"y_test_run_{run}.csv"), index=False)

    # Model pipeline, creates the RF and sets the parameters
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=run, # Sets the random state so it is the same
            n_estimators=400, # number of estimators
            max_depth=3,   # the maximum depth of trees
            max_features='sqrt',# max number of features considered at each split
            bootstrap=True    # sampling with replacement 
        ))
    ])

    # Establishes the full pipeline
    pipeline.fit(X_train, y_train)

    # Saves each model run in the models folder
    model_path = os.path.join(models_dir, f"full_rf_model_run_{run}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved full RF model {run+1}/{n_runs}") # Proress not

    y_pred = pipeline.predict(X_test)

    # Confusion matrix (order matches sorted unique labels unless you pass labels=...)
    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0,1], zero_division=0
    )
    # Store results
    results["conf_matrices"].append(cm)
    results["accuracies"].append(accuracy)
    results["balanced_accuracies"].append(balanced_acc)
    results["cohen_kappas"].append(kappa)
    results["precisions"].append(p)
    results["recalls"].append(r)
    results["f1s"].append(f1)

# Print lets you know when model training is completed
print(f"Done training {n_runs} models")

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
