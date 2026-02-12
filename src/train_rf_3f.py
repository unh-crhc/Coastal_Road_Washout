import os # Interact with operating system
import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing
from sklearn.pipeline import Pipeline # Chains model steps into one workflow
from sklearn.impute import SimpleImputer # Handles missing values by replacing them with a specified strategy (mean, median, or most frequent)
from sklearn.ensemble import RandomForestClassifier # Imports the random forest modeling package
from sklearn.metrics import ( # Imports for performance analysis
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, precision_recall_fscore_support
)

# Directory, provides paths to models and splits folder
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "../models")
splits_dir = os.path.join(script_dir, "../splits")
os.makedirs(models_dir, exist_ok=True)

# Path to Maine data
data_file = os.path.join(script_dir, "../data/processed/ME/processed.csv")
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min'] # Input features
target_column = 'Damage_Status' # Target feature (damage or no damage)
n_runs = 100 # Number of runs

# Loads Maine data and converts Damage_Status labels to binary, 0 for no damage and 1 for damage
# Techincally does not need to access the full data set as training will use the pre-saved splits
df = pd.read_csv(data_file)
df[target_column] = df[target_column].map({'No Damage': 0, 'Damage': 1})
df = df.dropna(subset=[target_column])

# Naming for reference
model_type = "three_feature_rf"

# Creates containers for the results
results = {
    "conf_matrices": [],
    "accuracies": [],
    "balanced_accuracies": [],
    "cohen_kappas": [],
    "precisions": [],
    "recalls": [],
    "f1s": []
}


# Trains models over 100 runs
for run in range(n_runs):
    # Loads the established train/test splits created in full RF training for consistency
    X_train = pd.read_csv(os.path.join(splits_dir, f"X_train_run_{run}.csv"))[input_features]
    y_train = pd.read_csv(os.path.join(splits_dir, f"y_train_run_{run}.csv")).squeeze()

    # Load pre-saved test splits
    X_test_path = os.path.join(splits_dir, f"X_test_run_{run}.csv")
    y_test_path = os.path.join(splits_dir, f"y_test_run_{run}.csv")

    # If files are missing, skips run and gives error message
    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print(f"Warning: Missing test split for run {run}. Skipping. Train full RF to establish splits")
        continue

    # Ensures that columns are numeric, no categorical columns in this model
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    X_test = pd.read_csv(X_test_path)[input_features].apply(pd.to_numeric, errors='coerce')
    y_test = pd.read_csv(y_test_path).squeeze()

    # Training pipeline
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Imputes missing values with the mean of the column, should not be any missing values in this data set
        ('classifier', RandomForestClassifier(
            random_state=run, # Same configuration as full RF
            n_estimators=400,
            max_depth=3,
            max_features='sqrt',
            bootstrap=True
        ))
    ])

    # Trains the RF models
    pipeline.fit(X_train, y_train)

    # Saves the models
    model_path = os.path.join(models_dir, f"three_feature_rf_model_run_{run}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved three-feature RF model {run+1}/{n_runs}") # Progress note
    
    # Runs models and gets outputs
    y_pred = pipeline.predict(X_test)

    # Compute performance metrics
    results["conf_matrices"].append(confusion_matrix(y_test, y_pred, labels=[0,1]))
    results["accuracies"].append(accuracy_score(y_test, y_pred))
    results["balanced_accuracies"].append(balanced_accuracy_score(y_test, y_pred))
    results["cohen_kappas"].append(cohen_kappa_score(y_test, y_pred))

    # Computes class level performance metrics, precision, recall, and F1
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0,1], zero_division=0
    )

    # Stores class level performance metrics
    results["precisions"].append(p)
    results["recalls"].append(r)
    results["f1s"].append(f1)

    # Prints a completion note for each run, progress tracking
    print(f"Completed run {run+1}/{n_runs}")


# Completion note
print(f"\nFinished training {n_runs} three-feature RF models using saved splits")
cms = np.array(results["conf_matrices"])
avg_cm = cms.mean(axis=0)
std_cm = cms.std(axis=0)

# Prints average confusion matrices and standard deviations
print("Average Confusion Matrix:")
print(avg_cm)
print("Std of Confusion Matrix:")
print(std_cm)

# Computes summary metrics with standard deviation
acc = np.mean(results["accuracies"])
acc_std = np.std(results["accuracies"])
bal = np.mean(results["balanced_accuracies"])
bal_std = np.std(results["balanced_accuracies"])
kap = np.mean(results["cohen_kappas"])
kap_std = np.std(results["cohen_kappas"])

# Prints summary statistics 
print(f"Accuracy: {acc:.3f} ± {acc_std:.3f}")
print(f"Balanced Accuracy: {bal:.3f} ± {bal_std:.3f}")
print(f"Cohen Kappa: {kap:.3f} ± {kap_std:.3f}")

# Computes class-level metrics
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
print(f"\nCompleted evaluation of {n_runs} runs for {model_type}")
