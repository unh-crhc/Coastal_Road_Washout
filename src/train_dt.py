import os # interact with operating system
import pandas as pd # Package for csvs and data
import joblib # Used for parallel computing
from sklearn.pipeline import Pipeline # Chains model steps into one workflow, described as an assembly line
from sklearn.impute import SimpleImputer # Handles missing values by replacing them with a specified strategy (mean, median, or most frequent)
from sklearn.tree import DecisionTreeClassifier # Imports the decision tree classifier
import numpy as np # Used for numerical and scientific computing
from sklearn.metrics import ( # Imports for performance analysis
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, precision_recall_fscore_support,roc_auc_score,
)

# Directory, foler locations
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "../models")
splits_dir = os.path.join(script_dir, "../splits")
os.makedirs(models_dir, exist_ok=True)

# Path to data, list of input features, the target (damage), and number of runs 
data_file = os.path.join(script_dir, "../data/processed/ME/processed.csv")
input_features = ['Z_Min', 'Distance_to_Coast_m', 'Rel_Elev_Min']
target_column = 'Damage_Status'
n_runs = 100
model_type = "simple_dt" # Model name

# Loads teh data and reads the csv, converts Damage_Status to binary, 0 for no damage and 1 for damage, drops rows where target column is missing
df = pd.read_csv(data_file)
df[target_column] = df[target_column].map({'No Damage': 0, 'Damage': 1})
df = df.dropna(subset=[target_column])

# Creates containers for te results
results = {
    "conf_matrices": [],
    "accuracies": [],
    "balanced_accuracies": [],
    "cohen_kappas": [],
    "AUC": [],
    "precisions": [],
    "recalls": [],
    "f1s": []
}

# Train models 
for run in range(n_runs):
    # Loads presaved train/test splits for consistency
    X_train = pd.read_csv(os.path.join(splits_dir, f"X_train_run_{run}.csv"))[input_features]
    y_train = pd.read_csv(os.path.join(splits_dir, f"y_train_run_{run}.csv")).squeeze()

    X_test_path = os.path.join(splits_dir, f"X_test_run_{run}.csv")
    y_test_path = os.path.join(splits_dir, f"y_test_run_{run}.csv")
    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print(f"Warning: Missing test split for run {run}. Skipping. Train full RF first")
        continue

    X_test = pd.read_csv(X_test_path)[input_features]
    y_test = pd.read_csv(y_test_path).squeeze()

    # Ensure numeric columns are numeric, no categorical variables for the DT in this case
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Numeric imputer and the DT pipeline, imputes missing data with the mean of the column, but there shouldnt be any missing data
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', DecisionTreeClassifier(random_state=run, max_depth=2, min_samples_leaf=10))
    ])

    # Fits the pipeline to the data
    pipeline.fit(X_train, y_train)

    # Saves model rins
    model_path = os.path.join(models_dir, f"simple_dt_model_run_{run}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved DT model {run+1}/{n_runs}") # Prints progress

    y_pred = pipeline.predict(X_test)

     # Probabilities for AUC
    y_prob = pipeline.predict_proba(X_test)[:,1]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_prob)

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0,1], zero_division=0
    )

    # Compute performance metrics
    results["conf_matrices"].append(confusion_matrix(y_test, y_pred, labels=[0,1]))
    results["accuracies"].append(accuracy_score(y_test, y_pred))
    results["balanced_accuracies"].append(balanced_accuracy_score(y_test, y_pred))
    results["cohen_kappas"].append(cohen_kappa_score(y_test, y_pred))
    results["AUC"].append(AUC)

    # Computes class level performance metrics, precision, recall, and F1
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0,1], zero_division=0
    )

    # Stores class level performance metrics
    results["precisions"].append(p)
    results["recalls"].append(r)
    results["f1s"].append(f1)

# Completion note
print(f"\nFinished training {n_runs} DT models using saved splits")
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
AUC = np.mean(results["AUC"])
AUC_std = np.std(results["AUC"])

# Prints summary statistics 
print(f"Accuracy: {acc:.3f} ± {acc_std:.3f}")
print(f"Balanced Accuracy: {bal:.3f} ± {bal_std:.3f}")
print(f"Cohen Kappa: {kap:.3f} ± {kap_std:.3f}")
print(f"AUC: {AUC:.3f} ± {AUC_std:.3f}")

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
