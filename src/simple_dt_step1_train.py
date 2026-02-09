import os # interact with operating system
import pandas as pd # Package for csvs and data
import joblib # Used for parallel computing
from sklearn.pipeline import Pipeline # Chains model steps into one workflow, described as an assembly line
from sklearn.impute import SimpleImputer # Handles missing values by replacing them with a specified strategy (mean, median, or most frequent)
from sklearn.tree import DecisionTreeClassifier # Imports the decision tree classifier

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
    "precisions": [],
    "recalls": [],
    "f1s": []
}

# Train models 
for run in range(n_runs):
    # Loads presaved train/test splits for consistency
    X_train = pd.read_csv(os.path.join(splits_dir, f"X_train_run_{run}.csv"))[input_features]
    y_train = pd.read_csv(os.path.join(splits_dir, f"y_train_run_{run}.csv")).squeeze()

    # Ensure numeric columns are numeric, no categorical variables for the DT in this case
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

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

# Completion message
print(f"\nFinished training {n_runs} decision trees using saved splits")
