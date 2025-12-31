import os # Interact with operating system
import pandas as pd # Package for csvs and data
import joblib # Used for parallel computing
from sklearn.pipeline import Pipeline # Chains model steps into one workflow
from sklearn.impute import SimpleImputer # Handles missing values by replacing them with a specified strategy (mean, median, or most frequent)
from sklearn.ensemble import RandomForestClassifier # Imports the random forest modeling package

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

# Trains models over 100 runs
for run in range(n_runs):
    # Loads the established train/test splits created in full RF training for consistency
    X_train = pd.read_csv(os.path.join(splits_dir, f"X_train_run_{run}.csv"))[input_features]
    y_train = pd.read_csv(os.path.join(splits_dir, f"y_train_run_{run}.csv")).squeeze()

    # Ensures that columns are numeric, no categorical columns in this model
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

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

# Completion note
print(f"\nFinished training {n_runs} three-feature RF models using saved splits")
