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

# Separates the data into damage and no damage subsets, this is done for controlled sampling of training and testing subsets
damage_df = df[df[target_column] == 1]
nodamage_df = df[df[target_column] == 0]

# Identifies the numerical and categorical columns for preprocessing
numerical_cols = df[input_features].select_dtypes(include=np.number).columns
categorical_cols = df[input_features].select_dtypes(include='object').columns

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

# Print lets you know when model training is completed
print(f"Done training {n_runs} models")
