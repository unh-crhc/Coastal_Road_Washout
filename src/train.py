import pandas as pd # Package for csvs and data
import numpy as np # Used for numerical and scientific computing
import joblib # Used for parallel computing, common in data science
from sklearn.compose import ColumnTransformer # Scikit-learn (sklearn) is a machine learning library, ColumnTransformer lets you apply different preprocessing steps to different types of columns
from sklearn.pipeline import Pipeline # Chains model steps into one workflow, described as an assembly line
from sklearn.preprocessing import OneHotEncoder # Converts categorical variables into binary indicators
from sklearn.impute import SimpleImputer # Handles missing values by replacing them with a specified strategy (mean, median, or most frequent)
from sklearn.ensemble import RandomForestClassifier # Imports the random forest modeling package
from sklearn.model_selection import train_test_split # Splits data sets into training and testing groups

# Load Maine data
df = pd.read_csv("../data/ME_roads_all_info_with_structure.csv")

# Select the input features and target column (Damage_Status)
input_features = [
    'aadt_type', 'priority', 'capacity', 'Z_Min', 'Distance_to_Coast_m',
    'Max_WVHT', 'Max_Dir', 'Avg_WVHT', 'Avg_Dir',
    'Max_WSPD', 'Max_WSPD_Dir', 'Avg_WSPD', 'Avg_WSPD_Dir',
    'Precipitation', 'Rel_Elev_Min', 'Max_Height', 'Return_Period'
]
target_column = 'Damage_Status'

# Convert Damage/No-Damage lables to binary 0 and 1
df[target_column] = df[target_column].map({'No Damage': 0, 'Damage': 1})
df = df.dropna(subset=[target_column])

# Separate damage and no damage data frames
damage_df = df[df[target_column] == 1]
nodamage_df = df[df[target_column] == 0]

# Separate and identify the numerical and categorical columns correctly
numerical_cols = df[input_features].select_dtypes(include=np.number).columns  # selects all columns in input_features that are numeric
categorical_cols = df[input_features].select_dtypes(include='object').columns # selects all columns that are categorical

# Preprocessor: defines how to handle missing data and categorical encoding:
# Missing numbers ('num') replaced with mean of the column
# Missing categorical ('cat') values replaced with most frequent in column
# Onehot encoding is employed to category into binary columns
# This data set is not missing any data (most probably), this is to cover bases
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (like OBJECTID if not in input_features)
)

# Train Save 100 Models
n_runs = 100 # Number of runs

for run in range(n_runs):
    # 1:1 balanced sampling with each model trained on a different subset of the data
    nodamage_sample = nodamage_df.sample(n=len(damage_df), random_state=run)
    balanced_df = pd.concat([damage_df, nodamage_sample], axis=0)

    X = balanced_df[input_features] # input features
    y = balanced_df[target_column] # target columns (Damage_Status)

    # Splits data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=run
    )

    # Pipeline for the random forest
    pipeline = Pipeline(steps=[ # applies the preprocessing steps defined earlier
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=run, # Each hyperparameter below was honed using grid search, also meant to be relativel simple to limit overfitting
            n_estimators=400, # Number of trees used in the forest
            max_depth=3, # Depth each tree is limited to
            max_features='sqrt', # Each split considers a random subset of features equal to the sqrt of the total number of features
            bootstrap=True # Each tree is trained on a random sample with replacement
        ))
    ])

    # Fit and trains the models
    pipeline.fit(X_train, y_train)

    # Save model pipeline
    model_path = f"../models/washout_model_run_{run}.pkl"
    joblib.dump(pipeline, model_path)

print(f"Done with {n_runs} models.") # Shows stuff is donw