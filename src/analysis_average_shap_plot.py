import os # Interact with operating system
import joblib # Used for parallel computing
import numpy as np # Used for numerical and scientific computing
import pandas as pd # Package for csvs and data
import shap # For SHAP plots (SHapley Additive exPlanations)
import matplotlib.pyplot as plt # For plotting
# The goal of this script is to create an averaged SHAP plot from the 100 model runs of the Full RF

# Directory, creates paths to models, splits, and figures folder
script_dir = os.path.dirname(__file__)
models_dir = os.path.join(script_dir, "../models")
splits_dir = os.path.join(script_dir, "../splits")
figures_dir = os.path.join(script_dir, "../figures")
os.makedirs(figures_dir, exist_ok=True)

# Control the number of runs averaged and number of samples
runs = list(range(100))        # number of runs to average
max_samples = 550      # samples (or number of roads) used per run, important for speed control if data sets were bigger

# Load the test set from the first run (consistent baseline)
X_test = pd.read_csv(os.path.join(splits_dir, "X_test_run_0.csv"))
X_test = X_test.sample(min(max_samples, len(X_test)), random_state=0)

# Creates list to store shap plots
all_shap = []

# Iterates through each trained RF
for run in runs:
    print(f"Computing SHAP for run {run}") # Shows which run is being worked on 

    # Loads the model pipeline for a given run
    model_file = os.path.join(models_dir, f"full_rf_model_run_{run}.pkl")
    pipeline = joblib.load(model_file)

    # Ensures the input has the correct columns and returns the probability of damage 
    def predict_damage(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=X_test.columns)
        return pipeline.predict_proba(X)[:, 1]

    # Selects a data set for SHAP baseline,  
    background = X_test.sample(min(50, len(X_test)), random_state=run) # randndom sate = run means a different reference population for each run

    # Creates a SHAP explainer using kernel approximation and computes SHAP values 
    explainer = shap.KernelExplainer(
    predict_damage,
    background,
    l1_reg="num_features(10)")
    shap_values = explainer.shap_values(X_test, nsamples=500) # nsamples is how many pertubations are used to estimate SHAP values for each sample (road), affects accuracy and run time

    # Store SHAP values
    all_shap.append(shap_values)

# Computes the mean SHAP value for each feature
mean_shap = np.mean(np.array(all_shap), axis=0)

# Creates the figure 
plt.figure(figsize=(12, 8))
shap.summary_plot(mean_shap, X_test, plot_type="dot", show=False) # Specifics about the figure, ie. dot plot

# Name and organization of the plot
plot_file = os.path.join(figures_dir, "shap_summary_FULL_RF_AVERAGED.png")
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.close()

# Completion note
print(f"\nSaved averaged SHAP plot to:\n{plot_file}")
