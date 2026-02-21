import os # Interact with operating system
import glob # For finding files (*.csv etc.)

# Directory, this script deletes all of the models, results, splits, and figures so they are not uploaded to GitHub
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Folders to clear 
targets = {
    "models": "*.pkl",      # Removes all trained models
    "results": "*.csv",     # Removes all csvs in results folder
    "splits": "*.csv",      # Remove all csvs in splits folder
    "figures": "*.*"        # Remove all files in figure folder (png, pdf, etc.)
}

# Clean up loop 
for folder, pattern in targets.items():
    path = os.path.join(base_dir, folder)
    if not os.path.exists(path):
        continue

    # Finds all files matching the target description and iterates through all matching files and deletes tehm
    files = glob.glob(os.path.join(path, pattern))
    for f in files:
        os.remove(f)
        print(f"Removed {f}") # Shows file is deleted

# Shows that script is done
print("\nCleanup done.")
