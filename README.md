# Coastal Road Resilience

## Project Overview
- [ ] Add project overview

---

## Requirements

### Software Dependencies
List of all the software dependencies required to run the project, including their minimum versions and any specific notes about installation or compatibility.

- [ ] Add dependency libraries (the ones you installed using pip)

| Component | Minimum Version | Notes |
| :--- | :--- | :--- |
| Python | 3.10 | Required for geospatial scripting. |
| numpy | 1.22 | Used for numerical operations. |
| pandas | 1.4 | Used for data manipulation and analysis. |
| scikit-learn | 1.0 | Used for machine learning modeling. |

### Python Packages

Install required Python libraries using the provided environment file:

```bash
pip install -r requirements.txt
```

### Data Inputs

1.  **Features:** .
- [ ] explain about your features and why you chose them
- [ ] Is there any specific preprocessing steps required for the features?
- [ ] Is there any features missing in the dataset (RI or NH has the same features as ME)? If so, how do you handle them?

---

## Reproduction

The analysis proceeds in three stages: Data Preparation, Modeling Execution, and Result Interpretation.

### Setup and Preparation

1.  Clone the repository:
    ```bash
    git clone https://github.com/user/coastal-resilience.git
    cd coastal-resilience
    ```
2.  Place all raw data inputs (DEMs, Shapefiles) into the `/data/input/` directory.

### Training RF Models
- [ ] Explain about the ensumble RF model and how it works in this project (e.g., how many models, how many trees, what features are used, etc.)

To train the Random Forest models, run the python src/train.py. This file prepares the `.pkl` files corresponding to each model.

```bash
python src/train_rf.py
```

*This script generates `washout_model_run_{run}.pkl`.*

### Evaluating RF Models
To evaluate the trained models, run the following command:

```bash
python src/evaluate_rf.py
```

Open the generated files in `/data/output/final_maps/` to view inundation depths and prioritized road segments requiring mitigation.

### Training Decision Tree Models
- [ ] Explain about the decision tree model and how it works in this project (e.g., how many models, what features are used, etc.)
To train the Decision Tree models, run the python src/train_dt.py. This file prepares the `.pkl` files corresponding to each model.

```bash
python src/train_dt.py
```

*This script generates `washout_model_run_{run}.pkl`.*

### Evaluating Decision Tree Models
To evaluate the trained models, run the following command:

```bash
python src/evaluate_dt.py
```
This will print the evaluation metrics (e.g., accuracy, precision, recall) for each model and save the results in `/results/predictions.csv`.

