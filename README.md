# Coastal Road Resilience

## Project Overview
This repository contains the code and workflows for predicting coastal road washout using machine learning, based on observed roadway damage data from Maine (ME), New Hampshire (NH), Rhode Island (RI), Texas (TX), and Mississippi (MS). The project explores both full-feature and simplified predictive models, evaluates their transferability across states, and includes a linear regression approach derived from Texas data.

The repository includes
1.	Full Random Forest (RF) Model:
      - Trained on Maine data using 31 roadway, topographic, and storm-related features.
      - Provides baseline performance for damage classification.
      - Evaluated on accuracy, balanced accuracy, Cohen’s kappa, AUC, and class-specific precision, recall, and F1.
  	
2.	Three-Feature Random Forest Model:
      - Trained on Maine data using three key features:
            - Minimum elevation (Z_Min)
            - Distance to the coast (Distance_to_Coast_m)
            - Relative minimum elevation (Rel_Elev_Min)
      - Demonstrates that a small set of key variables can strongly predict road vulnerability.
      - Maintains performance comparable to the full RF model.
        
3.	Simple Decision Tree (DT) Model
      - Trained on Maine data using the same three key features. 
      - Allows interpretable rules for road damage risk classification.
        
4.	Linear Regression Model
      - Developed in “Fragility Analysis of Coastal Roadways and Performance Assessment of Coastal Transportation Systems Subjected to Storm Hazards”          by Darestani et al., 2021.
      - Trained on Texas data using: Distance to the coast (Distance_to_Coast_m) and Inundation duration (Inundation_Duration_Min)
      - Performance metrics include accuracy, balanced accuracy, Cohen’s kappa, AUC, and class-specific precision, recall, and F1.
        
5.	Screening Decision Tree
      - A single, interpretable DT derived from median splits in the best-performing simple DTs across all states.
      - Encodes the most common thresholds governing road vulnerability.
      - Applied across ME, NH, RI, TX, and MS for consistent, rule-based predictions.
# Data
Several data sources are included in this repository to support the analysis and modeling of coastal roadway damage. The primary datasets consist of processed CSV files for each study state (Maine, New Hampshire, Rhode Island, Texas, and Mississippi). The term processed indicates that these datasets have been prepared for modeling and analysis. Processing steps included associating observed damage locations with the nearest roadway segments, selecting nearby no-damage road segments for comparison, and compiling predictor variables describing roadway characteristics, topography, and storm exposure. Each processed dataset contains both damaged and undamaged road segments along with the predictor variables used in model development and evaluation.

Predictor variables included in the processed datasets represent three general categories: roadway characteristics, topographic conditions, and storm exposure metrics. Roadway variables include attributes such as averaged annual daily traffic, roadway classification, number of lanes, and surface type. Topographic variables were derived from digital elevation models and include elevation and slope-based metrics, as well as measures of relative elevation. Storm exposure variables were derived from observational datasets and include metrics such as water levels, wave heights, and wind conditions. Inundation duration represents the estimated time that a roadway segment was submerged during a storm event and was calculated by comparing roadway elevations to time-series water level observations from nearby tide gauges. Not all roadway variables are available for every state because publicly available roadway GIS datasets vary in content; however, topographic and storm exposure variables were developed consistently across all study areas.

In addition to the processed state datasets, the repository includes a dataset named webb_data.csv, which was developed in prior research and is used for comparison with the machine learning models developed in this study. This dataset contains the variables used to construct a linear regression model based  on distance from the coastline and inundation duration. The linear regression model provides a baseline approach against which the performance of the machine learning models can be evaluated. It is further discussed in a paper titled "Fragility Analysis of Coastal Roadways and Performance Assessment of Coastal Transportation Systems Subjected to Storm Hazards" by Darestani and co-authors in 2021. 

Together, these datasets provide a consistent framework for analyzing coastal roadway damage across multiple geographic regions and storm events, while allowing comparison between traditional statistical models and machine learning approaches. The table below outlines the total number of damaged and undamaged locations in each data set as well as when the damage occured. 

| Document           | # of Damage Locations | # of No Damage Locations | Date(s) of Damage        |
|:-------------------|----------------------:|--------------------------:|:--------------------------|
| ME/processed.csv   | 110                  | 550                      | 1/10/2024 and 1/13/2024   |
| NH/processed.csv   | 7                    | 35                       | 1/10/2024                 |
| RI/processed.csv   | 8                    | 40                       | 2016–2024                 |
| TX/processed.csv   | 59                   | 16                       | 9/13/2008                 |
| MS/processed.csv   | 29                   | 145                      | 2/29/2025                 |

---

## Requirements

### Software Dependencies
List of all the software dependencies required to run the project, including their minimum versions and any specific notes about installation or compatibility.

| Component | Minimum Version | Notes |
| :--- | :--- | :--- |
| Python | 3.10 | Required for geospatial scripting. |
| numpy | 1.22 | Used for numerical operations. |
| pandas | 1.4 | Used for data manipulation and analysis. |
| scikit-learn | 1.0 | Used for machine learning modeling. |
| matplotlib | 3.5 | Used for plotting model results and visualizations. |
| joblib | 1.1 | Used for saving and loading trained models. |
| shap | 0.41 | Used for feature importance and model interpretation. |
| os | Built-in | Used for file paths and directory management. |
| glob | Built-in | Used for locating files with pattern matching. |

### Python Packages

Install required Python libraries using the provided environment file:

### requirements.txt

```text
numpy>=1.22
pandas>=1.4
scikit-learn>=1.0
matplotlib>=3.5
joblib>=1.1
shap>=0.41
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Inputs
The datasets include predictor variables describing roadway characteristics, topographic conditions, and storm exposure. These features were selected based on prior research and their relevance to coastal roadway damage processes. Roadway variables represent the physical and functional characteristics of each road segment. Topographic variables capture the susceptibility of road segments to flooding and wave action. Storm exposure variables characterize the intensity and duration of storm impacts at each location. Together, these features were chosen to represent the primary physical factors that influence whether a roadway experiences damage during coastal storm events.

Several preprocessing steps were required before the features could be used for modeling. Damage locations were first matched to the nearest roadway segments on public road GIS files, and nearby undamaged road segments were selected to provide comparison cases. Predictor variables were then compiled from multiple sources and merged into a single dataset for each state. Some derived variables, such as relative elevation and inundation duration, were calculated from other collected variables. 

Not all states include identical roadway attribute information because the publicly available roadway GIS datasets varied in content. The Maine dataset contains the most complete set of roadway variables, while datasets from other states may include fewer roadway-specific attributes. However, all datasets share a consistent set of topographic and storm exposure variables. When a feature was not available for a particular state, the preprocessing workflow retained the corresponding column and filled missing values with a representative constant value (the median from the training dataset for numerical variables, or most common for categorical variables). This approach ensured a consistent feature structure across datasets so that models could be applied without errors, while preventing unavailable variables from influencing model predictions or breaking model paths.

## Predictor varaibles
| Variable | Meaning |
|---------|---------|
| **Roadway Variables** | |
| aadt_type | Type/category of AADT (Annual Average Daily Traffic) on a scale of A–E |
| priority | Priority level of the road on a scale of 1–5 |
| fedfunccls | Federal Functional Classification (e.g., arterial, collector, local) |
| capacity | Maximum vehicle capacity of the road segment (vehicles/day) |
| jurisdictn | Jurisdiction responsible for the road (e.g., state, county, municipal) |
| num_lanes | Total number of lanes (both directions) |
| st_urbrur | Urban/rural classification |
| surfc_type | Surface type of the road (e.g., flexible, unimproved, other) |
| segment_length | Length of the road segment in the GIS file (ft) |
| **Topographic Variables** | |
| Z_Min | Minimum elevation along the segment (m) |
| Z_Max | Maximum elevation along the segment (m) |
| Z_Mean | Mean elevation of the road segment (m) |
| Min_Slope | Minimum slope along the segment (degrees) |
| Max_Slope | Maximum slope along the segment (degrees) |
| Avg_Slope | Average slope across the segment (degrees) |
| **Storm Exposure Variables** | |
| Max_Height | Maximum recorded tidal water elevation during the storm (m) |
| Return_Period | Storm return period based on water elevations (years) |
| Distance_to_Coast_m | Distance from road segment to coast (m) |
| Max_WVHT | Maximum recorded wave height (m) |
| Max_Dir | Direction of maximum wave height (degrees) |
| Avg_WVHT | Average recorded wave height (m) |
| Avg_Dir | Direction of average wave height (degrees) |
| Max_WSPD | Maximum recorded wind speed (knots) |
| Max_WSPD_Dir | Direction of maximum wind speed (degrees) |
| Avg_WSPD | Average recorded wind speed (knots) |
| Avg_WSPD_Dir | Direction of average wind speed (degrees) |
| Precipitation | Storm precipitation (cm) |
| Rel_Elev_Min | Max_Height minus minimum elevation (m) |
| Rel_Elev_Mean | Max_Height minus mean elevation (m) |
| Rel_Elev_Max | Max_Height minus maximum elevation (m) |
| Inundation_Duration_Min | Estimated inundation duration (minutes) |

---
## Structure
The project code is organized into scripts that support model training, model evaluation, model interpretation, and environment management. Each script is designed to perform a specific task within the modeling workflow.

### Model Training
Training scripts develop the predictive models and output performance metrics based on the training and testing datasets. The full random forest model should be trained first, as this script generates the training/testing splits and calibration datasets that are reused by the other models to ensure consistent comparisons.

**Full Random Forest Model**

Trains the full random forest model using the complete set of predictor variables and generates the dataset splits used by all models.
```bash
python3 src/train_rf.py
```

**Three-Feature Random Forest Model**

Trains a simplified random forest model using a reduced set of three key predictor variables.
```bash
python3 src/train_rf_3f.py
```

**Decision Tree Model**
Trains a single decision tree model using the standardized training and testing splits.
```bash
python3 src/train_dt.py
```

**Linear Regression Model**
Trains the linear regression model used as a baseline comparison to the machine learning approaches.
```bash
python3 src/train_linear_regression.py
```

### Model Evaluation
Evaluation scripts apply the trained models to datasets from other states and calculate performance metrics. Models are evaluated using overall accuracy, balanced accuracy, Cohen’s kappa, area under the receiver operating characteristic curve (AUC), and class-specific precision, recall, and F1 scores.

**Full Random Forest Model**

Applies the full random forest model to external state datasets and reports performance metrics.
```bash
python3 src/evaluate_rf.py
```

**Three-Feature Random Forest Model**

Applies the simplified random forest model to external state datasets.
```bash
python3 src/evaluate_rf_3f.py
```

**Decision Tree Model**

Applies the decision tree model to external state datasets.
```bash
python3 src/evaluate_dt.py
```

**Linear Regression Model**

Applies the linear regression model to external state datasets.
```bash
python3 src/evaluate_linear_regression.py
```

**Screening Decision Tree Model**

Applies the screening decision tree model, which is intended for simplified classification and screening-level assessments.
```bash
python3 src/evaluate_screening_dt.py
```

### Model Interpretation
Interpretation scripts generate SHAP (SHapley Additive exPlanations) plots to help interpret the influence of predictor variables in the full random forest model.

**Average SHAP Plot**

Generates SHAP plots representing the average feature importance across multiple model runs.
```bash
python3 src/analysis_average_shap_plot.py
```

**Individual SHAP Plot**

Generates SHAP plots for a single model run.
```bash
python3 src/analysis_individual_shap_plot.py
```

These scripts help identify which predictor variables contribute most strongly to model predictions.

### Environment Management

**Cleanup Script**

This script removes generated model files, dataset splits, and figure outputs, returning the project directory to a clean state. This can be useful when rerunning the full modeling workflow or preparing a fresh analysis environment.
```bash
python3 src/clean_up.py
```

## Expected Results
This section outlines the expected results. 


