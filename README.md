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
      - Trained on Texas data using:
            - Distance to the coast (Distance_to_Coast_m)
            -Inundation duration (Inundation_Duration_Min)
      - Performance metrics include accuracy, balanced accuracy, Cohen’s kappa, AUC, and class-specific precision, recall, and F1.
        
5.	Screening Decision Tree
      - A single, interpretable DT derived from median splits in the best-performing simple DTs across all states.
      - Encodes the most common thresholds governing road vulnerability.
      - Applied across ME, NH, RI, TX, and MS for consistent, rule-based predictions.
# Data
Several data sources are included in this repository to support the analysis and modeling of coastal roadway damage. The primary datasets consist of processed CSV files for each study state (Maine, New Hampshire, Rhode Island, Texas, and Mississippi). The term processed indicates that these datasets have been prepared for modeling and analysis. Processing steps included associating observed damage locations with the nearest roadway segments, selecting nearby no-damage road segments for comparison, and compiling predictor variables describing roadway characteristics, topography, and storm exposure. Each processed dataset contains both damaged and undamaged road segments along with the predictor variables used in model development and evaluation.

Predictor variables included in the processed datasets represent three general categories: roadway characteristics, topographic conditions, and storm exposure metrics. Roadway variables include attributes such as averaged annual daily traffic, roadway classification, number of lanes, and surface type. Topographic variables were derived from digital elevation models and include elevation and slope-based metrics, as well as measures of relative elevation. Storm exposure variables were derived from observational datasets and include metrics such as water levels, wave heights, and wind conditions. Inundation duration represents the estimated time that a roadway segment was submerged during a storm event and was calculated by comparing roadway elevations to time-series water level observations from nearby tide gauges. Not all roadway variables are available for every state because publicly available roadway GIS datasets vary in content; however, topographic and storm exposure variables were developed consistently across all study areas.

In addition to the processed state datasets, the repository includes a dataset named webb_data.csv, which was developed in prior research and is used for comparison with the machine learning models developed in this study. This dataset contains the variables used to construct a linear regression model based primarily on distance from the coastline and inundation duration. The linear regression model provides a simplified baseline approach against which the performance of the more complex machine learning models can be evaluated. It is further discussed in a paper titled "Fragility Analysis of Coastal Roadways and Performance Assessment of Coastal Transportation Systems Subjected to Storm Hazards" by Darestani and co-authors in 2021. 

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

1.  **Features:** .
The datasets include predictor variables describing roadway characteristics, topographic conditions, and storm exposure. These features were selected based on prior research and their relevance to coastal roadway damage processes. Roadway variables (e.g., traffic volume, roadway classification, number of lanes, and surface type) represent the physical and functional characteristics of each road segment. Topographic variables (e.g., minimum elevation, slope, and relative elevation) capture the susceptibility of road segments to flooding and wave action. Storm exposure variables (e.g., water levels, wave heights, wind conditions, precipitation, and inundation duration) characterize the intensity and duration of storm impacts at each location. Together, these features were chosen to represent the primary physical factors that influence whether a roadway experiences damage during coastal storm events.

Several preprocessing steps were required before the features could be used for modeling. Damage locations were first matched to the nearest roadway segments on public road GIS files, and nearby undamaged road segments were selected to provide comparison cases. Predictor variables were then compiled from multiple sources and merged into a single dataset for each state. Some derived variables, such as relative elevation and inundation duration, were calculated from raw elevation and water-level data to better represent storm exposure conditions.

Not all states include identical roadway attribute information because publicly available roadway GIS datasets vary in content. The Maine dataset contains the most complete set of roadway variables, while datasets from other states may include fewer roadway-specific attributes. However, all datasets share a consistent set of topographic and storm exposure variables. When a feature was not available for a particular state, the preprocessing workflow retained the corresponding column and filled missing values with a representative constant value (typically the median from the training dataset, or most common for categorical variables). This approach ensured a consistent feature structure across datasets so that models could be applied without errors, while preventing unavailable variables from exerting meaningful influence on model predictions.

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

Several models were developed to predict the likelihood of roadway washout during coastal storm events. Two of the approaches utilize random forest (RF) modeling. The RF approach combines multiple decision trees into a single ensemble model to improve predictive performance and reduce overfitting. Each tree is trained on a slightly different subset of the data, and final predictions are generated through majority voting across all trees.

The model is implemented using a preprocessing and modeling pipeline to ensure consistent data handling and reproducibility. Input variables include roadway characteristics, topographic variables, and storm exposure variables.

**Model configuration:**

- **Algorithm:** Random Forest Classifier  
- **Number of trees:** 400 decision trees  
- **Maximum tree depth:** 3 levels  
- **Feature sampling:** Square root of total features considered at each split (`max_features = sqrt`)  
- **Bootstrap sampling:** Enabled (training samples drawn with replacement)  
- **Random seed:** Controlled for reproducibility

The relatively shallow tree depth (maximum depth = 3) was selected to reduce model complexity and improve generalization across geographic regions. Using a large number of trees (400) stabilizes model predictions and reduces variance.

The Random Forest is trained using a preprocessing pipeline that standardizes feature handling and ensures consistent application of transformations during both training and model application.

Multiple Random Forest models are trained using different random seeds. This produces a set of independent models that collectively form an ensemble. The ensemble approach reduces sensitivity to individual training splits and improves robustness of the final predictions.


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

To train the Decision Tree models, run the python src/train_dt.py. This file prepares the `.pkl` files corresponding to each model.

Decision Tree (DT) classifiers are used in this project as simplified and interpretable models for predicting roadway washout. Unlike the Random Forest models, which combine many trees into an ensemble, each Decision Tree model consists of a single tree that partitions the feature space into decision rules. This approach provides transparent decision thresholds that can be directly interpreted and compared across regions.

Decision Tree models are trained using roadway, topographic, and storm exposure variables. The trees identify threshold values in the predictor variables that separate damaged and undamaged road segments.

The training process produces multiple Decision Tree models using different random seeds. This allows variability in the training process to be evaluated and helps identify consistent splitting patterns across models and states.

**Model configuration:**

- **Algorithm:** Decision Tree Classifier  
- **Tree depth:** Maximum depth of 3 levels  
- **Minimum samples per leaf:** 10 observations  
- **Missing data handling:** Mean-value imputation using `SimpleImputer`  
- **Random seed:** Controlled for reproducibility

The shallow tree depth was selected to maintain interpretability and prevent overfitting. Limiting the depth produces simple decision rules that can be easily visualized and compared between study regions.

Although a mean-value imputation step is included in the training pipeline, the processed datasets are expected to contain no missing values. The imputer is included primarily to ensure robustness and reproducibility.

Each trained Decision Tree model is saved as a `.pkl` file and can be applied to additional datasets during the model application stage.

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

