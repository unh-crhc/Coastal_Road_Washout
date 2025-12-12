# Coastal Road Resilience

## Project Overview

Geospatial analysis and hydraulic modeling to quantify the vulnerability of critical coastal infrastructure (e.g., Highway A1) to climate change stressors (sea-level rise, storm surge). The project utilizes high-resolution terrain data to simulate inundation scenarios and provide prioritized, cost-effective mitigation strategies for road segment reinforcement and adaptation.

---

## Requirements

### Software Dependencies

| Component | Minimum Version | Notes |
| :--- | :--- | :--- |
| Python | 3.10 | Required for geospatial scripting. |

### Python Packages

Install required Python libraries using the provided environment file:

```bash
pip install -r requirements.txt
```

(The key packages are `geopandas`, `rasterio`, `numpy`, and `scipy`.)

### Data Inputs

1.  **Features:** .

---

## Reproduction

The analysis proceeds in three stages: Data Preparation, Modeling Execution, and Result Interpretation.

### 1. Setup and Preparation

1.  Clone the repository:
    ```bash
    git clone https://github.com/user/coastal-resilience.git
    cd coastal-resilience
    ```
2.  Place all raw data inputs (DEMs, Shapefiles) into the `/data/input/` directory.

### 2. Modeling Execution

Execute the main script, which trains Random Forest models and prepares their `.pkl` files.

```bash
python src/train.py
```

*This script generates `washout_model_run_{run}.pkl` and required boundary condition files.*

### 3. Results and Analyze

Run the post-processing script to compute the confusion matrix for the models:

```bash
python src/evaluate.py
```

Open the generated files in `/data/output/final_maps/` to view inundation depths and prioritized road segments requiring mitigation.
