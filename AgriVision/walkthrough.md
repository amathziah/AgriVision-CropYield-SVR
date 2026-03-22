# AgriVision: Crop Yield Prediction Walkthrough

This document provides a detailed walkthrough of the **AgriVision** project, a machine learning pipeline designed to predict agricultural crop yields (tonnes per hectare) using climate data and Support Vector Regression (SVR).

## 🚀 Executive Summary
AgriVision achieves a **79.6% R² score**, meaning it explains nearly 80% of the variance in crop yields across 101 countries and 10 crop types. It significantly outperforms baseline predictors, reducing error (RMSE) by over **54%**.

---

## 🛠️ The 6-Step Pipeline

### 1. Data Ingestion & Cleaning (`step1_load.py`)
- **Action**: Loads raw FAO data, standardizes column names, and filters for relevant crops.
- **Key Decision**: Handled missing values by ensuring only complete records for climate (rainfall/temp) and pesticides are used.
- **Output**: `data/final_dataset.csv`

### 2. Exploratory Data Analysis (`step2_eda.py`)
- **Action**: Visualizes yield distributions and correlations.
- **Insight**: Discovered that yield is highly skewed, justifying the use of **log-transformation** for pesticide data and **per-crop normalization** for the target variable.
- **Artifacts**: Distribution plots and correlation heatmaps in `results/`.

### 3. Feature Engineering (`step3_features.py`)
- **Action**: Creates new interaction features.
- **Feature Highlight**: `rainfall_temp_ratio` (capturing humidity/evaporation effects) and `log_pesticides`.
- **Encoding**: Uses **Label Encoding** for high-cardinality categorical variables (Country, Crop) to preserve meaningful distances for the SVR kernel.

### 4. SVR Kernel Comparison (`step4_svr.py`)
- **Action**: Compares Linear, Polynomial, and RBF kernels.
- **Result**: The **RBF (Radial Basis Function)** kernel performed best, confirming that climate-yield relationships are non-linear.

### 5. Hyperparameter Tuning (`step5_tuning.py`)
- **Action**: Executes a `GridSearchCV` over `C`, `epsilon`, and `gamma`.
- **Optimization**: Uses a `Pipeline` to ensure `StandardScaler` is only fit on training data, preventing data leakage.
- **Outcome**: Optimal parameters saved in `models/best_svr_model.pkl`.

### 6. Validation & Ablation (`step6_validation.py`)
- **Action**: Compares the tuned model against Mean and Median baselines.
- **Validation**: Uses **Permutation Importance** to rank features.
- **Discovery**: `log_pesticides` and `avg_temp` are the primary drivers of yield prediction.

---

## 📊 Final Performance Metrics

| Metric | Baseline (Mean) | **AgriVision (SVR)** | Improvement |
| :--- | :--- | :--- | :--- |
| **MAE** | 6.48 t/ha | **2.30 t/ha** | 64.5% ↓ |
| **RMSE** | 8.51 t/ha | **3.84 t/ha** | 54.8% ↓ |
| **R²** | 0.000 | **0.796** | +0.796 |

---

## 🧠 Key Design Philosophy
1. **Target Normalization**: We normalize yield *per crop*. This prevents the model from being biased toward high-yield crops (like Cassava) and allows it to focus on climate impacts within each crop family.
2. **Temporal Integrity**: We purposefully dropped the `Year` column to avoid "learning the trend" (temporal leakage) and force the model to learn the underlying physics/climate relationships.
3. **Robustness**: SVR with an RBF kernel provides high resistance to outliers compared to standard Linear Regression.

---

## 📈 Visualizing Success
- **Predicted vs. Actual**: Most points cluster tightly around the 45-degree line.
- **Residuals**: The error distribution is centered tightly on zero with no significant bias.
- **Feature Importance**: Confirms that Pesticides and Temperature are more critical than Country/Crop identity alone.
