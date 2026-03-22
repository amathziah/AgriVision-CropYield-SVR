# AgriVision — Satellite-Based Precision Agriculture

Crop yield prediction using Support Vector Regression on FAO climate data.
Predicts yield in tonnes/ha for 101 countries and 10 crop types.

## Results

| Model | MAE (t/ha) | RMSE (t/ha) | R² |
|-------|-----------|------------|-----|
| Mean Predictor (baseline) | 6.48 | 8.51 | 0.000 |
| Median Predictor (baseline) | 5.81 | 9.31 | -0.196 |
| **Tuned SVR (ours)** | **2.30** | **3.84** | **0.796** |

SVR improves RMSE by **54.8%** over the mean predictor baseline.

## Dataset

Source: [FAO Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

- 28,242 rows — one row = one country, one crop, one year
- 101 countries, 10 crop types, years 1990–2013
- Features: rainfall (mm), temperature (°C), pesticides (tonnes)
- Target: yield in tonnes per hectare

## Project Structure

```
AgriVision/
├── data/
│   ├── yield_df.csv          ← raw dataset (download from Kaggle)
│   ├── final_dataset.csv     ← output of step1
│   └── features_dataset.csv  ← output of step3
├── src/
│   ├── step1_load.py         ← clean and rename columns
│   ├── step2_eda.py          ← distribution analysis and plots
│   ├── step3_features.py     ← feature engineering
│   ├── step4_svr.py          ← 3 kernel comparison
│   ├── step5_tuning.py       ← GridSearchCV hyperparameter tuning
│   └── step6_validation.py   ← ablation study and feature importance
├── models/
│   └── best_svr_model.pkl    ← saved trained model
├── results/                  ← all plots and ablation_results.csv
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

```bash
python src/step1_load.py
python src/step2_eda.py
python src/step3_features.py
python src/step4_svr.py
python src/step5_tuning.py     # takes 3-5 minutes (81 model fits)
python src/step6_validation.py
```

## Key Design Decisions

**Why SVR over Linear Regression?**
Climate-yield relationships are non-linear. The RBF kernel maps features to
an infinite-dimensional space where non-linearities become linear, without
explicitly computing the high-dimensional mapping (kernel trick).

**Why Label Encoding over One-Hot Encoding?**
OHE on 101 countries creates 101 sparse binary columns. SVR's RBF kernel
computes Euclidean distances — in sparse OHE space every country is
equidistant. Label encoding + StandardScaler preserves meaningful distances.

**Why drop the year column?**
Including year causes temporal leakage: the model learns yields increase over
time (technology trend) rather than the climate-yield relationship.

**Why per-crop target normalisation?**
Cassava averages 14 t/ha; Soybeans 1.7 t/ha. SVR cannot fit both on one
scale. Z-score normalisation within each crop group teaches the model
climate effects within crops, not just which crop produces more.

**Why StandardScaler inside Pipeline?**
Scaler must be fit only on training data. Pipeline ensures it never sees
test data during GridSearchCV, preventing data leakage.

## Feature Importance (Permutation Method)

| Feature | R² decrease when shuffled |
|---------|--------------------------|
| log_pesticides | 0.477 |
| avg_temp | 0.401 |
| rainfall_mm | 0.325 |
| rainfall_temp_ratio | 0.308 |
| country_enc | 0.279 |
# AgriVision-CropYield-SVR
