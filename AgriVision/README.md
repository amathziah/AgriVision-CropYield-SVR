# AgriVision — Satellite-Based Precision Agriculture

Crop yield prediction progressing from classical ML (SVR) to deep learning (BiLSTM + Attention)
on FAO climate data. Predicts yield in tonnes/ha for 101 countries and 10 crop types.

## Results

| Phase | Model | MAE (t/ha) | RMSE (t/ha) | R² |
|-------|-------|-----------|------------|-----|
| Baseline | Mean Predictor | 6.48 | 8.51 | 0.000 |
| Phase 1 | **Tuned SVR (RBF)** | **2.41** | **3.98** | **0.782** |
| Phase 2 | **BiLSTM + Attention** | **1.28** | **2.28** | **0.929** |

Phase 2 BiLSTM improves RMSE by **73.2%** over the mean predictor baseline.
SVR (Phase 1) improves RMSE by **53.3%** over the mean predictor baseline.

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
│   ├── phase1/best_svr_model.pkl     ← Phase 1 (SVR)
│   └── phase2/bilstm_artifact.pkl    ← Phase 2 (BiLSTM+Attention)
├── results/
│   ├── phase1/                       ← EDA + SVR plots, ablation_results.csv
│   └── phase2/                       ← DL learning curves, attention, per-crop R²
├── report/
│   └── phase1/                       ← Phase 1 report
├── presentation/
│   └── phase1/                       ← Phase 1 slides + simulator
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

```bash
# Phase 1 — SVR
python src/step1_load.py
python src/step2_eda.py
python src/step3_features.py
python src/step4_svr.py
python src/step5_tuning.py       # takes 3-5 minutes (81 model fits)
python src/step6_validation.py

# Phase 2 — Deep Learning
python src/step7_dl_model.py     # takes 10-20 minutes on CPU
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
| log_pesticides | 0.465 |
| rainfall_temp_ratio | 0.423 |
| avg_temp | 0.393 |
| rainfall_mm | 0.338 |
| country_enc | 0.292 |
| heat_stress_index | 0.262 |
| crop_enc | 0.234 |
| pesticides_tonnes | 0.169 |
# AgriVision-CropYield-SVR
