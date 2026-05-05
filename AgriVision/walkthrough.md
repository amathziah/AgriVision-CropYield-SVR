# AgriVision — Complete Project Walkthrough

AgriVision is a two-phase machine learning project that predicts agricultural crop yields
(tonnes per hectare) using climate and pesticide data. It progresses from classical ML (SVR)
through deep learning (BiLSTM + Attention).

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [File Structure](#2-file-structure)
3. [Phase 1 — Classical ML with SVR](#3-phase-1--classical-ml-with-svr)
4. [Phase 2 — Deep Learning with BiLSTM + Attention](#4-phase-2--deep-learning-with-bilstm--attention)
5. [Final Results Across All Phases](#5-final-results-across-all-phases)
6. [How to Run](#6-how-to-run)

---

## 1. Dataset

### Source

The dataset comes from **Kaggle — Crop Yield Prediction Dataset**
(uploaded by user `patelris`), which assembles data from two official UN sources:

| Column | Original Source |
|--------|----------------|
| `hg/ha_yield` | FAO FAOSTAT — Crop Production Statistics |
| `pesticides_tonnes` | FAO FAOSTAT — Pesticide Use Database |
| `average_rain_fall_mm_per_year` | World Bank Climate Data Portal |
| `avg_temp` | World Bank Climate Data Portal |

### What One Row Means

Each row represents **one country + one crop + one year**:

```
Albania | Maize | 1990 | rainfall=1485mm | temp=16.37°C | pesticides=121t → yield=3.66 t/ha
```

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Total rows | 28,242 |
| Countries | 101 (Albania → Zimbabwe) |
| Crops | 10 (see below) |
| Year range | 1990 – 2013 (24 years) |
| Missing values | 0 |
| Yield unit (raw) | hg/ha (hectograms per hectare) |
| Yield unit (used) | t/ha (tonnes per hectare, converted by ÷ 10,000) |

The 10 crops are: Cassava, Maize, Plantains, Potatoes, Rice (paddy), Sorghum,
Soybeans, Sweet Potatoes, Wheat, Yams.

### Important Limitation

Climate features are **country-level annual averages**, not field-level measurements.
"Albania has 1485mm rainfall" means the national average — not the specific field
where the crop grew. This limits granularity but is consistent across the full dataset.

---

## 2. File Structure

```
AgriVision/
│
├── data/
│   ├── yield_df.csv            Raw dataset — download from Kaggle
│   ├── final_dataset.csv       Output of step1 — cleaned, renamed columns
│   └── features_dataset.csv    Output of step3 — engineered features + year preserved
│
├── src/
│   ├── step1_load.py           Load, clean, convert yield units
│   ├── step2_eda.py            Exploratory data analysis + 3 plots
│   ├── step3_features.py       Feature engineering (label encode, interaction features)
│   ├── step4_svr.py            Compare Linear / RBF / Polynomial SVR kernels
│   ├── step5_tuning.py         GridSearchCV + per-crop target normalisation
│   ├── step6_validation.py     Ablation vs baselines + permutation importance
│   └── step7_dl_model.py       BiLSTM + Attention (Phase 2)
│
├── models/
│   ├── phase1/best_svr_model.pkl       Trained SVR (Phase 1)
│   └── phase2/bilstm_artifact.pkl      Trained BiLSTM + embeddings (Phase 2)
│
├── results/
│   ├── phase1/
│   │   ├── eda_yield_distribution.png
│   │   ├── eda_yield_by_crop.png
│   │   ├── eda_correlation.png
│   │   ├── svr_kernel_comparison.png
│   │   ├── svr_predicted_vs_actual.png
│   │   ├── svr_residuals.png
│   │   ├── ablation_comparison.png
│   │   ├── ablation_results.csv
│   │   └── feature_importance.png
│   └── phase2/
│       ├── dl_results.png              Learning curves + predicted vs actual
│       ├── dl_per_crop_r2.png          Per-crop R² for BiLSTM model
│       └── dl_attention_weights.png    Which lookback years matter most
│
├── presentation/
│   └── phase1/
│       ├── AgriVision_Phase1.pptx
│       └── AgriVision_Simulator.html
│
├── report/
│   └── phase1/AgriVision_Phase1_Report.docx
│
├── AgriVision.ipynb            Full notebook version of the pipeline
├── requirements.txt
└── walkthrough.md              This file
```

---

## 3. Phase 1 — Classical ML with SVR

**Goal**: Predict crop yield using Support Vector Regression on tabular climate features.

### Step 1 — Load and Clean (`step1_load.py`)

Reads `yield_df.csv`, renames columns to clean names, and converts yield units:

```
hg/ha  ×  0.0001  =  t/ha
```

Drops duplicates, reports nulls (zero), saves `data/final_dataset.csv`.

Key columns after cleaning:

| Column | Description |
|--------|-------------|
| `country` | Country name |
| `crop` | Crop type |
| `year` | Year (1990–2013) |
| `rainfall_mm` | Annual average rainfall (mm) |
| `avg_temp` | Annual average temperature (°C) |
| `pesticides_tonnes` | Total pesticides applied nationally (tonnes) |
| `yield_tonnes_per_ha` | Target variable |

### Step 2 — Exploratory Data Analysis (`step2_eda.py`)

Three plots saved to `results/`:

**eda_yield_distribution.png** — Yield is right-skewed (skewness ~4).
The log-transformed version is near-normal, justifying `log_pesticides` as a feature.

**eda_yield_by_crop.png** — Massive scale gap between crops:
Cassava averages ~14 t/ha; Soybeans average ~1.7 t/ha.
This motivates per-crop z-score normalisation of the target variable.

**eda_correlation.png** — Pesticides have the highest raw correlation with yield
(technology adoption proxy). Rainfall has the weakest, partially confounded by crop type.

### Step 3 — Feature Engineering (`step3_features.py`)

**Label encoding** for `country` (101 classes) and `crop` (10 classes) instead of
one-hot encoding. Reason: SVR's RBF kernel computes Euclidean distances. In OHE space
every country is equidistant (orthogonal sparse vectors). Label encoding + StandardScaler
lets the kernel find meaningful country-level proximity.

**Year** is excluded from `FEATURE_COLS` for SVR (temporal leakage — year encodes
rising technology trend, not climate effects) but is **preserved in the CSV** for
Phase 2 temporal sequence building.

Three engineered interaction features:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `rainfall_temp_ratio` | `rainfall_mm / (avg_temp + 1)` | Captures effective moisture: high rainfall under high temp still causes evapotranspiration stress |
| `log_pesticides` | `log1p(pesticides_tonnes)` | Diminishing returns on pesticides; compresses the 300,000-tonne outliers |
| `heat_stress_index` | `max(0, avg_temp × (1 - rainfall_mm/3000))` | Hot + dry conjunction; clipped at 0 only so the feature remains continuous for StandardScaler |

Final feature set (8 columns): `country_enc`, `crop_enc`, `rainfall_mm`, `avg_temp`,
`pesticides_tonnes`, `rainfall_temp_ratio`, `log_pesticides`, `heat_stress_index`.

### Step 4 — SVR Kernel Comparison (`step4_svr.py`)

Trains Linear, RBF, and Polynomial SVR kernels on an 8,000-row subsample
(SVR is O(n²) — full dataset would take hours).

StandardScaler lives **inside the Pipeline** so it is fit only on training data
during each cross-validation fold. Fitting the scaler before splitting would let
test statistics leak into training.

Result: **RBF wins** — confirms climate-yield relationships are non-linear.

### Step 5 — Hyperparameter Tuning (`step5_tuning.py`)

Per-crop z-score normalisation of the target before training:

```
yield_norm = (yield - crop_mean) / crop_std
```

This teaches the model "how far above/below average is this yield given these
climate conditions?" rather than "which crop produces more?". Predictions are
inverse-transformed back to t/ha for evaluation.

GridSearchCV over 81 combinations (C × gamma × epsilon, 3-fold CV):

| Parameter | Values searched |
|-----------|----------------|
| C | 1, 10, 100 |
| gamma | scale, 0.01, 0.1 |
| epsilon | 0.1, 0.5, 1.0 |

Best params: `C=10, epsilon=0.5, gamma=scale`. Model saved to `models/phase1/best_svr_model.pkl`.

### Step 6 — Validation and Ablation (`step6_validation.py`)

Compares Tuned SVR against Mean Predictor and Median Predictor baselines.
Permutation importance: shuffles each feature 5 times, measures average R² drop.

Phase 1 results:

| Model | MAE (t/ha) | RMSE (t/ha) | R² |
|-------|-----------|-------------|-----|
| Mean Predictor | 6.48 | 8.51 | 0.000 |
| Median Predictor | 5.81 | 9.31 | -0.196 |
| Tuned SVR | 2.41 | 3.98 | 0.782 |

Top features by permutation importance:

| Feature | R² drop when shuffled |
|---------|-----------------------|
| log_pesticides | 0.465 |
| rainfall_temp_ratio | 0.423 |
| avg_temp | 0.393 |
| rainfall_mm | 0.338 |
| heat_stress_index | 0.262 |

---

## 4. Phase 2 — Deep Learning with BiLSTM + Attention

**Goal**: Exploit the temporal dimension of the dataset that SVR ignores.
Each (country, crop) pair has up to 24 yearly observations — a natural time series.

### Key Idea: Lookback Sequences

For each row (country, crop, year t), a **5-year lookback window** is built:
```
[year t-4, year t-3, year t-2, year t-1, year t]  →  predict yield at year t
```
Years with fewer than 5 years of history (e.g., 1990–1993) are left-padded with zeros.
This gives 25,932 sequences of shape `(5 timesteps, 5 temporal features)`.

The 5 temporal features per timestep are:
`rainfall_mm`, `avg_temp`, `log_pesticides`, `heat_stress_index`, `rainfall_temp_ratio`

Static features (`country_enc`, `crop_enc`) are passed separately and merged
via a skip connection — so the LSTM focuses on climate patterns while the skip
path carries country/crop identity.

### Step 7 — BiLSTM + Attention (`step7_dl_model.py`)

**Architecture**:

```
Input sequences    (B, 5, 5)
        ↓
BiLSTM × 2 layers  (B, 5, 128)    hidden=64, bidirectional, dropout=0.3
        ↓
Temporal Attention (B, 128)        α = softmax(v · tanh(W · h))
        ↓
Concat + static    (B, 130)        append country_enc, crop_enc
        ↓
Dense(128) + BN + Dropout(0.3)
        ↓
Dense(64)  + BN
        ↓
+ Skip(static)                     residual from country/crop encodings
        ↓
Dense(1)   →  normalised yield prediction
```

**Why BiLSTM over standard LSTM**:
Forward pass captures "yields have been rising for 3 years → likely to continue".
Backward pass (during training) learns "high yield at t is preceded by 2 wet years".
Together they produce richer temporal representations.

**Why Attention**:
Not all years in the lookback window are equally informative. A drought 4 years ago
may matter more than last year's normal conditions. Attention learns a scalar weight
α_t for each timestep and computes a weighted sum: `context = Σ α_t × h_t`.

**Why the skip connection**:
Separates two concerns. The LSTM learns climate deviation patterns (domain-invariant).
The skip path carries country/crop baseline identity directly to the output layer,
preventing the LSTM from wasting capacity memorising which country is which.

**Training setup**:

| Setting | Value |
|---------|-------|
| Split | Stratified 70/15/15 by crop |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | Patience=15 epochs |
| Batch size | 256 |
| Gradient clipping | max_norm=1.0 |
| Target | Per-crop z-score normalised (same as SVR) |

**Data leakage prevention**: StandardScaler is fit only on training sequences.
The scaler never sees validation or test data.

**Phase 2 results**:

| Metric | Value |
|--------|-------|
| MAE | 1.28 t/ha |
| RMSE | 2.28 t/ha |
| R² | 0.929 |

The BiLSTM improves over SVR because it captures multi-year climate patterns:
a sequence of 3 dry years depresses yield differently than a single dry year,
something SVR cannot express by treating each row independently.

**Outputs**:
- `results/phase2/dl_results.png` — learning curves + predicted vs actual
- `results/phase2/dl_per_crop_r2.png` — per-crop R² breakdown
- `results/phase2/dl_attention_weights.png` — attention weights showing which lookback years matter
- `models/phase2/bilstm_artifact.pkl` — model weights, scalers, and 128-dim embeddings for every row

---

## 5. Final Results Across All Phases

| Phase | Model | RMSE (t/ha) | R² |
|-------|-------|-------------|-----|
| Baseline | Mean Predictor | 8.54 | 0.000 |
| Phase 1 | SVR (RBF, tuned) | 3.98 | 0.782 |
| Phase 2 | BiLSTM + Attention | 2.28 | 0.929 |

Each phase strictly improves over the previous. The progression demonstrates that:
1. Non-linear kernel methods (SVR) outperform trivial baselines on tabular climate data
2. Exploiting the temporal structure of the data (BiLSTM) yields a large additional gain

---

## 6. Phase 3 — Hybrid Neuro-Symbolic Model

**Goal**: Combine the temporal extraction power of the BiLSTM with the robust, non-linear margin regression of SVR to create a true Synergistic Hybrid Model.

### Architecture Diagram

```mermaid
graph TD
    A[Raw Data] --> B[Static Features]
    A --> C[5-Year Lookback Sequences (Batch, 5, 5)]
    
    subgraph Phase 2: Deep Learning (BiLSTM)
        C --> D[BiLSTM Layers]
        D --> E[Temporal Attention Context α]
        E --> F[Dense Layer 128]
        F --> G[Penultimate Embeddings 64-dim]
    end
    
    subgraph Phase 3: Hybrid Neuro-Symbolic Fusion
        B --> H((Concat))
        G --> H
        H --> I[Hybrid Feature Vector]
        I --> J[Tuned RBF SVR]
        J --> K[Final Yield Prediction]
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
```

### Diagnostic Ablation Study (`step10_ablation.py`)
To prove the necessity of the hybrid complexity, we removed the DL and ML components and evaluated them on the exact same test set:

| Model | Components | Impact of Removal |
|-------|------------|-------------------|
| Pure ML | Static SVR (No DL) | Fails to capture multi-year climate patterns. R² drops significantly. |
| Pure DL | BiLSTM (No SVR) | Good temporal understanding, but less robust to tabular outliers than SVR. |
| Embeddings Only | SVR on DL Embeddings (No Static) | Missing country/crop baselines causes a slight performance penalty. |
| **Hybrid** | **SVR on Embeddings + Static Features** | **Highest performance**. Synergistically leverages both components. |

---

## 7. How to Run (100% Reproducible)

### Using Docker (Recommended)
We have implemented a turn-key Docker environment for full reproducibility:

```bash
# Run the complete Phase 1-3 pipeline automatically
docker-compose up pipeline

# Launch the interactive Web UI (Gradio)
docker-compose up webui
# Open http://localhost:7860
```

### Manual Setup

```bash
pip install -r requirements.txt

# Run the full pipeline sequentially
bash run_pipeline.sh
```

### Interactive Web UI (`app.py`)
An extra mile addition: A functional Gradio Web UI that allows you to input custom 5-year climate histories for specific countries and crops, and runs a real-time inference using the Phase 3 Hybrid Model.

### Pretrained models
All models are already trained and saved in `models/`. The pipeline will automatically overwrite these with freshly trained instances when run.
