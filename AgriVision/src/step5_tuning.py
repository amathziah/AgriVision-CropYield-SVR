"""
STEP 5 — SVR with Per-Crop Normalised Target
WHY: Cassava averages 14 t/ha, Soybeans 1.7 t/ha.
SVR cannot fit both on one scale.
Fix: normalise yield within each crop (z-score per crop),
train SVR on normalised target, inverse-transform for evaluation.
This teaches the model climate effects WITHIN each crop.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH  = Path("data/features_dataset.csv")
MODEL_PATH = Path("models/phase1/best_svr_model.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "country_enc", "crop_enc", "rainfall_mm", "avg_temp",
    "pesticides_tonnes", "rainfall_temp_ratio", "log_pesticides", "heat_stress_index"
]
TARGET_COL   = "yield_tonnes_per_ha"
RANDOM_STATE = 42

PARAM_GRID = {
    "svr__C"       : [1, 10, 100],
    "svr__gamma"   : ["scale", 0.01, 0.1],
    "svr__epsilon" : [0.1, 0.5, 1.0],
}

def tune_svr():
    df = pd.read_csv(DATA_PATH)

    # ── Per-crop z-score normalisation of target ──────────────
    # WHY: Without this, SVR sees Cassava=14 and Soybean=1.7
    # as completely different problems. After normalisation,
    # both become "how far above/below the crop average is this
    # observation given these climate conditions?" — a single
    # learnable function.
    crop_stats = df.groupby("crop_enc")[TARGET_COL].agg(["mean","std"]).rename(
        columns={"mean":"crop_mean","std":"crop_std"}
    )
    crop_stats["crop_std"] = crop_stats["crop_std"].replace(0, 1)
    df = df.merge(crop_stats, on="crop_enc")
    df["yield_norm"] = (df[TARGET_COL] - df["crop_mean"]) / df["crop_std"]

    X = df[FEATURE_COLS]
    y_norm = df["yield_norm"]
    y_orig = df[TARGET_COL]

    # Same split for both normalised and original
    X_train, X_test, y_train_norm, y_test_norm = train_test_split(
        X, y_norm, test_size=0.20, random_state=RANDOM_STATE
    )
    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y_orig, test_size=0.20, random_state=RANDOM_STATE
    )
    _, _, df_train, df_test = train_test_split(
        X, df[["crop_enc","crop_mean","crop_std"]], test_size=0.20, random_state=RANDOM_STATE
    )

    print("=" * 60)
    print("  GRIDSEARCHCV — SVR with Per-Crop Normalised Target")
    print("=" * 60)
    print(f"  Normalised target range: {y_norm.min():.2f} to {y_norm.max():.2f}")
    print(f"  Total fits: 27 x 3 = 81")

    # Subsample for grid search
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), 5000, replace=False)
    X_gs = X_train.iloc[idx]
    y_gs = y_train_norm.iloc[idx]

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", max_iter=5000))
    ])

    gs = GridSearchCV(
        estimator  = base_pipe,
        param_grid = PARAM_GRID,
        cv         = 3,
        scoring    = "neg_root_mean_squared_error",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True
    )
    gs.fit(X_gs, y_gs)

    best_params  = {k.replace("svr__", ""): v for k, v in gs.best_params_.items()}
    print(f"\n  Best params : {best_params}")
    print(f"  Best CV RMSE (normalised): {-gs.best_score_:.4f}")

    # Retrain on full training set
    print(f"\n  Retraining on full {len(X_train)} rows...")
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", max_iter=5000, **best_params))
    ])
    final_pipe.fit(X_train, y_train_norm)

    # Inverse-transform predictions back to original scale
    preds_norm = final_pipe.predict(X_test)
    preds_orig = (preds_norm * df_test["crop_std"].values) + df_test["crop_mean"].values
    y_test_arr = np.array(y_test_orig)

    mae  = mean_absolute_error(y_test_arr, preds_orig)
    rmse = np.sqrt(mean_squared_error(y_test_arr, preds_orig))
    r2   = r2_score(y_test_arr, preds_orig)

    print(f"\n  Final Test Results (original scale):")
    print(f"  MAE  = {mae:.4f} t/ha")
    print(f"  RMSE = {rmse:.4f} t/ha")
    print(f"  R2   = {r2:.4f}")

    # Also show R2 per crop
    test_df = X_test.copy()
    test_df["actual"]    = y_test_arr
    test_df["predicted"] = preds_orig
    print("\n  R2 per crop:")
    for c in sorted(test_df["crop_enc"].unique()):
        sub = test_df[test_df["crop_enc"] == c]
        if len(sub) > 10:
            print(f"    crop_enc={c}  n={len(sub)}  R2={r2_score(sub['actual'], sub['predicted']):.3f}")

    joblib.dump({
        "pipeline"    : final_pipe,
        "best_params" : best_params,
        "X_test"      : X_test,
        "y_test"      : pd.Series(y_test_arr),
        "y_test_norm" : y_test_norm,
        "df_test_meta": df_test,
        "feature_cols": FEATURE_COLS,
        "crop_stats"  : crop_stats,
        "mae"         : mae,
        "rmse"        : rmse,
        "r2"          : r2,
    }, MODEL_PATH)

    print(f"\n  Model saved → {MODEL_PATH}")

if __name__ == "__main__":
    tune_svr()
