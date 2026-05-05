import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
import shutil
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH  = Path("data/hybrid_dataset.csv")
ARTIFACT_PATH = Path("models/phase2/bilstm_artifact.pkl")
MODEL_PATH = Path("models/phase3/hybrid_svr_model.pkl")

if MODEL_PATH.parent.exists():
    shutil.rmtree(MODEL_PATH.parent)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "country_enc", "crop_enc", "rainfall_mm", "avg_temp",
    "pesticides_tonnes", "rainfall_temp_ratio", "log_pesticides", "heat_stress_index"
]
TARGET_COL   = "yield_tonnes_per_ha"
RANDOM_STATE = 42

PARAM_GRID = {
    "svr__C"       : [10, 50, 100],
    "svr__gamma"   : ["scale", 0.01, 0.1],
    "svr__epsilon" : [0.1, 0.5, 1.0],
}

def train_hybrid_svr():
    print("=" * 60)
    print("  STEP 9 — HYBRID SVR TRAINING (Phase 3)")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    artifact = joblib.load(ARTIFACT_PATH)
    
    # Extract the exact same splits as Phase 2
    idx_tr = artifact["idx_train"]
    idx_vl = artifact["idx_val"]
    idx_te = artifact["idx_test"]
    
    # We will combine train and val for training the final SVR
    idx_train_full = np.concatenate([idx_tr, idx_vl])
    
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    hybrid_features = FEATURE_COLS + emb_cols
    
    X = df[hybrid_features]
    
    # Target normalization is already computed in df if we use yield_norm, but let's recompute or use crop_stats
    crop_stats = df.groupby("crop_enc")[TARGET_COL].agg(["mean","std"]).rename(
        columns={"mean":"crop_mean","std":"crop_std"}
    )
    crop_stats["crop_std"] = crop_stats["crop_std"].replace(0, 1)
    
    # For safety, ensure df has crop_mean and crop_std properly aligned
    df_with_stats = df.copy()
    if 'crop_mean' not in df_with_stats.columns:
        df_with_stats = df_with_stats.merge(crop_stats, on="crop_enc", how="left")
    
    y_norm = (df_with_stats[TARGET_COL] - df_with_stats["crop_mean"]) / df_with_stats["crop_std"]
    y_orig = df_with_stats[TARGET_COL]
    
    X_train = X.iloc[idx_train_full]
    y_train_norm = y_norm.iloc[idx_train_full]
    
    X_test = X.iloc[idx_te]
    y_test_orig = y_orig.iloc[idx_te]
    df_test_meta = df_with_stats.iloc[idx_te][["crop_enc", "crop_mean", "crop_std"]]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Hybrid feature dimension: {len(hybrid_features)} ({len(FEATURE_COLS)} static + {len(emb_cols)} embeddings)")
    
    # Subsample for grid search to speed it up
    idx_gs = np.random.RandomState(RANDOM_STATE).choice(len(X_train), 5000, replace=False)
    X_gs = X_train.iloc[idx_gs]
    y_gs = y_train_norm.iloc[idx_gs]
    
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
    print("\n  Running GridSearchCV...")
    gs.fit(X_gs, y_gs)
    
    best_params  = {k.replace("svr__", ""): v for k, v in gs.best_params_.items()}
    print(f"\n  Best params : {best_params}")
    print(f"  Best CV RMSE (normalised): {-gs.best_score_:.4f}")
    
    # Retrain on full training set
    print(f"\n  Retraining Hybrid Model on full {len(X_train)} rows...")
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", max_iter=8000, **best_params))
    ])
    final_pipe.fit(X_train, y_train_norm)
    
    # Evaluate
    preds_norm = final_pipe.predict(X_test)
    preds_orig = (preds_norm * df_test_meta["crop_std"].values) + df_test_meta["crop_mean"].values
    y_test_arr = np.array(y_test_orig)
    
    mae  = mean_absolute_error(y_test_arr, preds_orig)
    rmse = np.sqrt(mean_squared_error(y_test_arr, preds_orig))
    r2   = r2_score(y_test_arr, preds_orig)
    
    print(f"\n  Final Hybrid Test Results (original scale):")
    print(f"  MAE  = {mae:.4f} t/ha")
    print(f"  RMSE = {rmse:.4f} t/ha")
    print(f"  R2   = {r2:.4f}")
    
    # ── Plot: Predicted vs Actual & Residuals ─────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_PATH = Path("results/phase3")
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Predicted vs Actual
    sample = np.random.choice(len(y_test_arr), min(800, len(y_test_arr)), replace=False)
    axes[0].scatter(y_test_arr[sample], preds_orig[sample], alpha=0.35, s=12, color="#2E75B6")
    lim = [max(0, y_test_arr.min()-0.5), y_test_arr.max()+0.5]
    axes[0].plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Actual Yield (t/ha)")
    axes[0].set_ylabel("Predicted Yield (t/ha)")
    axes[0].set_title(f"Hybrid SVR: Predicted vs Actual\nRMSE={rmse:.3f}  R²={r2:.3f}", fontweight="bold")
    axes[0].legend()
    
    # 2. Residuals
    residuals = y_test_arr - preds_orig
    axes[1].hist(residuals, bins=60, color="#70AD47", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Prediction Error (t/ha)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Hybrid SVR: Residual Distribution", fontweight="bold")
    
    plt.suptitle("Phase 3: Hybrid Model (DL Embeddings + SVR)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "hybrid_results.png", dpi=150)
    plt.close()
    print(f"  ✅ Plot saved: {RESULTS_PATH}/hybrid_results.png")
    
    joblib.dump({
        "pipeline"    : final_pipe,
        "best_params" : best_params,
        "hybrid_features": hybrid_features,
        "mae"         : mae,
        "rmse"        : rmse,
        "r2"          : r2,
    }, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    train_hybrid_svr()
