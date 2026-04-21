"""
STEP 6 — Validation and Ablation Study
Works with per-crop normalised model from step5.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import os

MODEL_PATH   = Path("models/phase1/best_svr_model.pkl")
RESULTS_PATH = Path("results/phase1")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Surgical cleanup: only delete files this script produces to avoid removing step4 plots
for f in ["ablation_results.csv", "svr_predicted_vs_actual.png", "svr_residuals.png", "ablation_comparison.png", "feature_importance.png"]:
    file_path = RESULTS_PATH / f
    if file_path.exists():
        os.remove(file_path)

def run_validation():
    saved      = joblib.load(MODEL_PATH)
    pipeline   = saved["pipeline"]
    X_test     = saved["X_test"]
    y_test     = np.array(saved["y_test"])
    y_test_norm= np.array(saved["y_test_norm"])
    feat_cols  = saved["feature_cols"]
    df_meta    = saved["df_test_meta"]

    print("=" * 60)
    print("  VALIDATION AND ABLATION STUDY")
    print("=" * 60)

    # SVR predicts normalised, inverse-transform to original
    preds_norm = pipeline.predict(X_test)
    preds_orig = (preds_norm * df_meta["crop_std"].values) + df_meta["crop_mean"].values

    mean_preds   = np.full(len(y_test), y_test.mean())
    median_preds = np.full(len(y_test), np.median(y_test))

    models = {
        "Mean Predictor"   : mean_preds,
        "Median Predictor" : median_preds,
        "Tuned SVR"        : preds_orig,
    }

    rows = []
    for name, preds in models.items():
        rows.append({
            "Model" : name,
            "MAE"   : round(mean_absolute_error(y_test, preds), 4),
            "RMSE"  : round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            "R2"    : round(r2_score(y_test, preds), 4),
        })

    results_df = pd.DataFrame(rows)
    print("\n  ABLATION RESULTS:")
    print(results_df.to_string(index=False))

    svr_row  = results_df[results_df["Model"] == "Tuned SVR"].iloc[0]
    mean_row = results_df[results_df["Model"] == "Mean Predictor"].iloc[0]
    pct      = (mean_row["RMSE"] - svr_row["RMSE"]) / mean_row["RMSE"] * 100
    print(f"\n  SVR improves RMSE by {pct:.1f}% over mean predictor")
    print(f"  R2 = {svr_row['R2']} — model explains {svr_row['R2']*100:.1f}% of yield variance")

    results_df.to_csv(RESULTS_PATH / "ablation_results.csv", index=False)

    # Plot 1 — Predicted vs Actual
    plt.figure(figsize=(7, 6))
    idx = np.random.choice(len(y_test), min(800, len(y_test)), replace=False)
    plt.scatter(y_test[idx], preds_orig[idx], alpha=0.35, s=12, color="#2E75B6")
    lim = [max(0, y_test.min()-0.5), y_test.max()+0.5]
    plt.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Actual Yield (t/ha)")
    plt.ylabel("Predicted Yield (t/ha)")
    plt.title(f"SVR Predicted vs Actual\nRMSE={svr_row['RMSE']}  R2={svr_row['R2']}", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "svr_predicted_vs_actual.png", dpi=150)
    plt.close()

    # Plot 2 — Residuals
    residuals = y_test - preds_orig
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=60, color="#70AD47", edgecolor="white", linewidth=0.3)
    plt.axvline(0, color="red", linestyle="--", lw=1.5)
    plt.title("Residual Distribution (centred on 0 = no bias)", fontweight="bold")
    plt.xlabel("Prediction Error (t/ha)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "svr_residuals.png", dpi=150)
    plt.close()

    # Plot 3 — Ablation bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#C00000", "#ED7D31", "#2E75B6"]
    bars   = ax.bar(results_df["Model"], results_df["RMSE"],
                    color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylabel("RMSE (t/ha)")
    ax.set_title("Ablation Study — SVR vs Baselines", fontweight="bold")
    ax.set_ylim(0, results_df["RMSE"].max() * 1.25)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "ablation_comparison.png", dpi=150)
    plt.close()

    # Plot 4 — Feature importance (on normalised target)
    print("\n  Computing permutation importance...")
    n   = min(2000, len(X_test))
    idx = np.random.choice(len(X_test), n, replace=False)
    X_sub = X_test.iloc[idx]
    y_sub = y_test_norm[idx]

    perm       = permutation_importance(pipeline, X_sub, y_sub,
                                        n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = perm.importances_mean.argsort()[::-1]

    print("\n  Feature Importance:")
    for i in sorted_idx:
        print(f"    {feat_cols[i]:<30} {perm.importances_mean[i]:+.4f}")

    plt.figure(figsize=(8, 5))
    plt.barh([feat_cols[i] for i in sorted_idx],
             perm.importances_mean[sorted_idx],
             color="#2E75B6", edgecolor="white")
    plt.xlabel("Mean R2 decrease")
    plt.title("Feature Importance — Permutation Method", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "feature_importance.png", dpi=150)
    plt.close()

    print(f"\n  All plots saved to {RESULTS_PATH}/")
    print("=" * 60)

if __name__ == "__main__":
    run_validation()
