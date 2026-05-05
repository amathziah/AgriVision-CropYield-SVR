import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/hybrid_dataset.csv")
ARTIFACT_PATH = Path("models/phase2/bilstm_artifact.pkl")
HYBRID_MODEL_PATH = Path("models/phase3/hybrid_svr_model.pkl")
RESULTS_PATH = Path("results/phase3")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

TARGET_COL = "yield_tonnes_per_ha"


def per_crop_breakdown():
    print("=" * 60)
    print("  STEP 11 — PHASE 3 PER-CROP & CROSS-PHASE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    artifact = joblib.load(ARTIFACT_PATH)
    hybrid = joblib.load(HYBRID_MODEL_PATH)

    idx_te = artifact["idx_test"]
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    hybrid_features = hybrid["hybrid_features"]

    # Recompute crop normalisation
    crop_stats = df.groupby("crop_enc")[TARGET_COL].agg(["mean", "std"]).rename(
        columns={"mean": "crop_mean", "std": "crop_std"}
    )
    crop_stats["crop_std"] = crop_stats["crop_std"].replace(0, 1)
    df_meta = df.copy()
    if "crop_mean" not in df_meta.columns:
        df_meta = df_meta.merge(crop_stats, on="crop_enc", how="left")

    test_df = df_meta.iloc[idx_te].copy().reset_index(drop=True)
    X_test = df[hybrid_features].iloc[idx_te].reset_index(drop=True)

    pred_norm = hybrid["pipeline"].predict(X_test)
    pred_orig = pred_norm * test_df["crop_std"].values + test_df["crop_mean"].values
    test_df["pred"] = pred_orig

    # Per-crop R² / RMSE / MAE
    rows = []
    for crop, sub in test_df.groupby("crop"):
        if len(sub) < 5:
            continue
        y_true = sub[TARGET_COL].values
        y_pred = sub["pred"].values
        rows.append({
            "crop": crop,
            "n": len(sub),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
        })
    per_crop = pd.DataFrame(rows).sort_values("R2", ascending=False)
    per_crop.to_csv(RESULTS_PATH / "per_crop_phase3.csv", index=False)
    print("\n  Per-crop performance (Phase 3 winning model):")
    print(per_crop.to_string(index=False))

    # Per-crop bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(per_crop)))
    bars = ax.bar(per_crop["crop"], per_crop["R2"], color=colors, edgecolor="black", linewidth=0.4)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Crop")
    ax.set_title("Phase 3 Hybrid Model — Per-Crop R² on Held-Out Test Set", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "phase3_per_crop_r2.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_PATH}/phase3_per_crop_r2.png")

    # Cross-phase comparison
    cross = pd.DataFrame([
        {"Phase": "Baseline\n(Mean)", "RMSE": 8.51, "R2": 0.000},
        {"Phase": "Phase 1\nSVR (RBF)", "RMSE": 3.98, "R2": 0.782},
        {"Phase": "Phase 2\nBiLSTM+Attn", "RMSE": artifact["rmse"], "R2": artifact["r2"]},
        {"Phase": "Phase 3\nHybrid", "RMSE": hybrid["rmse"], "R2": hybrid["r2"]},
    ])
    print("\n  Cross-phase summary:")
    print(cross.to_string(index=False))
    cross.to_csv(RESULTS_PATH / "cross_phase_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    palette = ["#9aa0a6", "#FFC000", "#70AD47", "#2E75B6"]

    bars1 = axes[0].bar(cross["Phase"], cross["R2"], color=palette,
                        edgecolor="black", linewidth=0.4)
    axes[0].bar_label(bars1, fmt="%.3f", padding=3, fontsize=10)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("R² Progression Across Phases", fontweight="bold")

    bars2 = axes[1].bar(cross["Phase"], cross["RMSE"], color=palette,
                        edgecolor="black", linewidth=0.4)
    axes[1].bar_label(bars2, fmt="%.2f", padding=3, fontsize=10)
    axes[1].set_ylabel("RMSE (t/ha)")
    axes[1].set_title("RMSE Reduction Across Phases", fontweight="bold")

    plt.suptitle("AgriVision: Cumulative Performance Across Pipeline Phases",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "cross_phase_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {RESULTS_PATH}/cross_phase_comparison.png")

    print("=" * 60)


if __name__ == "__main__":
    per_crop_breakdown()
