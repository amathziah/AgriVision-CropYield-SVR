#!/bin/bash
set -e

echo "=========================================================="
echo "    AgriVision - Full Reproducible Pipeline (Phase 1-3)   "
echo "=========================================================="

echo "[1/11] Loading Data..."
python src/step1_load.py

echo "[2/11] Exploratory Data Analysis..."
python src/step2_eda.py

echo "[3/11] Engineering Features..."
python src/step3_features.py

echo "[4/11] SVR Kernel Comparison..."
python src/step4_svr.py

echo "[5/11] SVR Hyperparameter Tuning..."
python src/step5_tuning.py

echo "[6/11] SVR Validation & Ablation (Phase 1)..."
python src/step6_validation.py

echo "[7/11] Deep Learning Model (BiLSTM + Attention)..."
python src/step7_dl_model.py

echo "[8/11] Hybrid Feature Extraction (BiLSTM Embeddings)..."
python src/step8_hybrid_features.py

echo "[9/11] Hybrid Model Training (Phase 3)..."
python src/step9_hybrid_model.py

echo "[10/11] Deep Diagnostic Ablation Study..."
python src/step10_ablation.py

echo "[11/11] Phase 3 Per-Crop & Cross-Phase Analysis..."
python src/step11_phase3_analysis.py

echo "=========================================================="
echo "    Pipeline completed successfully! Results in results/  "
echo "=========================================================="
