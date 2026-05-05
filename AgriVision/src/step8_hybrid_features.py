import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from step7_dl_model import build_sequences, BiLSTMAttention, TEMPORAL_FEATURES, STATIC_FEATURES, TARGET_COL, DEVICE, LOOKBACK

def extract_hybrid_features():
    print("=" * 60)
    print("  STEP 8 — HYBRID FEATURE EXTRACTION (Phase 3)")
    print("=" * 60)
    
    data_path = Path("data/features_dataset.csv")
    artifact_path = Path("models/phase2/bilstm_artifact.pkl")
    out_path = Path("data/hybrid_dataset.csv")
    
    if not artifact_path.exists():
        print("Error: BiLSTM artifact not found. Run step7 first.")
        return
        
    df = pd.read_csv(data_path)
    artifact = joblib.load(artifact_path)
    
    # Per-crop z-score normalisation of target for BiLSTM
    crop_stats = df.groupby("crop_enc")[TARGET_COL].agg(["mean", "std"]).rename(
        columns={"mean": "crop_mean", "std": "crop_std"}
    )
    crop_stats["crop_std"] = crop_stats["crop_std"].replace(0, 1)
    df = df.merge(crop_stats, on="crop_enc")
    
    print(f"  Building lookback sequences (k={LOOKBACK})...")
    seqs, stats, tgts_orig, crop_means, crop_stds = build_sequences(df)
    
    # Load model
    model_config = artifact["model_config"]
    model = BiLSTMAttention(**model_config).to(DEVICE)
    model.load_state_dict(artifact["model_state"])
    model.eval()
    
    # We will use a forward hook to extract the 64-dim embeddings right before the final fc_out
    embeddings_list = []
    
    def hook(module, input, output):
        embeddings_list.append(input[0].detach().cpu().numpy())
        
    handle = model.fc_out.register_forward_hook(hook)
    
    seq_scaler = artifact["scalers"]["seq"]
    stat_scaler = artifact["scalers"]["stat"]
    
    B, T, F = seqs.shape
    
    # Process in batches to avoid OOM
    batch_size = 512
    for i in range(0, B, batch_size):
        s_batch = seqs[i:i+batch_size]
        st_batch = stats[i:i+batch_size]
        
        s_scaled = seq_scaler.transform(s_batch.reshape(-1, F)).reshape(len(s_batch), T, F).astype(np.float32)
        st_scaled = stat_scaler.transform(st_batch).astype(np.float32)
        
        with torch.no_grad():
            _ = model(torch.FloatTensor(s_scaled).to(DEVICE), torch.FloatTensor(st_scaled).to(DEVICE))
            
    handle.remove()
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"  Extracted embeddings shape: {embeddings.shape}")
    
    # Reconstruct the DataFrame in the same order as build_sequences outputs
    df_sorted = df.copy().sort_values(["country", "crop", "year"]).reset_index(drop=True)
    
    # Add the 64 embedding columns
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    
    df_hybrid = pd.concat([df_sorted, emb_df], axis=1)
    df_hybrid.to_csv(out_path, index=False)
    
    print(f"  Hybrid dataset saved → {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    extract_hybrid_features()
