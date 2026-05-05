"""
AgriVision Phase 3 — Interactive Web UI

Loads the frozen Phase 2 BiLSTM and the trained Phase 3 hybrid SVR, and exposes
two ways to play with the model:

  1. "Historical Prediction" — pick a real (country, crop, year), load the true
     5-year climate history from the dataset, and see the hybrid prediction
     side-by-side with the actual yield. Also surfaces attention weights so the
     user can see which year drove the prediction.

  2. "Custom Scenario" — type in your own 5-year climate history and crop and
     see what the hybrid would predict.

Run locally:    python app.py
Run in Docker:  docker-compose up webui  →  http://localhost:7860
"""
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from step7_dl_model import BiLSTMAttention, DEVICE, LOOKBACK  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Constants — must match training-time orderings exactly
# ─────────────────────────────────────────────────────────────
TEMPORAL_FEATURES = [
    "rainfall_mm", "avg_temp", "log_pesticides",
    "heat_stress_index", "rainfall_temp_ratio",
]
STATIC_HYBRID_FEATURES = [
    "country_enc", "crop_enc", "rainfall_mm", "avg_temp",
    "pesticides_tonnes", "rainfall_temp_ratio", "log_pesticides", "heat_stress_index",
]

# ─────────────────────────────────────────────────────────────
# Load artifacts once at startup
# ─────────────────────────────────────────────────────────────
ARTIFACTS_OK = True
LOAD_ERROR = ""
try:
    DATA = pd.read_csv("data/features_dataset.csv")
    DL_ART = joblib.load("models/phase2/bilstm_artifact.pkl")
    HYBRID = joblib.load("models/phase3/hybrid_svr_model.pkl")

    BILSTM = BiLSTMAttention(**DL_ART["model_config"]).to(DEVICE)
    BILSTM.load_state_dict(DL_ART["model_state"])
    BILSTM.eval()

    SEQ_SCALER = DL_ART["scalers"]["seq"]
    STAT_SCALER = DL_ART["scalers"]["stat"]
    HYBRID_PIPE = HYBRID["pipeline"]

    # Per-crop yield stats — for normalization inverse and "typical range" display
    CROP_STATS = (
        DATA.groupby(["crop", "crop_enc"])["yield_tonnes_per_ha"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    CROP_STATS["std"] = CROP_STATS["std"].replace(0, 1)

    COUNTRY_LIST = sorted(DATA["country"].unique().tolist())
    CROP_LIST = sorted(DATA["crop"].unique().tolist())
    YEAR_RANGE = (int(DATA["year"].min()), int(DATA["year"].max()))
    print(f"Models loaded. {len(COUNTRY_LIST)} countries, {len(CROP_LIST)} crops, "
          f"years {YEAR_RANGE[0]}–{YEAR_RANGE[1]}")
except Exception as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)
    DATA = pd.DataFrame()
    COUNTRY_LIST = ["India", "Brazil", "USA"]
    CROP_LIST = ["Maize", "Wheat", "Rice, paddy"]
    YEAR_RANGE = (1990, 2013)
    print(f"WARNING: artifacts not loaded — {LOAD_ERROR}")


# ─────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────
def _engineer(rain: float, temp: float, pest: float):
    """Compute the four engineered scalars from raw rainfall / temp / pesticides."""
    log_p = float(np.log1p(max(0.0, pest)))
    ratio = float(rain / (temp + 1.0))
    heat = float(max(0.0, temp * (1.0 - rain / 3000.0)))
    return log_p, ratio, heat


def _build_inputs(country: str, crop: str, years: list[tuple[float, float, float]]):
    """Build the BiLSTM sequence + static tensor and the SVR static block.

    `years` is a length-5 list of (rainfall, temp, pesticides) tuples ordered
    oldest → newest. Returns:
        seq_tensor  : (1, 5, 5) torch float — input to BiLSTM
        stat_tensor : (1, 2)   torch float — input to BiLSTM
        svr_static  : np.ndarray (8,)      — static block of the hybrid SVR vector
        meta        : dict with country_enc, crop_enc, crop_mean, crop_std
    """
    if country not in DATA["country"].values:
        raise ValueError(f"Unknown country '{country}'")
    if crop not in DATA["crop"].values:
        raise ValueError(f"Unknown crop '{crop}'")

    country_enc = int(DATA.loc[DATA["country"] == country, "country_enc"].iloc[0])
    crop_enc = int(DATA.loc[DATA["crop"] == crop, "crop_enc"].iloc[0])
    cs = CROP_STATS.loc[CROP_STATS["crop"] == crop].iloc[0]
    crop_mean, crop_std = float(cs["mean"]), float(cs["std"])

    seq = []
    for r, t, p in years:
        log_p, ratio, heat = _engineer(r, t, p)
        seq.append([r, t, log_p, heat, ratio])  # TEMPORAL_FEATURES order
    seq = np.asarray([seq], dtype=np.float32)  # (1, 5, 5)
    stat = np.asarray([[country_enc, crop_enc]], dtype=np.float32)

    # Apply Phase 2 scalers
    seq_scaled = SEQ_SCALER.transform(seq.reshape(-1, 5)).reshape(1, LOOKBACK, 5).astype(np.float32)
    stat_scaled = STAT_SCALER.transform(stat).astype(np.float32)

    # Static block for the hybrid SVR vector — uses year-t (last) values, raw scale
    r5, t5, p5 = years[-1]
    log_p5, ratio5, heat5 = _engineer(r5, t5, p5)
    svr_static = np.array([
        country_enc, crop_enc, r5, t5, p5, ratio5, log_p5, heat5,
    ], dtype=np.float32)

    return (
        torch.from_numpy(seq_scaled),
        torch.from_numpy(stat_scaled),
        svr_static,
        {"country_enc": country_enc, "crop_enc": crop_enc,
         "crop_mean": crop_mean, "crop_std": crop_std},
    )


def _predict(country: str, crop: str, years: list[tuple[float, float, float]]):
    """Run the hybrid model. Returns (yield_pred, attn_weights, embedding)."""
    seq_t, stat_t, svr_static, meta = _build_inputs(country, crop, years)

    # Forward pass with a hook to capture the 64-dim embedding
    captured = {}

    def hook(_module, inputs, _output):
        captured["e"] = inputs[0].detach().cpu().numpy()

    handle = BILSTM.fc_out.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _, attn = BILSTM(seq_t.to(DEVICE), stat_t.to(DEVICE))
    finally:
        handle.remove()

    embedding = captured["e"][0]                    # (64,)
    attn_w = attn.detach().cpu().numpy()[0]          # (5,)

    # Hybrid SVR prediction
    hybrid_x = np.concatenate([svr_static, embedding]).reshape(1, -1)
    pred_norm = HYBRID_PIPE.predict(hybrid_x)[0]
    pred = pred_norm * meta["crop_std"] + meta["crop_mean"]

    return float(pred), attn_w, meta


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────
def _attention_plot(attn_w: np.ndarray, year_labels: list[str]):
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    colors = plt.cm.viridis(attn_w / max(attn_w.max(), 1e-6))
    bars = ax.bar(year_labels, attn_w, color=colors, edgecolor="black", linewidth=0.4)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, max(attn_w.max() * 1.25, 0.4))
    ax.set_ylabel("Attention weight")
    ax.set_title("Which year drove the prediction?", fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def _context_plot(pred: float, crop: str):
    cs = CROP_STATS.loc[CROP_STATS["crop"] == crop].iloc[0]
    mean, std, lo, hi = float(cs["mean"]), float(cs["std"]), float(cs["min"]), float(cs["max"])

    fig, ax = plt.subplots(figsize=(6.0, 1.8))
    ax.axhline(0.5, color="#cccccc", linewidth=10, zorder=1)
    # Typical band = mean ± 1 std
    ax.plot([mean - std, mean + std], [0.5, 0.5], color="#70AD47", linewidth=10, zorder=2)
    ax.plot([mean], [0.5], marker="|", color="black", markersize=18, mew=2, zorder=3)
    ax.plot([pred], [0.5], marker="v", color="#C00000", markersize=14, zorder=4,
            label=f"Predicted: {pred:.2f} t/ha")
    ax.set_xlim(min(lo, pred) - 0.5, max(hi, pred) + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Yield (t/ha)")
    ax.set_title(
        f"{crop}: typical range {mean - std:.1f}–{mean + std:.1f} t/ha "
        f"(global mean {mean:.1f})", fontweight="bold", fontsize=10
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Tab 1 — Historical prediction
# ─────────────────────────────────────────────────────────────
def list_years_for(country, crop):
    if not country or not crop:
        return gr.Dropdown(choices=[], value=None)
    sub = DATA[(DATA["country"] == country) & (DATA["crop"] == crop)]
    years = sorted(set(int(y) for y in sub["year"].unique()))
    return gr.Dropdown(choices=years, value=years[-1] if years else None)


def predict_historical(country, crop, year):
    if not ARTIFACTS_OK:
        return f"⚠️ Models not loaded: {LOAD_ERROR}", "", None, None, None

    if year is None:
        return "Pick a year first.", "", None, None, None

    sub = (
        DATA[(DATA["country"] == country) & (DATA["crop"] == crop)]
        .sort_values("year")
        .drop_duplicates(subset=["year"], keep="first")
    )
    target_row = sub[sub["year"] == int(year)]
    if target_row.empty:
        return f"No record for {country} / {crop} in {year}.", "", None, None, None

    # Build 5-year lookback ending at `year` — left-pad with zeros if fewer years exist
    yr = int(year)
    history_rows = sub[sub["year"] <= yr].tail(LOOKBACK)
    years_data = []
    history_lines = []
    for _, row in history_rows.iterrows():
        years_data.append((float(row["rainfall_mm"]),
                           float(row["avg_temp"]),
                           float(row["pesticides_tonnes"])))
        history_lines.append(
            f"  {int(row['year'])}: rainfall={row['rainfall_mm']:.0f} mm, "
            f"temp={row['avg_temp']:.1f}°C, pesticides={row['pesticides_tonnes']:.0f} t"
        )
    while len(years_data) < LOOKBACK:
        years_data.insert(0, (0.0, 0.0, 0.0))
        history_lines.insert(0, "  (padded with zeros — first years of dataset)")

    pred, attn, _ = _predict(country, crop, years_data)
    actual = float(target_row["yield_tonnes_per_ha"].iloc[0])
    err = pred - actual

    headline = (
        f"### Predicted: **{pred:.2f} t/ha**   |   Actual: **{actual:.2f} t/ha**   |   "
        f"Error: **{err:+.2f} t/ha** ({err / actual * 100:+.1f}%)"
    )

    details = (
        f"**5-year history fed to the model** (oldest → newest):\n"
        + "\n".join(history_lines)
        + f"\n\n**Attention weights**: "
        + ", ".join(f"{w:.2f}" for w in attn)
        + f"\n\n**Model**: Phase 3 hybrid (BiLSTM 64-dim embeddings → tuned RBF SVR)."
    )

    year_labels = []
    seen_years = sorted([int(r) for r in history_rows["year"].tolist()])
    while len(year_labels) + len(seen_years) < LOOKBACK:
        year_labels.append("(pad)")
    year_labels.extend(str(y) for y in seen_years)

    return headline, details, _attention_plot(attn, year_labels), _context_plot(pred, crop), None


# ─────────────────────────────────────────────────────────────
# Tab 2 — Custom scenario
# ─────────────────────────────────────────────────────────────
def predict_custom(country, crop,
                   r1, t1, p1, r2, t2, p2, r3, t3, p3, r4, t4, p4, r5, t5, p5):
    if not ARTIFACTS_OK:
        return f"⚠️ Models not loaded: {LOAD_ERROR}", "", None, None

    years_data = [
        (float(r1), float(t1), float(p1)),
        (float(r2), float(t2), float(p2)),
        (float(r3), float(t3), float(p3)),
        (float(r4), float(t4), float(p4)),
        (float(r5), float(t5), float(p5)),
    ]
    try:
        pred, attn, _ = _predict(country, crop, years_data)
    except Exception as e:
        return f"❌ Error: {e}", "", None, None

    cs = CROP_STATS.loc[CROP_STATS["crop"] == crop].iloc[0]
    mean = float(cs["mean"])
    delta = pred - mean
    pct = delta / mean * 100

    headline = (
        f"### Predicted: **{pred:.2f} t/ha**   |   "
        f"Crop average: {mean:.2f} t/ha   |   "
        f"Δ: **{delta:+.2f} t/ha ({pct:+.1f}%)**"
    )
    details = (
        "**Attention weights** (oldest → newest year): "
        + ", ".join(f"{w:.2f}" for w in attn)
        + ".\n\n"
        + "Higher weights mean that year's climate had more influence on the prediction."
    )
    return headline, details, _attention_plot(attn, ["t-4", "t-3", "t-2", "t-1", "t"]), \
        _context_plot(pred, crop)


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
INTRO = """
# 🌱 AgriVision Phase 3 — Hybrid Crop Yield Predictor

This is the **Phase 3 hybrid model** — a frozen BiLSTM with attention extracts a
64-dimensional summary of the past 5 years of climate, and a tuned RBF Support
Vector Regressor maps that summary to a yield prediction in tonnes per hectare.

**Test-set performance**: $R^{2}=0.951$, RMSE = 1.90 t/ha (best of the project).

Use the **Historical Prediction** tab to load real (country, crop, year) data
from the FAO dataset and compare the hybrid's prediction to the true yield.
Use **Custom Scenario** to design your own 5-year climate history.
"""

with gr.Blocks(title="AgriVision — Phase 3 Hybrid Predictor",
               theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tabs():
        # ── Tab 1 ────────────────────────────────────────────
        with gr.Tab("📜 Historical Prediction"):
            gr.Markdown(
                "Pick a country, crop, and year. The app loads the **true 5-year "
                "climate history** ending at that year and runs the hybrid model. "
                "The actual recorded yield is shown side-by-side so you can see "
                "the error directly."
            )
            with gr.Row():
                hist_country = gr.Dropdown(choices=COUNTRY_LIST, label="Country",
                                           value="India" if "India" in COUNTRY_LIST else COUNTRY_LIST[0])
                hist_crop = gr.Dropdown(choices=CROP_LIST, label="Crop",
                                        value="Wheat" if "Wheat" in CROP_LIST else CROP_LIST[0])
                hist_year = gr.Dropdown(choices=[YEAR_RANGE[1]], value=YEAR_RANGE[1],
                                        label="Year")
            hist_btn = gr.Button("🔍 Load history & predict", variant="primary")

            hist_headline = gr.Markdown()
            hist_details = gr.Markdown()
            with gr.Row():
                hist_attn_plot = gr.Plot(label="Attention over the 5-year window")
                hist_context_plot = gr.Plot(label="Where this prediction sits for this crop")
            _hidden = gr.Plot(visible=False)

            # When country or crop changes, refresh the year dropdown
            hist_country.change(list_years_for, [hist_country, hist_crop], hist_year)
            hist_crop.change(list_years_for, [hist_country, hist_crop], hist_year)
            hist_btn.click(predict_historical,
                           [hist_country, hist_crop, hist_year],
                           [hist_headline, hist_details, hist_attn_plot, hist_context_plot, _hidden])

        # ── Tab 2 ────────────────────────────────────────────
        with gr.Tab("🛠️ Custom Scenario"):
            gr.Markdown(
                "Imagine a 5-year climate history for any country/crop pair and "
                "see what the hybrid would predict for the **last (current) year**."
            )
            with gr.Row():
                cust_country = gr.Dropdown(choices=COUNTRY_LIST, label="Country",
                                           value="India" if "India" in COUNTRY_LIST else COUNTRY_LIST[0])
                cust_crop = gr.Dropdown(choices=CROP_LIST, label="Crop",
                                        value="Wheat" if "Wheat" in CROP_LIST else CROP_LIST[0])

            year_inputs = []
            for offset, label in enumerate(["t-4 (oldest)", "t-3", "t-2", "t-1", "t (current)"]):
                with gr.Row():
                    gr.Markdown(f"**Year {label}**")
                    r = gr.Number(label="Rainfall (mm)", value=1200.0)
                    t = gr.Number(label="Avg temp (°C)", value=22.0)
                    p = gr.Number(label="Pesticides (t)", value=200.0)
                    year_inputs.extend([r, t, p])

            cust_btn = gr.Button("🌾 Predict yield", variant="primary")

            cust_headline = gr.Markdown()
            cust_details = gr.Markdown()
            with gr.Row():
                cust_attn_plot = gr.Plot(label="Attention over the 5-year window")
                cust_context_plot = gr.Plot(label="Where this prediction sits for this crop")

            cust_btn.click(predict_custom,
                           [cust_country, cust_crop, *year_inputs],
                           [cust_headline, cust_details, cust_attn_plot, cust_context_plot])

        # ── Tab 3 ────────────────────────────────────────────
        with gr.Tab("ℹ️ About this model"):
            gr.Markdown("""
### How it works

1. **Inputs.** A (country, crop) pair plus a 5-year window of (rainfall mm, avg temperature °C, pesticide use t).
2. **Phase 2 BiLSTM (frozen).** A bidirectional LSTM with attention reads the 5 yearly vectors and produces a 64-dimensional summary $\\mathbf{e}_i$. The attention weights you see in the plot tell you which year the network focused on.
3. **Phase 3 SVR.** A tuned RBF Support Vector Regressor maps $\\mathbf{e}_i$ (concatenated with 8 raw static features) to a *per-crop normalized* yield. The output is then inverse-normalized to t/ha.

### Why hybrid?

| Model | RMSE (t/ha) | $R^{2}$ |
|---|---|---|
| Phase 1 — SVR on static features | 3.98 | 0.782 |
| Phase 2 — BiLSTM with attention | 2.28 | 0.929 |
| **Phase 3 — Hybrid (this app)** | **1.90** | **0.951** |

The BiLSTM contributes temporal pattern recognition; the SVR contributes a smoother, more outlier-resistant final regressor.

### Limitations

- Climate data is country-level annual averages — not field-level. Two fields in the same country in the same year are indistinguishable to the model.
- The dataset ends in 2013, so the model does not see recent extreme weather.
- Per-crop $R^{2}$ ranges from 0.78 (Plantains) to 0.90 (Potatoes/Wheat); see the report for the breakdown.
""")

    gr.Markdown(
        "<sub>AgriVision — Phase 3. Code: "
        "<a href='https://github.com/'>repository</a>. Built with PyTorch + scikit-learn + Gradio.</sub>"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
