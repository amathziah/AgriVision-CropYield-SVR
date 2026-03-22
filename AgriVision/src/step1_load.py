"""
STEP 1 — Load & Clean Dataset
Rubric target: Dataset Quality & EDA — 8/10
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH   = Path("data/yield_df.csv")
OUTPUT_PATH = Path("data/final_dataset.csv")

def load_and_clean():
    df = pd.read_csv(DATA_PATH)

    # Drop the unnamed index column
    df.drop(columns=["Unnamed: 0"], inplace=True)

    # Rename columns to clean names
    df.rename(columns={
        "Area"                           : "country",
        "Item"                           : "crop",
        "Year"                           : "year",
        "hg/ha_yield"                    : "yield_hg_per_ha",
        "average_rain_fall_mm_per_year"  : "rainfall_mm",
        "pesticides_tonnes"              : "pesticides_tonnes",
        "avg_temp"                       : "avg_temp"
    }, inplace=True)

    # Convert yield from hg/ha → tonnes/ha  (1 hg/ha = 0.0001 t/ha)
    df["yield_tonnes_per_ha"] = (df["yield_hg_per_ha"] * 0.0001).round(4)
    df.drop(columns=["yield_hg_per_ha"], inplace=True)

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Duplicates removed: {before - len(df)}")

    # Report
    print(f"\n✅ Dataset loaded")
    print(f"   Shape      : {df.shape}")
    print(f"   Countries  : {df['country'].nunique()}")
    print(f"   Crops      : {df['crop'].nunique()} → {list(df['crop'].unique())}")
    print(f"   Years      : {df['year'].min()} – {df['year'].max()}")
    print(f"   Nulls      : {df.isnull().sum().sum()}")
    print(f"\n   Sample:\n{df.head(3).to_string(index=False)}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved → {OUTPUT_PATH}")
    return df

if __name__ == "__main__":
    load_and_clean()
