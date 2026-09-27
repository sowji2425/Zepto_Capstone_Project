import os
import pandas as pd
import numpy as np

DATA_URL = "https://githubusercontent.com"
LOCAL_PATH = "analytics/titanic.csv"

def load_data():
    """Fetches the raw dataset from remote storage or uses cached copy."""
    if os.path.exists(LOCAL_PATH):
        print(f"Loading data from local cache: {LOCAL_PATH}")
        return pd.read_csv(LOCAL_PATH)
    
    print(f"Streaming data from remote repository: {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL)
        # Cache local copy for module consistency
        os.makedirs("analytics", exist_ok=True)
        df.to_csv(LOCAL_PATH, index=False)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch dataset stream: {e}")

def profile_dataset():
    """Profiles and saves structural analysis metrics of the Titanic dataset."""
    df = load_data()
    
    print("\n=== DATASET SHAPE & METRICS ===")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    
    print("\n=== STRUCTURAL MISSINGNESS PROFILE ===")
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Counts": missing_counts, "Percentage (%)": missing_pct})
    print(missing_df[missing_df["Missing Counts"] > 0])
    
    print("\n=== SURVIVAL TARGET DISTRIBUTION ===")
    print(df["Survived"].value_counts(normalize=True))
    
    print("\n=== DEMOGRAPHIC PROFILE MATRIX ===")
    pivot = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
    print(pivot)

if __name__ == "__main__":
    profile_dataset()
