import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import re
import string
from utils.text_utils import preprocess_text

def load_and_preprocess_data(path):
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded data from {path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] File not found: {path}")
    except Exception as e:
        raise Exception(f"[ERROR] Failed to read CSV: {e}")

    # Rename 'humour' to 'label' if needed
    if "label" not in df.columns:
        if "humour" in df.columns:
            print("[INFO] Renaming 'humour' column to 'label'")
            df.rename(columns={"humour": "label"}, inplace=True)
        else:
            raise ValueError(f"[ERROR] Missing required 'label' column. Found columns: {df.columns.tolist()}")

    # Check if 'text' exists
    if "text" not in df.columns:
        raise ValueError(f"[ERROR] Missing required 'text' column. Found columns: {df.columns.tolist()}")

    # Drop rows with missing text or label
    df.dropna(subset=["text", "label"], inplace=True)

    # Apply text preprocessing
    df["clean_text"] = df["text"].apply(preprocess_text)

    return df

if __name__ == "__main__":
    data_path = "data/processed/processed_data(20k).csv"
    df = load_and_preprocess_data(data_path)
    print("[INFO] Sample processed data:")
    print(df.head())
