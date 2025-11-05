# ------------------------------------------------------------
# train.py
# ------------------------------------------------------------
# Exercises 1 & 2 (plus Bonus 4) trainer for the Coffee Analysis data.
# Saves the following models:
#   - model_1.pickle : LinearRegression on `100g_USD`
#   - model_2.pickle : DecisionTreeRegressor on [`100g_USD`, `roast_cat`]
#   - model_3.pickle : (bonus) TF-IDF text feature model on `desc_3`
# ------------------------------------------------------------

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------
# Utility: roast_category mapping (string -> numeric label)
# ------------------------------------------------------------

_ROAST_ORDER = [
    "very light", "light", "light-medium", "medium-light",
    "medium", "medium-dark", "dark", "very dark"
]
_ROAST_INDEX = {name: i for i, name in enumerate(_ROAST_ORDER, start=0)}

def _normalize_roast(val: Optional[str]) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    val = str(val).strip().lower()
    for sep in [" ", "_"]:
        val = val.replace(sep, "-")
    return val

def roast_category(val: Optional[str]) -> float:
    val = _normalize_roast(val)
    return float(_ROAST_INDEX.get(val, np.nan)) if val else np.nan

# ------------------------------------------------------------
# Load data (URL -> local fallback)
# ------------------------------------------------------------

DEFAULT_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

def load_data(url: Optional[str]) -> pd.DataFrame:
    url = url or os.getenv("DATA_URL") or DEFAULT_URL
    try:
        return pd.read_csv(url)
    except Exception:
        if Path("coffee_analysis.csv").exists():
            return pd.read_csv("coffee_analysis.csv")
        raise RuntimeError(f"Failed to load data from URL and local file.")

# ------------------------------------------------------------
# Training routines
# ------------------------------------------------------------

def train_model_1(df: pd.DataFrame, out: Path) -> None:
    data = df[["100g_USD", "rating"]].dropna()
    X = data[["100g_USD"]]
    y = data["rating"].values
    model = LinearRegression().fit(X, y)
    pickle.dump(model, out.open("wb"))

def train_model_2(df: pd.DataFrame, out: Path) -> None:
    if "roast" not in df.columns:
        df["roast"] = np.nan
    df["roast_cat"] = df["roast"].apply(roast_category)
    data = df[["100g_USD", "roast_cat", "rating"]].dropna()
    X = data[["100g_USD", "roast_cat"]]
    y = data["rating"].values
    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    pickle.dump(model, out.open("wb"))

def train_model_3(df: pd.DataFrame, out: Path) -> None:
    data = df[["desc_3", "rating"]].dropna()
    text = data["desc_3"].astype(str)
    y = data["rating"].values
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(text).toarray()
    model = LinearRegression().fit(X, y)
    pickle.dump({"vectorizer": vectorizer, "model": model}, out.open("wb"))

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-url", type=str)
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data_url)
    train_model_1(df, Path("model_1.pickle"))
    train_model_2(df, Path("model_2.pickle"))
    if not args.skip_text:
        train_model_3(df, Path("model_3.pickle"))

if __name__ == "__main__":
    main()


