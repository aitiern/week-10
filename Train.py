import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer


_ROAST_ORDER = [
    "very light",
    "light",
    "light-medium",
    "medium-light",
    "medium",
    "medium-dark",
    "dark",
    "very dark",
]

_ROAST_INDEX = {name: i for i, name in enumerate(_ROAST_ORDER, start=0)}


def _normalize_roast(val: Optional[str]) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if not isinstance(val, str):
        val = str(val)
    s = val.strip().lower()
    # Unify separators
    s = s.replace(" ", "-")
    s = s.replace("_", "-")
    # Common alternates
    s = s.replace("lightmedium", "light-medium")
    s = s.replace("mediumlight", "medium-light")
    s = s.replace("mediumdark", "medium-dark")
    return s


def roast_category(val: Optional[str]) -> float:
    """Map raw roast text to a numeric category label.

    Returns an int label if recognized, otherwise np.nan (allowed by spec).
    Examples:
        roast_category('Medium-Light') -> 3 (with the chosen scheme)
        roast_category(None)          -> np.nan
    """
    s = _normalize_roast(val)
    if s is None:
        return np.nan
    return float(_ROAST_INDEX.get(s, np.nan))


# ------------------------------------------------------------
# Core training logic
# ------------------------------------------------------------

def load_coffee_df(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV from {data_url}: {e}")
    # Minimal sanity check for required columns
    required = {"rating", "100g_USD"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Data is missing required columns: {sorted(missing)}. Columns present: {sorted(df.columns)}"
        )
    return df


def train_model_1(df: pd.DataFrame, out_path: Path) -> None:
    # LinearRegression on 100g_USD -> rating
    X = df[["100g_USD"]].values
    y = df["rating"].values

    # Drop rows with NA in features/target
    mask = ~np.isnan(X).ravel() & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    lr = LinearRegression()
    lr.fit(X, y)

    with out_path.open("wb") as f:
        pickle.dump(lr, f)


def train_model_2(df: pd.DataFrame, out_path: Path) -> None:
    # Need `roast` column; if missing, create with NaN so mapping -> NaN
    if "roast" not in df.columns:
        df = df.copy()
        df["roast"] = np.nan

    df = df.copy()
    df["roast_cat"] = df["roast"].apply(roast_category)

    # Features: 100g_USD and roast_cat (allow NaN per assignment)
    X = df[["100g_USD", "roast_cat"]].values
    y = df["rating"].values

    # For Decision Trees, NaN in a feature will raise; replace NaN with a sentinel
    # The prompt encourages trying odd values, so use large sentinel for NaN.
    X = np.where(np.isnan(X), 9_999.0, X)

    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X, y)

    artifact = {
        "model": dtr,
        "sentinel": 9_999.0,
        "roast_index": _ROAST_INDEX,  # for reference/debug
    }
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)


def train_model_3_text(df: pd.DataFrame, out_path: Path) -> None:
    # Bonus 4: TF-IDF on desc_3 -> LinearRegression
    if "desc_3" not in df.columns:
        raise ValueError("desc_3 column is required for the text model (model_3.pickle).")

    text = df["desc_3"].fillna("").astype(str)
    y = df["rating"].values

    mask = ~np.isnan(y)
    text = text[mask]
    y = y[mask]

    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
    X_sparse = vectorizer.fit_transform(text)

    # LinearRegression doesn't accept sparse; densify (dataset is small).
    X = X_sparse.toarray()

    lr = LinearRegression()
    lr.fit(X, y)

    artifact = {
        "vectorizer": vectorizer,
        "model": lr,
    }
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train coffee models (Exercises 1, 2, Bonus 4)")
    parser.add_argument("--data-url", type=str, default=os.getenv("DATA_URL"),
                        help="URL to the coffee analysis CSV (or set DATA_URL env var)")
    parser.add_argument("--skip-text", action="store_true", help="Skip training the text model (model_3.pickle)")
    args = parser.parse_args()

    if not args.data_url:
        print("ERROR: No data URL provided. Use --data-url or set DATA_URL env var.", file=sys.stderr)
        sys.exit(2)

    df = load_coffee_df(args.data_url)

    out1 = Path("model_1.pickle")
    out2 = Path("model_2.pickle")
    out3 = Path("model_3.pickle")

    print("Training model_1 (LinearRegression on 100g_USD -> rating)...")
    train_model_1(df, out1)
    print(f"Saved {out1.resolve()}")

    print("Training model_2 (DecisionTreeRegressor on [100g_USD, roast_cat])...")
    train_model_2(df, out2)
    print(f"Saved {out2.resolve()}")

    if not args.skip_text:
        print("Training model_3 (TF-IDF + LinearRegression on desc_3 text only)...")
        train_model_3_text(df, out3)
        print(f"Saved {out3.resolve()}")
    else:
        print("Skipping text model (model_3.pickle)")


if __name__ == "__main__":
    main()
