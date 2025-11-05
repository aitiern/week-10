# ------------------------------------------------------------
# train.py
# ------------------------------------------------------------
# Exercises 1 & 2 (plus Bonus 4) trainer for the Coffee Analysis data.
# - Exercise 1: trains LinearRegression on `100g_USD` -> `rating` and
#               saves model_1.pickle
# - Exercise 2: trains DecisionTreeRegressor on [`100g_USD`, `roast_cat`]
#               (with `roast_cat` derived from text `roast`) and
#               saves model_2.pickle
# - Bonus 4   : trains TF-IDF -> LinearRegression on `desc_3` (text only)
#               and saves model_3.pickle
#
# Usage examples:
#   python train.py --data-url https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv
#   python train.py              # if DATA_URL env var is set
#   python train.py --skip-text  # skip Bonus 4 text model
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Utility: roast_category mapping (string -> numeric label)
# ------------------------------------------------------------
# The assignment calls for a function named `roast_category`.
# We treat common roast labels (case-insensitive, hyphen/space tolerant).

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

DEFAULT_DATA_URL = (
    os.getenv("DATA_URL")
    or "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
)

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
    """LinearRegression on 100g_USD -> rating -> model_1.pickle"""
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
    """DecisionTree on [100g_USD, roast_cat] -> model_2.pickle"""
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
    """Bonus 4: TF-IDF on desc_3 -> LinearRegression -> model_3.pickle"""
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
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL,
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


# ------------------------------------------------------------
# apputil.py  (Bonus Exercise 3 & 4 helper)
# ------------------------------------------------------------
# Provides a `predict_rating` function with two modes:
#   1) Tabular mode: df_X with columns ["100g_USD", "roast" (text)]
#      - If roast is known (seen category), use model_2 (tree) on both features
#      - Else, fall back to model_1 (linear) using only 100g_USD
#   2) Text mode: predict_rating(X, text=True) where X is an array-like of
#      strings (in the style of desc_3). Uses model_3 (TF-IDF + LinearRegression).
# ------------------------------------------------------------

import pickle as _pickle
import numpy as _np
import pandas as _pd

# --- Duplicate the roast mapping helpers so apputil.py is standalone ---
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

def _normalize_roast(val):
    import numpy as _np  # local import to keep this file self-contained
    if val is None or (isinstance(val, float) and _np.isnan(val)):
        return None
    if not isinstance(val, str):
        val = str(val)
    s = val.strip().lower()
    s = s.replace(" ", "-").replace("_", "-")
    s = s.replace("lightmedium", "light-medium")
    s = s.replace("mediumlight", "medium-light")
    s = s.replace("mediumdark", "medium-dark")
    return s

def roast_category(val):
    s = _normalize_roast(val)
    if s is None:
        return _np.nan
    return float(_ROAST_INDEX.get(s, _np.nan))

# --- IO helpers ---

def _load_pickle(path: str):
    with open(path, "rb") as f:
        return _pickle.load(f)

# --- Public API ---

def predict_rating(X, text: bool = False):
    """Predict ratings using the saved models.

    Parameters
    ----------
    X :
        - If text == False: a pandas.DataFrame with columns ["100g_USD", "roast"]
          (roast may be missing or unseen; we will fall back to model_1 as needed).
        - If text == True: array-like of strings (or a DataFrame/Series with one text column).
    text : bool, default False
        When True, use the text-only model (model_3.pickle) on input strings.

    Returns
    -------
    numpy.ndarray of predicted ratings
    """
    if text:
        # Text mode (Bonus 4)
        art = _load_pickle("model_3.pickle")
        vectorizer = art["vectorizer"]
        model = art["model"]

        # Normalize X to a 1D array of strings
        if isinstance(X, _pd.DataFrame):
            if X.shape[1] == 1:
                texts = X.iloc[:, 0].astype(str).fillna("").values
            else:
                raise ValueError("For text=True, pass a single-column DataFrame or a 1D array of strings.")
        elif isinstance(X, _pd.Series):
            texts = X.astype(str).fillna("").values
        else:
            texts = _np.asarray(X, dtype=str)

        Xvec = vectorizer.transform(texts)  # unseen words ignored by TF-IDF
        Xdense = Xvec.toarray()             # match training path
        return model.predict(Xdense)

    # Tabular mode (Bonus 3)
    if not isinstance(X, _pd.DataFrame):
        raise ValueError("For tabular mode, X must be a pandas DataFrame with columns ['100g_USD', 'roast'].")

    if "100g_USD" not in X.columns or "roast" not in X.columns:
        raise ValueError("X must have columns ['100g_USD', 'roast'] in tabular mode.")

    # Load both models
    lr = _load_pickle("model_1.pickle")  # LinearRegression on price only
    art2 = _load_pickle("model_2.pickle")  # tree artifact
    dtr = art2["model"]
    sentinel = art2.get("sentinel", 9_999.0)

    # Compute roast_cat with the same mapping used in training
    X = X.copy()
    X["roast_cat"] = X["roast"].apply(roast_category)

    # Determine which rows use model_2 (known roast) vs model_1 (fallback)
    use_tree = ~X["roast_cat"].isna()

    # Prepare outputs
    y_pred = _np.empty(len(X), dtype=float)

    # Tree predictions where roast is known
    if use_tree.any():
        X_tree = X.loc[use_tree, ["100g_USD", "roast_cat"]].values
        # Replace NaN with sentinel for consistency with training
        X_tree = _np.where(_np.isnan(X_tree), sentinel, X_tree)
        y_pred[use_tree.values] = dtr.predict(X_tree)

    # Linear predictions fallback
    if (~use_tree).any():
        X_lin = X.loc[~use_tree, ["100g_USD"]].values
        y_pred[(~use_tree).values] = lr.predict(X_lin)

    return y_pred

