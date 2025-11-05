# ------------------------------------------------------------
# apputil.py — Bonus utility for model predictions
# ------------------------------------------------------------

import pickle
import numpy as np
import pandas as pd

# roast mapping
_ROAST_ORDER = [
    "very light", "light", "light-medium", "medium-light",
    "medium", "medium-dark", "dark", "very dark"
]
_ROAST_INDEX = {name: i for i, name in enumerate(_ROAST_ORDER, start=0)}

def roast_category(val: str) -> float:
    val = str(val).strip().lower().replace(" ", "-")
    return float(_ROAST_INDEX.get(val, np.nan))

# load models
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_rating(df, text=False):
    if text:
        art = load_pickle("model_3.pickle")
        X = art["vectorizer"].transform(df["text"]).toarray()
        return art["model"].predict(X)

    lr = load_pickle("model_1.pickle")
    tree = load_pickle("model_2.pickle")
    df["roast_cat"] = df["roast"].apply(roast_category)
    mask = ~df["roast_cat"].isna()

    y = np.zeros(len(df))
    if mask.any():
        X_tree = df.loc[mask, ["100g_USD", "roast_cat"]]
        y[mask] = tree.predict(X_tree)
    if (~mask).any():
        y[~mask] = lr.predict(df.loc[~mask, ["100g_USD"]])
    return y



