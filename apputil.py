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

import numpy as _np
import pandas as _pd
from pathlib import Path as _Path

# Reuse the same mapping helpers from above (they will exist when this file is saved as its own module).
# If you keep apputil.py as a separate file, copy the mapping code there as-is.


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


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

        Xvec = vectorizer.transform(texts)
        Xdense = Xvec.toarray()  # match training path
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

