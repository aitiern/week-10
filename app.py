# ------------------------------------------------------------
# app.py — Streamlit interface for Coffee Models
# ------------------------------------------------------------
# Requirements:
#   pip install streamlit scikit-learn pandas numpy
# Files expected in the same folder after running train.py:
#   - model_1.pickle  (LinearRegression on 100g_USD)
#   - model_2.pickle  (DecisionTree on 100g_USD + roast_cat)
#   - model_3.pickle  (TF-IDF + LinearRegression on desc_3)  [optional]
#   - apputil.py      (contains predict_rating)
# Run:
#   streamlit run app.py
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st

from apputil import predict_rating

# ------------------------------
# Small helpers (camelCase names)
# ------------------------------

def ensureColumns(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def makeExampleTabular() -> pd.DataFrame:
    return pd.DataFrame([
        {"100g_USD": 10.00, "roast": "Dark"},
        {"100g_USD": 15.00, "roast": "Medium-Light"},
        {"100g_USD": 8.50,  "roast": None},
    ])


def makeExampleText() -> pd.DataFrame:
    return pd.DataFrame([
        {"text": "A delightfully smooth cup with cocoa and caramel."},
        {"text": "Bold flavor with a smoky finish and lingering spice."},
    ])


# ------------------------------
# Sidebar
# ------------------------------
st.set_page_config(page_title="Coffee Rating Predictor", page_icon="☕")

st.sidebar.title("☕ Coffee Rating Predictor")
mode = st.sidebar.radio(
    "Choose input mode:",
    ["Price + Roast (tabular)", "Review Text (bonus)"]
)

st.sidebar.markdown("---")
with st.sidebar.expander("Models expected"):
    st.write("- model_1.pickle (required)")
    st.write("- model_2.pickle (required)")
    st.write("- model_3.pickle (optional for text mode)")

missing = []
for fname in ["model_1.pickle", "model_2.pickle"]:
    if not os.path.exists(fname):
        missing.append(fname)
if missing:
    st.sidebar.warning(
        "Missing model files: " + ", ".join(missing) + ".\nRun `python train.py` first."
    )

# ------------------------------
# Main content
# ------------------------------
st.title("Coffee Rating Predictor")

if mode == "Price + Roast (tabular)":
    st.markdown("Enter one or more rows with **100g_USD** and **roast** (text).\n\n"
                "If *roast* is missing or unrecognized, the app falls back to the price-only model.")

    df_input = st.data_editor(
        makeExampleTabular(),
        use_container_width=True,
        num_rows="dynamic",
        key="tabular_editor"
    )

    df_input = ensureColumns(df_input, ["100g_USD", "roast"])  # keep only required columns

    colA, colB = st.columns([1, 3])
    with colA:
        run_pred = st.button("Predict ratings")
    with colB:
        st.caption("Tip: Add more rows in the table before predicting.")

    if run_pred:
        try:
            y_hat = predict_rating(df_input)
            df_out = df_input.copy()
            df_out["predicted_rating"] = np.round(y_hat, 2)
            st.success("Predictions ready")
            st.dataframe(df_out, use_container_width=True)
        except FileNotFoundError as e:
            st.error(f"Missing model file: {e}")
        except Exception as e:
            st.exception(e)

else:  # Review Text (bonus)
    st.markdown("Paste review-like text (one row per input). Uses the **text model** if available.")

    df_text = st.data_editor(
        makeExampleText(),
        use_container_width=True,
        num_rows="dynamic",
        key="text_editor"
    )

    # Accept either a single-column DF labeled 'text', or coerce first column to 'text'
    if df_text.shape[1] == 1 and df_text.columns[0] != "text":
        df_text.columns = ["text"]

    if "text" not in df_text.columns:
        st.warning("Please include a single column named 'text'.")
    else:
        run_pred_text = st.button("Predict ratings from text")
        if run_pred_text:
            try:
                y_hat = predict_rating(df_text[["text"]], text=True)
                df_out = df_text.copy()
                df_out["predicted_rating"] = np.round(y_hat, 2)
                st.success("Text predictions ready")
                st.dataframe(df_out, use_container_width=True)
            except FileNotFoundError:
                st.error("model_3.pickle not found. Train it with `python train.py` (do not use --skip-text).")
            except Exception as e:
                st.exception(e)

# Footer
st.markdown("---")
st.caption(
    "Models trained from the Coffee Analysis dataset. This demo is for coursework;"
    " model performance is not the focus."
)
