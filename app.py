# ------------------------------------------------------------
# app.py — Enhanced Streamlit UI for Coffee Models
# ------------------------------------------------------------
# Requirements (same env as before):
#   pip install streamlit scikit-learn pandas numpy
# Files expected after running train.py:
#   - model_1.pickle, model_2.pickle, (optional) model_3.pickle
#   - apputil.py  (with predict_rating)
# Run:  streamlit run app.py
# ------------------------------------------------------------

import io
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from typing import List

from apputil import predict_rating, roast_category

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Coffee Rating Predictor",
    page_icon="☕",
    layout="wide",
)

# ------------------------------
# Cached loaders & utilities
# ------------------------------

@st.cache_resource(show_spinner=False)
def loadModel(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def modelStatusBadge(name: str, exists: bool):
    return f"✅ {name}" if exists else f"❌ {name}"


def ensureTabular(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["100g_USD", "roast"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df[needed]


def exampleTabular() -> pd.DataFrame:
    return pd.DataFrame([
        {"100g_USD": 10.00, "roast": "Dark"},
        {"100g_USD": 15.00, "roast": "Medium-Light"},
        {"100g_USD": 8.50,  "roast": None},
    ])


def exampleText() -> pd.DataFrame:
    return pd.DataFrame([
        {"text": "A delightfully smooth cup with cocoa and caramel."},
        {"text": "Bold flavor with a smoky finish and lingering spice."},
    ])


def downloadCsvButton(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# ------------------------------
# Sidebar — model status & roast guide
# ------------------------------

st.sidebar.title("☕ Coffee Models")
model1 = loadModel("model_1.pickle")  # LinearRegression
model2 = loadModel("model_2.pickle")  # DecisionTreeRegressor
model3 = loadModel("model_3.pickle")  # {vectorizer, model} or None

st.sidebar.markdown(
    "\n".join([
        modelStatusBadge("model_1.pickle (price)", model1 is not None),
        modelStatusBadge("model_2.pickle (price+roast)", model2 is not None),
        modelStatusBadge("model_3.pickle (text)", model3 is not None),
    ])
)

with st.sidebar.expander("Roast label guide"):
    roastLabels: List[str] = [
        "very light", "light", "light-medium", "medium-light",
        "medium", "medium-dark", "dark", "very dark"
    ]
    st.write(pd.DataFrame({"roast": roastLabels, "roast_cat": list(range(len(roastLabels)))}))
    st.caption("Unknown/missing roast falls back to the price-only model.")

st.sidebar.markdown("---")
st.sidebar.info("Run `python train.py` to (re)create the model files.")

# ------------------------------
# Header
# ------------------------------

st.title("Coffee Rating Predictor")
st.caption("Predict coffee ratings from price + roast or review text. Coursework demo — performance not the focus.")

# ------------------------------
# Tabs
# ------------------------------

tab1, tab2, tab3 = st.tabs(["Price + Roast", "Review Text", "Model Info"])

# ------------------------------
# Tab 1 — Price + Roast
# ------------------------------
with tab1:
    colL, colR = st.columns([2, 1], gap="large")

    with colL:
        st.subheader("Input rows")
        st.caption("Edit the table, paste, or upload a CSV with columns: 100g_USD, roast")

        upload = st.file_uploader("Upload CSV (optional)", type=["csv"], accept_multiple_files=False)
        if upload:
            try:
                df_tab = pd.read_csv(upload)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                df_tab = exampleTabular()
        else:
            df_tab = exampleTabular()

        df_tab = ensureTabular(df_tab)
        df_tab = st.data_editor(df_tab, num_rows="dynamic", use_container_width=True, key="tabular_editor")

        c1, c2 = st.columns([1, 1])
        with c1:
            runPred = st.button("Predict ratings", type="primary")
        with c2:
            downloadCsvButton(df_tab, "coffee_input_tabular.csv", "Download template CSV")

        if runPred:
            if model1 is None or model2 is None:
                st.warning("Model files missing. Please run `python train.py` first.")
            else:
                try:
                    y_hat = predict_rating(df_tab)
                    df_out = df_tab.copy()
                    df_out["predicted_rating"] = np.round(y_hat, 2)
                    st.success(f"Predicted {len(df_out)} rows")
                    st.dataframe(df_out, use_container_width=True)
                    downloadCsvButton(df_out, "coffee_predictions_tabular.csv", "Download predictions CSV")
                except Exception as e:
                    st.exception(e)

    with colR:
        st.subheader("Quick add")
        price = st.number_input("100g_USD", min_value=0.0, max_value=200.0, value=12.50, step=0.25)
        roast_in = st.selectbox(
            "roast",
            ["", "Very Light", "Light", "Light-Medium", "Medium-Light", "Medium", "Medium-Dark", "Dark", "Very Dark"],
            index=5,
        )
        if st.button("Predict single row"):
            row = pd.DataFrame([[price, roast_in if roast_in else np.nan]], columns=["100g_USD", "roast"])
            try:
                y1 = predict_rating(row)
                st.metric("Predicted rating", f"{float(y1[0]):.2f}")
            except Exception as e:
                st.error("Need trained model files.")
                st.stop()
        with st.expander("How roast is mapped"):
            if roast_in:
                st.write({"input": roast_in, "roast_cat": roast_category(roast_in)})
            st.caption("Mapping uses case-insensitive, hyphen/space tolerant labels.")

# ------------------------------
# Tab 2 — Review Text (Bonus)
# ------------------------------
with tab2:
    st.subheader("Predict from review-like text")
    if model3 is None:
        st.warning("Text model not found. Train with `python train.py` (without --skip-text).")
    
    upload_txt = st.file_uploader("Upload CSV with a 'text' column (optional)", type=["csv"], accept_multiple_files=False, key="txt_up")
    if upload_txt:
        try:
            df_text = pd.read_csv(upload_txt)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_text = exampleText()
    else:
        df_text = exampleText()

    if df_text.shape[1] == 1 and df_text.columns[0] != "text":
        df_text.columns = ["text"]
    if "text" not in df_text.columns:
        df_text = df_text.rename(columns={df_text.columns[0]: "text"})

    df_text = st.data_editor(df_text, num_rows="dynamic", use_container_width=True, key="text_editor")

    c1, c2 = st.columns([1, 1])
    with c1:
        runTxt = st.button("Predict from text", type="primary")
    with c2:
        downloadCsvButton(df_text, "coffee_input_text.csv", "Download template CSV")

    if runTxt:
        try:
            y_hat = predict_rating(df_text[["text"]], text=True)
            df_out = df_text.copy()
            df_out["predicted_rating"] = np.round(y_hat, 2)
            st.success(f"Predicted {len(df_out)} rows")
            st.dataframe(df_out, use_container_width=True)
            downloadCsvButton(df_out, "coffee_predictions_text.csv", "Download predictions CSV")
        except FileNotFoundError:
            st.error("model_3.pickle not found. Train it with `python train.py` (do not use --skip-text).")
        except Exception as e:
            st.exception(e)

# ------------------------------
# Tab 3 — Model Info
# ------------------------------
with tab3:
    st.subheader("Artifacts & simple diagnostics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear model (model_1)** — 100g_USD → rating")
        if model1 is None:
            st.info("Not loaded.")
        else:
            try:
                # Show coefficient & intercept if available
                coef = getattr(model1, "coef_", None)
                intercept = getattr(model1, "intercept_", None)
                if coef is not None:
                    st.write(pd.DataFrame({"feature": ["100g_USD"], "coef": coef}))
                if intercept is not None:
                    st.write({"intercept": float(intercept)})
            except Exception:
                st.write("Model loaded but coefficients not available.")

    with col2:
        st.markdown("**Decision tree (model_2)** — [100g_USD, roast_cat] → rating")
        if model2 is None:
            st.info("Not loaded.")
        else:
            try:
                fi = getattr(model2, "feature_importances_", None)
                if fi is not None and len(fi) == 2:
                    st.write(pd.DataFrame({"feature": ["100g_USD", "roast_cat"], "importance": fi}))
                st.caption("Basic impurity-based importances from scikit-learn.")
            except Exception:
                st.write("Model loaded but feature importances unavailable.")

    st.markdown("---")
    st.caption("This dashboard is a lightweight demo. For robust analysis, add cross-validation, error bars, and calibration plots.")
