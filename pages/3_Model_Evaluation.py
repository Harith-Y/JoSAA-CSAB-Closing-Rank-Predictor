import os
import pickle

import pandas as pd
import streamlit as st

from pipeline.config import PREDICT_YEAR, SOURCES, MODEL_DIR, CURRENT_ROUND_DATA
from pipeline.predict import load_all_actuals, evaluate_round

st.set_page_config(page_title="Model Evaluation", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    cfg  = SOURCES["josaa"]
    path = os.path.join(MODEL_DIR, cfg["model"])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def _load_actuals() -> dict[int, dict]:
    return load_all_actuals(CURRENT_ROUND_DATA)


@st.cache_data(show_spinner=False)
def _evaluate(_model, actuals: dict, round_num: int, year: int) -> pd.DataFrame:
    return evaluate_round(_model, actuals, round_num, year)


st.title("Model Evaluation")
st.caption(f"JoSAA {PREDICT_YEAR}: predicted vs actual closing ranks")

model = _load_model()
if model is None:
    st.error("Model not found. The model will be trained automatically on the next Predictor page load.")
    st.stop()

all_actuals = _load_actuals()
if not all_actuals:
    st.info("No actual round data is available yet for the current counselling year.")
    st.stop()

for rn, r_actuals in sorted(all_actuals.items()):
    st.subheader(f"Round {rn} {PREDICT_YEAR}")
    with st.spinner(f"Evaluating R{rn}…"):
        eval_df = _evaluate(model, r_actuals, rn, PREDICT_YEAR)

    if eval_df.empty:
        st.caption("No matched slots.")
        continue

    pred_col = f"Predicted_R{rn}"
    act_col  = f"Actual_R{rn}"

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",        f"{int(eval_df['Abs_Error'].mean()):,}")
    c2.metric("Median AE",  f"{int(eval_df['Abs_Error'].median()):,}")
    c3.metric("Slots matched", f"{len(eval_df):,}")

    st.caption("Slots within error threshold:")
    cols = st.columns(4)
    for i, (thresh, label) in enumerate([(500, "±500"), (1000, "±1,000"), (2000, "±2,000"), (5000, "±5,000")]):
        frac = (eval_df["Abs_Error"] <= thresh).mean()
        with cols[i]:
            st.markdown(f"**{label}**")
            st.progress(frac)
            st.caption(f"{frac * 100:.1f}%")

    st.markdown("")

    with st.expander(f"Worst-predicted R{rn} slots (top 20)"):
        st.dataframe(
            eval_df[["Institute", "Academic Program Name", pred_col, act_col, "Abs_Error", "Pct_Error"]].head(20),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander(f"All R{rn} evaluated slots ({len(eval_df):,} total)"):
        st.dataframe(
            eval_df[["Institute", "Academic Program Name", pred_col, act_col, "Abs_Error", "Pct_Error"]],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
