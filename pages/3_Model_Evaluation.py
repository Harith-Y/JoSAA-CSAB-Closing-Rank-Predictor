import os
import pickle

import pandas as pd
import streamlit as st

from pipeline.config import PREDICT_YEAR, SOURCES, MODEL_DIR, CURRENT_ROUND_DATA
from pipeline.predict import load_all_actuals, evaluate_round
from pipeline.display import _display_program_name, _short_institute_name, abbr_guide_df


@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    cfg  = SOURCES["josaa"]
    path = os.path.join(MODEL_DIR, cfg["model"])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


_ROUND_DATA_KEY = tuple(sorted(CURRENT_ROUND_DATA.items()))

@st.cache_data(show_spinner=False)
def _load_actuals(round_data: tuple = _ROUND_DATA_KEY) -> dict[int, dict]:
    return load_all_actuals(dict(round_data))


@st.cache_data(show_spinner=False)
def _evaluate(_model, actuals: dict, round_num: int, year: int) -> pd.DataFrame:
    return evaluate_round(_model, actuals, round_num, year)


def _add_display_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Inst"]    = df["Institute"].apply(_short_institute_name)
    df["Program"] = df["Academic Program Name"].apply(_display_program_name)
    return df


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
    st.subheader(f"Round {rn} — {PREDICT_YEAR}")
    with st.spinner(f"Evaluating R{rn}…"):
        eval_df = _evaluate(model, r_actuals, rn, PREDICT_YEAR)

    if eval_df.empty:
        st.caption("No matched slots.")
        continue

    eval_df = _add_display_cols(eval_df)
    pred_col = f"Predicted_R{rn}"
    act_col  = f"Actual_R{rn}"

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",           f"{int(eval_df['Abs_Error'].mean()):,}")
    c2.metric("Median AE",     f"{int(eval_df['Abs_Error'].median()):,}")
    c3.metric("Slots matched", f"{len(eval_df):,}")

    st.caption("Slots within error threshold:")
    cols = st.columns(4)
    for i, (thresh, label) in enumerate(
        [(500, "±500"), (1000, "±1,000"), (2000, "±2,000"), (5000, "±5,000")]
    ):
        frac = (eval_df["Abs_Error"] <= thresh).mean()
        with cols[i]:
            st.markdown(f"**{label}**")
            st.progress(frac)
            st.caption(f"{frac * 100:.1f}%")

    st.markdown("")

    _disp_cols = ["Inst", "Program", pred_col, act_col, "Abs_Error", "Pct_Error"]
    _col_cfg = {
        "Inst":    st.column_config.TextColumn("Institute"),
        "Program": st.column_config.TextColumn("Program"),
        pred_col:  st.column_config.NumberColumn(pred_col, format="%d"),
        act_col:   st.column_config.NumberColumn(act_col,  format="%d"),
        "Abs_Error":  st.column_config.NumberColumn("Abs Error", format="%d"),
        "Pct_Error":  st.column_config.NumberColumn("% Error",   format="%.1f%%"),
    }

    with st.expander(f"Worst-predicted R{rn} slots (top 20)"):
        with st.popover("📖 Abbr. guide"):
            st.dataframe(abbr_guide_df(), hide_index=True, height=300)
        st.dataframe(
            eval_df[_disp_cols].head(20),
            hide_index=True,
            use_container_width=True,
            column_config=_col_cfg,
        )

    with st.expander(f"All R{rn} evaluated slots ({len(eval_df):,} total)"):
        with st.popover("📖 Abbr. guide"):
            st.dataframe(abbr_guide_df(), hide_index=True, height=300)
        st.dataframe(
            eval_df[_disp_cols],
            hide_index=True,
            use_container_width=True,
            column_config=_col_cfg,
        )

    st.markdown("---")
