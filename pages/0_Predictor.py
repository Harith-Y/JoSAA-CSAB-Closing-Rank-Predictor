import os
import pickle
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.config import PREDICT_YEAR, SOURCES, MODEL_DIR
from pipeline.predict import predict

# Constants
SEAT_TYPES = [
    "OPEN", "OPEN (PwD)",
    "EWS", "EWS (PwD)",
    "OBC-NCL", "OBC-NCL (PwD)",
    "SC", "SC (PwD)",
    "ST", "ST (PwD)",
]

QUOTAS = {
    "josaa": ["AI", "HS", "OS", "GO", "JK", "LA"],
    "csab":  ["AI", "HS", "OS", "JK", "LA"],
}

CAT_COLOR = {"safe": "#27ae60", "match": "#f39c12", "reach": "#e74c3c"}
CAT_ICON  = {"safe": "🟢", "match": "🟡", "reach": "🔴"}

# Model cache
@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(source: str):
    cfg  = SOURCES[source]
    path = os.path.join(MODEL_DIR, cfg["model"])
    if not os.path.exists(path):
        csv_path = cfg["csv"]
        if os.path.exists(csv_path):
            os.makedirs(MODEL_DIR, exist_ok=True)
            from pipeline.train import train
            from pipeline.config import DEFAULT_TREND_MODEL
            train(csv_path, model_path=path,
                  trend_model=cfg.get("trend_model", DEFAULT_TREND_MODEL))
        else:
            return None, path
    with open(path, "rb") as f:
        return pickle.load(f), path


# Helpers
def _slot_label(row: pd.Series) -> str:
    inst = _short_institute_name(row["Institute"])
    prog = _short_program_name(row["Academic Program Name"])
    return f"{inst} · {prog}"


def _short_institute_name(inst: str) -> str:
    inst = inst.split("(")[0].strip()
    replacements = [
        (r"^Indian Institute of Technology, Design & Manufacturing\b", "IIITDM"),
        (r"^Indian Institute of Information Technology, Design & Manufacturing\b", "IIITDM"),
        (r"^Indian Institute of Information Technology\b", "IIIT"),
        (r"^International Institute of Information Technology\b", "IIIT"),
        (r"^National Institute of Technology\b", "NIT"),
        (r"^Indian Institute of Technology\b", "IIT"),
        (r"^Indian Institute of Engineering Science and Technology\b", "IIEST"),
        (r"^Indian Institute of Science Education and Research\b", "IISER"),
        (r"^Indian Institute of Science\b", "IISc"),
    ]
    for pattern, replacement in replacements:
        inst = re.sub(pattern, replacement, inst, count=1)
    inst = re.sub(r"\bUniversity\b", "Univ.", inst)
    inst = re.sub(r"\bInstitute\b", "Inst.", inst)
    inst = re.sub(r"\s+,\s+", ", ", inst)
    if len(inst) > 32:
        inst = inst[:29] + "…"
    return inst


def _short_program_name(prog: str) -> str:
    prog = prog.strip()
    replacements = [
        (r"Computer Science and Engineering", "CSE"),
        (r"Computer Science & Engineering", "CSE"),
        (r"Computer Engineering", "CSE"),
        (r"Information Technology", "IT"),
        (r"Electronics and Communication Engineering", "ECE"),
        (r"Electronics & Communication Engineering", "ECE"),
        (r"Electrical Engineering", "EE"),
        (r"Mechanical Engineering", "ME"),
        (r"Civil Engineering", "CE"),
        (r"Chemical Engineering", "CHE"),
        (r"Aerospace Engineering", "AE"),
        (r"Biotechnology", "BT"),
        (r"Metallurgical Engineering", "Met. Engg."),
        (r"Mathematics and Computing", "MnC"),
        (r"Artificial Intelligence", "AI"),
        (r"Data Science", "DS"),
        (r"Electronics and Electrical Engineering", "EEE"),
        (r"Electronics & Electrical Engineering", "EEE"),
    ]
    for pattern, replacement in replacements:
        prog = re.sub(pattern, replacement, prog, flags=re.IGNORECASE)

    prog = re.sub(r"\s*\(4 Years?, Bachelor of Technology\)", "", prog, flags=re.IGNORECASE)
    prog = re.sub(r"\s*\(4 Years?, Bachelor of Engineering\)", "", prog, flags=re.IGNORECASE)
    prog = re.sub(r"\s*\(5 Years?, Bachelor of Technology\)", "", prog, flags=re.IGNORECASE)

    if len(prog) > 28:
        prog = prog[:25] + "…"
    return prog


def _slot_legend_label(row: pd.Series) -> str:
    inst = _short_institute_name(row["Institute"])
    prog = _short_program_name(row["Academic Program Name"])
    return f"{inst}<br>{prog}"


def _build_trajectory_fig(
    df: pd.DataFrame,
    round_cols: list[str],
    student_rank: int,
    source: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_hline(
        y=student_rank,
        line_dash="dash",
        line_color="royalblue",
        line_width=2,
        annotation_text=f"  Your rank: {student_rank:,}",
        annotation_position="top right",
    )

    for _, row in df.iterrows():
        ys   = [row[r] for r in round_cols if isinstance(row.get(r), (int, float))]
        xs   = [r      for r in round_cols if isinstance(row.get(r), (int, float))]
        cat  = row["Category"]
        name = _slot_label(row)
        legend_name = _slot_legend_label(row)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=legend_name,
            line=dict(color=CAT_COLOR[cat], width=2),
            marker=dict(size=8),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Round: %{x}<br>"
                "Predicted closing rank: %{y:,}<br>"
                f"Category: {cat}"
                "<extra></extra>"
            ),
        ))

    _fg = "#1a1a1a"

    fig.update_layout(
        title=dict(
            text=f"Closing-rank trajectory: {source.upper()} {PREDICT_YEAR}",
            font=dict(size=16, color=_fg),
        ),
        font=dict(color=_fg),
        xaxis=dict(
            title=dict(text="Round", font=dict(color=_fg)),
            showgrid=True,
            gridcolor="#dddddd",
            tickmode="array",
            tickvals=round_cols,
            tickfont=dict(color=_fg),
            linecolor=_fg,
        ),
        yaxis=dict(
            title=dict(text="Predicted Closing Rank", font=dict(color=_fg)),
            autorange="reversed",
            showgrid=True,
            gridcolor="#dddddd",
            tickformat=",",
            tickfont=dict(color=_fg),
            linecolor=_fg,
        ),
        height=540,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="left",
            x=0,
            font=dict(size=11, color=_fg),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=60, b=170),
    )
    return fig


# Sidebar
with st.sidebar:
    st.title("College Predictor")
    st.caption(f"Predictions for **{PREDICT_YEAR}** counselling")
    st.markdown("---")

    source = st.radio("Counselling source", ["JoSAA", "CSAB"],
                      horizontal=True).lower()
    cfg = SOURCES[source]

    if source == "csab":
        exam_type = "mains"
        st.info(
            "CSAB covers NITs / IIITs / GFTIs only (JEE Mains ranks). "
            "**CSAB uses CRL for all categories**. Enter your CRL rank below, not your category rank."
        )
    else:
        exam_label = st.radio(
            "Exam",
            ["JEE Mains  →  NIT / IIIT / GFTI", "JEE Advanced  →  IIT"],
            help="JEE Advanced is exclusively for IIT admissions.",
        )
        exam_type = "advanced" if "Advanced" in exam_label else "mains"

    if source == "csab":
        rank_help = (
            "Enter your JEE Mains **CRL (Common Rank List)** rank. "
            "CSAB uses CRL for all seat types; including OBC-NCL, SC, ST, and EWS. "
            "Do not enter your category rank here."
        )
    else:
        rank_help = (
            "Enter the rank corresponding to your selected Seat Type. "
            "Use CRL for OPEN / OPEN (PwD), and category rank for "
            "OBC-NCL / SC / ST / EWS (including PwD variants)."
        )

    rank = st.number_input(
        "Your rank",
        min_value=1, max_value=1_000_000,
        value=10_000, step=100,
        help=rank_help,
    )

    quota = st.selectbox(
        "Quota",
        QUOTAS[source],
        help=(
            "**AI**: All India (open to everyone). IITs and IIITs use this exclusively. "
            "NITs have very few AI seats and they're highly competitive.\n\n"
            "**HS**: Home State. NIT seats for students from the same state as the NIT. "
            "Largest share of NIT seats: pick this if you're from that state.\n\n"
            "**OS**: Other State. NIT seats for students from outside the NIT's state.\n\n"
            "**GO**: Goa. special quota at NIT Goa for students from Goa.\n\n"
            "**JK**: Jammu & Kashmir. reserved for students domiciled in J&K.\n\n"
            "**LA**: Ladakh. reserved for students domiciled in Ladakh.\n\n"
        ),
    )
    seat_type = st.selectbox("Seat Type", SEAT_TYPES)

    gender_raw = st.radio("Gender", ["Gender-Neutral", "Female-only"])
    gender = (
        "Female-only (including Supernumerary)"
        if gender_raw == "Female-only"
        else "Gender-Neutral"
    )

    include_reach = st.checkbox("Include reach colleges", value=True)

    coverage = 0.95

    st.markdown("---")
    predict_btn = st.button("Predict", width="stretch", type="primary")

    current_inputs = (source, exam_type, rank, quota, seat_type, gender, include_reach)
    if st.session_state.get("last_inputs") != current_inputs:
        st.session_state.pop("results_df", None)
        st.session_state.pop("last_rank", None)


# Main area
st.title("JoSAA / CSAB Closing Rank Predictor")

if cfg.get("disclaimer"):
    st.warning(cfg["disclaimer"])

if predict_btn:
    model, model_path = load_model_cached(source)
    if model is None:
        st.error(
            f"**Model and data not found.**\n\n"
            f"The app needs either:\n"
            f"- `models/{source}_model.pkl` (pre-trained, commit to the repo), **or**\n"
            f"- `{source}_ranks.csv` (raw data, app will auto-train on first load)\n\n"
            f"Run locally: `python predict_cli.py train --source {source}` "
            f"then commit `models/{source}_model.pkl`."
        )
        st.stop()

    with st.spinner("Computing predictions…"):
        df = predict(
            rank            = rank,
            exam_type       = exam_type,
            quota           = quota,
            seat_type       = seat_type,
            gender          = gender,
            model           = model,
            rounds          = cfg["rounds"],
            include_reach   = include_reach,
            safe_threshold  = cfg["safe_threshold"],
            reach_threshold = cfg["reach_threshold"],
            coverage        = coverage,
        )

    st.session_state["results_df"]  = df
    st.session_state["last_rank"]   = rank
    st.session_state["last_inputs"] = current_inputs

# Display results
df = st.session_state.get("results_df")

if df is None:
    st.markdown(
        """
        ### How to use

        1. Choose your counselling source in the sidebar (**JoSAA** for main
           counselling, **CSAB** for the supplementary round).
        2. Fill in your exam type, rank, quota, seat type, and gender.
        3. Click **Predict** to see eligible colleges with predicted closing
           ranks for every round.
        4. Switch to the **Trajectory Plot** tab to compare how closing ranks
           evolve across rounds for your shortlisted colleges.
        """,
    )
    st.stop()

if df is None or df.empty:
    st.info("No matching colleges found for the given profile. Try widening your criteria.")
    st.stop()

round_cols = [c for c in df.columns if c.startswith("R") and c[1:].isdigit()]
student_rank = st.session_state.get("last_rank", rank)

# Summary metrics
counts = df["Category"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Safe",  counts.get("safe",  0))
c2.metric("Match", counts.get("match", 0))
c3.metric("Reach", counts.get("reach", 0))
c4.metric("Total", len(df))

export_cols = [
    "Category", "Institute", "Academic Program Name", "Quota", "Seat Type", "Gender",
    *round_cols, "Final Pred", "Lower", "Upper", "Years", "Seats",
]
export_df = df[[c for c in export_cols if c in df.columns]].copy()
csv_data = export_df.to_csv(index=False).encode("utf-8")

with st.expander("Export results", expanded=False):
    selected_export_cats = st.multiselect(
        "Categories to include in filtered export",
        options=["safe", "match", "reach"],
        default=["safe", "match"],
        help="Choose one or more categories for the filtered CSV download.",
    )

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Export all categories as CSV",
            data=csv_data,
            file_name=f"{source}_{exam_type}_rank_{student_rank}_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

    with e2:
        filtered_export_df = export_df[export_df["Category"].isin(selected_export_cats)]
        filtered_csv_data = filtered_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export selected categories as CSV",
            data=filtered_csv_data,
            file_name=(
                f"{source}_{exam_type}_rank_{student_rank}_"
                f"{'-'.join(selected_export_cats) if selected_export_cats else 'none'}_predictions.csv"
            ),
            mime="text/csv",
            width="stretch",
            disabled=not selected_export_cats,
        )

tab_table, tab_plot = st.tabs(["Results Table", "Trajectory Plot"])

# Results table
with tab_table:
    f1, f2 = st.columns(2)

    with f1:
        prog_kw = st.text_input(
            "Filter by branch",
            key="filter_prog",
            placeholder="e.g. CSE, Mechanical, Data Science, AI",
            help="Matches any part of the program name. 'CSE' finds all CS variants at once.",
        )

    with f2:
        inst_kw = st.text_input(
            "Filter by institute",
            key="filter_inst",
            placeholder="e.g. NIT Trichy, IIT Bombay, IIIT Hyderabad",
            help="Matches any part of the institute name.",
        )

    table_df = df.copy()
    if prog_kw:
        table_df = table_df[
            table_df["Academic Program Name"].str.contains(prog_kw, case=False, na=False)
        ]
    if inst_kw:
        table_df = table_df[
            table_df["Institute"].str.contains(inst_kw, case=False, na=False)
        ]

    if table_df.empty:
        st.info("No rows match the selected program/institute filter.")

    has_seats     = "Seats" in df.columns and df["Seats"].notna().any()
    has_intervals = "Lower" in df.columns and "Upper" in df.columns
    display_cols = (
        ["Institute", "Academic Program Name"]
        + round_cols
        + ["Final Pred"]
        + (["Lower", "Upper"] if has_intervals else [])
        + ["Years"]
        + (["Seats"] if has_seats else [])
    )
    cov_pct = 95
    col_cfg = {
        "Final Pred": st.column_config.NumberColumn("Final", format="%d"),
        "Lower":      st.column_config.NumberColumn(
                          f"Lower ({cov_pct}%)",
                          format="%d",
                          help=f"Lower bound of the {cov_pct}% prediction interval. "
                               "Your rank below this means the slot is Safe."),
        "Upper":      st.column_config.NumberColumn(
                          f"Upper ({cov_pct}%)",
                          format="%d",
                          help=f"Upper bound of the {cov_pct}% prediction interval. "
                               "Your rank above this means the slot is out of reach."),
        "Years":      st.column_config.NumberColumn("Yrs",
                          help="Years of historical data for this slot"),
        "Seats":      st.column_config.NumberColumn("Seats",
                          help="Total seats available in this slot for the prediction year"),
        **{r: st.column_config.NumberColumn(r, format="%d") for r in round_cols},
    }

    for cat in ["safe", "match", "reach"]:
        subset = table_df[table_df["Category"] == cat]
        if subset.empty:
            continue
        color = CAT_COLOR[cat]
        icon  = CAT_ICON[cat]
        st.markdown(
            f"<h4 style='color:{color};margin-bottom:6px'>"
            f"{icon} {cat.upper()} &nbsp; <small>({len(subset)} options)</small>"
            "</h4>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            subset[display_cols].reset_index(drop=True),
            width="stretch",
            hide_index=True,
            column_config=col_cfg,
        )
        st.markdown("")

# Trajectory plot
with tab_plot:
    st.markdown(
        """
        <div style="padding:0.75rem 1rem;border:1px solid #d0d0d0;border-radius:0.5rem;
                    background:rgba(255,255,255,0.04);margin-bottom:0.75rem;">
        <ul style="margin:0;padding-left:1.2rem;">
            <li>Select colleges from your results to compare their predicted closing-rank trajectories across rounds.</li>
            <li>The dashed blue line marks <strong>your rank</strong>. Traces below the line indicate rounds where you would be eligible for that seat.</li>
            <li>Use <strong>full-screen</strong> view for the cleanest chart when plotting many institutes.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df["_label"] = df.apply(_slot_label, axis=1)

    default_labels = (
        df[df["Category"].isin(["safe", "match"])]
        .head(3)["_label"]
        .tolist()
    )

    chosen_labels = st.multiselect(
        "Colleges to plot",
        options=df["_label"].tolist(),
        default=default_labels,
        help="You can select up to ~15 colleges; more than that becomes cluttered.",
    )

    if not chosen_labels:
        st.info("Select at least one college above to see the trajectory plot.")
    else:
        chosen_df = df[df["_label"].isin(chosen_labels)].copy()
        fig = _build_trajectory_fig(chosen_df, round_cols, student_rank, source)
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Y-axis is **inverted**: lower position on the chart = higher rank number "
            "= more accessible seat. Your rank line divides eligible seats (below) "
            "from too-competitive seats (above). Round R1 typically has the highest "
            "(most accessible) closing ranks; later rounds tighten as floating students "
            "fill seats."
        )
