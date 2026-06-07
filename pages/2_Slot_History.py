"""
Slot History page: Shows historical closing ranks for a single predicted slot.
Reached via the "📊 view" link in the Predictor results table.

Expected query params:
    source, inst, prog, quota, seat_type, gender,
    pred, lower, upper, cat, rank
"""

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pipeline.config import PREDICT_YEAR, SOURCES

CAT_COLOR = {"safe": "#27ae60", "match": "#f39c12", "reach": "#e74c3c"}
_ROUND_PALETTE = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#e74c3c", "#1abc9c"]


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _supabase_client():
    try:
        from supabase import create_client
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_supabase(
    table: str, inst: str, prog: str, quota: str, seat_type: str, gender: str,
) -> pd.DataFrame:
    client = _supabase_client()
    if client is None:
        return pd.DataFrame()
    try:
        rows: list[dict] = []
        page = 0
        while True:
            resp = (
                client.table(table)
                .select("Year,Round,Opening Rank,Closing Rank")
                .eq("Institute", inst)
                .eq("Academic Program Name", prog)
                .eq("Quota", quota)
                .eq("Seat Type", seat_type)
                .eq("Gender", gender)
                .range(page * 1000, page * 1000 + 999)
                .execute()
            )
            rows.extend(resp.data)
            if len(resp.data) < 1000:
                break
            page += 1
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ["Year", "Round", "Opening Rank", "Closing Rank"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return (
            df.dropna(subset=["Year", "Round", "Closing Rank"])
            .sort_values(["Year", "Round"])
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _fetch_csv(source: str) -> pd.DataFrame:
    cfg = SOURCES[source]
    csv_path = cfg["csv"]
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        from pipeline.loader import load as _load
        return _load(csv_path)
    except Exception:
        return pd.DataFrame()


def _fetch(source, inst, prog, quota, seat_type, gender) -> pd.DataFrame:
    full = _fetch_csv(source)
    if not full.empty:
        mask = (
            (full["Institute"] == inst)
            & (full["Academic Program Name"] == prog)
            & (full["Quota"] == quota)
            & (full["Seat Type"] == seat_type)
            & (full["Gender"] == gender)
        )
        sub = full[mask][["Year", "Round", "Opening Rank", "Closing Rank"]].copy()
        if not sub.empty:
            return sub.sort_values(["Year", "Round"]).reset_index(drop=True)
    return _fetch_supabase(source, inst, prog, quota, seat_type, gender)


# ── Chart builder ─────────────────────────────────────────────────────────────

def _build_fig(
    hist: pd.DataFrame,
    pred: int,
    lower: int,
    upper: int,
    cat: str,
    student_rank: int,
    final_only: bool,
) -> go.Figure:
    fig = go.Figure()
    _fg = "#1a1a1a"

    fig.add_hline(
        y=student_rank,
        line_dash="dash",
        line_color="royalblue",
        line_width=2,
        annotation_text=f"  Your rank: {student_rank:,}",
        annotation_position="top right",
    )

    if final_only:
        max_r = hist.groupby("Year")["Round"].max().rename("MaxR")
        hdf = hist.join(max_r, on="Year")
        hdf = hdf[hdf["Round"] == hdf["MaxR"]].sort_values("Year")
        fig.add_trace(go.Scatter(
            x=hdf["Year"], y=hdf["Closing Rank"],
            mode="lines+markers",
            name="Final round (actual)",
            line=dict(color="#3498db", width=2.5),
            marker=dict(size=8, color="#3498db"),
            hovertemplate="<b>Year %{x}</b><br>Closing Rank: %{y:,}<extra></extra>",
        ))
    else:
        for i, r in enumerate(sorted(hist["Round"].unique())):
            rdf = hist[hist["Round"] == r].sort_values("Year")
            color = _ROUND_PALETTE[i % len(_ROUND_PALETTE)]
            fig.add_trace(go.Scatter(
                x=rdf["Year"], y=rdf["Closing Rank"],
                mode="lines+markers",
                name=f"R{r} (actual)",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
                hovertemplate=f"<b>R{r} - Year %{{x}}</b><br>Closing Rank: %{{y:,}}<extra></extra>",
            ))

    pred_color = CAT_COLOR.get(cat, "#888")
    fig.add_trace(go.Scatter(
        x=[PREDICT_YEAR], y=[pred],
        mode="markers",
        name=f"{PREDICT_YEAR} prediction ({cat})",
        marker=dict(size=18, symbol="star", color=pred_color, line=dict(color="#333", width=1)),
        error_y=dict(
            type="data", symmetric=False,
            array=[upper - pred], arrayminus=[pred - lower],
            visible=True, color=pred_color, thickness=2.5, width=8,
        ),
        hovertemplate=(
            f"<b>{PREDICT_YEAR} Prediction</b><br>"
            f"Final Pred: {pred:,}<br>"
            f"95% interval: {lower:,} – {upper:,}<br>"
            f"Category: {cat}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="Actual closing ranks vs model prediction", font=dict(size=15, color=_fg)),
        font=dict(color=_fg),
        xaxis=dict(
            title=dict(text="Year", font=dict(color=_fg)),
            showgrid=True, gridcolor="#dddddd",
            tickmode="linear", dtick=1,
            tickfont=dict(color=_fg), linecolor=_fg,
        ),
        yaxis=dict(
            title=dict(text="Closing Rank", font=dict(color=_fg)),
            autorange="reversed",
            showgrid=True, gridcolor="#dddddd",
            tickformat=",",
            tickfont=dict(color=_fg), linecolor=_fg,
        ),
        height=480,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0,
            font=dict(size=11, color=_fg),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=55, b=120),
    )
    return fig


# ── Page ──────────────────────────────────────────────────────────────────────

def _qp(key: str, default: str = "") -> str:
    val = st.query_params.get(key)
    return str(val) if val is not None else default


def _qp_int(key: str, default: int = 0) -> int:
    try:
        return int(st.query_params.get(key, default))
    except (ValueError, TypeError):
        return default


source     = _qp("source", "josaa")
inst       = _qp("inst")
prog       = _qp("prog")
quota      = _qp("quota")
seat_type  = _qp("seat_type")
gender     = _qp("gender")
pred       = _qp_int("pred")
lower      = _qp_int("lower")
upper      = _qp_int("upper")
cat        = _qp("cat", "match")
student_rank = _qp_int("rank")

if not inst or not prog:
    st.warning("No slot specified. Please open this page from the Predictor results table.")
    st.stop()

# Sidebar starts collapsed on this page via initial_sidebar_state in app.py
# (detected from the URL), so no JS toggling is needed here.

st.title("Historical Closing Ranks")
st.markdown(f"**{inst}** - {prog}")
st.caption(f"Quota: {quota} · Seat Type: {seat_type} · Gender: {gender} · Source: {source.upper()}")
st.markdown("---")

with st.spinner("Fetching historical data…"):
    hist = _fetch(source, inst, prog, quota, seat_type, gender)

if hist.empty:
    st.info(
        "No historical data found for this slot. "
        "It may be a new programme or named differently in older data."
    )
    st.stop()

# Groundedness badge
n_yrs  = int(hist["Year"].nunique())
yr_min = int(hist["Year"].min())
yr_max = int(hist["Year"].max())
if n_yrs >= 5:
    gc, gt = "#27ae60", f"Strong, {n_yrs} years of data ({yr_min}–{yr_max})"
elif n_yrs >= 3:
    gc, gt = "#f39c12", f"Moderate, {n_yrs} years of data ({yr_min}–{yr_max})"
else:
    gc, gt = "#e74c3c", f"Weak, Only {n_yrs} year(s) of data"

st.markdown(
    f"<span style='background:{gc};color:white;padding:4px 12px;"
    f"border-radius:12px;font-size:0.85rem;font-weight:600'>"
    f"Groundedness: {gt}</span>",
    unsafe_allow_html=True,
)
st.markdown("")

show_all = st.checkbox(
    "Show all rounds",
    value=False,
    help="Default shows only the final round per year (what the model predicts). Enable to overlay all rounds.",
)

fig = _build_fig(hist, pred, lower, upper, cat, student_rank, final_only=not show_all)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Y-axis **inverted** (lower = more accessible). "
    f"★ = {PREDICT_YEAR} prediction with 95% interval. "
    "If the trend line and the star align closely, the prediction is well-grounded."
)

with st.expander("Raw data", expanded=False):
    dh = hist.rename(columns={
        "Closing Rank": "Closing Rank (actual)",
        "Opening Rank": "Opening Rank (actual)",
    }).sort_values(["Year", "Round"]).reset_index(drop=True)
    st.dataframe(
        dh, hide_index=True, width="stretch",
        column_config={
            "Year":  st.column_config.NumberColumn("Year",  format="%d"),
            "Round": st.column_config.NumberColumn("Round", format="%d"),
            "Opening Rank (actual)": st.column_config.NumberColumn("Opening Rank", format="%d"),
            "Closing Rank (actual)": st.column_config.NumberColumn("Closing Rank", format="%d"),
        },
    )
