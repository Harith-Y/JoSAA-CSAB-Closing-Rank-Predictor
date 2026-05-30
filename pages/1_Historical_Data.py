"""
Historical closing-rank browser.
"""

import pandas as pd
import streamlit as st

from pipeline.config import IIT_KEYWORDS, PREDICT_YEAR


def _qp(key: str, default: str) -> str:
    val = st.query_params.get(key)
    return str(val) if val is not None else default


def _qp_list(key: str, default: list, cast=str) -> list:
    val = st.query_params.get(key)
    if not val:
        return default
    try:
        return [cast(v.strip()) for v in str(val).split(",") if v.strip()]
    except (ValueError, AttributeError):
        return default


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


@st.cache_resource(show_spinner=False)
def _supabase_client():
    from supabase import create_client
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_years(table: str) -> list[int]:
    """Return sorted list of years by querying min and max (avoids loading all rows)."""
    client = _supabase_client()
    lo = client.table(table).select("Year").order("Year", desc=False).limit(1).execute()
    hi = client.table(table).select("Year").order("Year", desc=True).limit(1).execute()
    if not lo.data or not hi.data:
        return []
    return list(range(int(lo.data[0]["Year"]), int(hi.data[0]["Year"]) + 1))


@st.cache_data(ttl=3600, show_spinner="Fetching data…")
def fetch_data(table: str, years: tuple[int, ...]) -> pd.DataFrame | None:
    """Returns None on a database error (e.g. statement timeout)."""
    client = _supabase_client()
    rows: list[dict] = []
    try:
        for year in years:
            page = 0
            while True:
                resp = (
                    client.table(table)
                    .select("*")
                    .eq("Year", year)
                    .range(page * 1000, page * 1000 + 999)
                    .execute()
                )
                rows.extend(resp.data)
                if len(resp.data) < 1000:
                    break
                page += 1
    except Exception:
        st.error("Couldn't load data right now. Please try again in a moment.")
        return None
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _exam_type(inst: str) -> str:
    inst_lower = inst.lower()
    for kw in IIT_KEYWORDS:
        if kw in inst_lower:
            return "advanced"
    return "mains"


# Page

st.title("Historical Closing Ranks")
st.caption("Browse actual closing ranks from past JoSAA / CSAB counselling rounds.")

# Sidebar
with st.sidebar:
    st.header("Filters")

    source = st.radio(
        "Source", ["JoSAA", "CSAB"],
        index=1 if _qp("source", "josaa") == "csab" else 0,
        horizontal=True,
    ).lower()
    table  = source  # table names match source names

    available_years = _fetch_years(table)
    if not available_years:
        st.error("No data found. Please try again later.")
        st.stop()

    _years_saved = _qp_list("years", [available_years[-1]], cast=int)
    _years_default = [y for y in _years_saved if y in available_years] or [available_years[-1]]
    selected_years = st.multiselect(
        "Year(s)",
        options=available_years,
        default=_years_default,
        help="Select one or more years.",
    )
    if not selected_years:
        st.warning("Select at least one year to load data.")
        st.stop()

    _exam_opts = ["All", "JEE Mains (NIT / IIIT / GFTI)", "JEE Advanced (IIT)"]
    _exam_saved = _qp("exam_filter", "All")
    exam_filter = st.radio(
        "Exam type",
        _exam_opts,
        index=_exam_opts.index(_exam_saved) if _exam_saved in _exam_opts else 0,
    )

    _quota_saved = [q for q in _qp_list("quota", []) if q in QUOTAS[source]]
    quota_filter = st.multiselect(
        "Quota", QUOTAS[source], default=_quota_saved,
        help="Leave empty to show all quotas.",
    )

    _seat_saved = [s for s in _qp_list("seat_type", []) if s in SEAT_TYPES]
    seat_filter = st.multiselect(
        "Seat Type", SEAT_TYPES, default=_seat_saved,
        help="Leave empty to show all seat types.",
    )

    _gender_opts = ["Gender-Neutral", "Female-only (including Supernumerary)"]
    _gender_saved = [g for g in _qp_list("gender", []) if g in _gender_opts]
    gender_filter = st.multiselect(
        "Gender", _gender_opts, default=_gender_saved,
    )

    rounds_available = [1, 2, 3, 4, 5, 6] if source == "josaa" else [1, 2]
    round_col = "Round" if source == "josaa" else "Special Round"
    _rounds_saved = [r for r in _qp_list("rounds", [], cast=int) if r in rounds_available]
    round_filter = st.multiselect(
        "Round", rounds_available, default=_rounds_saved,
        help="Leave empty to show all rounds.",
    )

# Pre-populate text inputs from URL on fresh load
if "hist_inst" not in st.session_state:
    st.session_state["hist_inst"] = _qp("inst", "")
if "hist_prog" not in st.session_state:
    st.session_state["hist_prog"] = _qp("prog", "")

# Inline keyword filters
c1, c2 = st.columns(2)
with c1:
    inst_kw = st.text_input(
        "Filter by institute",
        key="hist_inst",
        placeholder="e.g. NIT Trichy, IIT Bombay, IIIT Hyderabad",
        help="Matches any part of the institute name.",
    )
with c2:
    prog_kw = st.text_input(
        "Filter by branch",
        key="hist_prog",
        placeholder="e.g. CSE, Mechanical, Data Science, AI",
        help="Matches any part of the program name.",
    )

# Sync all current filter state to URL
st.query_params.update({
    "source":      source,
    "years":       ",".join(str(y) for y in sorted(selected_years)),
    "exam_filter": exam_filter,
    "quota":       ",".join(quota_filter),
    "seat_type":   ",".join(seat_filter),
    "gender":      ",".join(gender_filter),
    "rounds":      ",".join(str(r) for r in sorted(round_filter)),
    "inst":        inst_kw,
    "prog":        prog_kw,
})

# Load + filter
df = fetch_data(table, tuple(sorted(selected_years)))

if df is None:
    st.stop()

if df.empty:
    st.warning("No data returned. Please try again later.")
    st.stop()

# Derive exam type for filtering
df["_et"] = df["Institute"].apply(_exam_type)
if exam_filter == "JEE Mains (NIT / IIIT / GFTI)":
    df = df[df["_et"] == "mains"]
elif exam_filter == "JEE Advanced (IIT)":
    df = df[df["_et"] == "advanced"]

if quota_filter:
    df = df[df["Quota"].isin(quota_filter)]
if seat_filter:
    df = df[df["Seat Type"].isin(seat_filter)]
if gender_filter:
    df = df[df["Gender"].isin(gender_filter)]
if round_filter and round_col in df.columns:
    df = df[df[round_col].isin(round_filter)]
if inst_kw:
    df = df[df["Institute"].str.contains(inst_kw, case=False, na=False)]
if prog_kw:
    df = df[df["Academic Program Name"].str.contains(prog_kw, case=False, na=False)]

df = df.drop(columns=["_et", "id"], errors="ignore")

# Summary row
st.caption(f"**{len(df):,}** rows matching current filters")

if df.empty:
    st.info("No rows match the current filters. Try relaxing some selections.")
    st.stop()

# Column order + display
DISPLAY_COLS = [
    "Year", "Round", "Special Round", "Institute", "Academic Program Name",
    "Quota", "Seat Type", "Gender", "Opening Rank", "Closing Rank",
]
display_cols = [c for c in DISPLAY_COLS if c in df.columns]

sort_round_col = round_col if round_col in df.columns else "Year"
st.dataframe(
    df[display_cols].sort_values(
        ["Year", sort_round_col, "Closing Rank"], ascending=[False, True, True]
    ).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Year":          st.column_config.NumberColumn("Year",          format="%d"),
        "Round":         st.column_config.NumberColumn("Round",         format="%d"),
        "Special Round": st.column_config.NumberColumn("Special Round", format="%d"),
        "Opening Rank":  st.column_config.NumberColumn("Opening Rank",  format="%d"),
        "Closing Rank":  st.column_config.NumberColumn("Closing Rank",  format="%d"),
    },
)

# Download
csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
year_str = "-".join(str(y) for y in sorted(selected_years))
st.download_button(
    "Download as CSV",
    data=csv_bytes,
    file_name=f"{source}_historical_{year_str}.csv",
    mime="text/csv",
)
