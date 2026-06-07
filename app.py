import streamlit as st

# Collapse the sidebar by default on the per-slot history page (reached via a
# full-page "View" link from the Predictor results table). Keep it expanded on
# every other page / on first load. Detected from the URL before page_config so
# it applies natively on the initial render, no fragile JS toggling.
try:
    _url = st.context.url or ""
except Exception:
    _url = ""
_sidebar_state = "collapsed" if "Slot_History" in _url else "expanded"

st.set_page_config(
    page_title="JoSAA / CSAB Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state=_sidebar_state,
)

pg = st.navigation([
    st.Page("pages/0_Predictor.py",       title="Predictor",      icon="🎓"),
    st.Page("pages/1_Historical_Data.py", title="Historical Data", icon="📊"),
    st.Page("pages/2_Slot_History.py",    title="Slot History",    icon="📈"),
])
pg.run()
