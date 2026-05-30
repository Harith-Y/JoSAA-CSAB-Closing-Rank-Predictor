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
_BRANCH_ABBRS = [
    # ── Computer / IT ──────────────────────────────────────────────────────────
    (r"Computer Science and Engineering", "CSE"),
    (r"Computer Science & Engineering", "CSE"),
    (r"Computer Science Engineering", "CSE"),
    (r"Computer Science and Artificial Intelligence", "CSAI"),
    (r"Computer Science and Business", "CSB"),
    (r"Computer Science and Technology", "CST"),
    (r"Computer Science", "CS"),
    (r"Computer Engineering", "CE"),
    (r"Information Technology", "IT"),
    (r"Computational Engineering", "CompE"),
    (r"Computational Mathematics", "CM"),
    (r"Computational and Data Science", "CDS"),
    # ── Electronics ────────────────────────────────────────────────────────────
    (r"Electronics and Communication Engineering", "ECE"),
    (r"Electronics & Communication Engineering", "ECE"),
    (r"Electronics and Electrical Communication Engineering", "EECE"),
    (r"Electronics and Electrical Engineering", "EEE"),
    (r"Electronics & Electrical Engineering", "EEE"),
    (r"Electrical and Electronics Engineering", "EEE"),
    (r"Electronics and Instrumentation Engineering", "EIE"),
    (r"Electrical and Instrumentation Engineering", "EIE"),
    (r"Electronics and Telecommunication Engineering", "ETE"),
    (r"Electronics and VLSI Engineering", "EVLSIE"),
    (r"Electronics System Engineering", "ESE"),
    (r"Electronics Engineering", "ElecE"),
    (r"Microelectronics.*VLSI", "MVLSI"),
    (r"Integrated Circuit Design", "ICD"),
    # ── Electrical ─────────────────────────────────────────────────────────────
    (r"Electrical Engineering", "EE"),
    (r"Electronic Engineering", "ElecE"),
    # ── Mechanical / Manufacturing ─────────────────────────────────────────────
    (r"Mechanical Engineering", "ME"),
    (r"Mechatronics and Automation Engineering", "MechT"),
    (r"Mechatronics Engineering", "MechT"),
    (r"Mechatronics", "MechT"),
    (r"Manufacturing Science and Engineering", "MfgE"),
    (r"Manufacturing Engineering", "MfgE"),
    (r"Production and Industrial Engineering", "PIE"),
    (r"Production Engineering", "ProdE"),
    (r"Industrial and Production Engineering", "IPE"),
    (r"Industrial and Systems Engineering", "ISE"),
    (r"Industrial Engineering and Operations Research", "IEOR"),
    (r"Industrial Engineering", "IE"),
    (r"Engineering Design", "ED"),
    (r"Quality Engineering Design and Manufacturing", "QEDM"),
    # ── Civil / Architecture ───────────────────────────────────────────────────
    (r"Civil and Environmental Engineering", "CEE"),
    (r"Civil and Infrastructure Engineering", "CIE"),
    (r"Civil Engineering", "CE"),
    (r"Architecture and Planning", "AP"),
    (r"Architecture, Town and Regional Planning", "ATRP"),
    (r"Architecture", "Arch"),
    (r"Planning", "Planning"),
    # ── Chemical / Materials ───────────────────────────────────────────────────
    (r"Chemical and Biochemical Engineering", "CBE"),
    (r"Chemical Engineering \(Plastic and Polymer\)", "CEPP"),
    (r"Chemical Engineering", "ChE"),
    (r"Chemical Science and Technology", "ChST"),
    (r"Chemical Sciences?", "ChemSci"),
    (r"Chemical Technology", "ChemT"),
    (r"Chemistry", "Chem"),
    (r"Metallurgical and Materials Engineering", "MetMatE"),
    (r"Metallurgical Engineering and Materials Science", "MetMatE"),
    (r"Metallurgy and Materials Engineering", "MetMatE"),
    (r"Metallurgical Engineering", "MetE"),
    (r"Mineral and Metallurgical Engineering", "MinMetE"),
    (r"Materials and Metallurgical Engineering", "MatME"),
    (r"Materials Science and Metallurgical Engineering", "MSME"),
    (r"Materials Science and Engineering", "MatSciE"),
    (r"Materials Science and Technology", "MatSciT"),
    (r"Materials Engineering", "MatE"),
    (r"Material Science and Engineering", "MatSciE"),
    (r"Material Science", "MatSci"),
    (r"Polymer Science and Engineering", "PolyE"),
    (r"Polymer Science and Technology", "PolyT"),
    (r"Rubber Technology", "RT"),
    (r"Ceramic Engineering", "CerE"),
    # ── Aerospace / Ocean / Naval ──────────────────────────────────────────────
    (r"Aeronautical Engineering", "AeroE"),
    (r"Aerospace Engineering", "AE"),
    (r"Naval Architecture and Ocean Engineering", "NAOE"),
    (r"Ocean Engineering and Naval Architecture", "OENA"),
    (r"Ocean Engineering", "OE"),
    (r"Naval Architecture", "NA"),
    (r"Space Science and Engineering", "SSE"),
    (r"Space Sciences and Engineering", "SSsE"),
    # ── Mining / Petroleum / Energy ────────────────────────────────────────────
    (r"Mining Machinery Engineering", "MinMachE"),
    (r"Mining Safety Engineering", "MSE"),
    (r"Mining Engineering", "MineE"),
    (r"Mineral Engineering", "MineE"),
    (r"Petroleum Engineering", "PetroE"),
    (r"Nuclear Engineering", "NucE"),
    (r"Energy and Electrical Vehicle Engineering", "EEEV"),
    (r"Energy Engineering", "EnergyE"),
    # ── Bio / Agriculture / Food ───────────────────────────────────────────────
    (r"Biochemical Engineering and Biotechnology", "BEBT"),
    (r"Biochemical Engineering", "BioChemE"),
    (r"Bioengineering", "BioE"),
    (r"Biological Engineering", "BioE"),
    (r"Biological Sciences? and Bioengineering", "BSBE"),
    (r"Biosciences? and Bioengineering", "BBE"),
    (r"Biological Sciences?", "BioSci"),
    (r"Biomedical Engineering", "BME"),
    (r"Bio Medical Engineering", "BME"),
    (r"Bio Engineering", "BioE"),
    (r"Biotechnology and Biochemical Engineering", "BT+BioChmE"),
    (r"Biotechnology and Bioinformatics", "BT+BioInf"),
    (r"Biotechnology", "BT"),
    (r"Bio Technology", "BT"),
    (r"Agricultural and Food Engineering", "AgriFood E"),
    (r"Agricultural Engineering", "AgriE"),
    (r"Food Engineering and Technology", "FoodET"),
    (r"Food Process Engineering", "FoodProcE"),
    (r"Food Technology and Management", "FTM"),
    (r"Food Technology", "FoodT"),
    (r"Pharmaceutical Engineering & Technology", "PharmET"),
    (r"Pharmaceutical Engineering", "PharmE"),
    (r"Pharmaceutics", "Pharm"),
    (r"Dairy Engineering", "DairyE"),
    # ── Mathematics / Physics / Statistics ─────────────────────────────────────
    (r"Mathematics and Computing", "MnC"),
    (r"Mathematics & Computing", "MnC"),
    (r"Mathematics and Data Science", "MDS"),
    (r"Mathematics and Scientific Computing", "MSC"),
    (r"Mathematics Computing Technology", "MnCT"),
    (r"Applied Mathematics", "AM"),
    (r"Quantitative Economics.*Data Science", "QEDS"),
    (r"Statistics and Data Science", "SDS"),
    (r"Mathematics", "Maths"),
    (r"Physics and Computational Engineering", "PCE"),
    (r"Engineering Physics", "EngPhys"),
    (r"Applied Geophysics", "AGP"),
    (r"Exploration Geophysics", "EG"),
    (r"Geophysical Technology", "GeophysT"),
    (r"Applied Geology", "AGL"),
    (r"Geological Technology", "GeoT"),
    (r"Physics", "Physics"),
    # ── Data / AI / Interdisciplinary ──────────────────────────────────────────
    (r"Artificial Intelligence and Data Analytics", "AIDA"),
    (r"Artificial Intelligence and Data Engineering", "AIDE"),
    (r"Artificial Intelligence and Data Science", "AI+DS"),
    (r"Artificial Intelligence and Machine Learning", "AIML"),
    (r"Artificial Intelligence", "AI"),
    (r"Data Science and Artificial Intelligence", "DSAI"),
    (r"Data Science and Engineering", "DSE"),
    (r"Data Science", "DS"),
    (r"Instrumentation and Biomedical Engineering", "IBME"),
    (r"Instrumentation and Control Engineering", "InstrCE"),
    (r"Instrumentation Engineering", "InstrE"),
    (r"Engineering Science", "EngSci"),
    (r"Interdisciplinary Sciences", "ISci"),
    (r"Industrial Internet of Things", "IIoT"),
    (r"Robotics and AI", "RAI"),
    # ── Specialisation terms (appear after "with specialisation in" stripping) ──
    (r"Cyber Security", "CyberSec"),
    (r"Cyber Physical System", "CPS"),
    (r"Quantum Technologies?", "Quantum"),
    (r"VLSI and Embedded Systems?", "VLSIEmbedded"),
    (r"VLSI Design", "VLSI"),
    (r"Microelectronics and VLSI", "MicroVLSI"),
    (r"Signal Processing and Communication", "SPC"),
    (r"Design and Manufacturing", "D&M"),
    (r"Advanced Manufacturing", "Adv.Mfg"),
    (r"Product Design", "ProdDesign"),
    (r"Power System", "PowerSys"),
    (r"Embedded Systems?", "Embedded"),
    (r"Wearable Electronics", "Wearable"),
    (r"Rail Engineering", "Rail"),
    (r"Transportation and Logistics", "TransLog"),
    (r"Nano Science", "NanoSci"),
    (r"Internet of Things", "IoT"),
    (r"AI and ML", "AI+ML"),
    (r"AI and Robotics", "AIR"),
    (r"Communication Systems?", "CommSys"),
    (r"Systems? Design", "SD"),
    (r"Construction Technology and Management", "CTM"),
    (r"Machine Learning", "ML"),
    # ── Other ──────────────────────────────────────────────────────────────────
    (r"Environmental Science and Engineering", "EnvSciE"),
    (r"Environmental Engineering", "EnvE"),
    (r"Textile Technology", "TT"),
    (r"Textile Engineering", "TE"),
    (r"Carpet and Textile Technology", "CTT"),
    (r"Fashion and Apparel Engineering", "FashionE"),
    (r"Handloom and Textile Technology", "HTT"),
    (r"Printing and Packaging Technology", "PPT"),
    (r"Industrial Chemistry", "IndChem"),
    (r"Industrial Design", "IndDesign"),
    (r"Design Engineering", "DE"),
    (r"Bachelor of Design", "B.Des"),
    (r"Design", "Design"),
    (r"Economics", "Econ"),
    (r"Earth Sciences?", "ES"),
    (r"Life Science", "LS"),
    (r"Physical Science", "PS"),
    (r"Animation and VFX", "AVFX"),
    (r"Smart Manufacturing", "SM"),
    (r"Digital Agriculture", "DA"),
]

# Search-query aliases: what user types → what to search in Program + raw name.
# Used when the typed code isn't a substring of either the Program column or the raw name.
_PROG_FILTER_ALIASES = {
    # Branch shortcuts
    "OE":     "OE",            # Ocean Engineering
    "NA":     "NA",            # Naval Architecture
    "NAOE":   r"NAOE|OENA",    # both orderings of Naval Arch + Ocean Engg
    "OENA":   r"NAOE|OENA",
    "NE":     "NucE",          # Nuclear Engineering
    "NUC":    "Nuclear",
    "BME":    "BME",           # Biomedical Engineering
    "METE":   "MetE",          # Metallurgical Engineering
    "MET":    "Metallurg",
    "MINE":   "MineE",         # Mining Engineering
    "PETRO":  "PetroE",        # Petroleum Engineering
    "PETE":   "Petroleum",
    "AE":     "AE",            # Aerospace Engineering
    "AERO":   "Aero",          # Aerospace or Aeronautical
    "MECHAT": "MechT",         # Mechatronics
    "IE":     "IE",            # Industrial Engineering
    "PIE":    "PIE",           # Production & Industrial Engineering
    # Degree shortcuts
    "DUAL":       r"\+|Integrated|Dual",   # dual = integrated (both are 5Y combined degrees)
    "INT":        r"\+|Integrated|Dual",
    "INTEGRATED": r"\+|Integrated|Dual",
    "BTECH":  "B.Tech",
    "MTECH":  "M.Tech",
    "BARCH":  "B.Arch",
    # Subject shortcuts
    "MECH":   "Mechanical",    # ME in Program, but user may type MECH
    "CHEM":   "Chemical",      # ChE in Program, but user may type CHEM
    "ELEC":   "Electrical",
    "COMP":   "Computer",
    "ENV":    "Environmental",
    "GEO":    "Geo",           # Geology, Geophysics, Geotechnical
    "BIO":    "Bio",           # Bio-related programs
    "ML":     "Machine Learning",
    "IOT":    "Internet of Things",
    "VLSI":   "VLSI",
    "MFG":    "Manufacturing",
    "PHARMA": "Pharmaceut",
    "FOOD":   "Food",
    "TEXT":   "Textile",
    "SPACE":  "Space",
    "DS":     "DS",
    "AI":     "AI",
    "IT":     "IT",
    "CE":     r"\bCE\b|CompE",  # Civil Engineering AND Computer Engineering
    "CSE":    "CSE",
    "ECE":    "ECE",
    "EEE":    "EEE",
}

# Institute filter aliases: typed code → substring to search in raw Institute column.
_INST_FILTER_ALIASES = {
    # Institute types
    "NIT":    "National Institute of Technology",
    "IIT":    "Indian Institute of Technology",
    "IIIT":   r"Indian Institute of Information Technology|International Institute of Information Technology",
    "IIEST":  "Indian Institute of Engineering Science",
    "IISER":  "Indian Institute of Science Education",
    "IISC":   "Indian Institute of Science",
    # Specific IITs
    "IITB":   "Indian Institute of Technology Bombay",
    "IITD":   "Indian Institute of Technology Delhi",
    "IITM":   "Indian Institute of Technology Madras",
    "IITK":   "Indian Institute of Technology Kanpur",
    "IITKGP": "Indian Institute of Technology Kharagpur",
    "IITG":   "Indian Institute of Technology Guwahati",
    "IITR":   "Indian Institute of Technology Roorkee",
    "IITH":   "Indian Institute of Technology Hyderabad",
    "IITBHU": "Indian Institute of Technology (BHU)",
    "IITISM": "Indian Institute of Technology (ISM)",
    "ISM":    "Indian Institute of Technology (ISM)",
    # Specific NITs
    "NITK":   "National Institute of Technology Karnataka",
    "NITC":   "National Institute of Technology Calicut",
    "NITT":   "National Institute of Technology, Tiruchirappalli",
    "NITR":   "National Institute of Technology, Rourkela",
    "NITW":   "National Institute of Technology, Warangal",
    "NITJ":   "National Institute of Technology, Jamshedpur",
    "VNIT":   "Visvesvaraya National Institute of Technology",
    "SVNIT":  "Sardar Vallabhbhai National Institute",
    "MNIT":   "Malaviya National Institute of Technology",
    "MANIT":  "Maulana Azad National Institute",
    "MNNIT":  "Motilal Nehru National Institute",
    "NITIE":  "National Institute of Industrial Engineering",
    # Specific IIITs
    "IIITA":  "Indian Institute of Information Technology, Allahabad",
    "IIITDM": r"Design.*Manufacturing|Design & Manufacturing",
    "ABV":    "Atal Bihari Vajpayee",
    # GFTIs
    "BIT":    "Birla Institute of Technology",
    "SPA":    "School of Planning",
    "PEC":    "Punjab Engineering College",
    "SLIET":  "Sant Longowal",
    "NERIST": "North Eastern Regional Institute",
    "NIFFT":  "National Institute of Foundry",
    "NIFTEM": "National Institute of Food Technology",
}

_DEGREE_PATTERNS = [
    (r"(\d+)\s+Years?.*B\.?\s*Tech\.?\s*/\s*B\.?\s*Tech\.?\s*\(Hons",
     lambda m: f"{m.group(1)}Y B.Tech (Hons.)"),
    (r"(\d+)\s+Years?.*Bachelor and Master of Technology",
     lambda m: f"{m.group(1)}Y B.Tech+M.Tech"),
    (r"(\d+)\s+Years?.*Bachelor and Master of Pharmaceutics",
     lambda m: f"{m.group(1)}Y B.Pharm+M.Pharm"),
    (r"(\d+)\s+Years?.*Bachelor of Science and Master of Science",
     lambda m: f"{m.group(1)}Y B.Sc+M.Sc"),
    (r"(\d+)\s+Years?.*Bachelor of Science and MBA",
     lambda m: f"{m.group(1)}Y B.Sc+MBA"),
    (r"(\d+)\s+Years?.*Bachelor of Technology and MBA",
     lambda m: f"{m.group(1)}Y B.Tech+MBA"),
    (r"(\d+)\s+Years?.*B\.Tech\.\s*\+\s*M\.Tech\./MS",
     lambda m: f"{m.group(1)}Y B.Tech+M.Tech/MS"),
    (r"(\d+)\s+Years?.*Integrated B\.?\s*Tech\.? and M\.?\s*Tech\.?/MBA",
     lambda m: f"{m.group(1)}Y B.Tech+M.Tech/MBA"),
    (r"(\d+)\s+Years?.*Integrated B\.?\s*Tech\.? and MBA",
     lambda m: f"{m.group(1)}Y B.Tech+MBA"),
    (r"(\d+)\s+Years?.*Integrated B\.?\s*Tech\.? and M\.?\s*Tech",
     lambda m: f"{m.group(1)}Y B.Tech+M.Tech"),
    (r"(\d+)\s+Years?.*Integrated Bachelor of Science.Master of Science",
     lambda m: f"{m.group(1)}Y B.Sc+M.Sc"),
    (r"(\d+)\s+Years?.*Integrated Masters? in Technology",
     lambda m: f"{m.group(1)}Y M.Tech (Int.)"),
    (r"(\d+)\s+Years?.*Integrated Master of Technology",
     lambda m: f"{m.group(1)}Y M.Tech (Int.)"),
    (r"(\d+)\s+Years?.*Integrated Master of Science",
     lambda m: f"{m.group(1)}Y M.Sc (Int.)"),
    (r"(\d+)\s+Years?.*Bachelor of Architecture",
     lambda m: f"{m.group(1)}Y B.Arch"),
    (r"(\d+)\s+Years?.*Bachelor of Technology",
     lambda m: f"{m.group(1)}Y B.Tech"),
    (r"(\d+)\s+Years?.*Bachelor of Science",
     lambda m: f"{m.group(1)}Y B.Sc"),
    (r"(\d+)\s+Years?.*Bachelor of Planning",
     lambda m: f"{m.group(1)}Y B.Plan"),
    (r"(\d+)\s+Years?.*Bachelor of Design",
     lambda m: f"{m.group(1)}Y B.Des"),
    (r"(\d+)\s+Years?.*Bachelor of Pharmaceutics",
     lambda m: f"{m.group(1)}Y B.Pharm"),
    (r"(\d+)\s+Years?.*Bachelor of Engineering",
     lambda m: f"{m.group(1)}Y B.E."),
]


def _abbreviate_branch(prog: str) -> str:
    """Abbreviate the branch name only, stripping the degree-type suffix."""
    branch = re.split(r"\s*\(\d+\s+Year", prog)[0].strip()

    # Fix known typos in source data
    branch = re.sub(r"\blntelligence\b", "Intelligence", branch)
    branch = re.sub(r"\bIntelligenece\b", "Intelligence", branch, flags=re.IGNORECASE)

    # Strip "B.Tech in / B. Tech. in" prefixes (Rail / Gati Shakti programmes)
    branch = re.sub(r"^B\.?\s*Tech\.?\s+in\s+", "", branch, flags=re.IGNORECASE)

    # "with specialization/minor/major in X [+ M.Tech...]" → "(X)"
    branch = re.sub(
        r"\s+with\s+(?:specialization|minor|major)\s+(?:in|of)\s+([^+]+?)(?:\s*\+.*)?$",
        r" (\1)",
        branch,
        flags=re.IGNORECASE,
    )
    # "(with Specialization of X)" parenthetical variant
    branch = re.sub(
        r"\s*\(with\s+(?:specialization|minor|major)\s+(?:in|of)\s+(.+?)\)",
        r" (\1)",
        branch,
        flags=re.IGNORECASE,
    )

    # "X and M.Tech (X) Spl. Y" → "X+M.Tech (Y)"  (IIIT Allahabad integrated programs)
    branch = re.sub(
        r"\s+and\s+M\.Tech\.?\s*\([^)]+\)\s*Spl\.?\s*(.+)?$",
        lambda m: f"+M.Tech ({m.group(1).strip()})" if m.group(1) else "+M.Tech",
        branch,
        flags=re.IGNORECASE,
    )

    branch = re.sub(r"\s+", " ", branch).strip()

    for pattern, replacement in _BRANCH_ABBRS:
        branch = re.sub(pattern, replacement, branch, flags=re.IGNORECASE)
    return branch


def _degree_abbr(prog: str) -> str:
    """Extract and abbreviate the degree-type label from a full program name."""
    for pattern, fmt in _DEGREE_PATTERNS:
        m = re.search(pattern, prog, re.IGNORECASE)
        if m:
            return fmt(m)
    return ""


def _display_program_name(prog: str) -> str:
    """Abbreviated branch + short degree label for table display and filtering."""
    branch = _abbreviate_branch(prog)
    degree = _degree_abbr(prog)
    result = f"{branch} ({degree})" if degree else branch
    if len(result) > 50:
        result = result[:47] + "…"
    return result


def _short_program_name(prog: str) -> str:
    """Abbreviated branch only, no degree suffix; used for compact chart labels."""
    branch = _abbreviate_branch(prog)
    if len(branch) > 28:
        branch = branch[:25] + "…"
    return branch


def _short_institute_name(inst: str) -> str:
    # Remove parenthetical clarifications but keep text that follows them
    # e.g. "IIT (BHU) Varanasi" → "IIT Varanasi", "IIIT (IIIT) Nagpur" → "IIIT Nagpur"
    inst = re.sub(r"\s*\([^)]*\)", "", inst).strip()
    replacements = [
        # ── Named NITs (must come before generic NIT pattern) ─────────────────
        (r"^Dr\.?\s*B\.?\s*R\.?\s*Ambedkar National Institute of Technology[,\s]+", "NIT "),
        (r"^Malaviya National Institute of Technology[,\s]+", "MNIT "),
        (r"^Maulana Azad National Institute of Technology[,\s]+", "MANIT "),
        (r"^Motilal Nehru National Institute of Technology[,\s]+", "MNNIT "),
        (r"^Sardar Vallabhbhai National Institute of Technology[,\s]+", "SVNIT "),
        (r"^Visvesvaraya National Institute of Technology[,\s]+", "VNIT "),
        # ── Standard patterns ─────────────────────────────────────────────────
        (r"^Indian Institute of Technology, Design & Manufacturing\b", "IIITDM"),
        (r"^Indian Institute of Information Technology, Design & Manufacturing\b", "IIITDM"),
        (r"^Indian Institute of Information Technology[,\s]+", "IIIT "),
        (r"^Indian institute of information technology[,\s]+", "IIIT "),
        (r"^INDIAN INSTITUTE OF INFORMATION TECHNOLOGY[,\s]+", "IIIT "),
        (r"^International Institute of Information Technology[,\s]+", "IIIT "),
        (r"^National Institute of Technology[,\s]+", "NIT "),
        (r"^Indian Institute of Technology[,\s]+", "IIT "),
        (r"^Indian Institute of Engineering Science and Technology\b", "IIEST"),
        (r"^Indian Institute of Science Education and Research\b", "IISER"),
        (r"^Indian Institute of Science\b", "IISc"),
    ]
    for pattern, replacement in replacements:
        inst = re.sub(pattern, replacement, inst, count=1)
    inst = re.sub(r"\bUniversity\b", "Univ.", inst)
    inst = re.sub(r"\bInstitute\b", "Inst.", inst)
    inst = re.sub(r"\s+,\s+|\s{2,}", " ", inst).strip()
    if len(inst) > 32:
        inst = inst[:29] + "…"
    return inst


_WELL_KNOWN_INST_RE = re.compile(
    r"^("
    r"Indian Institute of Technology|"
    r"Indian Institute of Information Technology|"
    r"International Institute of Information Technology|"
    r"National Institute of Technology|"
    r"Dr\.?\s*B\.?\s*R\.?\s*Ambedkar National Institute of Technology|"
    r"Malaviya National Institute of Technology|"
    r"Maulana Azad National Institute of Technology|"
    r"Motilal Nehru National Institute of Technology|"
    r"Sardar Vallabhbhai National Institute of Technology|"
    r"Visvesvaraya National Institute of Technology|"
    r"Indian Institute of Engineering Science and Technology|"
    r"Indian Institute of Science Education and Research|"
    r"Indian Institute of Science\b"
    r")",
    re.IGNORECASE,
)


def _display_institute_name(inst: str, abbr: str) -> str:
    """Use abbreviated name for well-known institutes; full name for all others."""
    clean = re.sub(r"\s*\([^)]*\)", "", inst).strip()
    return abbr if _WELL_KNOWN_INST_RE.match(clean) else inst


def _slot_label(row: pd.Series) -> str:
    inst = _short_institute_name(row["Institute"])
    prog = _display_program_name(row["Academic Program Name"])
    return f"{inst} · {prog}"


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


# ── URL query-param helpers ──────────────────────────────────────────────────
# Snapshot BEFORE the sidebar writes defaults. True only when the user
# navigated here with a previously-saved URL (refresh / shared link).
_HAD_URL_STATE = all(
    k in st.query_params for k in ("rank", "quota", "seat_type", "gender")
)


def _qp(key: str, default: str) -> str:
    val = st.query_params.get(key)
    return str(val) if val is not None else default


# Sidebar
with st.sidebar:
    st.title("College Predictor")
    st.caption(f"Predictions for **{PREDICT_YEAR}** counselling")
    st.markdown("---")

    source = st.radio(
        "Counselling source",
        ["JoSAA", "CSAB"],
        index=1 if _qp("source", "josaa") == "csab" else 0,
        horizontal=True,
        help=(
            "**JoSAA** (Joint Seat Allocation Authority): The main annual counselling "
            "process for IITs, NITs, IIITs, and GFTIs. Open to both JEE Mains and "
            "JEE Advanced rank holders.\n\n"
            "**CSAB** (Central Seat Allocation Board): A supplementary round held after "
            "JoSAA to fill vacant seats in NITs, IIITs, and GFTIs. Only for JEE Mains "
            "candidates. Choose this if you missed JoSAA or want to try for remaining seats."
        ),
    ).lower()
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
            index=1 if _qp("exam", "mains") == "advanced" else 0,
            help=(
                "**JEE Mains → NIT / IIIT / GFTI**: Use your JEE Mains rank. "
                "Covers National Institutes of Technology (NITs), Indian Institutes of "
                "Information Technology (IIITs), and Government Funded Technical Institutes (GFTIs).\n\n"
                "**JEE Advanced → IIT**: Use your JEE Advanced rank. "
                "Exclusively for Indian Institutes of Technology (IITs). "
                "You must have qualified JEE Advanced to use this option."
            ),
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

    try:
        _rank_default = max(1, min(1_000_000, int(_qp("rank", "10000"))))
    except ValueError:
        _rank_default = 10_000
    rank = st.number_input(
        "Your rank",
        min_value=1, max_value=1_000_000,
        value=_rank_default, step=100,
        help=rank_help,
    )

    if exam_type == "advanced":
        quota = "AI"
        st.info("IITs use the **AI** quota exclusively.")
    else:
        _quota_default = _qp("quota", QUOTAS[source][0])
        _quota_idx = QUOTAS[source].index(_quota_default) if _quota_default in QUOTAS[source] else 0
        quota = st.selectbox(
            "Quota",
            QUOTAS[source],
            index=_quota_idx,
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
    _seat_default = _qp("seat_type", "OPEN")
    _seat_idx = SEAT_TYPES.index(_seat_default) if _seat_default in SEAT_TYPES else 0
    seat_type = st.selectbox(
        "Seat Type",
        SEAT_TYPES,
        index=_seat_idx,
        help=(
            "Select the category that matches what is printed on your JEE rank card.\n\n"
            "**OPEN**: General category, i.e. no reservation. Any candidate can compete.\n\n"
            "**EWS**: Economically Weaker Section; for candidates with annual family "
            "income below ₹8 lakh and no other reservation benefit.\n\n"
            "**OBC-NCL**: Other Backward Classes (Non-Creamy Layer); for OBC candidates "
            "whose family income is below the creamy layer limit (₹8 lakh/year).\n\n"
            "**SC**: Scheduled Caste reservation.\n\n"
            "**ST**: Scheduled Tribe reservation.\n\n"
            "**(PwD)** variants: Person with Disability sub-quota within each category. "
            "Select only if you have a valid PwD certificate from a recognised authority."
        ),
    )

    gender_raw = st.radio(
        "Gender",
        ["Gender-Neutral", "Female-only"],
        index=1 if _qp("gender", "GN") == "FO" else 0,
        help=(
            "**Gender-Neutral**: Seats open to candidates of all genders. "
            "The vast majority of seats fall in this category.\n\n"
            "**Female-only**: Supernumerary seats reserved exclusively for female candidates. "
            "These are *extra* seats created to improve female enrolment. They do not reduce "
            "seats available to others. If you are female, check both options separately "
            "to see your full range of choices."
        ),
    )
    gender = (
        "Female-only (including Supernumerary)"
        if gender_raw == "Female-only"
        else "Gender-Neutral"
    )

    coverage = 0.95

    st.markdown("---")
    predict_btn = st.button("Predict", width="stretch", type="primary")

    current_inputs = (source, exam_type, rank, quota, seat_type, gender)
    if st.session_state.get("last_inputs") != current_inputs:
        st.session_state.pop("results_df", None)
        st.session_state.pop("last_rank", None)

    # Sync current inputs to URL (survives refresh; makes URL shareable)
    st.query_params.update({
        "source":    source,
        "exam":      exam_type,
        "rank":      str(rank),
        "quota":     quota,
        "seat_type": seat_type,
        "gender":    "FO" if gender_raw == "Female-only" else "GN",
    })

# Auto-predict on fresh load when URL already has saved inputs
_auto_predict = _HAD_URL_STATE and "results_df" not in st.session_state


# Main area
st.title("JoSAA / CSAB Closing Rank Predictor")

if cfg.get("disclaimer"):
    st.warning(cfg["disclaimer"])

if predict_btn or _auto_predict:
    model, model_path = load_model_cached(source)
    if model is None:
        st.error(
            f"**Model and data not found.**\n\n"
            f"The app needs either:\n"
            f"- `models/{source}_model.pkl` (pre-trained, commit to the repo), **or**\n"
            f"- `{source}_ranks.csv` (raw data, app will auto-train on first load)\n\n"
            f"Run locally: `python scripts/predict_cli.py train --source {source}` "
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
            include_reach   = True,
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

df["Program"]   = df["Academic Program Name"].apply(_display_program_name)
df["InstAbbr"]  = df["Institute"].apply(_short_institute_name)
df["InstDisplay"] = df.apply(lambda r: _display_institute_name(r["Institute"], r["InstAbbr"]), axis=1)

# Summary metrics
counts = df["Category"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Safe", counts.get("safe", 0),
    help="Your rank is comfortably below the predicted closing rank. High likelihood of getting a seat here.",
)
c2.metric(
    "Match", counts.get("match", 0),
    help="Your rank is close to the predicted closing rank. Moderate chance. Worth applying but not guaranteed.",
)
c3.metric(
    "Reach", counts.get("reach", 0),
    help="Your rank is slightly above the predicted closing rank. Lower probability, but possible if the cutoff relaxes compared to last year.",
)
c4.metric(
    "Total", len(df),
    help="Total number of college–program combinations matching your profile across all categories.",
)

export_cols = [
    "Category", "Institute", "Academic Program Name", "Quota", "Seat Type", "Gender",
    *round_cols, "Final Pred", "Lower", "Upper", "Years", "Seats",
]
export_df = df[[c for c in export_cols if c in df.columns]].copy()
_cat_order = {"reach": 0, "match": 1, "safe": 2}
export_df = export_df.sort_values("Category", key=lambda s: s.map(_cat_order), kind="stable").reset_index(drop=True)
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
            type="primary",
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
            placeholder="e.g. CSE, OE, Dual, ChE, MnC",
            help="Supports short codes and full words. Click '📖 Abbr. guide' below to see all codes.",
        )

    with f2:
        inst_kw = st.text_input(
            "Filter by institute",
            key="filter_inst",
            placeholder="e.g. NIT Trichy, IIT Bombay, IIIT Hyderabad",
            help="Matches any part of the institute name.",
        )

    with st.popover("📖 Abbr. guide", use_container_width=False):
        g_prog, g_inst = st.tabs(["Programs", "Institutes"])

        with g_prog:
            st.caption("Short codes used in the **Program** column. Full names also work in the filter.")
            _seen_codes: dict[str, str] = {}
            for _pat, _code in _BRANCH_ABBRS:
                if _code not in _seen_codes:
                    _name = re.sub(r"\\.|\?|\.\*", "", _pat).strip()
                    _seen_codes[_code] = _name
            _guide_df = pd.DataFrame(
                sorted(_seen_codes.items(), key=lambda x: x[0]),
                columns=["Code", "Branch"],
            )
            st.dataframe(_guide_df, hide_index=True, use_container_width=True, height=300)

            st.caption("**Degree types** in the Program column:")
            st.markdown(
                "| Code | Meaning |\n|---|---|\n"
                "| `4Y B.Tech` | 4-year Bachelor of Technology |\n"
                "| `5Y B.Tech+M.Tech` | 5-year Dual Degree |\n"
                "| `5Y M.Tech (Int.)` | 5-year Integrated M.Tech |\n"
                "| `5Y M.Sc (Int.)` | 5-year Integrated M.Sc |\n"
                "| `5Y B.Arch` | 5-year Bachelor of Architecture |\n"
                "| `4Y B.Des` | 4-year Bachelor of Design |\n"
                "| `4Y B.Plan` | 4-year Bachelor of Planning |"
            )
            st.caption("**Search shortcuts** (not branch codes):")
            st.markdown(
                "| Type | Searches for |\n|---|---|\n"
                "| `DUAL` or `INT` | All dual / integrated programmes |\n"
                "| `BIO` | All Bio-related branches |\n"
                "| `GEO` | Geology / Geophysics |\n"
                "| `ENV` | Environmental Engineering |\n"
                "| `ML` | Machine Learning |\n"
                "| `IOT` | Internet of Things |\n"
                "| `VLSI` | VLSI-related programmes |\n"
                "| `MECH` | Mechanical (alias for ME) |\n"
                "| `CHEM` | Chemical (alias for ChE) |"
            )

        with g_inst:
            st.caption("Short codes used in the **Institute** column. Full names also work in the filter.")
            st.markdown(
                "**IITs**\n\n"
                "| Code | Institute |\n|---|---|\n"
                "| `IIT Bombay` | Indian Institute of Technology Bombay |\n"
                "| `IIT Delhi` | Indian Institute of Technology Delhi |\n"
                "| `IIT Madras` | Indian Institute of Technology Madras |\n"
                "| `IIT Kanpur` | Indian Institute of Technology Kanpur |\n"
                "| `IIT Kharagpur` | Indian Institute of Technology Kharagpur |\n"
                "| `IIT Roorkee` | Indian Institute of Technology Roorkee |\n"
                "| `IIT Guwahati` | Indian Institute of Technology Guwahati |\n"
                "| `IIT Hyderabad` | Indian Institute of Technology Hyderabad |\n"
                "| `IIT Varanasi` | IIT (BHU) Varanasi |\n"
                "| `IIT Dhanbad` | IIT (ISM) Dhanbad |\n\n"
                "**Named NITs**\n\n"
                "| Code | Institute |\n|---|---|\n"
                "| `MNIT Jaipur` | Malaviya NIT Jaipur |\n"
                "| `MANIT Bhopal` | Maulana Azad NIT Bhopal |\n"
                "| `MNNIT Allahabad` | Motilal Nehru NIT Allahabad |\n"
                "| `SVNIT Surat` | Sardar Vallabhbhai NIT Surat |\n"
                "| `VNIT Nagpur` | Visvesvaraya NIT Nagpur |\n"
                "| `NIT Trichy` | NIT Tiruchirappalli |\n"
                "| `NIT Surathkal` | NIT Karnataka, Surathkal |\n\n"
                "**Institute-type search shortcuts**\n\n"
                "| Type | Finds |\n|---|---|\n"
                "| `NIT` | All NITs |\n"
                "| `IIT` | All IITs (not IIITs) |\n"
                "| `IIIT` | All IIITs (both Indian & International) |\n"
                "| `IIEST` | IIE Science & Technology, Shibpur |\n"
                "| `BIT` | Birla Inst. of Technology |\n"
                "| `SPA` | School of Planning & Architecture |\n"
                "| `PEC` | Punjab Engineering College |\n"
                "| `SLIET` | Sant Longowal Inst. of Engg. |\n"
                "| `NIFFT` | Natl. Inst. of Foundry & Forge Tech. |"
            )

    table_df = df.copy()
    if prog_kw:
        search_kw = _PROG_FILTER_ALIASES.get(prog_kw.strip().upper(), prog_kw)
        table_df = table_df[
            table_df["Program"].str.contains(search_kw, case=False, na=False)
            | table_df["Academic Program Name"].str.contains(search_kw, case=False, na=False)
        ]
    if inst_kw:
        inst_search = _INST_FILTER_ALIASES.get(inst_kw.strip().upper(), inst_kw)
        table_df = table_df[
            table_df["Institute"].str.contains(inst_search, case=False, na=False)
            | table_df["InstAbbr"].str.contains(inst_kw, case=False, na=False)
        ]

    if table_df.empty:
        st.info("No rows match the selected program/institute filter.")

    has_seats     = "Seats" in df.columns and df["Seats"].notna().any()
    has_intervals = "Lower" in df.columns and "Upper" in df.columns
    display_cols = (
        ["InstDisplay", "Program"]
        + round_cols
        + ["Final Pred"]
        + (["Lower", "Upper"] if has_intervals else [])
        + ["Years"]
        + (["Seats"] if has_seats else [])
    )
    cov_pct = 95
    col_cfg = {
        "InstDisplay": st.column_config.TextColumn(
            "Institute",
            width="large",
        ),
        "Program": st.column_config.TextColumn(
            "Program",
            help="Abbreviated branch name and degree type. "
                 "e.g. 'CSE (4Y B.Tech)' = Computer Science & Engineering, 4-year B.Tech.",
        ),
        "Final Pred": st.column_config.NumberColumn(
            "Final Pred",
            format="%d",
            help="The model's predicted closing rank for the final counselling round. "
                 "Your rank must be below (numerically smaller than) this number for a realistic chance.",
        ),
        "Lower":      st.column_config.NumberColumn(
                          f"Lower ({cov_pct}%)",
                          format="%d",
                          help=f"Lower bound of the {cov_pct}% prediction interval. "
                               "If your rank is below this number, the seat is categorised as Safe. "
                               "The college is very likely to remain within reach."),
        "Upper":      st.column_config.NumberColumn(
                          f"Upper ({cov_pct}%)",
                          format="%d",
                          help=f"Upper bound of the {cov_pct}% prediction interval. "
                               "If your rank is above this number, the seat is out of reach. "
                               "The closing rank is unlikely to rise that high."),
        "Years":      st.column_config.NumberColumn(
            "Yrs",
            help="Number of years of historical closing-rank data used to train the prediction for this slot. "
                 "More years = more reliable prediction.",
        ),
        "Seats":      st.column_config.NumberColumn(
            "Seats",
            help="Total seats available in this college–program–quota–category combination for the current year.",
        ),
        **{r: st.column_config.NumberColumn(
            r, format="%d",
            help=f"Predicted closing rank for Round {r[1:]}. "
                 "Closing ranks typically tighten (get smaller) in later rounds as floating candidates fill seats.",
        ) for r in round_cols},
    }

    CAT_DESC = {
        "safe":  "Your rank is comfortably below the predicted closing rank. High likelihood of admission.",
        "match": "Your rank is close to the predicted closing rank. Moderate chance. Apply but have a backup.",
        "reach": "Your rank is slightly above the predicted closing rank. Lower probability, but possible if cutoffs relax.",
    }

    for cat in ["reach", "match", "safe"]:
        subset = table_df[table_df["Category"] == cat]
        if subset.empty:
            continue
        color = CAT_COLOR[cat]
        icon  = CAT_ICON[cat]
        st.markdown(
            f"<h4 style='color:{color};margin-bottom:2px'>"
            f"{icon} {cat.upper()} &nbsp; <small>({len(subset)} options)</small>"
            "</h4>"
            f"<p style='color:#666;font-size:0.85rem;margin-top:0;margin-bottom:6px'>{CAT_DESC[cat]}</p>",
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
