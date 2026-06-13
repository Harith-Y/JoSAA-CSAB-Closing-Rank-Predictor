"""Shared display helpers: institute/program name abbreviation for all pages."""

import re

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
    (r"Microelectronics.*VLSI(?:\s+Systems?)?", "MVLSI"),
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
    (r"Bio-Medical Engineering", "BME"),
    (r"Biomedical Engineering", "BME"),
    (r"Bio Medical Engineering", "BME"),
    (r"Bio Engineering", "BioE"),
    (r"Biotechnology and Biochemical Engineering", "BT+BioChmE"),
    (r"Biotechnology and Bioinformatics", "BT+BioInf"),
    (r"Bio-?Informatics", "BioInf"),
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
    # ── Specialisation terms ───────────────────────────────────────────────────
    (r"Cyber Security\s*(?:including\s+Block\s*Chain\s*Technology)?", "CyberSec"),
    (r"Block\s*Chain\s*Technology", "BlockChain"),
    (r"Cyber Physical System", "CPS"),
    (r"Quantum Technologies?", "Quantum"),
    (r"VLSI\s*(?:and|&)\s*Electronic\s+Systems?\s*Design", "VLSIElecSD"),
    (r"VLSI and Embedded Systems?", "VLSIEmbedded"),
    (r"VLSI Design", "VLSI"),
    (r"Microelectronics and VLSI", "MicroVLSI"),
    (r"Micro\s+Electronics", "MicroE"),
    (r"Signal Processing\s*(?:and|&)\s*Communication", "SPC"),
    (r"Design and Manufacturing", "D&M"),
    (r"\bD\s+and\s+M\b", "D&M"),
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
    (r"Wireless Communication Engineering", "WCE"),
    (r"Human Computer\s+Interaction\s+and\s+Gaming\s+Technology", "HCI"),
    (r"Human Computer\s+Interaction\b", "HCI"),
    (r"Cyber Law\s*(?:and|&)\s*Information Security", "CLIS"),
    (r"Intelligent System", "IntelSys"),
    (r"Software Engineering", "SoftE"),
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

_PROG_FILTER_ALIASES = {
    "OE":     "OE",
    "NA":     "NA",
    "NAOE":   r"NAOE|OENA",
    "OENA":   r"NAOE|OENA",
    "NE":     "NucE",
    "NUC":    "Nuclear",
    "BME":    "BME",
    "METE":   "MetE",
    "MET":    "Metallurg",
    "MINE":   "MineE",
    "PETRO":  "PetroE",
    "PETE":   "Petroleum",
    "AE":     "AE",
    "AERO":   "Aero",
    "MECHAT": "MechT",
    "IE":     "IE",
    "PIE":    "PIE",
    "DUAL":       r"\+|Integrated|Dual",
    "INT":        r"\+|Integrated|Dual",
    "INTEGRATED": r"\+|Integrated|Dual",
    "BTECH":  "B.Tech",
    "MTECH":  "M.Tech",
    "BARCH":  "B.Arch",
    "MECH":   "Mechanical",
    "CHEM":   "Chemical",
    "ELEC":   "Electrical",
    "COMP":   "Computer",
    "ENV":    "Environmental",
    "GEO":    "Geo",
    "BIO":    "Bio",
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
    "CE":     r"\bCE\b|CompE",
    "CSE":    "CSE",
    "ECE":    "ECE",
    "EEE":    "EEE",
}

_INST_FILTER_ALIASES = {
    "NIT":    "National Institute of Technology",
    "IIT":    "Indian Institute of Technology",
    "IIIT":   r"Indian Institute of Information Technology|International Institute of Information Technology",
    "IIEST":  "Indian Institute of Engineering Science",
    "IISER":  "Indian Institute of Science Education",
    "IISC":   "Indian Institute of Science",
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
    "IIITA":  "Indian Institute of Information Technology, Allahabad",
    "IIITDM": r"Design.*Manufacturing|Design & Manufacturing",
    "ABV":    "Atal Bihari Vajpayee",
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

_TRAILING_STATE_RE = re.compile(
    r",\s*(?:Andhra Pradesh|Andra Pradesh|Arunachal Pradesh|Assam|Bihar|"
    r"Chandigarh|Chhattisgarh|Goa|Gujarat|Gujrat|Haryana|Himachal Pradesh|"
    r"Jammu\s*(?:&|and)\s*Kashmir|Jharkhand|Karnataka|Kerala|Madhya Pradesh|"
    r"Maharashtra|Manipur|Meghalaya|Mizoram|Nagaland|Odisha|Orissa|Punjab|"
    r"Rajasthan|Sikkim|Tamil\s*Nadu|Tamil\s*Naidu|Tamilnadu|Telangana|Tripura|"
    r"Uttar Pradesh|Uttarakhand|West Bengal|Delhi|Puducherry|Pondicherry|Ladakh)"
    r"\s*$",
    re.IGNORECASE,
)

_WELL_KNOWN_INST_RE = re.compile(
    r"^("
    r"Indian Institute of Technology|"
    r"Indian Institute of Information Technology|"
    r"International Institute of Information Technology|"
    r"National Institute of Technology|"
    r"National Institute of Electronics and Information Technology|"
    r"Dr\.?\s*B\.?\s*R\.?\s*Ambedkar National Institute of Technology|"
    r"Malaviya National Institute of Technology|"
    r"Maulana Azad National Institute of Technology|"
    r"Motilal Nehru National Institute of Technology|"
    r"Sardar Vallabhbhai National Institute of Technology|"
    r"Visvesvaraya National Institute of Technology|"
    r"Indian Institute of Engineering Science and Technology|"
    r"Indian Institute of Science Education and Research|"
    r"Indian Institute of Science\b|"
    r"Atal Bihari Vajpayee Indian Institute of Information Technology|"
    r"Pt\.?\s*Dwarka Prasad Mishra Indian Institute of Information Technology|"
    r"Birla Institute of Technology|"
    r"Indian School of Mines|"
    r"Jawaharlal Nehru University|"
    r"Shri G\.?\s*S\.?\s*Institute of Technology|"
    r"Pondicherry Engineering College|"
    r"INDIAN INSTITUTE OF INFORMATION TECHNOLOGY SENAPATI"
    r")",
    re.IGNORECASE,
)


def _abbreviate_branch(prog: str) -> str:
    branch = re.split(r"\s*\(\d+\s+Year", prog)[0].strip()

    branch = re.sub(r"\blntelligence\b", "Intelligence", branch)
    branch = re.sub(r"\bIntelligenece\b", "Intelligence", branch, flags=re.IGNORECASE)
    branch = re.sub(r"\bIntellegent\b", "Intelligent", branch, flags=re.IGNORECASE)
    branch = re.sub(r"\blnteraction\b", "Interaction", branch)

    branch = re.sub(
        r"^Integrated\s+B\.?\s*Tech\.?\s*[-–]\s*M\.?\s*Tech\.?\s+in\s+",
        "", branch, flags=re.IGNORECASE,
    )
    branch = re.sub(
        r"^Integrated\s+B\.?\s*Tech\.?\s*\(([^)]+)\)\s+and\s+M\.?\s*Tech\b.*$",
        r"\1+M.Tech", branch, flags=re.IGNORECASE,
    )
    branch = re.sub(
        r"^Integrated\s+B\.?\s*Tech\.?\s*\(([^)]+)\)\s+and\s+MBA\b.*$",
        r"\1+MBA", branch, flags=re.IGNORECASE,
    )
    branch = re.sub(
        r"^B\.?\s*Tech\.?\s*\(([^)]+)\)\s*[-–]\s*MBA\b.*$",
        r"\1+MBA", branch, flags=re.IGNORECASE,
    )
    branch = re.sub(r"^B\.?\s*Tech\.?\s+in\s+", "", branch, flags=re.IGNORECASE)
    branch = re.sub(
        r"\s+with\s+(?:specialization|minor|major)\s+(?:in|of)\s+([^+]+?)(?:\s*\+.*)?$",
        r" (\1)", branch, flags=re.IGNORECASE,
    )
    branch = re.sub(
        r"\s*\(with\s+(?:specialization|minor|major)\s+(?:in|of)\s+(.+?)\)",
        r" (\1)", branch, flags=re.IGNORECASE,
    )
    branch = re.sub(
        r"\s+and\s+M\.Tech\.?\s*\([^)]+\)\s*Spl\.?\s*(.+)?$",
        lambda m: (
            f"+M.Tech ({m.group(1).strip().strip('()')})" if m.group(1) else "+M.Tech"
        ),
        branch, flags=re.IGNORECASE,
    )

    branch = re.sub(r"\s+", " ", branch).strip()
    for pattern, replacement in _BRANCH_ABBRS:
        branch = re.sub(pattern, replacement, branch, flags=re.IGNORECASE)
    branch = re.sub(r"\s+\)", ")", branch).strip()
    branch = re.sub(r"^B\.?\s*Tech\.?\s+", "", branch, flags=re.IGNORECASE)

    m_dual = re.match(
        r"^(.+?)\s+and\s+M\.?\s*Tech\.?\s+(?:in\s+)?(.+)$", branch, re.IGNORECASE,
    )
    if m_dual:
        btpart = m_dual.group(1).strip()
        mtpart = m_dual.group(2).strip()
        if mtpart.startswith(btpart):
            mtpart = mtpart[len(btpart):].strip()
            branch = f"{btpart}+M.Tech {mtpart}".strip() if mtpart else f"{btpart}+M.Tech"
        elif mtpart.startswith("(") and mtpart.endswith(")"):
            branch = f"{btpart}+M.Tech {mtpart}"
        else:
            branch = f"{btpart}+M.Tech ({mtpart})"
    else:
        branch = re.sub(
            r"\s*\+\s*M\.?\s*Tech\.?(?:\s*[-–]\s*|\s+)(.+)$",
            lambda m: (
                f"+M.Tech {m.group(1).strip()}"
                if m.group(1).strip().startswith("(")
                else f"+M.Tech ({m.group(1).strip()})"
            ),
            branch, flags=re.IGNORECASE,
        )

    return branch


def _degree_abbr(prog: str) -> str:
    for pattern, fmt in _DEGREE_PATTERNS:
        m = re.search(pattern, prog, re.IGNORECASE)
        if m:
            return fmt(m)
    return ""


def _display_program_name(prog: str) -> str:
    branch = _abbreviate_branch(prog)
    degree = _degree_abbr(prog)
    result = f"{branch} ({degree})" if degree else branch
    if len(result) > 50:
        result = result[:47] + "…"
    return result


def _short_program_name(prog: str) -> str:
    branch = _abbreviate_branch(prog)
    if len(branch) > 28:
        branch = branch[:25] + "…"
    return branch


def _short_institute_name(inst: str) -> str:
    inst = re.sub(r"\s*\([^)]*\)", "", inst).strip()
    replacements = [
        (r"^Atal Bihari Vajpayee\b.*", "ABV-IIIT Gwalior"),
        (r"^Pt\.?\s+Dwarka Prasad Mishra\b.*", "IIITDM Jabalpur"),
        (r"^Birla Institute of Technology[,\s]+Mesra\b.*", "BIT Mesra, Ranchi"),
        (r"^Birla Institute of Technology[,\s]+Patna\b.*", "BIT Patna Off-Campus"),
        (r"^Birla Institute of Technology[,\s]+Deoghar\b.*", "BIT Deoghar Off-Campus"),
        (r"^Birla Institute of Technology\b.*", "BIT Mesra"),
        (r"^Indian School of Mines\b.*", "ISM Dhanbad"),
        (r"^Jawaharlal Nehru University\b.*", "JNU Delhi"),
        (r"^Shri G\.?\s*S\.?\s*Institute of Technology\b.*", "SGSITS Indore"),
        (r"^Pondicherry Engineering College\b.*", "PEC Puducherry"),
        (r"^National Institute of Electronics and Information Technology[,\s]+", "NIELIT "),
        (r"^INDIAN INSTITUTE OF INFORMATION TECHNOLOGY SENAPATI\b.*", "IIIT Senapati"),
        (r"^Dr\.?\s*B\.?\s*R\.?\s*Ambedkar National Institute of Technology[,\s]+", "NIT "),
        (r"^Malaviya National Institute of Technology[,\s]+", "MNIT "),
        (r"^Maulana Azad National Institute of Technology[,\s]+", "MANIT "),
        (r"^Motilal Nehru National Institute of Technology[,\s]+", "MNNIT "),
        (r"^Sardar Vallabhbhai National Institute of Technology[,\s]+", "SVNIT "),
        (r"^Visvesvaraya National Institute of Technology[,\s]+", "VNIT "),
        (r"^Indian Institute of Technology,?\s*Design & Manufacturing\b", "IIITDM"),
        (r"^Indian Institute of Information Technology,?\s*Design & Manufacturing\b", "IIITDM"),
        (r"^Indian Institute of Information Technology[,\s]+", "IIIT "),
        (r"^Indian Institute of Information Technology(?=[A-Z])", "IIIT "),
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
    inst = _TRAILING_STATE_RE.sub("", inst).strip()
    inst = re.sub(r",?\s+District\s*$", "", inst, flags=re.IGNORECASE).strip()
    if len(inst) > 32:
        inst = inst[:29] + "…"
    return inst


def _display_institute_name(inst: str, abbr: str) -> str:
    clean = re.sub(r"\s*\([^)]*\)", "", inst).strip()
    return abbr if _WELL_KNOWN_INST_RE.match(clean) else inst


def _pat_to_name(pat: str) -> str:
    """Convert a regex pattern string to a human-readable branch name."""
    n = pat
    for _seq in (r"\s+", r"\s*", r"\s?", r"\s"):
        n = n.replace(_seq, " ")
    n = n.replace(r"\b", "")
    n = re.sub(r"\(\?:([^)]*)\)", lambda m: m.group(1).split("|")[0], n)
    n = re.sub(r"\[.+?\]", "", n)
    n = n.replace(".*", " ")
    for _ch in r"()?+*^${}|\\.":
        n = n.replace(_ch, "")
    return re.sub(r"\s+", " ", n).strip()


def abbr_guide_df():
    """Return a sorted DataFrame of branch Code → full name for the abbr guide popover."""
    import pandas as pd
    seen: dict[str, str] = {}
    for pat, code in _BRANCH_ABBRS:
        if code not in seen:
            seen[code] = _pat_to_name(pat)
    return pd.DataFrame(sorted(seen.items(), key=lambda x: x[0]), columns=["Code", "Branch"])
