"""Centralized configuration constants."""

DASHBOARD_PER_PAGE = 24

ARXIV_API_DELAY = 3
ARXIV_API_BATCH_SIZE = 50

DEFAULT_MAX_WORKERS = 8
DEFAULT_LLM_MODEL = "google/gemma-3-27b-it:free"

# ---------------------------------------------------------------------------
# Human-friendly arXiv category names
# ---------------------------------------------------------------------------
ARXIV_CATEGORY_NAMES: dict[str, str] = {
    # Computer Science
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation & Language",
    "cs.CR": "Cryptography & Security",
    "cs.CV": "Computer Vision",
    "cs.CY": "Computers & Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures & Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.NI": "Networking & Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social & Information Networks",
    "cs.SY": "Systems & Control",
    # Statistics
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning (Statistics)",
    "stat.OT": "Other Statistics",
    "stat.TH": "Statistics Theory",
    # Electrical Engineering & Systems Science
    "eess.AS": "Audio & Speech Processing",
    "eess.IV": "Image & Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems & Control",
    # Mathematics (commonly seen in ML/AI papers)
    "math.OC": "Optimization & Control",
    "math.ST": "Statistics Theory",
    "math.NA": "Numerical Analysis",
    "math.PR": "Probability",
    # Quantitative Biology
    "q-bio.BM": "Biomolecules",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons & Cognition",
    "q-bio.QM": "Quantitative Methods",
    # Quantitative Finance
    "q-fin.CP": "Computational Finance",
    "q-fin.ST": "Statistical Finance",
    # Physics
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis & Statistics",
}


def friendly_category_name(code: str) -> str:
    """Return human-friendly name for an arXiv category, falling back to the raw code."""
    return ARXIV_CATEGORY_NAMES.get(code, code)
