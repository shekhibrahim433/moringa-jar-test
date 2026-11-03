from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

RAW_FILE = DATA_RAW / "DataSet.xlsx"
SHEET = "Planilha1"

COLS = {
    "Sampling date (YYYY-MM-DD)": "date",
    "Mixing speed (rpm)": "speed",
    "Mixing time (s)": "mtime",
    "Coagulant concentration (mg/L)": "dose",
    "Sedimentation time (min)": "sed",
    "Trial number (-)": "trial",
    "Turbidity (NTU)": "turb",
}

KEYS = ["date", "speed", "mtime", "dose"]
SED_TIMES = [0, 10, 20, 30, 40, 50, 60]
TRIALS = [1, 2, 3]

SEED = 42
