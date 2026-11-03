import numpy as np
import pandas as pd

BASE = ["dose", "log_t0_raw", "sed", "speed", "mtime"]
EXTRA = ["dose_per_t0"]
META = ["run", "day", "batch", "date", "rep_run"]


def raw_t0(df):
    """Raw-water turbidity proxy per run: median jar T0 within the batch (day x speed x mixing time)."""
    t0 = df[df["sed"] == 0].groupby("run")["turb"].mean()
    batch = df.drop_duplicates("run").set_index("run")["batch"].loc[t0.index]
    return t0.groupby(batch).transform("median")


def make_table(df, level="cell", sed_min=10):
    """One row per run x settling time (cell means) or per replicate (trial). Targets are log10 scale."""
    d = df[df["sed"] >= sed_min]
    if level == "cell":
        first = {c: (c, "first") for c in ["day", "batch", "date", "rep_run", "speed", "mtime", "dose"]}
        d = d.groupby(["run", "sed"], as_index=False).agg(turb=("turb", "mean"), **first)
    elif level != "trial":
        raise ValueError("level must be 'cell' or 'trial'")
    d = d.copy()
    d["t0_raw"] = d["run"].map(raw_t0(df))
    d["log_t0_raw"] = np.log10(d["t0_raw"])
    d["dose_per_t0"] = d["dose"] / d["t0_raw"]
    d["log_turb"] = np.log10(d["turb"])
    d["log_ratio"] = np.log10(d["turb"] / d["t0_raw"])
    return d.reset_index(drop=True)
