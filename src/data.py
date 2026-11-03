import pandas as pd

from .config import RAW_FILE, SHEET, COLS, KEYS, SED_TIMES, TRIALS

BLOCK = len(SED_TIMES) * len(TRIALS)


def load_raw():
    df = pd.read_excel(RAW_FILE, sheet_name=SHEET).rename(columns=COLS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_ids(df):
    """Assign run (jar) and day ids. The file stores each run as a contiguous 21-row block."""
    df = df.reset_index(drop=True).copy()
    if len(df) % BLOCK:
        raise ValueError("row count is not a multiple of the run block size")
    df["run"] = df.index // BLOCK

    if (df.groupby("run")[KEYS].nunique() > 1).any().any():
        raise ValueError("settings change inside a run block")
    if not df.groupby(["run", "sed", "trial"]).size().eq(1).all():
        raise ValueError("run blocks are not complete sed x trial grids")

    days = sorted(df["date"].unique())
    df["day"] = df["date"].map({d: i for i, d in enumerate(days)})
    df["batch"] = df.groupby(["day", "speed", "mtime"]).ngroup()
    df["rep_run"] = df.groupby(KEYS)["run"].rank(method="dense").astype(int)

    t0 = df[df["sed"] == 0].groupby(["run", "trial"])["turb"].mean().rename("t0_trial")
    df = df.join(t0, on=["run", "trial"])
    df["t0"] = df.groupby("run")["t0_trial"].transform("mean")
    df["removal"] = 100 * (1 - df["turb"] / df["t0"])
    return df


def run_meta(df):
    return df.drop_duplicates("run").set_index("run")[["day", "batch", *KEYS, "rep_run"]]


def cell_stats(df):
    out = (df.groupby(["run", "sed"])["turb"]
             .agg(mean="mean", sd="std", n="size")
             .reset_index())
    out["cv"] = 100 * out["sd"] / out["mean"]
    return out.join(run_meta(df), on="run")


def run_table(df):
    wide = df.pivot_table(index="run", columns="sed", values="turb", aggfunc="mean")
    wide.columns = [f"T{c}" for c in wide.columns]
    out = run_meta(df).join(wide)
    for t in SED_TIMES[1:]:
        out[f"rem{t}"] = 100 * (1 - out[f"T{t}"] / out["T0"])
    return out.reset_index()
