import numpy as np
import pandas as pd


def scores(y_log, p_log):
    """Metrics on log10 turbidity and on the NTU scale."""
    y_log, p_log = np.asarray(y_log, float), np.asarray(p_log, float)
    e = p_log - y_log
    y, p = 10 ** y_log, 10 ** p_log
    ss_tot = ((y_log - y_log.mean()) ** 2).sum()
    return {
        "rmse_log": np.sqrt(np.mean(e ** 2)),
        "mae_log": np.mean(np.abs(e)),
        "r2_log": 1 - (e ** 2).sum() / ss_tot if ss_tot > 0 else np.nan,
        "rmse_ntu": np.sqrt(np.mean((p - y) ** 2)),
        "mape": 100 * np.mean(np.abs(p - y) / y),
    }


def removal(log_t, t0):
    return 100 * (1 - 10 ** np.asarray(log_t, float) / np.asarray(t0, float))


def summarise(g):
    """Scores for one model's predictions (columns: day, sed, t0_raw, log_turb, pred)."""
    s = scores(g["log_turb"], g["pred"])
    day_rmse = np.sqrt(g.assign(se=(g["pred"] - g["log_turb"]) ** 2).groupby("day")["se"].mean())
    g60 = g[g["sed"] == 60]
    s.update({
        "day_rmse_mean": day_rmse.mean(), "day_rmse_sd": day_rmse.std(),
        "rmse_log_60": np.sqrt(np.mean((g60["pred"] - g60["log_turb"]) ** 2)),
        "mae_removal60_pp": np.mean(np.abs(removal(g60["pred"], g60["t0_raw"]) - removal(g60["log_turb"], g60["t0_raw"]))),
    })
    return s, day_rmse


def non_monotone_share(g, tol=1e-9):
    """Share of runs whose predicted turbidity rises between consecutive settling times."""
    w = g.pivot_table(index="run", columns="sed", values="pred")
    return float((w.diff(axis=1).iloc[:, 1:] > tol).any(axis=1).mean())


def by_group(df, y="y", p="pred", group="day"):
    return pd.DataFrame({g: scores(d[y], d[p]) for g, d in df.groupby(group)}).T.rename_axis(group)
