from itertools import product

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

# f = T_inf / T0, so every model is written relative to the measured T0


def no_floor(t, T0, k):
    return T0 * np.exp(-k * t)


def first_order(t, T0, k, f):
    return T0 * (f + (1 - f) * np.exp(-k * t))


def first_order_lag(t, T0, k, f, lag):
    return T0 * (f + (1 - f) * np.exp(-k * np.clip(t - lag, 0, None)))


def stretched(t, T0, k, f, beta):
    return T0 * (f + (1 - f) * np.exp(-((k * t) ** beta)))


def second_order(t, T0, k, f):
    return T0 * (f + (1 - f) / (1 + k * t))


def san(t, T0, tau, b):
    """San (1989) Eq. 1 at one sampling depth: a*H^k becomes tau^b, removal P = 1 - T/T0."""
    return T0 / (1 + (t / tau) ** b)


def san_floor(t, T0, tau, b, f):
    return T0 * (f + (1 - f) / (1 + (t / tau) ** b))


SPECS = {
    "no_floor": (no_floor, ["k"], [1e-5], [1.0]),
    "first_order": (first_order, ["k", "f"], [1e-5, 0.0], [1.0, 0.999]),
    "first_order_lag": (first_order_lag, ["k", "f", "lag"], [1e-5, 0.0, 0.0], [1.0, 0.999, 20.0]),
    "stretched": (stretched, ["k", "f", "beta"], [1e-5, 0.0, 0.2], [1.0, 0.999, 4.0]),
    "second_order": (second_order, ["k", "f"], [1e-5, 0.0], [5.0, 0.999]),
    "san": (san, ["tau", "b"], [0.1, 0.1], [1e4, 5.0]),
    "san_floor": (san_floor, ["tau", "b", "f"], [0.1, 0.1, 0.0], [1e4, 5.0, 0.999]),
}

STARTS = {"k": [0.01, 0.05, 0.2], "f": [0.05, 0.3, 0.6], "lag": [0.0, 5.0, 9.0], "beta": [0.6, 1.0, 1.8],
          "tau": [5.0, 20.0, 60.0], "b": [0.5, 1.0, 2.0]}


def predict(model, t, T0, params):
    fun = SPECS[model][0]
    return fun(np.asarray(t, float), T0, *params)


def aicc(sse, n, p):
    return n * np.log(sse / n) + 2 * p + 2 * p * (p + 1) / (n - p - 1)


def pure_error(t, y):
    """Replicate (pure-error) sum of squares of log10 y around the mean at each time."""
    ly = np.log10(y)
    _, inv = np.unique(t, return_inverse=True)
    means = np.bincount(inv, ly) / np.bincount(inv)
    return ((ly - means[inv]) ** 2).sum(), len(ly) - inv.max() - 1


def fit(model, t, y, T0):
    """Least squares on log10 turbidity with T0 fixed, multi-start, bounded."""
    fun, names, lo, hi = SPECS[model]
    t, y = np.asarray(t, float), np.asarray(y, float)
    ly = np.log10(y)

    def resid(p):
        return np.log10(fun(t, T0, *p)) - ly

    eps = 1e-6
    best = None
    for x0 in product(*(STARTS[n] for n in names)):
        x0 = np.clip(x0, np.add(lo, eps), np.subtract(hi, eps))
        r = least_squares(resid, x0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r

    n, p = len(t), len(names)
    sse = 2 * best.cost
    cov = sse / (n - p) * np.linalg.pinv(best.jac.T @ best.jac)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    ss_pe, df_pe = pure_error(t, y)
    df_lof = len(np.unique(t)) - p
    ss_lof = max(sse - ss_pe, 0.0)
    lof_p = stats.f.sf((ss_lof / df_lof) / (ss_pe / df_pe), df_lof, df_pe) if df_lof > 0 and ss_pe > 0 else np.nan

    return {
        "params": dict(zip(names, best.x)),
        "se": dict(zip(names, se)),
        "at_bound": dict(zip(names, best.active_mask != 0)),
        "n": n, "p": p, "sse": sse, "rmse": np.sqrt(sse / n), "aicc": aicc(sse, n, p),
        "ss_pe": ss_pe, "df_pe": df_pe, "ss_lof": ss_lof, "df_lof": df_lof, "lof_p": lof_p,
    }
