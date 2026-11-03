import numpy as np
import optuna
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from . import kinetics as kin
from . import models as M
from .config import SEED

FORMS = ["san", "san_floor"]
LOG_PARAMS = {"tau", "b", "k"}
REGRESSORS = ["linear", "ridge_poly", "svr", "et", "mlp"]


def params_of(form):
    return kin.SPECS[form][1]


def to_targets(P, form):
    Z = np.array(P, float).copy()
    for j, name in enumerate(params_of(form)):
        if name in LOG_PARAMS:
            Z[:, j] = np.log(Z[:, j])
    return Z


def from_targets(Z, form):
    _, names, lo, hi = kin.SPECS[form]
    P = np.array(Z, float).copy()
    for j, name in enumerate(names):
        if name in LOG_PARAMS:
            P[:, j] = np.exp(P[:, j])
        P[:, j] = np.clip(P[:, j], lo[j], hi[j])
    return P


def log_shape(form, P, times):
    """log10(T/T0) for each jar (rows) at each time (columns)."""
    fun = kin.SPECS[form][0]
    t = np.asarray(times, float)[None, :]
    cols = [P[:, j][:, None] for j in range(P.shape[1])]
    return np.log10(fun(t, 1.0, *cols))


def make_reg(name, p, features):
    base = M.make(name, p, features)
    if name in ("svr", "hgb"):
        base = MultiOutputRegressor(base)
    return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())


def tune(name, form, X, Z, Y, groups, times, features, n_trials, k=4):
    if name not in M.TUNED:
        return {}
    cv = list(GroupKFold(k).split(X, groups=groups))

    def objective(trial):
        p = M.suggest(trial, name)
        err = []
        for tr, te in cv:
            reg = make_reg(name, p, features).fit(X[tr], Z[tr])
            pred = log_shape(form, from_targets(reg.predict(X[te]), form), times)
            err.append(np.sqrt(np.mean((pred - Y[te]) ** 2)))
        return float(np.mean(err))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def run_lodo(name, form, X, Z, Y, groups, times, features, n_trials):
    """Outer leave-one-day-out on jars; inner tuning scores reconstructed curves, not parameters."""
    P_hat = np.full((len(X), Z.shape[1]), np.nan)
    params = []
    for i, g in enumerate(np.unique(groups)):
        tr, te = groups != g, groups == g
        p = tune(name, form, X[tr], Z[tr], Y[tr], groups[tr], times, features, n_trials)
        reg = make_reg(name, p, features).fit(X[tr], Z[tr])
        P_hat[te] = from_targets(reg.predict(X[te]), form)
        params.append({"fold": i, **p})
    return P_hat, params
