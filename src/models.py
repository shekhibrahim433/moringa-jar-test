import numpy as np
import optuna
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from .config import SEED

FIXED = ["sed_mean", "linear", "gp"]
TUNED = ["ridge_poly", "svr", "rf", "et", "hgb", "mlp"]
ALL = FIXED + TUNED


class SedMean(BaseEstimator, RegressorMixin):
    """Baseline: mean target at each settling time, ignoring dose and water."""

    def __init__(self, sed_col=0):
        self.sed_col = sed_col

    def fit(self, X, y):
        s = X[:, self.sed_col]
        self.means_ = {v: y[s == v].mean() for v in np.unique(s)}
        return self

    def predict(self, X):
        return np.array([self.means_[v] for v in X[:, self.sed_col]])


def suggest(trial, name):
    if name == "ridge_poly":
        return {"degree": trial.suggest_int("degree", 1, 3),
                "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)}
    if name == "svr":
        return {"C": trial.suggest_float("C", 0.1, 100, log=True),
                "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 0.1, log=True)}
    if name in ("rf", "et"):
        return {"max_features": trial.suggest_float("max_features", 0.3, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_depth": trial.suggest_categorical("max_depth", [None, 4, 8, 16])}
    if name == "hgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 31),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 600)}
    if name == "mlp":
        return {"width": trial.suggest_int("width", 8, 64),
                "depth": trial.suggest_int("depth", 1, 3),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
    raise ValueError(name)


def make(name, p, features):
    if name == "sed_mean":
        return SedMean(features.index("sed"))
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "ridge_poly":
        return make_pipeline(StandardScaler(), PolynomialFeatures(p["degree"], include_bias=False),
                             StandardScaler(), Ridge(alpha=p["alpha"]))
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"]))
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, **p)
    if name == "et":
        return ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, **p)
    if name == "hgb":
        mono = [-1 if f == "sed" else 0 for f in features]
        return HistGradientBoostingRegressor(monotonic_cst=mono, random_state=SEED, **p)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(p["width"],) * p["depth"], alpha=p["alpha"],
            learning_rate_init=p["learning_rate_init"], max_iter=1000, early_stopping=True,
            random_state=SEED))
    if name == "gp":
        kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.ones(len(features)), (1e-2, 1e3))
                  + WhiteKernel(1e-2, (1e-6, 1.0)))
        return make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel, normalize_y=True, n_restarts_optimizer=1, random_state=SEED))
    raise ValueError(name)


def inner_folds(X, groups, split, k=4):
    if split == "lodo":
        return list(GroupKFold(k).split(X, groups=groups))
    return list(KFold(k, shuffle=True, random_state=SEED).split(X))


def tune(name, X, y, groups, split, features, n_trials):
    cv = inner_folds(X, groups, split)

    def objective(trial):
        p = suggest(trial, name)
        err = []
        for tr, te in cv:
            m = make(name, p, features).fit(X[tr], y[tr])
            err.append(np.sqrt(np.mean((m.predict(X[te]) - y[te]) ** 2)))
        return float(np.mean(err))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def run_cv(name, X, y, groups, folds, split, features, n_trials):
    """Outer CV; tuning sees only the training part of each outer fold."""
    pred, std, params = np.full(len(y), np.nan), np.full(len(y), np.nan), []
    for i, (tr, te) in enumerate(folds):
        p = tune(name, X[tr], y[tr], groups[tr], split, features, n_trials) if name in TUNED else {}
        m = make(name, p, features).fit(X[tr], y[tr])
        if name == "gp":
            pred[te], std[te] = m.predict(X[te], return_std=True)
        else:
            pred[te] = m.predict(X[te])
        params.append({"fold": i, **p})
    return pred, std, params
