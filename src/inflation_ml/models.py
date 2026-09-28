"""The forecasting models: two benchmarks, a factor model, three penalized regressions, two tree ensembles.

Each model is built for one horizon h and fitted on one window. `fit` takes the full design of the
window (see features.window_design) and picks the columns it needs. Models that can say how much
each predictor moved a forecast implement `contributions`: SHAP values for the trees, coefficient
times standardized value for the linear models. Both add up to the forecast minus a baseline.

Tuning follows the replicated paper where it can: penalties chosen by BIC along the path, not by
cross-validation, and the paper's random forest and boosting settings.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import lars_path

from .data import TARGET
from .features import AR_MAX_LAG, AUX_PREFIX, predictor_columns


class Model(Protocol):
    def fit(self, D: pd.DataFrame, y: pd.Series) -> Model: ...
    def predict(self, D: pd.DataFrame) -> np.ndarray: ...


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Coefficients (constant first) and residual sum of squares."""
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return beta, float(((y - Xc @ beta) ** 2).sum())


def _bic(rss: np.ndarray | float, df: np.ndarray | float, n: int) -> np.ndarray | float:
    return n * np.log(rss / n) + df * np.log(n)


class AtkesonOhanian:
    """Random walk in the form of Atkeson and Ohanian (2001): the past 12 months' average, h times."""

    def __init__(self, h: int):
        self.h = h

    def fit(self, D, y):
        return self

    def predict(self, D):
        return self.h * D[f"{AUX_PREFIX}ma12"].to_numpy()


class AutoRegression:
    """Direct AR(p) on inflation lags, p from 1 to 12 chosen by BIC."""

    def fit(self, D, y):
        n = len(y)
        best = np.inf
        for p in range(1, AR_MAX_LAG + 1):
            cols = [f"{AUX_PREFIX}pi_L{lag}" for lag in range(p)]
            beta, rss = _ols(D[cols].to_numpy(), y.to_numpy())
            crit = _bic(rss, p + 1, n)
            if crit < best:
                best, self.cols_, self.beta_ = crit, cols, beta
        return self

    def predict(self, D):
        return self.beta_[0] + D[self.cols_].to_numpy() @ self.beta_[1:]


class DiffusionIndex:
    """Stock and Watson (2002): OLS on 4 lags of inflation and the 4 current principal components."""

    def fit(self, D, y):
        factors = [c for c in D if re.fullmatch(r"F\d+_L0", c)]
        self.cols_ = [f"{TARGET}_L{lag}" for lag in range(4)] + factors
        self.beta_, _ = _ols(D[self.cols_].to_numpy(), y.to_numpy())
        return self

    def predict(self, D):
        return self.beta_[0] + D[self.cols_].to_numpy() @ self.beta_[1:]


class Standardized:
    """Base for the penalized regressions: standardize with the window's moments, centre y."""

    def _prepare(self, D, y):
        self.cols_ = predictor_columns(D)
        X = D[self.cols_].to_numpy()
        self.mean_, self.scale_ = X.mean(0), X.std(0)
        self.scale_[self.scale_ == 0] = 1
        self.y_mean_ = float(y.mean())
        return (X - self.mean_) / self.scale_, y.to_numpy() - self.y_mean_

    def _z(self, D):
        return (D[self.cols_].to_numpy() - self.mean_) / self.scale_

    def predict(self, D):
        return self.y_mean_ + self._z(D) @ self.coef_

    def contributions(self, D) -> pd.DataFrame:
        return pd.DataFrame(self._z(D) * self.coef_, index=D.index, columns=self.cols_)


def bic_lasso(Z: np.ndarray, yc: np.ndarray, max_df_share: float = 0.5) -> np.ndarray:
    """LASSO: the exact path by LARS, the penalty picked by BIC over its knots.

    Degrees of freedom are the non-zero coefficients (Zou, Hastie and Tibshirani, 2007). With more
    predictors than observations the residuals reach zero at the end of the path and BIC would
    favour it, so the path stops after `max_df_share` * n steps.
    """
    n = len(yc)
    _, _, coefs = lars_path(Z, yc, method="lasso", max_iter=int(max_df_share * n))
    rss = ((yc[:, None] - Z @ coefs) ** 2).sum(0)
    df = (coefs != 0).sum(0)
    return coefs[:, int(np.argmin(_bic(rss, df, n)))]


class Lasso(Standardized):
    def fit(self, D, y):
        self.coef_ = bic_lasso(*self._prepare(D, y))
        return self


class AdaptiveLasso(Standardized):
    """Zou (2006): a second LASSO with penalty weights 1 / (|b| + 1/sqrt(n)) from a first LASSO."""

    def fit(self, D, y):
        Z, yc = self._prepare(D, y)
        scale = np.abs(bic_lasso(Z, yc)) + 1 / np.sqrt(len(yc))
        self.coef_ = bic_lasso(Z * scale, yc) * scale
        return self


class Ridge(Standardized):
    """Ridge with the penalty chosen by BIC, degrees of freedom = trace of the hat matrix."""

    def fit(self, D, y):
        Z, yc = self._prepare(D, y)
        n = len(yc)
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        uy = U.T @ yc
        best, self.coef_ = np.inf, None
        for alpha in s[0] ** 2 * np.logspace(-6, 0, 100):
            shrink = s**2 / (s**2 + alpha)
            rss = float(((yc - U @ (shrink * uy)) ** 2).sum())
            df = shrink.sum()
            crit = _bic(rss, df, n) if df <= n / 2 else np.inf
            if crit < best:
                best, self.coef_ = crit, Vt.T @ (s / (s**2 + alpha) * uy)
        return self


class RandomForest:
    """The paper's setting: 500 trees, a third of the predictors tried at each split, leaves of 5."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, D, y):
        self.cols_ = predictor_columns(D)
        self.model_ = RandomForestRegressor(
            n_estimators=500, max_features=1 / 3, min_samples_leaf=5, n_jobs=1, random_state=self.seed
        ).fit(D[self.cols_].to_numpy(), y.to_numpy())
        return self

    def predict(self, D):
        return self.model_.predict(D[self.cols_].to_numpy())

    def contributions(self, D) -> pd.DataFrame:
        import shap

        values = shap.TreeExplainer(self.model_).shap_values(D[self.cols_].to_numpy(), check_additivity=False)
        return pd.DataFrame(values, index=D.index, columns=self.cols_)


class Boosting:
    """XGBoost with the paper's setting: 1000 rounds, learning rate 0.05, depth 4, 2/3 of columns per level."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, D, y):
        from xgboost import XGBRegressor

        self.cols_ = predictor_columns(D)
        self.model_ = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=4,
            colsample_bylevel=2 / 3,
            min_child_weight=len(y) / 200,
            tree_method="hist",
            n_jobs=1,
            random_state=self.seed,
        ).fit(D[self.cols_].to_numpy(), y.to_numpy())
        return self

    def predict(self, D):
        return self.model_.predict(D[self.cols_].to_numpy())

    def contributions(self, D) -> pd.DataFrame:
        from xgboost import DMatrix

        values = self.model_.get_booster().predict(DMatrix(D[self.cols_].to_numpy()), pred_contribs=True)
        return pd.DataFrame(values[:, :-1], index=D.index, columns=self.cols_)


MODELS = {
    "rw": ("Random walk (Atkeson-Ohanian)", lambda h: AtkesonOhanian(h)),
    "ar": ("AR, BIC lags", lambda h: AutoRegression()),
    "factor": ("Factor model", lambda h: DiffusionIndex()),
    "ridge": ("Ridge", lambda h: Ridge()),
    "lasso": ("LASSO", lambda h: Lasso()),
    "adalasso": ("Adaptive LASSO", lambda h: AdaptiveLasso()),
    "rf": ("Random forest", lambda h: RandomForest()),
    "xgb": ("XGBoost", lambda h: Boosting()),
}
BENCHMARK = "ar"


def make_model(name: str, h: int) -> Model:
    return MODELS[name][1](h)


def label(name: str) -> str:
    return MODELS[name][0]
