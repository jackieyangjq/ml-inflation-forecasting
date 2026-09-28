"""Direct h-step targets and the design matrix of one estimation window.

Forecasts are direct: one model per horizon h maps what is known at month t to inflation over the
next h months, y_t = pi_{t+1} + ... + pi_{t+h}. The predictors, as in Medeiros et al. (2021), are
4 lags (t, t-1, t-2, t-3) of every series and of 4 principal components of the panel. The
components are estimated on the window alone, so nothing after the forecast origin enters them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGET

AUX_PREFIX = "aux_"  # columns only the benchmarks read: 12 lags of inflation and its 12-month mean
AR_MAX_LAG = 12


def cumulative_target(inflation: pd.Series, h: int) -> pd.Series:
    """Inflation over the h months after t, indexed by t (missing for the last h months)."""
    return sum(inflation.shift(-j) for j in range(1, h + 1)).rename(f"y_h{h}")


def add_lags(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """Columns <name>_L0 ... <name>_L{n_lags-1}; L0 is the value at t itself."""
    return pd.concat([df.shift(lag).add_suffix(f"_L{lag}") for lag in range(n_lags)], axis=1)


def base_name(feature: str) -> str:
    """'INDPRO_L2' -> 'INDPRO'."""
    return feature.rsplit("_L", 1)[0]


def principal_components(X: pd.DataFrame, k: int) -> pd.DataFrame:
    """First k principal components of the standardized panel, F1 ... Fk."""
    std = X.std().replace(0, 1)
    Z = ((X - X.mean()) / std).to_numpy()
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    scores = U[:, :k] * s[:k]
    return pd.DataFrame(scores, index=X.index, columns=[f"F{i + 1}" for i in range(k)])


def window_design(X: pd.DataFrame, n_lags: int = 4, n_factors: int = 4) -> pd.DataFrame:
    """Design matrix of one window; its first rows lack lags and are dropped.

    Every row t uses only data up to t, and the factors use only this window.
    """
    F = principal_components(X, n_factors)
    design = add_lags(pd.concat([X, F], axis=1), n_lags)
    pi = X[TARGET]
    aux = pd.concat([pi.shift(lag).rename(f"{AUX_PREFIX}pi_L{lag}") for lag in range(AR_MAX_LAG)], axis=1)
    aux[f"{AUX_PREFIX}ma12"] = pi.rolling(12).mean()
    return pd.concat([design, aux], axis=1).dropna()


def predictor_columns(design: pd.DataFrame) -> list[str]:
    """The columns the data-rich models use: everything but the benchmarks' helpers."""
    return [c for c in design.columns if not c.startswith(AUX_PREFIX)]
