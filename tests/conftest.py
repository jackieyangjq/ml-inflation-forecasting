import numpy as np
import pandas as pd
import pytest

from inflation_ml.data import TARGET, Panel


def synthetic_panel(n_months: int = 200, n_series: int = 12, seed: int = 0) -> Panel:
    """Inflation driven by an AR(1) and the lagged first series, plus noise series."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("1990-01-01", periods=n_months, freq="MS")
    X = pd.DataFrame(rng.normal(size=(n_months, n_series)), index=index, columns=[f"S{i}" for i in range(n_series)])
    pi = np.zeros(n_months)
    for t in range(1, n_months):
        pi[t] = 0.2 + 0.5 * pi[t - 1] + 0.3 * X["S0"].iloc[t - 1] + 0.1 * rng.normal()
    X.insert(0, TARGET, pi)
    return Panel(X=X, inflation=X[TARGET].rename("inflation"), dropped=[], interpolated={}, n_outliers=0)


@pytest.fixture
def panel() -> Panel:
    return synthetic_panel()
