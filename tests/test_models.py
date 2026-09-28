import numpy as np
import pandas as pd
import pytest

from inflation_ml.backtest import Config, forecast_one
from inflation_ml.features import add_lags, base_name, cumulative_target, window_design
from inflation_ml.models import MODELS, bic_lasso, make_model


def test_cumulative_target_sums_the_next_h_months():
    pi = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(cumulative_target(pi, 1), [2, 3, 4, 5, np.nan])
    np.testing.assert_allclose(cumulative_target(pi, 3), [9, 12, np.nan, np.nan, np.nan])


def test_add_lags_and_base_name():
    df = pd.DataFrame({"CUSR0000SA0L2": [1.0, 2.0, 3.0]})
    lagged = add_lags(df, 2)
    assert list(lagged.columns) == ["CUSR0000SA0L2_L0", "CUSR0000SA0L2_L1"]
    assert lagged["CUSR0000SA0L2_L1"].tolist()[1:] == [1.0, 2.0]
    assert base_name("CUSR0000SA0L2_L1") == "CUSR0000SA0L2"


def test_bic_lasso_finds_a_sparse_signal():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 400))
    y = 1.0 * Z[:, 3] - 0.8 * Z[:, 10] + 0.1 * rng.normal(size=300)
    coef = bic_lasso(Z, y - y.mean())
    assert set(np.flatnonzero(np.abs(coef) > 0.1)) == {3, 10}


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_fits_and_predicts(panel, name):
    D = window_design(panel.X, n_lags=4, n_factors=2)
    y = cumulative_target(panel.inflation, 3).reindex(D.index)
    train = y.notna()
    model = make_model(name, 3).fit(D[train], y[train])
    forecast = model.predict(D.iloc[[-1]])
    assert forecast.shape == (1,) and np.isfinite(forecast[0])


@pytest.mark.parametrize("name", ["lasso", "adalasso", "ridge", "xgb", "rf"])
def test_contributions_add_up_to_the_forecast_minus_a_baseline(panel, name):
    D = window_design(panel.X, n_lags=4, n_factors=2)
    y = cumulative_target(panel.inflation, 1).reindex(D.index)
    train = y.notna()
    model = make_model(name, 1).fit(D[train], y[train])
    x = D.iloc[-3:]
    baseline = model.predict(x) - model.contributions(x).sum(axis=1).to_numpy()
    np.testing.assert_allclose(baseline, baseline[0], atol=1e-3)


def test_the_lasso_picks_the_true_predictor(panel):
    D = window_design(panel.X, n_lags=2, n_factors=2)
    y = cumulative_target(panel.inflation, 1).reindex(D.index)
    train = y.notna()
    model = make_model("lasso", 1).fit(D[train], y[train])
    top = model.contributions(D.iloc[-40:]).abs().mean().nlargest(2).index
    assert set(top) == {"S0_L0", "CPIAUCSL_L0"}


def test_no_look_ahead(panel):
    """A forecast made at T must not change when the data after T change."""
    cfg = Config(window=120, n_lags=2, n_factors=2)
    origin = panel.X.index[150]
    before = forecast_one("lasso", panel.X, panel.inflation, origin, 6, cfg)[0]
    X = panel.X.copy()
    X.iloc[151:] = 1e3
    after = forecast_one("lasso", X, X["CPIAUCSL"], origin, 6, cfg)[0]
    assert before == after


def test_window_longer_than_the_data_is_refused(panel):
    with pytest.raises(ValueError, match="window needs"):
        forecast_one("ar", panel.X, panel.inflation, panel.X.index[50], 1, Config(window=120))
