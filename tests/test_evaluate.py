import numpy as np
import pandas as pd

from inflation_ml.evaluate import accuracy_table, diebold_mariano, ratio_markdown


def test_diebold_mariano_detects_a_better_model():
    rng = np.random.default_rng(0)
    bench = rng.normal(size=300)
    better = 0.6 * bench + 0.2 * rng.normal(size=300)
    stat, p = diebold_mariano(better, bench, h=1)
    assert stat < 0 and p < 0.01
    stat_rev, p_rev = diebold_mariano(bench, better, h=1)
    assert stat_rev == -stat and p_rev > 0.99


def test_diebold_mariano_with_overlapping_errors_is_less_confident():
    rng = np.random.default_rng(1)
    shocks = rng.normal(size=312)
    bench = np.convolve(shocks, np.ones(12), "valid")  # 12-month sums overlap like h = 12 errors
    model = 0.9 * bench + 0.3 * rng.normal(size=len(bench))
    assert diebold_mariano(model, bench, h=12)[1] > diebold_mariano(model, bench, h=1)[1]


def fake_forecasts(n=120):
    rng = np.random.default_rng(2)
    origins = pd.date_range("2005-01-01", periods=n, freq="MS")
    actual = rng.normal(size=n)
    rows = []
    for name, noise in [("ar", 1.0), ("rf", 0.5), ("lasso", 0.9)]:
        rows.append(
            pd.DataFrame(
                {
                    "origin": origins.strftime("%Y-%m"),
                    "target_end": (origins + pd.DateOffset(months=1)).strftime("%Y-%m"),
                    "horizon": 1,
                    "model": name,
                    "forecast": actual + noise * rng.normal(size=n),
                    "actual": actual,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_accuracy_table_and_markdown():
    table = accuracy_table(fake_forecasts())
    full = table[table["period"] == "2001-2026"].set_index("model")
    assert full.loc["ar", "rmse_ratio"] == 1.0
    assert full.loc["rf", "rmse_ratio"] < 0.7 and full.loc["rf", "in_mcs"]
    assert not full.loc["ar", "in_mcs"]
    text = ratio_markdown(table, "2001-2026")
    assert "| Random forest | **0.5" in text and "\\*\\* †**" in text
