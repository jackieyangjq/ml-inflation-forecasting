import numpy as np
import pandas as pd
import pytest

from inflation_ml.data import TARGET, build_panel, series_table, transform


def test_transform_codes():
    x = pd.Series([100.0, 110.0, 99.0, 120.0])
    assert transform(x, 1).equals(x)
    np.testing.assert_allclose(transform(x, 2).iloc[1:], [10, -11, 21])
    np.testing.assert_allclose(transform(x, 3).iloc[2:], [-21, 32])
    np.testing.assert_allclose(transform(x, 5).iloc[1:], np.diff(np.log(x)))
    np.testing.assert_allclose(transform(x, 6).iloc[2:], np.diff(np.log(x), 2))
    np.testing.assert_allclose(transform(x, 7).iloc[2], (99 / 110 - 1) - (110 / 100 - 1))
    with pytest.raises(ValueError):
        transform(x, 8)


def test_series_table_has_every_group():
    table = series_table()
    assert TARGET in table.index
    assert set(table["group"]) == set(range(1, 9))


def write_vintage(path, n=60):
    """A small file in the FRED-MD layout: a header, a transform-code row, then monthly levels."""
    index = pd.date_range("1959-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(1)
    cpi = 100 * np.exp(np.cumsum(0.003 + 0.001 * rng.normal(size=n)))
    ip = 50 * np.exp(np.cumsum(0.002 + 0.01 * rng.normal(size=n)))
    rate = 5 + np.cumsum(0.1 * rng.normal(size=n))
    late = np.where(np.arange(n) < 30, np.nan, 1.0 + np.arange(n))
    df = pd.DataFrame({"CPIAUCSL": cpi, "INDPRO": ip, "FEDFUNDS": rate, "LATE": late}, index=index)
    df.loc["1962-06-01", "INDPRO"] = np.nan  # a single missing month
    df.loc["1963-01-01", "FEDFUNDS"] += 500  # an outlier in the change
    rows = [["Transform:", 6, 5, 2, 5]]
    rows += [[d.strftime("%-m/%-d/%Y"), *r] for d, r in zip(df.index, df.to_numpy(), strict=True)]
    pd.DataFrame(rows, columns=["sasdate", *df.columns]).to_csv(path, index=False)
    return df


def test_build_panel(tmp_path):
    levels = write_vintage(tmp_path / "v.csv")
    panel = build_panel(tmp_path / "v.csv", start="1959-03", outlier_reference_end="1962-12")

    assert panel.dropped == ["LATE"]
    assert panel.interpolated == {"INDPRO": ["1962-06"]}
    # inflation is 100 * dlog CPI, not the change in inflation that CPI's own code 6 would give
    expected = 100 * np.log(levels["CPIAUCSL"]).diff().loc["1959-03":]
    np.testing.assert_allclose(panel.inflation, expected)
    # the interpolated month splits the two-month growth of INDPRO in half
    two_month = np.log(levels.loc["1962-07-01", "INDPRO"] / levels.loc["1962-05-01", "INDPRO"])
    assert panel.X.loc["1962-06-01", "INDPRO"] == pytest.approx(two_month / 2)
    # the jump into and out of the outlier month are replaced by the previous change
    assert panel.n_outliers == 2
    assert panel.X.loc["1963-01-01", "FEDFUNDS"] == panel.X.loc["1962-12-01", "FEDFUNDS"]
    assert not panel.X.isna().any().any()
