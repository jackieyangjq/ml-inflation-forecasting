"""FRED-MD: download a vintage, apply the stationarity transformations, clean outliers.

FRED-MD (McCracken and Ng, 2016) is a monthly panel of about 126 US macro series kept by the
St. Louis Fed. The first data row of each vintage holds a transformation code per series:

    1 x    2 dx    3 d2x    4 log x    5 dlog x    6 d2log x    7 d(x_t / x_{t-1} - 1)

The target is CPI inflation, 100 * dlog CPIAUCSL, in percent per month. CPIAUCSL's own code is 6
(the change in inflation); here it is read with code 5 instead, so the inflation series itself
enters the predictors and its lags make up the autoregressive part of every model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TARGET = "CPIAUCSL"

# The vintage list on https://www.stlouisfed.org/research/economists/mccracken/fred-databases names
# files irregularly (a revised release is "2026-rev-08-md.csv"), so known vintages are listed here.
BASE_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/"
VINTAGE_FILES = {
    "2026-08": "2026-rev-08-md.csv",
    "2026-07": "2026-07-md.csv",
}
DEFAULT_VINTAGE = "2026-08"

GROUP_NAMES = {
    1: "Output and income",
    2: "Labor market",
    3: "Housing",
    4: "Consumption, orders and inventories",
    5: "Money and credit",
    6: "Interest and exchange rates",
    7: "Prices",
    8: "Stock market",
}


def series_table() -> pd.DataFrame:
    """Group, transformation code and description of each series, from the FRED-MD appendix."""
    text = resources.files("inflation_ml").joinpath("fredmd_series.csv").read_text()
    return pd.read_csv(io.StringIO(text)).set_index("fred")


def download(vintage: str = DEFAULT_VINTAGE, data_dir: Path = Path("data")) -> Path:
    """Save the vintage as data/fredmd-<vintage>.csv (skipped when the file is already there)."""
    path = data_dir / f"fredmd-{vintage}.csv"
    if path.exists():
        return path
    if vintage not in VINTAGE_FILES:
        raise ValueError(f"unknown vintage {vintage!r}; known: {', '.join(VINTAGE_FILES)}")
    resp = requests.get(BASE_URL + VINTAGE_FILES[vintage], timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    if not resp.text.startswith("sasdate"):
        raise RuntimeError(f"{resp.url} did not return a FRED-MD csv")
    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    return path


def read_raw(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Levels indexed by month start, and the transformation code of each column."""
    raw = pd.read_csv(path)
    tcodes = raw.iloc[0, 1:].astype(int)
    levels = raw.iloc[1:].dropna(subset=["sasdate"]).copy()
    levels.index = pd.to_datetime(levels.pop("sasdate"), format="%m/%d/%Y")
    levels.index.name = "date"
    levels = levels.dropna(how="all").astype(float)
    return levels, tcodes


def transform(x: pd.Series, tcode: int) -> pd.Series:
    """Apply one FRED-MD transformation code (logs are natural logs)."""
    if tcode in (4, 5, 6):
        x = np.log(x.where(x > 0))
    match tcode:
        case 1 | 4:
            return x
        case 2 | 5:
            return x.diff()
        case 3 | 6:
            return x.diff().diff()
        case 7:
            return x.pct_change(fill_method=None).diff()
    raise ValueError(f"unknown transformation code {tcode}")


def flag_outliers(df: pd.DataFrame, reference_end: str, k: float = 10.0) -> pd.DataFrame:
    """True where |x - median| > k * IQR, the rule McCracken and Ng use for FRED-MD.

    Median and IQR come from the data up to `reference_end` only, so that the cleaning of
    observations in the evaluation period does not use anything after them.
    """
    ref = df.loc[:reference_end]
    med = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    return (df - med).abs() > k * iqr


@dataclass
class Panel:
    """Transformed predictors and the inflation target on a common monthly index."""

    X: pd.DataFrame  # stationary, outlier-cleaned predictors; column TARGET is raw inflation
    inflation: pd.Series  # 100 * dlog CPI, percent per month
    dropped: list[str]  # series left out because they start after `start`
    interpolated: dict[str, list[str]]  # series -> months filled inside the sample
    n_outliers: int


def build_panel(path: Path, start: str = "1960-01", outlier_reference_end: str = "2000-12") -> Panel:
    """Transform every series, drop the ones with gaps, replace outliers by the last clean value.

    - A single month missing inside a series is filled log-linearly from its two neighbours, in
      levels, before transforming. In the 2026-08 vintage that is October 2025, when the federal
      shutdown stopped the CPI and the household survey (21 series), and April 2020 for two
      commercial paper rates. Longer gaps are not filled.
    - Series with any other missing value between `start` and the ragged edge are dropped (in the
      2026-08 vintage: ACOGNO, ANDENOx, TWEXAFEGSMTHx, UMCSENTx and VIXCLSx), leaving 121.
    - The panel ends in the last month with CPI; a series not yet released for the last month
      or two (the "ragged edge") carries its last value forward.
    - Outliers are set to missing and filled with the previous value, which looks only backwards.
      Inflation itself is not cleaned: it is what we forecast.
    """
    levels, tcodes = read_raw(path)
    tcodes[TARGET] = 5
    single = levels.isna() & levels.shift(1).notna() & levels.shift(-1).notna()
    interpolated = {c: [d.strftime("%Y-%m") for d in levels.index[single[c]]] for c in levels if single[c].any()}
    positive = (levels.gt(0) | levels.isna()).all()
    between = levels.ffill().add(levels.bfill()) / 2
    log_between = np.exp(np.log(levels.loc[:, positive]).ffill().add(np.log(levels.loc[:, positive]).bfill()) / 2)
    between.loc[:, positive] = log_between
    levels = levels.mask(single, between)

    X = pd.DataFrame({c: transform(levels[c], int(tcodes[c])) for c in levels.columns})
    X = X.loc[start:]
    X = X.loc[: X[TARGET].last_valid_index()]
    ragged_edge = 3
    has_gap = X.iloc[:-ragged_edge].isna().any()
    dropped = sorted(has_gap[has_gap].index)
    X = X.drop(columns=dropped)

    inflation = 100 * X[TARGET]
    outliers = flag_outliers(X.drop(columns=TARGET), outlier_reference_end)
    cleaned = X.drop(columns=TARGET).mask(outliers).ffill()
    X = pd.concat([inflation.rename(TARGET), cleaned], axis=1)
    interpolated = {c: months for c, months in interpolated.items() if c in X.columns}
    return Panel(
        X=X,
        inflation=inflation.rename("inflation"),
        dropped=dropped,
        interpolated=interpolated,
        n_outliers=int(outliers.sum().sum()),
    )
