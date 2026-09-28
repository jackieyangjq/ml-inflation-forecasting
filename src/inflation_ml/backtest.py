"""Rolling-window pseudo out-of-sample forecasts.

At each forecast origin T from December 2000 to the last month of data, every model is refitted
on the 480 months (40 years) that end at T and forecasts inflation over the next h months. Only
data up to T is used: the targets of the training rows are cut at T (so the last h rows of the
window have none) and the principal components come from the window alone.

The data are the latest vintage, revised, not what forecasters saw in real time; the paper does
the same.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from .data import Panel
from .features import base_name, cumulative_target, window_design
from .models import make_model


@dataclass(frozen=True)
class Config:
    window: int = 480
    n_lags: int = 4
    n_factors: int = 4
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    first_origin: str = "2000-12"


def forecast_one(
    name: str, X: pd.DataFrame, inflation: pd.Series, origin: pd.Timestamp, h: int, cfg: Config
) -> tuple[float, pd.Series | None]:
    """Fit `name` on the window ending at `origin`; return its forecast and, if it has them, the
    contributions of each series (summed over its lags) to that forecast."""
    end = X.index.get_loc(origin)
    if end + 1 < cfg.window:
        raise ValueError(f"only {end + 1} months up to {origin:%Y-%m}, the window needs {cfg.window}")
    D = window_design(X.iloc[end + 1 - cfg.window : end + 1], cfg.n_lags, cfg.n_factors)
    y = cumulative_target(inflation.loc[:origin], h).reindex(D.index)
    train = y.notna()
    model = make_model(name, h).fit(D[train], y[train])
    x_new = D.loc[[origin]]
    forecast = float(model.predict(x_new)[0])
    contrib = None
    if hasattr(model, "contributions"):
        contrib = model.contributions(x_new).iloc[0].groupby(base_name, sort=False).sum()
    return forecast, contrib


def run(models: list[str], panel: Panel, cfg: Config, out_dir: Path, n_jobs: int = -1, overwrite: bool = False) -> None:
    """Write results/forecasts/<model>_h<h>.csv (and contributions/) for every model and horizon.

    A file that already exists is kept unless `overwrite`, so an interrupted run picks up where it
    stopped.
    """
    origins = panel.X.loc[cfg.first_origin :].index
    (out_dir / "forecasts").mkdir(parents=True, exist_ok=True)
    (out_dir / "contributions").mkdir(parents=True, exist_ok=True)
    for h in cfg.horizons:
        actual = cumulative_target(panel.inflation, h)
        for name in models:
            path = out_dir / "forecasts" / f"{name}_h{h}.csv"
            if path.exists() and not overwrite:
                continue
            start = time.perf_counter()
            results = Parallel(n_jobs=n_jobs)(
                delayed(forecast_one)(name, panel.X, panel.inflation, T, h, cfg) for T in origins
            )
            out = pd.DataFrame(
                {
                    "origin": origins.strftime("%Y-%m"),
                    "target_end": (origins + pd.DateOffset(months=h)).strftime("%Y-%m"),
                    "horizon": h,
                    "model": name,
                    "forecast": [f for f, _ in results],
                    "actual": actual.reindex(origins).to_numpy(),
                }
            )
            out.to_csv(path, index=False, float_format="%.6f")
            contribs = [c for _, c in results]
            if all(c is not None for c in contribs):
                table = pd.DataFrame(contribs, index=origins.strftime("%Y-%m"))
                table.index.name = "origin"
                table.to_csv(out_dir / "contributions" / f"{name}_h{h}.csv", float_format="%.6f")
            print(f"{name:>9} h={h:<2} {len(origins)} origins in {time.perf_counter() - start:6.1f}s", flush=True)
