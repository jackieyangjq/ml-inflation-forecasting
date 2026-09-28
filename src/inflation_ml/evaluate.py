"""Forecast accuracy: RMSE relative to the AR benchmark, Diebold-Mariano tests, model confidence sets.

Forecasts are grouped by the month their target period ends in. The 2001-2015 period is the
second out-of-sample period of Medeiros et al. (2021); 2016-2019 and 2020 onward are new.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .models import BENCHMARK, MODELS, label

PERIODS = {
    "2001-2026": ("2001-01", "2100-12"),
    "2001-2015": ("2001-01", "2015-12"),
    "2016-2019": ("2016-01", "2019-12"),
    "2020-2026": ("2020-01", "2100-12"),
}


def load_forecasts(results: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((results / "forecasts").glob("*_h*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no forecasts in {results / 'forecasts'}; run `inflation-ml backtest` first")
    return pd.concat(frames, ignore_index=True)


def in_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = PERIODS[period]
    return df[(df["target_end"] >= start) & (df["target_end"] <= end) & df["actual"].notna()]


def error_matrix(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forecast errors (actual - forecast), one column per model, one row per origin."""
    sub = df[df["horizon"] == h]
    return sub.pivot(index="origin", columns="model", values="actual") - sub.pivot(
        index="origin", columns="model", values="forecast"
    )


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal squared error with the Harvey, Leybourne and Newbold (1997)
    small-sample correction. The long-run variance uses Newey-West weights over h - 1 lags, since
    h-step errors overlap. Returns the statistic and the one-sided p-value against the alternative
    that the model is more accurate than the benchmark."""
    d = e_model**2 - e_bench**2
    n = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for lag in range(1, h):
        lrv += 2 * (1 - lag / h) * (dc[lag:] @ dc[:-lag]) / n
    dm = d.mean() / np.sqrt(lrv / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(stats.t.cdf(dm, df=n - 1))


def model_confidence_set(errors: pd.DataFrame, h: int, size: float = 0.10, seed: int = 0) -> list[str]:
    """Hansen, Lunde and Nason (2011) model confidence set on squared errors, with a block
    bootstrap (blocks of max(h, 6) months): the models that cannot be told apart from the best
    at the `size` level."""
    from arch.bootstrap import MCS

    mcs = MCS(errors**2, size=size, reps=2000, block_size=max(h, 6), method="R", seed=seed)
    mcs.compute()
    return list(mcs.included)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per period, horizon and model: RMSE, its ratio to AR, the DM test, MCS membership."""
    rows = []
    for period in PERIODS:
        sub = in_period(df, period)
        for h in sorted(sub["horizon"].unique()):
            errors = error_matrix(sub, h).dropna()
            in_mcs = set(model_confidence_set(errors, h))
            bench_rmse = np.sqrt((errors[BENCHMARK] ** 2).mean())
            for name in [m for m in MODELS if m in errors]:
                rmse = np.sqrt((errors[name] ** 2).mean())
                dm, p = (np.nan, np.nan)
                if name != BENCHMARK:
                    dm, p = diebold_mariano(errors[name].to_numpy(), errors[BENCHMARK].to_numpy(), h)
                rows.append(
                    {
                        "period": period,
                        "horizon": h,
                        "model": name,
                        "n": len(errors),
                        "rmse": rmse,
                        "rmse_ratio": rmse / bench_rmse,
                        "dm_stat": dm,
                        "dm_pvalue": p,
                        "in_mcs": name in in_mcs,
                    }
                )
    return pd.DataFrame(rows)


def ratio_markdown(table: pd.DataFrame, period: str) -> str:
    """Markdown table of RMSE ratios to AR for one period: bold = lowest in its column,
    * and ** = DM p-value below 0.10 and 0.05, dagger = in the 90% model confidence set."""
    sub = table[table["period"] == period]
    horizons = sorted(sub["horizon"].unique())
    lines = ["| Model | " + " | ".join(f"h = {h}" for h in horizons) + " |", "|---|" + "---:|" * len(horizons)]
    best = sub.loc[sub.groupby("horizon")["rmse_ratio"].idxmin()].set_index("horizon")["model"]
    for name in [m for m in MODELS if m in set(sub["model"])]:
        cells = []
        for h in horizons:
            match = sub[(sub["model"] == name) & (sub["horizon"] == h)]
            if match.empty:
                cells.append("–")
                continue
            r = match.iloc[0]
            text = f"{r.rmse_ratio:.2f}"
            if r.dm_pvalue < 0.05:
                text += "\\*\\*"
            elif r.dm_pvalue < 0.10:
                text += "\\*"
            if r.in_mcs:
                text += " †"
            if best[h] == name:
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {label(name)} | " + " | ".join(cells) + " |")
    n = sub.groupby("horizon")["n"].first()
    ar = sub[sub["model"] == BENCHMARK].set_index("horizon")["rmse"]
    counts = ", ".join(f"h={h}: {n[h]}" for h in horizons)
    ar_rmse = ", ".join(f"h={h}: {ar[h]:.3f}" for h in horizons)
    lines += ["", f"Forecasts per horizon: {counts}. AR RMSE (percentage points): {ar_rmse}."]
    return "\n".join(lines)


def latest_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """The forecasts made at the last origin, whose outcome is not known yet, in percent."""
    last = df["origin"].max()
    sub = df[df["origin"] == last].pivot(index="model", columns="horizon", values="forecast")
    return sub.reindex([m for m in MODELS if m in sub.index])


def markdown(frame: pd.DataFrame, index_name: str, column_format: str = "{}") -> str:
    lines = [f"| {index_name} | " + " | ".join(column_format.format(c) for c in frame.columns) + " |"]
    lines.append("|---|" + "---:|" * len(frame.columns))
    lines += [f"| {i} | " + " | ".join(f"{v:.2f}" for v in row) + " |" for i, row in frame.iterrows()]
    return "\n".join(lines)


def write_tables(results: Path) -> pd.DataFrame:
    df = load_forecasts(results)
    table = accuracy_table(df)
    (results / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "tables" / "accuracy.csv", index=False, float_format="%.4f")
    parts = [f"## {period}\n\n{ratio_markdown(table, period)}\n" for period in PERIODS]
    latest = latest_forecasts(df)
    origin = df["origin"].max()
    parts.append(
        f"## Forecasts made in {origin}\n\nCPI inflation over the next h months, percent (not annualized).\n\n"
        + markdown(latest.rename(index=label), "Model", "h = {}")
        + "\n"
    )
    (results / "tables" / "summary.md").write_text("\n".join(parts))
    print((results / "tables" / "summary.md").read_text())
    return table
