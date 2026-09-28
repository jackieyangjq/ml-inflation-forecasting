"""The README figures, drawn from results/ (no refitting)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .data import GROUP_NAMES, TARGET, series_table  # noqa: E402
from .evaluate import PERIODS, error_matrix, load_forecasts  # noqa: E402
from .models import BENCHMARK, MODELS, label  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
# A model keeps its colour in every figure.
MODEL_COLORS = {"rf": SERIES[0], BENCHMARK: SERIES[1], "adalasso": SERIES[2], "xgb": SERIES[3]}
DIVERGING = LinearSegmentedColormap.from_list("ratio", ["#2a78d6", "#f0efec", "#e34948"])

OWN_LAGS = "Own lags of CPI inflation"
FACTORS = "Principal components"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
    }
)


def _dates(values: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(values, format="%Y-%m")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def forecast_paths(df: pd.DataFrame, models: list[str], h: int, path: Path) -> None:
    """Realized inflation over the next h months against a few models' forecasts, by target month."""
    sub = df[df["horizon"] == h]
    actual = sub[sub["model"] == BENCHMARK].dropna(subset=["actual"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(_dates(actual["target_end"]), actual["actual"], color=INK, label="Realized", zorder=3)
    for name in models:
        s = sub[sub["model"] == name]
        ax.plot(_dates(s["target_end"]), s["forecast"], color=MODEL_COLORS[name], label=label(name), linewidth=1.6)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    span = "12-month" if h == 12 else f"{h}-month"
    ax.set_title(f"US CPI inflation over the next {h} months: realized and forecast")
    ax.set_ylabel(f"{span} change, %")
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", ncol=len(models) + 1)
    ax.margins(x=0.01)
    _save(fig, path)


def ratio_heatmaps(table: pd.DataFrame, path: Path) -> None:
    """RMSE relative to AR by model and horizon, one panel per evaluation period."""
    models = [m for m in MODELS if m in set(table["model"])]
    horizons = sorted(table["horizon"].unique())
    fig, axes = plt.subplots(1, len(PERIODS), figsize=(2.2 * len(PERIODS) + 1.6, 0.42 * len(models) + 1.2), sharey=True)
    for ax, period in zip(axes, PERIODS, strict=True):
        sub = table[table["period"] == period]
        grid = sub.pivot(index="model", columns="horizon", values="rmse_ratio").reindex(models)[horizons]
        ax.imshow(grid.to_numpy(), cmap=DIVERGING, vmin=0.6, vmax=1.4, aspect="auto")
        for i, name in enumerate(models):
            for j, h in enumerate(horizons):
                v = grid.loc[name, h]
                row = sub[(sub["model"] == name) & (sub["horizon"] == h)].iloc[0]
                text = f"{v:.2f}" + ("*" if row.dm_pvalue < 0.05 else "")
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color="white" if abs(v - 1) > 0.3 else INK)
        ax.set_xticks(range(len(horizons)), [f"h={h}" for h in horizons])
        ax.set_yticks(range(len(models)), [label(m) for m in models])
        ax.set_title(period, fontsize=10)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "RMSE relative to the AR benchmark (below 1 = more accurate; * = Diebold-Mariano p < 0.05)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, path)


def cumulative_gain(df: pd.DataFrame, models: list[str], horizons: list[int], path: Path) -> None:
    """Running sum of AR's squared error minus the model's: rising = the model is doing better."""
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.6 * len(horizons), 3.6), sharex=True)
    for ax, h in zip(np.atleast_1d(axes), horizons, strict=True):
        errors = error_matrix(df[df["actual"].notna()], h).sort_index()
        ends = _dates(pd.Series(errors.index)) + pd.DateOffset(months=h)
        for name in models:
            gain = (errors[BENCHMARK] ** 2 - errors[name] ** 2).cumsum()
            ax.plot(ends, gain.to_numpy(), color=MODEL_COLORS[name], label=label(name), linewidth=1.6)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_title(f"h = {h} month{'s' if h > 1 else ''}", fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    np.atleast_1d(axes)[0].set_ylabel("cumulative squared-error gain over AR")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(models), bbox_to_anchor=(1, 1.02))
    fig.suptitle("Where the gains over AR come from", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def group_of(variable: str, groups: pd.Series) -> str:
    if variable == TARGET:
        return OWN_LAGS
    if variable.startswith("F") and variable[1:].isdigit():
        return FACTORS
    return GROUP_NAMES[int(groups[variable])]


def group_shares(contrib: pd.DataFrame) -> pd.DataFrame:
    """Share of each group in the absolute contributions to each forecast (rows sum to 1)."""
    groups = series_table()["group"]
    by_group = contrib.abs().T.groupby(lambda v: group_of(v, groups)).sum().T
    return by_group.div(by_group.sum(axis=1), axis=0)


def importance_bars(results: Path, model: str, horizons: list[int], path: Path) -> None:
    """Average share of each group of series in the model's forecasts, per horizon."""
    shares = {
        h: group_shares(pd.read_csv(results / "contributions" / f"{model}_h{h}.csv", index_col="origin")).mean()
        for h in horizons
    }
    table = pd.DataFrame(shares).fillna(0).sort_values(horizons[-1])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(table) + 1.2))
    y = np.arange(len(table))
    height = 0.8 / len(horizons)
    for k, (color, h) in enumerate(zip(SERIES, horizons, strict=False)):
        offset = ((len(horizons) - 1) / 2 - k) * height  # first horizon on top
        ax.barh(y + offset, table[h], height=height * 0.85, color=color, label=f"h = {h}")
    ax.set_yticks(y, table.index)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What the {label(model).lower()} forecasts rest on: share of absolute SHAP value by group")
    ax.legend(loc="lower right")
    _save(fig, path)


def contributions_over_time(results: Path, model: str, h: int, start: str, path: Path) -> None:
    """Contributions of the main groups to each forecast since `start`, stacked, with their sum."""
    contrib = pd.read_csv(results / "contributions" / f"{model}_h{h}.csv", index_col="origin").loc[start:]
    groups = series_table()["group"]
    by_group = contrib.T.groupby(lambda v: group_of(v, groups)).sum().T
    main = ["Housing", "Interest and exchange rates", "Labor market"]
    by_group["All other series"] = by_group.drop(columns=main).sum(axis=1)
    by_group = by_group[[*main, "All other series"]]
    colors = [*SERIES[:3], "#b5b3ad"]
    dates = _dates(pd.Series(by_group.index))
    fig, ax = plt.subplots(figsize=(9, 4))
    pos = np.zeros(len(by_group))
    neg = np.zeros(len(by_group))
    for color, name in zip(colors, by_group.columns, strict=True):
        v = by_group[name].to_numpy()
        up, down = np.clip(v, 0, None), np.clip(v, None, 0)
        ax.bar(dates, up, bottom=pos, width=25, color=color, label=name, linewidth=0)
        ax.bar(dates, down, bottom=neg, width=25, color=color, linewidth=0)
        pos, neg = pos + up, neg + down
    ax.plot(dates, by_group.sum(axis=1), color=INK, linewidth=1.6, label="Sum: forecast minus the model's average")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_title(f"What moved the {label(model).lower()}'s {h}-month forecast, by forecast month (SHAP values)")
    ax.set_ylabel("percentage points")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    _save(fig, path)


def make_figures(results: Path, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    df = load_forecasts(results)
    table = pd.read_csv(results / "tables" / "accuracy.csv")
    focus = ["rf", "adalasso", "xgb"]
    forecast_paths(df, ["rf", BENCHMARK], 12, figures / "forecasts_h12.png")
    ratio_heatmaps(table, figures / "rmse_ratios.png")
    cumulative_gain(df, focus, [1, 12], figures / "cumulative_gain.png")
    importance_bars(results, "rf", [1, 12], figures / "shap_groups_rf.png")
    contributions_over_time(results, "rf", 12, "2019-01", figures / "shap_2019_2026_rf.png")
