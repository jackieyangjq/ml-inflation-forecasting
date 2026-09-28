"""Command line: inflation-ml download | backtest | evaluate | figures | all."""

from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import Config
from .data import DEFAULT_VINTAGE, build_panel, download
from .models import MODELS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="inflation-ml", description=__doc__)
    parser.add_argument("command", choices=["download", "backtest", "evaluate", "figures", "all"])
    parser.add_argument("--vintage", default=DEFAULT_VINTAGE, help="FRED-MD vintage, YYYY-MM (default %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(Config.horizons))
    parser.add_argument("--jobs", type=int, default=-1, help="parallel fits (default: all cores)")
    parser.add_argument("--overwrite", action="store_true", help="refit even if a forecast file exists")
    args = parser.parse_args(argv)

    if args.command == "download":
        print(f"saved {download(args.vintage, args.data_dir)}")
        return
    if args.command in ("backtest", "all"):
        from .backtest import run

        panel = build_panel(download(args.vintage, args.data_dir))
        print(
            f"FRED-MD {args.vintage}: {panel.X.shape[1]} series, "
            f"{panel.X.index[0]:%Y-%m} to {panel.X.index[-1]:%Y-%m}; "
            f"dropped {', '.join(panel.dropped)}; {panel.n_outliers} outliers replaced"
        )
        run(args.models, panel, Config(horizons=tuple(args.horizons)), args.results, args.jobs, args.overwrite)
    if args.command in ("evaluate", "all"):
        from .evaluate import write_tables

        write_tables(args.results)
    if args.command in ("figures", "all"):
        from .plots import make_figures

        make_figures(args.results, args.figures)


if __name__ == "__main__":
    main()
