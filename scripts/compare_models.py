"""
Compare year-trend model variants for JoSAA and CSAB.

Single-year mode (default):
    Runs backtest for every combination of source x trend_model and prints
    a side-by-side MAE table.

Multi-year mode (--years):
    Runs backtest across multiple held-out years for the selected models and
    prints a cross-tabulated summary (models x years).  Designed for the
    SVR RBF multi-year validation task.

Usage:
    python scripts/compare_models.py                              # both sources, most recent year
    python scripts/compare_models.py josaa                        # JoSAA only
    python scripts/compare_models.py josaa --year 2024            # single year
    python scripts/compare_models.py josaa --years 2022 2023 2024 2025          # multi-year, all models
    python scripts/compare_models.py josaa --years 2022 2023 2024 2025 \\
                                   --models median svr_rbf               # focused comparison
"""

import os
import argparse
import pandas as pd

from pipeline.config import SOURCES
from pipeline.evaluate import backtest
from pipeline.loader import load
from pipeline.train import TREND_MODELS

DESCRIPTIONS = {
    "ols":          "OLS         - straight-line trend, equal weight all years",
    "theil_sen":    "Theil-Sen   - robust OLS, median of pairwise slopes",
    "weighted_ols": "Weighted OLS - recent years weighted ~10x more",
    "median":       "Median      - historical average, no trend extrapolation",
    "ridge":        "Ridge       - L2-regularised OLS, slope shrunk toward zero",
    "svr_linear":   "SVR Linear  - SVM regression, linear kernel (≈ Ridge)",
    "svr_rbf":      "SVR RBF     - SVM regression, RBF kernel (reverts to mean on extrapolation)",
}


# ---------------------------------------------------------------------------
# Single-year comparison (original behaviour)
# ---------------------------------------------------------------------------

def compare(source: str, test_year: int | None = None,
            models: list[str] | None = None) -> pd.DataFrame:
    cfg  = SOURCES[source]
    csv  = cfg["csv"]
    rnds = cfg["rounds"]

    if not os.path.exists(csv):
        print(f"[{source}] CSV not found: {csv} - skipping.")
        return pd.DataFrame()

    model_list = models if models else TREND_MODELS
    df_data    = load(csv)   # load once

    rows = []
    for tm in model_list:
        print(f"\n{'-'*60}")
        print(f"[{source.upper()}]  trend_model = {tm}")
        print(f"{'-'*60}")
        result = backtest(csv, test_year=test_year, rounds=rnds,
                          trend_model=tm, quiet=False, _df=df_data)
        row = {"Model": tm, "Overall MAE": result["overall_mae"],
               "Test Year": result["test_year"]}
        for r in rnds:
            row[f"R{r}"] = result["round_maes"].get(r, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    mae_cols  = ["Overall MAE"] + [c for c in df.columns if c.startswith("R")]
    df[mae_cols] = df[mae_cols].round(1)
    best_idx  = df["Overall MAE"].idxmin()
    year_label = df["Test Year"].iloc[0]

    print(f"\n{'='*80}")
    print(f"  {source.upper()} - test year {year_label}  |  "
          f"best: {best_idx}  ({DESCRIPTIONS[best_idx]})")
    print(f"{'='*80}")
    print(df[["Overall MAE"] + [c for c in df.columns if c.startswith("R")]].to_string())
    print()
    for tm in model_list:
        marker = " <- best" if tm == best_idx else ""
        print(f"  {tm:<14}  {DESCRIPTIONS.get(tm, tm)}{marker}")
    return df


# ---------------------------------------------------------------------------
# Multi-year cross-tabulated comparison
# ---------------------------------------------------------------------------

def compare_multi_year(source: str, test_years: list[int],
                       models: list[str] | None = None) -> pd.DataFrame:
    """
    Run backtest for every (model, year) combination, loading the CSV once.

    Returns a DataFrame with shape (n_models, n_years) containing overall MAE,
    and prints a formatted cross-tab plus a per-year winner summary.
    """
    cfg  = SOURCES[source]
    csv  = cfg["csv"]
    rnds = cfg["rounds"]

    if not os.path.exists(csv):
        print(f"[{source}] CSV not found: {csv} - skipping.")
        return pd.DataFrame()

    model_list = models if models else TREND_MODELS

    print(f"\n{'='*80}")
    print(f"  {source.upper()}  -  multi-year validation")
    print(f"  models : {model_list}")
    print(f"  years  : {test_years}")
    print(f"{'='*80}\n")

    df_data = load(csv)   # load once; passed to every backtest call

    # results[model][year] = overall_mae
    results: dict[str, dict[int, float]] = {tm: {} for tm in model_list}

    for tm in model_list:
        for year in test_years:
            print(f"  [{source.upper()}] trend={tm:14s}  test_year={year} ...", end=" ", flush=True)
            res = backtest(csv, test_year=year, rounds=rnds,
                           trend_model=tm, quiet=True, _df=df_data)
            mae = res["overall_mae"]
            results[tm][year] = mae
            print(f"MAE = {mae:,.1f}")

    # Build cross-tab DataFrame
    table = pd.DataFrame(results).T          # rows=models, cols=years
    table.index.name = "Model"
    table = table[test_years]                # ensure column order

    # Compute row average
    table["Avg"] = table.mean(axis=1).round(1)
    table[test_years] = table[test_years].round(1)

    # Determine per-year winners
    winners = {y: table[y].idxmin() for y in test_years}
    avg_winner = table["Avg"].idxmin()

    print(f"\n{'='*80}")
    print(f"  {source.upper()}  -  Overall MAE by (model, year)")
    print(f"{'='*80}")
    print(table.to_string())

    print(f"\n  Per-year winner:")
    for y in test_years:
        w     = winners[y]
        w_mae = table.loc[w, y]
        # compare with median baseline if present
        if "median" in table.index and w != "median":
            med_mae = table.loc["median", y]
            delta   = med_mae - w_mae
            pct     = delta / med_mae * 100
            note    = f"  ({delta:+.0f} vs median, {pct:+.1f}%)"
        else:
            note    = ""
        print(f"    {y}: {w:<14}  MAE={w_mae:,.1f}{note}")

    print(f"\n  Average over all years:")
    for tm in model_list:
        avg = table.loc[tm, "Avg"]
        marker = " (best)" if tm == avg_winner else ""
        print(f"    {tm:<14}  {avg:,.1f}{marker}")

    # Verdict
    print(f"\n  Verdict:")
    if avg_winner == "svr_rbf":
        wins = sum(1 for y in test_years if winners[y] == "svr_rbf")
        print(f"    SVR RBF wins {wins}/{len(test_years)} years and has the best average MAE.")
        if "median" in table.index:
            med_avg  = table.loc["median", "Avg"]
            rbf_avg  = table.loc["svr_rbf", "Avg"]
            pct      = (med_avg - rbf_avg) / med_avg * 100
            print(f"    Average improvement over Median: {pct:.1f}%")
            if wins == len(test_years):
                print("    Recommendation: promote SVR RBF to deployed default.")
            else:
                print("    Recommendation: SVR RBF is strong but does not win every year; "
                      "keep Median as default or run more years.")
    elif avg_winner == "median" or "median" not in table.index:
        print(f"    {avg_winner} has the best average MAE.")
        if "svr_rbf" in table.index and avg_winner != "svr_rbf":
            rbf_avg = table.loc["svr_rbf", "Avg"]
            win_avg = table.loc[avg_winner, "Avg"]
            print(f"    SVR RBF average: {rbf_avg:,.1f}  vs  {avg_winner}: {win_avg:,.1f}")
        print(f"    Recommendation: keep {avg_winner} as deployed default.")
    else:
        print(f"    {avg_winner} has the best average MAE across {len(test_years)} years.")

    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare year-trend model variants across sources and years.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sources", nargs="*", default=["josaa", "csab"],
                        choices=list(SOURCES), help="Counselling sources to compare")
    parser.add_argument("--year", type=int, default=None,
                        help="Single hold-out test year (default: most recent)")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        metavar="YEAR",
                        help="Multiple held-out years for cross-year validation "
                             "(activates multi-year mode; overrides --year)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=TREND_MODELS, metavar="MODEL",
                        help="Subset of models to run (default: all). "
                             f"Choices: {TREND_MODELS}")
    args = parser.parse_args()

    all_results = {}

    for src in args.sources:
        if args.years:
            # Multi-year cross-tab mode
            tbl = compare_multi_year(src, args.years, models=args.models)
            if not tbl.empty:
                all_results[src] = tbl
        else:
            # Single-year mode (original behaviour)
            df = compare(src, test_year=args.year, models=args.models)
            if not df.empty:
                all_results[src] = df

    # Combined winner summary for single-year mode with multiple sources
    if not args.years and len(all_results) > 1:
        print(f"\n{'='*80}")
        print("  COMBINED SUMMARY")
        print(f"{'='*80}")
        for src, df in all_results.items():
            best     = df["Overall MAE"].idxmin()
            best_mae = df.loc[best, "Overall MAE"]
            ols_mae  = df.loc["ols", "Overall MAE"] if "ols" in df.index else best_mae
            pct      = (ols_mae - best_mae) / ols_mae * 100 if ols_mae != best_mae else 0
            print(f"  {src.upper():<8}  best={best:<14}  "
                  f"MAE={best_mae:.1f}  ({pct:+.1f}% vs OLS)")
