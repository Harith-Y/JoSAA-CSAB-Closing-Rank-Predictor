"""
Full 2022-2025 multi-year evaluation of GPMLPEnsemble with:
  - Tuned architecture: hidden=(512, 256, 128), dropout=0.15
  - alpha sweep [0.0, 0.1, ..., 1.0] per held-out year (no retraining per alpha)

For each test year the ensemble is trained once; GP and MLP predictions are
stored separately, then alpha is swept cheaply by linear combination.

Usage:
    python scripts/eval_mlp_ensemble.py
    python scripts/eval_mlp_ensemble.py --source csab
    python scripts/eval_mlp_ensemble.py --alpha-step 0.05
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.loader import load
from pipeline.config import (
    SOURCES, COL_YEAR, COL_ROUND, COL_CLOSE_RANK,
    COL_INSTITUTE, COL_PROGRAM, COL_QUOTA, COL_SEAT_TYPE,
    COL_GENDER, COL_EXAM_TYPE,
)
from pipeline.train import SLOT_COLS, GPMLPEnsemble

MLP_HIDDEN  = (512, 256, 128)
MLP_DROPOUT = 0.15


def _run_year(
    df: pd.DataFrame,
    test_year: int,
    rounds: list[int],
    alpha_grid: list[float],
) -> dict:
    train_df = df[df[COL_YEAR] < test_year].copy()
    test_df  = df[df[COL_YEAR] == test_year].copy()

    train_years = sorted(train_df[COL_YEAR].unique())
    n_train = train_df[SLOT_COLS].drop_duplicates().shape[0]
    n_test  = test_df[SLOT_COLS].drop_duplicates().shape[0]
    print(f"\n{'='*65}")
    print(f"  Test year {test_year}  |  train: {train_years}")
    print(f"  Train slots: {n_train:,}   Test slots: {n_test:,}")
    print(f"  arch={MLP_HIDDEN}, dropout={MLP_DROPOUT}")
    print(f"{'='*65}")

    ens = GPMLPEnsemble(blend_alpha=0.5)
    ens.fit(train_df, mlp_hidden=MLP_HIDDEN, mlp_dropout=MLP_DROPOUT)

    # Get raw GP and MLP signals separately (no retraining needed per alpha)
    gp_preds, mlp_preds, gp_mask = ens.predict_df_components(test_df)

    # Index predictions by original test_df index for slot-group alignment
    gp_series  = pd.Series(gp_preds,  index=test_df.index)
    mlp_series = pd.Series(mlp_preds, index=test_df.index)
    mask_series = pd.Series(gp_mask,  index=test_df.index)

    # Collect per-row (actual, gp_pred, mlp_pred) using the same filtering as backtest()
    # Note: for cold-start slots predict_df_components sets gp_pred == mlp_pred,
    # so blending alpha*gp + (1-alpha)*mlp degrades gracefully to pure MLP there.
    rows_act, rows_gp, rows_mlp = [], [], []
    for key, test_grp in test_df.groupby(SLOT_COLS, sort=False):
        if tuple(key) not in ens.slot_stats:
            continue
        for _, row in test_grp.iterrows():
            r = int(row[COL_ROUND])
            if r not in rounds:
                continue
            rows_act.append(float(row[COL_CLOSE_RANK]))
            rows_gp.append(float(gp_series.loc[row.name]))
            rows_mlp.append(float(mlp_series.loc[row.name]))

    if not rows_act:
        print("  No valid rows - skipping.")
        return {}

    act_arr = np.array(rows_act)
    gp_arr  = np.maximum(1.0, np.array(rows_gp))
    mlp_arr = np.maximum(1.0, np.array(rows_mlp))

    print(f"  Valid (slot, round) pairs: {len(act_arr):,}")

    alpha_mae: dict[float, float] = {}
    for alpha in alpha_grid:
        blended = np.maximum(1.0, alpha * gp_arr + (1.0 - alpha) * mlp_arr)
        alpha_mae[alpha] = float(mean_absolute_error(act_arr, blended))

    best_alpha = min(alpha_mae, key=alpha_mae.get)
    print(f"\n  {'alpha':>6}  {'MAE':>10}")
    print(f"  {'-'*18}")
    for a in alpha_grid:
        marker = " <-- best" if a == best_alpha else ""
        print(f"  {a:>6.2f}  {alpha_mae[a]:>10.1f}{marker}")
    print(f"\n  Best alpha={best_alpha:.2f}  MAE={alpha_mae[best_alpha]:.1f}")
    print(f"  alpha=0.50 MAE={alpha_mae[0.5]:.1f}  "
          f"alpha=0.00 (MLP) MAE={alpha_mae[0.0]:.1f}  "
          f"alpha=1.00 (GP)  MAE={alpha_mae[1.0]:.1f}")

    return {"year": test_year, "alpha_mae": alpha_mae, "best_alpha": best_alpha,
            "n_pairs": len(act_arr)}


def main():
    parser = argparse.ArgumentParser(description="GP-MLP ensemble full 2022-2025 evaluation")
    parser.add_argument("--source", default="josaa", choices=["josaa", "csab"])
    parser.add_argument("--alpha-step", type=float, default=0.1,
                        help="Alpha grid step size (default 0.1)")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Test years (default: 2022 2023 2024 2025)")
    args = parser.parse_args()

    cfg = SOURCES[args.source]
    rounds   = cfg["rounds"]

    root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = cfg["csv"]
    for candidate in [csv_path, os.path.join(root, "data", cfg["csv"]),
                      os.path.join(root, cfg["csv"])]:
        if os.path.exists(candidate):
            csv_path = candidate
            break
    else:
        print(f"CSV not found: {cfg['csv']} (searched root and data/)")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    df = load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())
    print(f"Years available: {all_years}")

    test_years = args.years or [y for y in [2022, 2023, 2024, 2025] if y in all_years]
    steps = round(1.0 / args.alpha_step)
    alpha_grid = [round(i * args.alpha_step, 4) for i in range(steps + 1)]

    print(f"\nSource: {args.source}  |  test years: {test_years}")
    print(f"Architecture: {MLP_HIDDEN}, dropout={MLP_DROPOUT}")
    print(f"Alpha grid ({len(alpha_grid)} values): {alpha_grid}")

    results = []
    for year in test_years:
        if year not in all_years:
            print(f"\nYear {year} not in dataset - skipping.")
            continue
        r = _run_year(df, year, rounds, alpha_grid)
        if r:
            results.append(r)

    if not results:
        print("No results.")
        return

    # Summary table
    print(f"\n\n{'='*65}")
    print(f"  SUMMARY - GP-MLP ensemble  arch={MLP_HIDDEN}, dropout={MLP_DROPOUT}")
    print(f"{'='*65}")
    header = f"  {'Year':>6}  {'BestAlpha':>10}  {'BestMAE':>10}  {'a=0.5':>10}  {'a=0.0(MLP)':>12}  {'a=1.0(GP)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_maes, a05_maes = [], []
    for r in results:
        am = r["alpha_mae"]
        best_mae = am[r["best_alpha"]]
        a05  = am[0.5]
        a00  = am[0.0]
        a10  = am[1.0]
        best_maes.append(best_mae)
        a05_maes.append(a05)
        print(f"  {r['year']:>6}  {r['best_alpha']:>10.2f}  {best_mae:>10.1f}  "
              f"{a05:>10.1f}  {a00:>12.1f}  {a10:>10.1f}")

    avg_best = np.mean(best_maes)
    avg_a05  = np.mean(a05_maes)
    print(f"  {'Avg':>6}  {'':>10}  {avg_best:>10.1f}  {avg_a05:>10.1f}")
    print(f"\n  Improvement (avg best vs alpha=0.5): {avg_a05 - avg_best:.1f} rank positions "
          f"({(avg_a05 - avg_best) / avg_a05 * 100:.1f}%)")


if __name__ == "__main__":
    main()
