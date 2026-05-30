"""
Backtesting: train on years 2016–(N-1), predict year N, measure per-round MAE.

For each slot present in the test year:
  - Predict all rounds using only training-year data
  - Compare predicted vs actual closing rank per round
  - Report MAE per round and overall
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

from .config import COL_YEAR, COL_ROUND, COL_CLOSE_RANK, ALL_ROUNDS, DEFAULT_TREND_MODEL
from .loader import load
from .train import SLOT_COLS, SlotModel

# Tier ordering for consistent display
_TIER_ORDER = ["IIT", "NIT", "IIIT", "GFTI"]
_EXAM_ORDER = ["advanced", "mains"]


def _get_tier(institute: str, exam_type: str) -> str:
    """Classify a slot into IIT / NIT / IIIT / GFTI based on exam type and name."""
    if exam_type == "advanced":
        return "IIT"
    name = institute.lower()
    if "national institute of technology" in name:
        return "NIT"
    if "indian institute of information technology" in name:
        return "IIIT"
    return "GFTI"


def _strata_mae_table(strata_abs_errors: dict, rounds: list[int],
                      dim: str, order: list[str]) -> pd.DataFrame:
    """Build a DataFrame of MAE by stratum x round for one stratification dimension."""
    rows = []
    for stratum in order:
        errs_by_round = strata_abs_errors.get(stratum, {})
        all_errs: list[float] = []
        row: dict = {"Stratum": stratum}
        for r in rounds:
            errs = errs_by_round.get(r, [])
            if errs:
                row[f"R{r}"] = round(float(np.mean(errs)), 1)
                row[f"N{r}"] = len(errs)
                all_errs.extend(errs)
            else:
                row[f"R{r}"] = float("nan")
                row[f"N{r}"] = 0
        row["Overall MAE"] = round(float(np.mean(all_errs)), 1) if all_errs else float("nan")
        row["N"]           = len(all_errs)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Stratum")


def backtest(
    csv_path:    str,
    test_year:   int | None  = None,
    rounds:      list[int] | None = None,
    trend_model: str = DEFAULT_TREND_MODEL,
    normalize:   bool = False,
    quiet:       bool = False,
    _df:         pd.DataFrame | None = None,
    stratify:    bool = True,
    blend_alpha: float = 0.5,
    mlp_hidden:  tuple = (256, 128, 64),
    mlp_dropout: float = 0.2,
) -> dict:
    def log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    df = _df if _df is not None else load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if test_year is None:
        test_year = all_years[-1]

    if rounds is None:
        rounds = ALL_ROUNDS

    train_years = [y for y in all_years if y < test_year]
    log(f"Backtest  |  train: {train_years}  |  test: {test_year}  |  "
        f"trend={trend_model}  |  normalize={normalize}")

    train_df = df[df[COL_YEAR].isin(train_years)]
    test_df  = df[df[COL_YEAR] == test_year]

    log("Grouping training data...")
    train_groups = {
        key: grp for key, grp in train_df.groupby(SLOT_COLS, sort=False)
    }
    log(f"Training slots: {len(train_groups):,}  |  "
        f"Test slots: {test_df[SLOT_COLS].drop_duplicates().shape[0]:,}")

    # Global model (MLP or GP-MLP ensemble): train once, batch-predict test rows
    _global_mlp = None
    _mlp_preds: pd.Series | None = None
    if trend_model == "mlp":
        from .mlp_model import GlobalMLPModel
        _global_mlp = GlobalMLPModel()
        _global_mlp.fit(train_df)
        _pred_arr  = _global_mlp.predict_df(test_df)
        _mlp_preds = pd.Series(_pred_arr, index=test_df.index)
        log("  Batch predictions computed.")
    elif trend_model == "mlp_ensemble":
        from .train import GPMLPEnsemble
        _global_mlp = GPMLPEnsemble(blend_alpha=blend_alpha)
        _global_mlp.fit(train_df, mlp_hidden=mlp_hidden, mlp_dropout=mlp_dropout)
        _pred_arr  = _global_mlp.predict_df(test_df)
        _mlp_preds = pd.Series(_pred_arr, index=test_df.index)
        log("  Batch predictions computed.")

    round_errors: dict[int, tuple[list, list]] = {r: ([], []) for r in rounds}

    # strata_abs_errors["exam_type"]["advanced"][r] = [abs_err, ...]
    # strata_abs_errors["tier"]["NIT"][r] = [abs_err, ...]
    strata_abs_errors: dict[str, dict[str, dict[int, list]]] = {
        "exam_type": defaultdict(lambda: defaultdict(list)),
        "tier":      defaultdict(lambda: defaultdict(list)),
    }

    for i, (key, test_grp) in enumerate(test_df.groupby(SLOT_COLS, sort=False)):
        if _global_mlp is not None:
            if tuple(key) not in _global_mlp.slot_stats:
                continue
            grp_pred_vals = _mlp_preds.loc[test_grp.index].values
            preds = {int(r): float(p)
                     for r, p in zip(test_grp[COL_ROUND].values, grp_pred_vals)
                     if int(r) in rounds}
        else:
            train_grp = train_groups.get(key)
            if train_grp is None:
                continue
            m = SlotModel(trend_model=trend_model, normalize=normalize)
            m.fit(train_grp)
            preds = m.predict_all_rounds(test_year, rounds)

        inst, _prog, _q, _st, _g, exam_type = key
        tier = _get_tier(inst, exam_type)

        test_by_round = test_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
        for r in rounds:
            if r not in test_by_round or r not in preds:
                continue
            act  = float(test_by_round[r])
            pred = float(preds[r])
            round_errors[r][0].append(act)
            round_errors[r][1].append(pred)
            if stratify:
                abs_err = abs(act - pred)
                strata_abs_errors["exam_type"][exam_type][r].append(abs_err)
                strata_abs_errors["tier"][tier][r].append(abs_err)

        if (i + 1) % 1000 == 0:
            log(f"  {i + 1:,} slots processed...")

    log(f"\n{'Round':<8} {'N slots':>8} {'MAE':>10}")
    log("-" * 30)
    all_act, all_pred = [], []
    round_maes: dict[int, float] = {}
    for r in rounds:
        act, pred = round_errors[r]
        if not act:
            continue
        mae = mean_absolute_error(act, pred)
        round_maes[r] = mae
        log(f"R{r:<7} {len(act):>8,} {mae:>10.1f}")
        all_act.extend(act)
        all_pred.extend(pred)

    overall_mae = mean_absolute_error(all_act, all_pred) if all_act else float("nan")
    log(f"{'Overall':<8} {len(all_act):>8,} {overall_mae:>10.1f}")

    # Compute and log stratified MAE tables
    strata_tables: dict[str, pd.DataFrame] = {}
    if stratify:
        for dim, order, label in [
            ("exam_type", _EXAM_ORDER, "Exam Type"),
            ("tier",      _TIER_ORDER, "Institute Tier"),
        ]:
            tbl = _strata_mae_table(
                strata_abs_errors[dim], rounds, dim, order
            )
            strata_tables[dim] = tbl
            if not quiet:
                r_cols = [f"R{r}" for r in rounds if f"R{r}" in tbl.columns]
                log(f"\n  Stratified MAE by {label}:")
                log(f"  {'Stratum':<12} {'N':>8}  " +
                    "  ".join(f"{c:>8}" for c in r_cols) +
                    f"  {'Overall':>10}")
                log("  " + "-" * (12 + 10 + 10 * len(r_cols) + 12))
                for stratum, row in tbl.iterrows():
                    r_vals = "  ".join(
                        f"{row[c]:>8.0f}" if not np.isnan(row[c]) else f"{'-':>8}"
                        for c in r_cols
                    )
                    log(f"  {stratum:<12} {int(row['N']):>8}  {r_vals}  "
                        f"{row['Overall MAE']:>10.1f}")

    return {
        "test_year":     test_year,
        "overall_mae":   overall_mae,
        "round_maes":    round_maes,
        "round_errors":  round_errors,
        "strata_tables": strata_tables,
    }


def tune_ensemble_weight(
    csv_path:    str,
    val_year:    int | None = None,
    rounds:      list[int] | None = None,
    trend_model: str = DEFAULT_TREND_MODEL,
    w_grid:      list[float] | None = None,
    quiet:       bool = False,
) -> dict:
    """
    Find the optimal ensemble weight w by grid search on a held-out year.

    Slot models are trained once on years < val_year; the weight sweep is then
    free (no retraining per candidate w).

    Returns:
        {
          "best_w":   float,
          "val_year": int,
          "results":  pd.DataFrame  (columns: w, overall_mae, R1, R2, ...),
        }
    """
    def log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    if w_grid is None:
        w_grid = [round(x * 0.05, 2) for x in range(21)]  # 0.00 … 1.00 step 0.05

    df = load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if val_year is None:
        val_year = all_years[-1]

    if rounds is None:
        rounds = ALL_ROUNDS

    train_years = [y for y in all_years if y < val_year]
    log(f"Weight tune  |  train: {train_years}  |  val: {val_year}  |  trend={trend_model}")
    log(f"w grid: {w_grid}")

    train_df = df[df[COL_YEAR].isin(train_years)]
    val_df   = df[df[COL_YEAR] == val_year]

    # Train all slot models once
    log("Training slot models...")
    train_groups = {key: grp for key, grp in train_df.groupby(SLOT_COLS, sort=False)}
    slot_models: dict = {}
    for key, train_grp in train_groups.items():
        m = SlotModel(trend_model=trend_model)
        m.fit(train_grp)
        slot_models[key] = m
    log(f"Trained {len(slot_models):,} slot models.")

    # Pre-collect (actual, direct_signal, ratio_signal) per slot per round
    # so we can evaluate any w without re-running prediction
    actuals:      dict[int, list[float]] = {r: [] for r in rounds}
    directs:      dict[int, list[float]] = {r: [] for r in rounds}
    via_ratios:   dict[int, list[float]] = {r: [] for r in rounds}

    for key, val_grp in val_df.groupby(SLOT_COLS, sort=False):
        m = slot_models.get(key)
        if m is None:
            continue
        test_by_round = val_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
        for r in rounds:
            if r not in test_by_round:
                continue
            # Decompose the two signals for this round
            if r in m.round_year_models:
                direct = float(m.round_year_models[r].predict([[val_year]])[0])
            else:
                direct = m.round_medians.get(r, m.round_medians.get(m.max_round, 0))

            if m.max_round in m.round_year_models:
                pred_final = float(m.round_year_models[m.max_round].predict([[val_year]])[0])
            else:
                pred_final = m.round_medians.get(m.max_round, direct)

            ratio     = m.round_ratios.get(r, m.round_ratios.get(m.max_round, 1.0))
            via_ratio = pred_final * ratio

            actuals[r].append(float(test_by_round[r]))
            directs[r].append(max(1.0, direct))
            via_ratios[r].append(max(1.0, via_ratio))

    # Sweep over w values
    rows = []
    for w in w_grid:
        all_act, all_pred = [], []
        row = {"w": w}
        for r in rounds:
            if not actuals[r]:
                continue
            preds = [max(1.0, w * d + (1 - w) * vr)
                     for d, vr in zip(directs[r], via_ratios[r])]
            mae = mean_absolute_error(actuals[r], preds)
            row[f"R{r}"] = round(mae, 1)
            all_act.extend(actuals[r])
            all_pred.extend(preds)
        row["overall_mae"] = round(mean_absolute_error(all_act, all_pred), 1) if all_act else float("nan")
        rows.append(row)

    results = pd.DataFrame(rows).set_index("w")
    best_w  = float(results["overall_mae"].idxmin())

    log(f"\n{'='*60}")
    log(f"  Ensemble weight tuning  |  val year {val_year}")
    log(f"{'='*60}")
    r_cols = [c for c in results.columns if c.startswith("R")]
    log(results[["overall_mae"] + r_cols].to_string())
    log(f"\n  Best w = {best_w}  (overall MAE {results.loc[best_w, 'overall_mae']:.1f})")
    log(f"  Default w = 0.5  (overall MAE {results.loc[0.5, 'overall_mae']:.1f})")
    improvement = (results.loc[0.5, "overall_mae"] - results.loc[best_w, "overall_mae"])
    log(f"  Improvement over default: {improvement:.1f} rank positions")

    return {"best_w": best_w, "val_year": val_year, "results": results}


def tune_blend_alpha(
    csv_path:    str,
    val_year:    int | None = None,
    rounds:      list[int] | None = None,
    mlp_hidden:  tuple = (256, 128, 64),
    mlp_dropout: float = 0.2,
    alpha_grid:  list[float] | None = None,
    quiet:       bool = False,
    _df:         pd.DataFrame | None = None,
) -> dict:
    """
    Find the optimal GP-MLP blend_alpha by sweeping over alpha_grid on a
    held-out year.  The ensemble is trained once; GP and MLP predictions are
    stored separately so the sweep is free (no retraining per candidate alpha).

    Returns:
        {
          "best_alpha": float,
          "val_year":   int,
          "results":    pd.DataFrame  (index=alpha, columns: overall_mae, R1, ...),
        }
    """
    def log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    from .train import GPMLPEnsemble, SLOT_COLS
    from .config import COL_INSTITUTE, COL_PROGRAM, COL_QUOTA, COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE

    if alpha_grid is None:
        alpha_grid = [round(i * 0.1, 1) for i in range(11)]  # 0.0 … 1.0

    df = _df if _df is not None else load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if val_year is None:
        val_year = all_years[-1]

    if rounds is None:
        rounds = ALL_ROUNDS

    train_df = df[df[COL_YEAR] < val_year]
    val_df   = df[df[COL_YEAR] == val_year]

    log(f"Alpha tune  |  train: {sorted(train_df[COL_YEAR].unique())}  |  val: {val_year}")
    log(f"  arch={mlp_hidden}, dropout={mlp_dropout}  |  alpha grid: {alpha_grid}")

    ens = GPMLPEnsemble(blend_alpha=0.5)
    ens.fit(train_df, mlp_hidden=mlp_hidden, mlp_dropout=mlp_dropout)

    gp_preds, mlp_preds, _ = ens.predict_df_components(val_df)
    gp_series  = pd.Series(gp_preds,  index=val_df.index)
    mlp_series = pd.Series(mlp_preds, index=val_df.index)

    # Collect (actual, gp_pred, mlp_pred) per valid (slot, round) pair
    rows_act, rows_gp, rows_mlp = [], [], []
    round_tags: list[int] = []
    for key, grp in val_df.groupby(SLOT_COLS, sort=False):
        if tuple(key) not in ens.slot_stats:
            continue
        for _, row in grp.iterrows():
            r = int(row[COL_ROUND])
            if r not in rounds:
                continue
            rows_act.append(float(row[COL_CLOSE_RANK]))
            rows_gp.append(float(gp_series.loc[row.name]))
            rows_mlp.append(float(mlp_series.loc[row.name]))
            round_tags.append(r)

    act_arr = np.array(rows_act)
    gp_arr  = np.maximum(1.0, np.array(rows_gp))
    mlp_arr = np.maximum(1.0, np.array(rows_mlp))
    round_arr = np.array(round_tags)

    log(f"  Valid (slot, round) pairs: {len(act_arr):,}")

    table_rows = []
    for alpha in alpha_grid:
        blended = np.maximum(1.0, alpha * gp_arr + (1.0 - alpha) * mlp_arr)
        row = {"alpha": alpha,
               "overall_mae": float(mean_absolute_error(act_arr, blended))}
        for r in rounds:
            mask = round_arr == r
            if mask.any():
                row[f"R{r}"] = float(mean_absolute_error(act_arr[mask], blended[mask]))
        table_rows.append(row)

    results   = pd.DataFrame(table_rows).set_index("alpha")
    best_alpha = float(results["overall_mae"].idxmin())

    log(f"\n{'='*55}")
    log(f"  Blend-alpha sweep  |  val year {val_year}")
    log(f"{'='*55}")
    r_cols = [c for c in results.columns if c.startswith("R")]
    log(results[["overall_mae"] + r_cols].to_string())
    log(f"\n  Best alpha = {best_alpha}  "
        f"(overall MAE {results.loc[best_alpha, 'overall_mae']:.1f})")
    log(f"  Default alpha = 0.5  "
        f"(overall MAE {results.loc[0.5, 'overall_mae']:.1f})")
    improvement = results.loc[0.5, "overall_mae"] - results.loc[best_alpha, "overall_mae"]
    log(f"  Improvement over default: {improvement:.1f} rank positions")

    return {"best_alpha": best_alpha, "val_year": val_year, "results": results}


def tune_blend_alpha_loo(
    csv_path:    str,
    cal_years:   list[int] | None = None,
    rounds:      list[int] | None = None,
    mlp_hidden:  tuple = (256, 128, 64),
    mlp_dropout: float = 0.2,
    alpha_grid:  list[float] | None = None,
    quiet:       bool = False,
) -> dict:
    """
    Multi-year leave-one-out blend_alpha calibration.

    For each year in cal_years, trains on all preceding years and finds the
    best alpha.  The recommended deployed alpha is the average across years,
    rounded to the nearest grid step.

    cal_years defaults to all years except the most recent (held back as the
    true test year).

    Returns:
        {
          "best_alpha":    float,   # rounded average — recommended for deployment
          "per_year":      dict,    # {year: best_alpha}
          "per_year_mae":  dict,    # {year: best_mae}
          "alpha_grid":    list,
        }
    """
    def log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    if alpha_grid is None:
        alpha_grid = [round(i * 0.1, 1) for i in range(11)]
    step = alpha_grid[1] - alpha_grid[0]

    df = load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if cal_years is None:
        cal_years = all_years[:-1]   # all except the most recent test year

    # require at least 2 training years before each cal year
    cal_years = [y for y in cal_years if sum(1 for yr in all_years if yr < y) >= 2]

    log(f"\nLOO alpha calibration  |  cal years: {cal_years}")
    log(f"  arch={mlp_hidden}, dropout={mlp_dropout}  |  grid: {alpha_grid}")

    per_year: dict[int, float] = {}
    per_year_mae: dict[int, float] = {}

    for year in cal_years:
        log(f"\n--- Calibration year {year} ---")
        result = tune_blend_alpha(
            csv_path, val_year=year, rounds=rounds,
            mlp_hidden=mlp_hidden, mlp_dropout=mlp_dropout,
            alpha_grid=alpha_grid, quiet=quiet,
            _df=df,
        )
        per_year[year]     = result["best_alpha"]
        per_year_mae[year] = float(result["results"].loc[result["best_alpha"], "overall_mae"])

    avg_raw  = float(np.mean(list(per_year.values())))
    # Round to the nearest grid step
    best_alpha = round(round(avg_raw / step) * step, 10)
    best_alpha = min(alpha_grid, key=lambda a: abs(a - best_alpha))

    log(f"\n{'='*55}")
    log(f"  LOO alpha calibration summary")
    log(f"{'='*55}")
    log(f"  {'Year':>6}  {'Best α':>8}  {'MAE at best α':>15}")
    for y in cal_years:
        log(f"  {y:>6}  {per_year[y]:>8.1f}  {per_year_mae[y]:>15.1f}")
    log(f"\n  Raw average α = {avg_raw:.3f}  →  recommended α = {best_alpha}")

    return {
        "best_alpha":   best_alpha,
        "per_year":     per_year,
        "per_year_mae": per_year_mae,
        "alpha_grid":   alpha_grid,
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    backtest(csv_path)
