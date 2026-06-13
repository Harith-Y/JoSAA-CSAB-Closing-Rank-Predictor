"""
CLI entry point for college predictions.

Usage examples:

# Train JOSAA model (default):
  python scripts/predict_cli.py train
  python scripts/predict_cli.py train --source josaa

# Train CSAB model:
  python scripts/predict_cli.py train --source csab

# Backtest:
  python scripts/predict_cli.py backtest
  python scripts/predict_cli.py backtest --source csab --year 2024

# Predict (JOSAA):
  python scripts/predict_cli.py predict \
      --rank 5000 \
      --exam mains \
      --quota AI \
      --seat-type OPEN \
      --gender Gender-Neutral

# Predict (CSAB):
  python scripts/predict_cli.py predict \
      --source csab \
      --rank 8000 \
      --exam mains \
      --quota AI \
      --seat-type OPEN \
      --gender Gender-Neutral
"""

import torch  # must import before pandas/numpy to avoid DLL conflicts on Windows
import argparse
import os
import sys
import pandas as pd

from pipeline.config import SOURCES, MODEL_DIR, DEFAULT_TREND_MODEL


def _resolve_source(source: str) -> dict:
    """Return SOURCES entry with model_path resolved to absolute path."""
    cfg = SOURCES[source].copy()
    cfg["model_path"] = os.path.join(MODEL_DIR, cfg["model"])
    return cfg


def cmd_train(args):
    from pipeline.train import train
    cfg = _resolve_source(args.source)
    trend_model = args.trend_model or cfg.get("trend_model", DEFAULT_TREND_MODEL)
    train(cfg["csv"], model_path=cfg["model_path"], trend_model=trend_model,
          normalize=args.normalize_ranks,
          tune_mlp=getattr(args, "tune_mlp_hparams", False),
          mlp_hidden=cfg.get("mlp_hidden"),
          mlp_dropout=cfg.get("mlp_dropout"),
          blend_alpha=cfg.get("blend_alpha", 0.5))


def cmd_backtest(args):
    from pipeline.evaluate import backtest
    cfg = _resolve_source(args.source)
    trend_model = args.trend_model or cfg.get("trend_model", DEFAULT_TREND_MODEL)
    backtest(cfg["csv"], test_year=args.year, rounds=cfg["rounds"],
             trend_model=trend_model, normalize=args.normalize_ranks)


def cmd_tune(args):
    import pickle
    from pipeline.evaluate import tune_ensemble_weight

    cfg = _resolve_source(args.source)
    result = tune_ensemble_weight(
        cfg["csv"],
        val_year    = args.val_year,
        rounds      = cfg["rounds"],
        trend_model = args.trend_model,
    )
    best_w = result["best_w"]

    if args.save and os.path.exists(cfg["model_path"]):
        with open(cfg["model_path"], "rb") as f:
            model = pickle.load(f)
        model["ensemble_weight"] = best_w
        with open(cfg["model_path"], "wb") as f:
            pickle.dump(model, f)
        print(f"\nSaved ensemble_weight={best_w} into {cfg['model_path']}")
    elif args.save:
        print(f"\nModel not found at {cfg['model_path']}: run train first, then tune --save.")


def _save_blend_alpha(best_alpha: float, model_path: str) -> None:
    import pickle
    if not os.path.exists(model_path):
        print(f"\nModel not found at {model_path}: run train first, then tune-alpha --save.")
        return
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    gm = model.get("global_model")
    if gm is not None and hasattr(gm, "blend_alpha"):
        gm.blend_alpha = best_alpha
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nSaved blend_alpha={best_alpha} into {model_path}")
    else:
        print("\nModel is not a GPMLPEnsemble. blend_alpha not applicable.")


def cmd_tune_alpha(args):
    from pipeline.evaluate import tune_blend_alpha, tune_blend_alpha_loo

    cfg         = _resolve_source(args.source)
    mlp_hidden  = cfg.get("mlp_hidden",  (256, 128, 64))
    mlp_dropout = cfg.get("mlp_dropout", 0.2)

    if args.loo:
        result     = tune_blend_alpha_loo(
            cfg["csv"],
            rounds      = cfg["rounds"],
            mlp_hidden  = mlp_hidden,
            mlp_dropout = mlp_dropout,
        )
        best_alpha = result["best_alpha"]
    else:
        result     = tune_blend_alpha(
            cfg["csv"],
            val_year    = args.val_year,
            rounds      = cfg["rounds"],
            mlp_hidden  = mlp_hidden,
            mlp_dropout = mlp_dropout,
        )
        best_alpha = result["best_alpha"]

    if args.save:
        _save_blend_alpha(best_alpha, cfg["model_path"])


def cmd_predict(args):
    from pipeline.predict import predict, load_model

    cfg = _resolve_source(args.source)

    if not os.path.exists(cfg["model_path"]):
        print(f"Model not found: {cfg['model_path']}")
        print(f"Run:  python scripts/predict_cli.py train --source {args.source}")
        sys.exit(1)

    # Normalise gender input
    gender = args.gender
    if gender.lower() in ("female", "f", "female-only"):
        gender = "Female-only (including Supernumerary)"
    else:
        gender = "Gender-Neutral"

    model = load_model(cfg["model_path"])
    results = predict(
        rank            = args.rank,
        exam_type       = args.exam.lower(),
        quota           = args.quota.upper(),
        seat_type       = args.seat_type,
        gender          = gender,
        model           = model,
        rounds          = cfg["rounds"],
        include_reach   = not args.no_reach,
        safe_threshold  = cfg["safe_threshold"],
        reach_threshold = cfg["reach_threshold"],
        coverage        = args.coverage,
    )

    if results.empty:
        print("No matching colleges found for the given profile.")
        return

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.width", 0)

    round_cols    = [c for c in results.columns if c.startswith("R") and c[1:].isdigit()]
    interval_cols = ["Lower", "Upper"] if "Lower" in results.columns else []
    seat_col      = ["Seats"] if "Seats" in results.columns else []
    display_cols  = (["Institute", "Academic Program Name"] + round_cols
                     + ["Final Pred"] + interval_cols + ["Years"] + seat_col)

    if interval_cols:
        cov_pct = int(args.coverage * 100)
        print(f"\nPrediction intervals at {cov_pct}% coverage  "
              f"(Safe: rank <= Lower  |  Match: Lower < rank <= Final  |  "
              f"Reach: Final < rank <= Upper)")

    for cat in ["safe", "match", "reach"]:
        subset = results[results["Category"] == cat]
        if subset.empty:
            continue
        print(f"\n{'-'*80}")
        print(f"  {cat.upper()} ({len(subset)} options)")
        print(f"{'-'*80}")
        print(subset[display_cols].to_string(index=False))

    print(f"\nTotal: {len(results)} options  "
          f"(source={args.source}, rank={args.rank}, exam={args.exam}, "
          f"quota={args.quota}, seat={args.seat_type}, gender={gender})")

    if cfg["disclaimer"]:
        print(f"\n{'!'*80}")
        print(f"  {cfg['disclaimer']}")
        print(f"{'!'*80}")


def main():
    parser = argparse.ArgumentParser(description="JOSAA / CSAB College Predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    source_kwargs = dict(
        type=str, default="josaa", choices=["josaa", "csab"],
        help="Data source (default: josaa)",
    )
    trend_kwargs = dict(
        type=str, default=None,
        choices=["ols", "theil_sen", "weighted_ols", "median",
                 "ridge", "svr_linear", "svr_rbf", "ar1", "arp", "gp_rbf",
                 "mlp", "mlp_ensemble"],
        help="Year-trend model (default: per-source - gp_rbf for josaa, mlp_ensemble for csab). "
             "mlp_ensemble blends per-slot GP RBF with the global MLP.",
    )

    normalize_kwargs = dict(
        dest="normalize_ranks", action="store_true", default=False,
        help="Normalise closing ranks by JEE pool size before fitting "
             "(converts absolute ranks to pool-relative percentiles to remove "
             "year-to-year applicant-count bias)",
    )

    # train
    tr = sub.add_parser("train", help="Train model from CSV data")
    tr.add_argument("--source", **source_kwargs)
    tr.add_argument("--trend-model", dest="trend_model", **trend_kwargs)
    tr.add_argument("--normalize-ranks", **normalize_kwargs)
    tr.add_argument(
        "--tune-mlp-hparams", dest="tune_mlp_hparams",
        action="store_true", default=False,
        help="Run random hyperparameter search before final MLP training "
             "(applies when --trend-model is mlp or mlp_ensemble).",
    )

    # backtest
    bt = sub.add_parser("backtest", help="Evaluate model accuracy on a held-out year")
    bt.add_argument("--source", **source_kwargs)
    bt.add_argument("--trend-model", dest="trend_model", **trend_kwargs)
    bt.add_argument("--year", type=int, default=None,
                    help="Year to use as test set (default: most recent)")
    bt.add_argument("--normalize-ranks", **normalize_kwargs)

    # predict
    pr = sub.add_parser("predict", help="Predict colleges for a student profile")
    pr.add_argument("--source",    **source_kwargs)
    pr.add_argument("--rank",      type=int, required=True,  help="Your rank")
    pr.add_argument("--exam",      type=str, required=True,
                    choices=["advanced", "mains"],           help="Exam type")
    pr.add_argument("--quota",     type=str, required=True,
                    help="Quota: AI, HS, OS ...")
    pr.add_argument("--seat-type", type=str, required=True,
                    help="Seat type: OPEN, OBC-NCL, SC, ST, EWS ...")
    pr.add_argument("--gender",    type=str, default="Gender-Neutral",
                    help="Gender-Neutral (default) or Female")
    pr.add_argument("--no-reach",  action="store_true",
                    help="Exclude reach colleges from output")
    pr.add_argument("--coverage",  type=float, default=0.90,
                    metavar="LEVEL",
                    help="Prediction interval coverage (0.80/0.85/0.90/0.95, default 0.90). "
                         "Determines Safe/Match/Reach boundaries based on historical variability.")

    # tune
    tn = sub.add_parser("tune", help="Find optimal ensemble weight w via held-out validation")
    tn.add_argument("--source",     **source_kwargs)
    tn.add_argument("--trend-model", dest="trend_model", **trend_kwargs)
    tn.add_argument("--val-year",   dest="val_year", type=int, default=None,
                    help="Validation year (default: most recent year in CSV)")
    tn.add_argument("--save",       action="store_true",
                    help="Write best w into the model pickle for automatic use")

    # tune-alpha
    ta = sub.add_parser("tune-alpha",
                        help="Find optimal GP-MLP blend_alpha via held-out validation "
                             "(mlp_ensemble models only)")
    ta.add_argument("--source",   **source_kwargs)
    ta.add_argument("--val-year", dest="val_year", type=int, default=None,
                    help="Single validation year (default: most recent). "
                         "Ignored when --loo is set.")
    ta.add_argument("--loo",      action="store_true",
                    help="Multi-year leave-one-out calibration: find best alpha on "
                         "each year independently, then deploy the rounded average.")
    ta.add_argument("--save",     action="store_true",
                    help="Write best blend_alpha into the model pickle for automatic use")

    args = parser.parse_args()
    {"train": cmd_train, "backtest": cmd_backtest,
     "predict": cmd_predict, "tune": cmd_tune,
     "tune-alpha": cmd_tune_alpha}[args.cmd](args)


if __name__ == "__main__":
    main()
