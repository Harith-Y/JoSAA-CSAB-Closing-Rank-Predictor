"""
Global MLP trend model: one network trained on all slots simultaneously.

Features
--------
Categorical (embedded):
  tier (IIT/NIT/IIIT/GFTI), exam_type, quota, seat_type, gender,
  institute, programme
Continuous (normalised):
  year  → (year  - mean_year) / std_year
  round → (round - 1)         / 5

Target: log(closing_rank)  [inverse-transform: exp(pred)]

Default architecture (HP-search winner, full dataset)
------------------------------------------------------
  concat(embeddings, conts) → Linear(512) → BN → ReLU → Dropout(0.15)
                             → Linear(256) → BN → ReLU → Dropout(0.15)
                             → Linear(128) → BN → ReLU → Dropout(0.15)
                             → Linear(1)
  (221,573 params; selected by random search over 20 trials)

After training, the network is moved to CPU so the pickle is self-contained
and per-student inference needs no GPU transfer overhead.

Hyperparameter search
---------------------
tune_mlp_hyperparams(df, n_trials=20) runs a random search over the
architecture and optimiser search space, using the same internal 15%
validation split that fit() uses.  Returns the best config dict which
can be passed as **kwargs to fit().
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import (
    COL_YEAR, COL_ROUND, COL_INSTITUTE, COL_PROGRAM,
    COL_QUOTA, COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE,
    COL_CLOSE_RANK,
)

SLOT_COLS = [COL_INSTITUTE, COL_PROGRAM, COL_QUOTA,
             COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE]

# Categorical feature names and embedding dimensions (index order must match _featurize)
_CAT_KEYS = ["tier", "exam_type", "quota", "seat_type", "gender", "institute", "programme"]
_EMB_DIMS  = [4,      4,           8,       8,           4,        16,           32]


def _tier(institute: str, exam_type: str) -> str:
    """IITs are identified by exam type; NITs/IIITs by name."""
    if exam_type == "advanced":
        return "IIT"
    name = institute.lower()
    if "national institute of technology" in name:
        return "NIT"
    if "indian institute of information technology" in name or "iiit" in name:
        return "IIIT"
    return "GFTI"


class _CatEncoder:
    """Integer encoder with <UNK> (index 0) for unseen values at inference."""

    def __init__(self, values):
        uniq = ["<UNK>"] + sorted({str(v) for v in values})
        self._map = {v: i for i, v in enumerate(uniq)}
        self.n = len(uniq)

    def encode(self, values) -> np.ndarray:
        return np.array([self._map.get(str(v), 0) for v in values], dtype=np.int64)


class _Net(nn.Module):
    def __init__(self, emb_specs: list[tuple[int, int]], n_cont: int,
                 hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in emb_specs])
        in_dim = sum(d for _, d in emb_specs) + n_cont
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, cats: torch.Tensor, conts: torch.Tensor) -> torch.Tensor:
        embs = [emb(cats[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.net(torch.cat(embs + [conts], dim=1)).squeeze(-1)


class GlobalMLPModel:
    """
    Single MLP trained on the full dataset; all slots share the network.
    Slot identity is encoded as learned embeddings (institute, programme, etc.).

    Advantages over per-slot models
    --------------------------------
    - Shares statistical strength across similar slots (same tier/programme).
    - Handles cold-start: new slots in 2025 get predictions from their
      embedding neighbourhood even with zero personal history.
    """

    def __init__(self):
        self.encoders:    dict[str, _CatEncoder] = {}
        self.year_mean:   float = 2020.0
        self.year_std:    float = 3.0
        self.net:         _Net | None = None
        self.best_val_mse: float = float("inf")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.slot_stats: dict[tuple, dict] = {}  # per-slot round stats + n_years
        # Residual formulation: target = log(rank) - log(slot_round_median).
        # _log_median[(slot_key, round_no)] = log(historical median rank).
        # Unknown slot/round falls back to _global_log_median.
        self._log_median: dict[tuple, float] = {}
        self._global_log_median: float = 0.0

    # ------------------------------------------------------------------ #
    # Feature engineering                                                  #
    # ------------------------------------------------------------------ #

    def _featurize(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        tier_vals = [_tier(inst, et) for inst, et in
                     zip(df[COL_INSTITUTE], df[COL_EXAM_TYPE])]
        cats = np.stack([
            self.encoders["tier"].encode(tier_vals),
            self.encoders["exam_type"].encode(df[COL_EXAM_TYPE]),
            self.encoders["quota"].encode(df[COL_QUOTA]),
            self.encoders["seat_type"].encode(df[COL_SEAT_TYPE]),
            self.encoders["gender"].encode(df[COL_GENDER]),
            self.encoders["institute"].encode(df[COL_INSTITUTE]),
            self.encoders["programme"].encode(df[COL_PROGRAM]),
        ], axis=1).astype(np.int64)

        year_n  = (df[COL_YEAR].values.astype(float)  - self.year_mean) / self.year_std
        round_n = (df[COL_ROUND].values.astype(float) - 1.0)            / 5.0

        # Log of the historical slot+round median - the anchor the residual is
        # measured from.  Unknown slots fall back to the global median.
        slot_keys = list(zip(df[COL_INSTITUTE], df[COL_PROGRAM], df[COL_QUOTA],
                             df[COL_SEAT_TYPE], df[COL_GENDER], df[COL_EXAM_TYPE]))
        log_med_n = np.array([
            self._log_median.get((sk, int(r)), self._global_log_median)
            for sk, r in zip(slot_keys, df[COL_ROUND].values)
        ], dtype=np.float32)

        conts = np.stack([year_n, round_n, log_med_n], axis=1).astype(np.float32)
        return cats, conts

    def _tensors(self, cats: np.ndarray, conts: np.ndarray,
                 y: np.ndarray | None = None):
        tc = torch.tensor(cats,  device=self.device)
        tf = torch.tensor(conts, device=self.device)
        if y is not None:
            return tc, tf, torch.tensor(y, device=self.device)
        return tc, tf

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame,
            epochs: int = 200, lr: float = 1e-3, batch_size: int = 2048,
            hidden: tuple[int, ...] = (512, 256, 128),
            dropout: float = 0.15) -> None:
        # Defaults tuned on JoSAA (~514k rows, 20-trial random search).
        # For smaller datasets (e.g. CSAB ~47k rows) consider a smaller
        # architecture and/or run tune_mlp_hyperparams() on that dataset.
        all_years = sorted(df[COL_YEAR].unique())
        self.year_mean = float(np.mean(all_years))
        self.year_std  = max(float(np.std(all_years)), 1.0)

        # Build vocabulary encoders
        tier_vals = [_tier(i, e) for i, e in zip(df[COL_INSTITUTE], df[COL_EXAM_TYPE])]
        for vals, key in [
            (tier_vals,         "tier"),
            (df[COL_EXAM_TYPE], "exam_type"),
            (df[COL_QUOTA],     "quota"),
            (df[COL_SEAT_TYPE], "seat_type"),
            (df[COL_GENDER],    "gender"),
            (df[COL_INSTITUTE], "institute"),
            (df[COL_PROGRAM],   "programme"),
        ]:
            self.encoders[key] = _CatEncoder(vals)

        # Per-slot round stats (medians, abs deviations, round ratios) for prediction intervals
        all_log_medians: list[float] = []
        for key, grp in df.groupby(SLOT_COLS):
            r_stats: dict = {}
            for r, rgrp in grp.groupby(COL_ROUND):
                closes = rgrp[COL_CLOSE_RANK].values.astype(float)
                med = float(np.median(closes))
                r_stats[int(r)] = {
                    "median":   med,
                    "abs_devs": sorted(abs(c - med) for c in closes),
                }
                lm = float(np.log(max(med, 1.0)))
                self._log_median[(tuple(key), int(r))] = lm
                all_log_medians.append(lm)
            max_r = int(grp[COL_ROUND].max())
            r_stats["max_round"] = max_r
            r_stats["n_years"]   = int(grp[COL_YEAR].nunique())
            # Round ratios: mean(close_r / close_final) across all training years
            ratio_accum: dict[int, list] = {}
            for _yr, yr_grp in grp.groupby(COL_YEAR):
                yr_rounds = yr_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
                fin_r = int(yr_grp[COL_ROUND].max())
                fin_c = float(yr_rounds.get(fin_r, 0))
                if fin_c <= 0:
                    continue
                for _r, _c in yr_rounds.items():
                    ratio_accum.setdefault(int(_r), []).append(float(_c) / fin_c)
            r_stats["round_ratios"] = {r: float(np.mean(v)) for r, v in ratio_accum.items()}
            self.slot_stats[tuple(key)] = r_stats
        self._global_log_median = float(np.mean(all_log_medians)) if all_log_medians else 0.0

        # Validation split: random 15% of rows (stratified by year) for early
        # stopping.  A full-year holdout causes temporal mismatch between the
        # year used for model selection and the unseen test year.
        rng  = np.random.default_rng(42)
        mask = rng.random(len(df)) < 0.15
        train_df = df[~mask]
        val_df   = df[mask]

        def make_tensors(frame):
            cats, conts = self._featurize(frame)
            # Residual target: log(rank) - log(slot_round_median).
            # Net output = 0 → predict the historical median (mean-reversion default).
            log_rank = np.log(frame[COL_CLOSE_RANK].values.clip(1).astype(np.float32))
            log_med  = conts[:, 2]  # third continuous feature is log_median
            y = (log_rank - log_med).astype(np.float32)
            return self._tensors(cats, conts, y)

        tr_cats, tr_conts, tr_y = make_tensors(train_df)
        va_cats, va_conts, va_y = make_tensors(val_df)

        emb_specs = [(self.encoders[k].n, d) for k, d in zip(_CAT_KEYS, _EMB_DIMS)]
        self.net  = _Net(emb_specs, n_cont=3, hidden=hidden, dropout=dropout).to(self.device)

        opt     = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
        loss_fn = nn.MSELoss()
        N       = len(tr_y)

        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"  GlobalMLPModel  |  {N:,} train rows  val={len(val_df):,} (random 15%)"
              f"  device={self.device}  params={n_params:,}")

        best_val   = float("inf")
        best_state: dict | None = None
        no_improve = 0

        for ep in range(1, epochs + 1):
            self.net.train()
            perm    = torch.randperm(N, device=self.device)
            ep_loss = 0.0
            n_b     = 0
            for s in range(0, N, batch_size):
                idx  = perm[s : s + batch_size]
                loss = loss_fn(self.net(tr_cats[idx], tr_conts[idx]), tr_y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                n_b     += 1
            sched.step()

            if ep % 10 == 0 or ep == epochs:
                self.net.eval()
                with torch.no_grad():
                    vp    = self.net(va_cats, va_conts)
                    v_mse = loss_fn(vp, va_y).item()
                    # Residual → rank: rank = median * exp(residual)
                    log_med_va = va_conts[:, 2].cpu().numpy()
                    v_mae = float(np.mean(np.abs(
                        np.exp(vp.cpu().numpy() + log_med_va) -
                        np.exp(va_y.cpu().numpy() + log_med_va)
                    )))
                print(f"  ep {ep:3d}/{epochs}  train={ep_loss/n_b:.4f}"
                      f"  val_mse={v_mse:.4f}  val_MAE={v_mae:.0f}")

                if v_mse < best_val:
                    best_val   = v_mse
                    best_state = {k: v.clone().cpu() for k, v in self.net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= 5:  # patience = 5 × 10 = 50 epochs
                        print(f"  Early stop at epoch {ep}")
                        break

        if best_state:
            self.net.load_state_dict(best_state)
        self.best_val_mse = best_val
        # Move to CPU: pickle works, inference needs no GPU transfer
        self.net   = self.net.cpu().eval()
        self.device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """Batch predict closing ranks for all rows in df (float array)."""
        cats, conts = self._featurize(df)
        tc, tf = self._tensors(cats, conts)
        with torch.no_grad():
            residual = self.net(tc, tf).numpy()
        # rank = median * exp(residual);  conts[:,2] = log(median)
        log_rank = residual + conts[:, 2]
        return np.maximum(1.0, np.exp(log_rank))

    def predict_round(self, slot_key: tuple, round_no: int, year: int) -> float:
        row = dict(zip(SLOT_COLS, slot_key))
        row.update({COL_YEAR: year, COL_ROUND: round_no})
        return float(self.predict_df(pd.DataFrame([row]))[0])

    def predict_interval(self, slot_key: tuple, round_no: int, year: int,
                         coverage: float = 0.90) -> tuple[float, float]:
        pred  = self.predict_round(slot_key, round_no, year)
        stats = self.slot_stats.get(slot_key, {})
        devs  = stats.get(round_no, {}).get("abs_devs", [])
        if len(devs) >= 2:
            half_w = float(np.quantile(devs, coverage))
        else:
            half_w = 0.20 * pred
        return max(1.0, pred - half_w), pred + half_w

    def make_slot_adapter(self, slot_key: tuple) -> "_GlobalSlotAdapter":
        stats   = self.slot_stats.get(slot_key, {})
        max_r   = stats.get("max_round", 6)
        n_yrs   = stats.get("n_years",   0)
        medians = {r: v["median"]   for r, v in stats.items() if isinstance(r, int)}
        devs    = {r: v["abs_devs"] for r, v in stats.items() if isinstance(r, int)}
        return _GlobalSlotAdapter(self, slot_key, max_r, n_yrs, medians, devs)


class _GlobalSlotAdapter:
    """
    Makes GlobalMLPModel look like a SlotModel for predict.py.
    One lightweight adapter per slot key; just references, no data copy.
    """

    def __init__(self, model: GlobalMLPModel, slot_key: tuple,
                 max_round: int, n_years: int,
                 round_medians: dict, round_abs_deviations: dict):
        self._model               = model
        self._slot_key            = slot_key
        self.max_round            = max_round
        self.n_years              = n_years
        self.round_medians        = round_medians
        self.round_abs_deviations = round_abs_deviations

    def predict_round(self, round_no: int, year: int, w=None) -> float:
        return self._model.predict_round(self._slot_key, round_no, year)

    def predict_all_rounds(self, year: int, rounds: list[int],
                           w=None) -> dict[int, int]:
        return {r: int(round(self.predict_round(r, year))) for r in rounds}

    def predict_interval(self, round_no: int, year: int,
                         coverage: float = 0.90) -> tuple[float, float]:
        return self._model.predict_interval(self._slot_key, round_no, year, coverage)

    def get_round_ratios(self) -> dict[int, float]:
        return self._model.slot_stats.get(self._slot_key, {}).get("round_ratios", {})


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

_HP_SEARCH_SPACE: dict[str, list] = {
    "hidden":     [(256, 128, 64), (512, 256, 128), (128, 64, 32),
                   (512, 256, 128, 64), (256, 128), (384, 192, 96)],
    "dropout":    [0.1, 0.15, 0.2, 0.25, 0.3],
    "lr":         [3e-4, 5e-4, 1e-3, 2e-3],
    "batch_size": [2048, 4096, 8192],
}


def tune_mlp_hyperparams(
    df: "pd.DataFrame",
    n_trials: int = 20,
    n_epochs: int = 80,
    quiet: bool = False,
) -> dict:
    """
    Random search over MLP architecture and optimiser hyperparameters.

    Each trial trains a GlobalMLPModel for up to n_epochs epochs (with the
    same internal 15% validation split and early-stopping used by fit()), then
    records the best validation MSE.  The config that achieves the lowest
    val MSE across all trials is returned.

    Parameters
    ----------
    df        : full training DataFrame (same one you would pass to fit()).
    n_trials  : number of random configurations to try.
    n_epochs  : maximum epochs per trial (fewer than the final 200 to keep
                the search fast; early stopping usually cuts it shorter).
    quiet     : suppress per-trial output.

    Returns
    -------
    dict with keys: hidden, dropout, lr, batch_size
    """
    import random

    def log(*a, **kw):
        if not quiet:
            print(*a, **kw)

    rng = random.Random(42)
    best_mse: float = float("inf")
    best_cfg: dict  = {
        "hidden": (512, 256, 128), "dropout": 0.15, "lr": 1e-3, "batch_size": 2048
    }

    log(f"MLP hyperparameter search  |  {n_trials} trials  |  max_epochs={n_epochs}")
    log(f"Search space: {_HP_SEARCH_SPACE}")

    for trial in range(1, n_trials + 1):
        cfg = {k: rng.choice(v) for k, v in _HP_SEARCH_SPACE.items()}
        log(f"\n  Trial {trial}/{n_trials}: {cfg}")
        m = GlobalMLPModel()
        m.fit(
            df,
            epochs=n_epochs,
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            hidden=cfg["hidden"],
            dropout=cfg["dropout"],
        )
        log(f"    val_mse={m.best_val_mse:.6f}"
            + ("  ← best" if m.best_val_mse < best_mse else ""))
        if m.best_val_mse < best_mse:
            best_mse = m.best_val_mse
            best_cfg = cfg

    log(f"\nBest config (val_mse={best_mse:.6f}): {best_cfg}")
    return best_cfg
