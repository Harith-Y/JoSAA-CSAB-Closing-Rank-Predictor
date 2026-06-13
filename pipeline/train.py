"""
Train the ensemble model that predicts closing rank for every round.

Two signals per slot (institute x program x quota x seat_type x gender):

  1. Year-trend per round
       For each round r, fit LinearRegression(year -> closing_rank).
       Captures how the cutoff for *that specific round* has drifted year-over-year.

  2. Round-progression ratios
       For each year, record ratio[r] = close[r] / close[last_round_in_year].
       Average ratios across years to learn the typical R1->R2->...->Rfinal shape.
       At prediction time: predicted_close[r] = predicted_final_close x ratio[r].

Ensemble prediction for round r in target year Y:
       pred = w * direct_year_trend[r](Y)  +  (1-w) * (final_trend(Y) x ratio[r])
  where w = ENSEMBLE_WEIGHT (default 0.5).

Slots with < MIN_YEARS_FOR_TREND data points for a given round fall back to
the historical median for that round.
"""

import os
import pickle
from typing import TYPE_CHECKING
import torch  # must load before sklearn/numpy to avoid DLL conflicts on Windows
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from .mlp_model import GlobalMLPModel

from .config import (
    COL_YEAR, COL_ROUND, COL_INSTITUTE, COL_PROGRAM,
    COL_QUOTA, COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE,
    COL_CLOSE_RANK,
    MIN_YEARS_FOR_TREND, ENSEMBLE_WEIGHT,
    MODEL_DIR, MODEL_PATH, DEFAULT_TREND_MODEL,
)
from .loader import load

SLOT_COLS = [COL_INSTITUTE, COL_PROGRAM, COL_QUOTA,
             COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE]

# Supported trend models for the year-signal component
TREND_MODELS = ["ols", "theil_sen", "weighted_ols", "median",
                "ridge", "svr_linear", "svr_rbf", "ar1", "arp", "gp_rbf", "mlp",
                "mlp_ensemble"]

_ARP_MAX_P = 3   # maximum AR order tried by AIC selection

# Exponential decay rate for weighted_ols - weight = exp(λ x (year - min_year))
# λ=0.3 gives the most recent year ~10x the weight of the oldest year (9-year span).
_DECAY_LAMBDA = 0.3


class _ARpEstimator:
    """
    AR(p) mean-reversion estimator with AIC-based order selection.

    Fits  (C_t - mu) = phi_1*(C_{t-1}-mu) + ... + phi_p*(C_{t-p}-mu)
    by ridge-regularised OLS in demeaned space.  Order p is chosen from
    {1 ... _ARP_MAX_P} by AIC (penalises added lags).  A small ridge term
    (1e-6 * I) guards against near-singular design matrices when observations
    are nearly collinear.

    Prediction is always one step ahead from the last p observed closing ranks;
    the `year` argument is ignored for the same reason as _AR1Estimator.
    """
    def __init__(self, mu: float, phi: np.ndarray, last_closes: np.ndarray):
        self._mu   = mu
        self._phi  = phi          # shape (p,): coefficients for lags 1 … p
        self._last = last_closes  # shape (p,): [most_recent, …, p-th most recent]

    def predict(self, X) -> np.ndarray:
        pred = self._mu + float(np.dot(self._phi, self._last - self._mu))
        return np.full(len(X), pred)


class _AR1Estimator:
    """
    Explicit AR(1) mean-reversion estimator.

    Fits  C_t = mu + phi * (C_{t-1} - mu) + eps  on consecutive year pairs
    via OLS without intercept in demeaned space.  At prediction time the
    forecast is always one step ahead from the last observed closing rank:

        pred = mu + phi * (last_close - mu)

    The `year` argument passed to predict() is ignored intentionally; AR(1)
    is not a year-indexed function but a state-space update from the most
    recent observation.  phi is clamped to [-1, 1] to guarantee stationarity.
    """
    def __init__(self, mu: float, phi: float, last_close: float):
        self._mu   = mu
        self._phi  = phi
        self._last = last_close

    def predict(self, X) -> np.ndarray:
        pred = self._mu + self._phi * (self._last - self._mu)
        return np.full(len(X), pred)


class _FixedGPEstimator:
    """
    GP regression with a fixed RBF kernel; no hyperparameter optimisation.

    Operates entirely in StandardScaler-normalised space (X and y both have
    mean≈0, std≈1).  One np.linalg.solve call at fit time; O(n) predict.

    Posterior mean:  k(x*, X) @ (K + noise·I)⁻¹ @ y
    where k_ij = exp(-0.5·(xi-xj)²/ℓ²).

    With ℓ=1.0 (normalised) the kernel decays to ~0 roughly 2 standard
    deviations outside the training range, giving the same mean-reversion
    property as SVR RBF but with a proper Bayesian posterior.
    """
    def __init__(self, X_s: np.ndarray, y_s: np.ndarray,
                 length_scale: float = 1.0, noise: float = 0.1):
        x = X_s.ravel()
        diffs = x[:, None] - x[None, :]
        K = np.exp(-0.5 * diffs ** 2 / length_scale ** 2)
        self._alpha = np.linalg.solve(K + noise * np.eye(len(x)), y_s)
        self._x     = x
        self._ls    = length_scale

    def predict(self, X) -> np.ndarray:
        x_q = np.asarray(X).ravel()
        diffs  = x_q[:, None] - self._x[None, :]
        k_star = np.exp(-0.5 * diffs ** 2 / self._ls ** 2)
        return k_star @ self._alpha


class _ScaledEstimator:
    """
    Wraps a sklearn estimator with per-feature StandardScaler on X and y.

    SVR and Ridge work in normalized space; this wrapper handles the
    transform/inverse-transform so predict([[year]]) returns raw rank values.
    The RBF kernel has a particularly useful property: when the query year is
    outside the training range, the kernel weights decay to zero and the
    prediction reverts toward the training-target mean - approximating median
    behaviour naturally.
    """
    def __init__(self, estimator, scaler_X, scaler_y):
        self._est = estimator
        self._sx  = scaler_X
        self._sy  = scaler_y

    def predict(self, X):
        X_s = self._sx.transform(np.asarray(X).reshape(-1, 1))
        y_s = self._est.predict(X_s).reshape(-1, 1)
        return self._sy.inverse_transform(y_s).ravel()


class SlotModel:
    """
    Per-slot ensemble of year-trend models and round-progression ratios.
    """

    def __init__(self, trend_model: str = DEFAULT_TREND_MODEL,
                 normalize: bool = False):
        if trend_model not in TREND_MODELS:
            raise ValueError(f"trend_model must be one of {TREND_MODELS}")
        self.trend_model = trend_model
        self.normalize:   bool = normalize
        self._exam_type:  str  = ""   # set during fit() when normalize=True
        self._institute:  str  = ""
        self._quota:      str  = ""
        # {round_no: sklearn estimator}  - year -> close_rank for that round
        self.round_year_models: dict[int, object] = {}
        # {round_no: float}  - close[r] / close[last_round], averaged across years
        self.round_ratios:      dict[int, float] = {}
        # {round_no: float}  - fallback median closing rank per round
        self.round_medians:     dict[int, float] = {}
        self.max_round: int = 1
        self.n_years:   int = 0   # number of distinct years in training data

    def _fit_trend(self, years: np.ndarray, closes: np.ndarray):
        """Return a fitted sklearn estimator, or None (falls back to median)."""
        if len(years) < MIN_YEARS_FOR_TREND or self.trend_model == "median":
            return None
        # Sort by year so consecutive pairs and last_close are correct
        order  = np.argsort(years)
        years  = years[order]
        closes = closes[order]

        if self.trend_model == "ar1":
            mu = float(np.median(closes))
            demeaned = closes - mu
            x_lag = demeaned[:-1]   # C_{t-1} - mu
            y_lag = demeaned[1:]    # C_t     - mu
            # OLS without intercept: phi = (x' y) / (x' x)
            denom = float(np.dot(x_lag, x_lag))
            phi   = float(np.dot(x_lag, y_lag) / denom) if denom > 0 else 0.0
            phi   = float(np.clip(phi, -1.0, 1.0))
            return _AR1Estimator(mu, phi, float(closes[-1]))

        if self.trend_model == "arp":
            mu = float(np.median(closes))
            d  = closes - mu          # demeaned series, sorted by year
            n  = len(d)

            best_p, best_phi, best_aicc = 1, np.array([0.0]), float("inf")
            for p in range(1, min(_ARP_MAX_P, n - 1) + 1):
                n_pairs = n - p
                # AICc requires n_pairs - p - 1 >= 1, i.e. n_pairs >= p + 2
                if n_pairs < p + 2:
                    break
                # Design matrix: column j contains lag-(j+1) of demeaned series
                X = np.column_stack(
                    [d[p - j - 1 : n - j - 1] for j in range(p)]
                )
                y = d[p:]
                # lstsq handles near-singular design matrices via SVD truncation
                phi_p, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-10)
                # AICc = n*log(σ²) + 2k*n/(n-k-1)  (corrected AIC for small n)
                resid  = y - X @ phi_p
                sigma2 = max(float(np.var(resid, ddof=0)), 1e-10)
                aicc   = (n_pairs * np.log(sigma2)
                          + 2 * p * n_pairs / (n_pairs - p - 1))
                if aicc < best_aicc:
                    best_aicc, best_p, best_phi = aicc, p, phi_p

            # Store the last best_p closing ranks: [most_recent … p-th most recent]
            last_closes = closes[-best_p:][::-1].copy()
            return _ARpEstimator(mu, best_phi, last_closes)

        Y = years.reshape(-1, 1)
        if self.trend_model == "ols":
            return LinearRegression().fit(Y, closes)
        if self.trend_model == "theil_sen":
            from sklearn.linear_model import TheilSenRegressor
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                return TheilSenRegressor(random_state=0, max_iter=1000).fit(Y, closes)
        if self.trend_model == "weighted_ols":
            w = np.exp(_DECAY_LAMBDA * (years - years.min()))
            return LinearRegression().fit(Y, closes, sample_weight=w)

        # Models that need normalisation (SVR and Ridge operate in standardised space)
        from sklearn.preprocessing import StandardScaler
        sx = StandardScaler()
        sy = StandardScaler()
        X_s = sx.fit_transform(Y)
        y_s = sy.fit_transform(closes.reshape(-1, 1)).ravel()

        if self.trend_model == "ridge":
            from sklearn.linear_model import Ridge
            # alpha=1.0 in normalised space gives moderate slope shrinkage
            est = Ridge(alpha=1.0).fit(X_s, y_s)
        elif self.trend_model == "svr_linear":
            from sklearn.svm import SVR
            # Linear kernel SVR - regularised linear fit, C=1 is sklearn default
            est = SVR(kernel="linear", C=1.0, epsilon=0.1).fit(X_s, y_s)
        elif self.trend_model == "svr_rbf":
            from sklearn.svm import SVR
            # RBF kernel SVR - key property: predictions outside the training
            # year range revert toward the training-target mean (≈ median).
            est = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale").fit(X_s, y_s)
        elif self.trend_model == "gp_rbf":
            # Fixed-hyperparameter RBF GP in normalised space.
            # No kernel optimisation; one np.linalg.solve per slot.
            # Outside the training range the posterior mean reverts toward
            # 0 (≈ median rank), matching SVR RBF's mean-reversion property.
            est = _FixedGPEstimator(X_s, y_s)
        else:
            raise ValueError(f"Unknown trend_model: {self.trend_model!r}")

        return _ScaledEstimator(est, sx, sy)

    def _denorm(self, val: float, year: int) -> float:
        """Convert percentile back to absolute rank; no-op when normalize=False."""
        if not self.normalize:
            return val
        from .pool_sizes import get_pool_size_for_slot
        return val * get_pool_size_for_slot(self._exam_type, self._quota, self._institute, year)

    def fit(self, slot_df: pd.DataFrame) -> None:
        """
        slot_df: all rows for one slot, all years, all rounds.
        When normalize=True ranks are divided by pool_size[year] before fitting,
        storing everything in fractional-percentile space.  predict_round() and
        predict_interval() call _denorm() to convert outputs back to absolute ranks.
        """
        if self.normalize:
            from .pool_sizes import get_pool_size_for_slot
            self._exam_type = str(slot_df[COL_EXAM_TYPE].iloc[0])
            self._institute = str(slot_df[COL_INSTITUTE].iloc[0])
            self._quota     = str(slot_df[COL_QUOTA].iloc[0])
            slot_df = slot_df.copy()
            pool_sizes = slot_df[COL_YEAR].map(
                lambda y: get_pool_size_for_slot(
                    self._exam_type, self._quota, self._institute, int(y))
            )
            slot_df[COL_CLOSE_RANK] = slot_df[COL_CLOSE_RANK] / pool_sizes

        self.max_round = int(slot_df[COL_ROUND].max())
        self.n_years   = int(slot_df[COL_YEAR].nunique())

        # Per-round year-trend models
        for r, grp in slot_df.groupby(COL_ROUND):
            r = int(r)
            closes = grp[COL_CLOSE_RANK].values.astype(float)
            years  = grp[COL_YEAR].values.astype(float)
            self.round_medians[r] = float(np.median(closes))
            m = self._fit_trend(years, closes)
            if m is not None:
                self.round_year_models[r] = m

        # Round-progression ratios
        # For each year, compute ratio[r] = close[r] / close[max_round_in_year].
        # Only use years that have *both* the round r and the max round.
        ratio_accum: dict[int, list[float]] = {}
        for year, year_grp in slot_df.groupby(COL_YEAR):
            year_rounds = year_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
            final_r = int(year_grp[COL_ROUND].max())
            final_close = year_rounds.get(final_r)
            if not final_close or final_close == 0:
                continue
            for r, close in year_rounds.items():
                r = int(r)
                ratio_accum.setdefault(r, []).append(close / final_close)

        self.round_ratios = {r: float(np.mean(v)) for r, v in ratio_accum.items()}

        # Absolute residuals from the year-trend model per round.
        # Using model residuals (not deviations from median) prevents trend-inflated
        # intervals: a slot trending down from 12k→5k over 10 years would otherwise
        # produce half-widths of ~4k even though year-to-year noise is only ~500.
        # Falls back to median-based deviations when no trend model is available for
        # a round, or when normalize=True (predictions are in fractional space there).
        self.round_abs_deviations: dict[int, list[float]] = {}
        for r, grp in slot_df.groupby(COL_ROUND):
            r = int(r)
            closes = grp[COL_CLOSE_RANK].values.astype(float)
            if not self.normalize and r in self.round_year_models:
                preds = np.array([
                    float(self.round_year_models[r].predict([[yr]])[0])
                    for yr in grp[COL_YEAR].values
                ])
                residuals = np.abs(closes - preds)
            else:
                med = float(np.median(closes))
                residuals = np.abs(closes - med)
            self.round_abs_deviations[r] = sorted(float(x) for x in residuals)

    def predict_interval(self, round_no: int, year: int,
                         coverage: float = 0.90) -> tuple[float, float]:
        """
        Prediction interval at the requested coverage level.

        Uses the sorted absolute deviations of historical closing ranks from
        their per-round median as a non-parametric proxy for future prediction
        uncertainty.  The coverage quantile of those deviations is the
        half-width: lower = pred - half_width, upper = pred + half_width.

        Note: this is an in-sample (optimistic) calibration; the true
        leave-one-out coverage may be lower, especially for volatile slots.
        Falls back to ±20 % of the prediction when fewer than 2 observations
        are available.
        """
        pred = self.predict_round(round_no, year)
        devs = self.round_abs_deviations.get(round_no, [])
        if len(devs) >= 2:
            # np.quantile interpolates so adjacent coverage levels always differ.
            half_w = self._denorm(float(np.quantile(devs, coverage)), year)
        else:
            half_w = 0.20 * pred
        return max(1.0, pred - half_w), pred + half_w

    def predict_round(self, round_no: int, year: int,
                      w: float | None = None) -> float:
        """
        Ensemble prediction for a single round.
        w: override for ENSEMBLE_WEIGHT; pass explicitly during weight tuning.
        """
        ew = ENSEMBLE_WEIGHT if w is None else w

        # Signal 1: direct year trend for this round
        if round_no in self.round_year_models:
            direct = float(self.round_year_models[round_no].predict([[year]])[0])
        else:
            direct = self.round_medians.get(round_no,
                     self.round_medians.get(self.max_round, 0))

        # Signal 2: scale from predicted final-round close
        if self.max_round in self.round_year_models:
            pred_final = float(
                self.round_year_models[self.max_round].predict([[year]])[0]
            )
        else:
            pred_final = self.round_medians.get(self.max_round, direct)

        ratio = self.round_ratios.get(round_no,
                self.round_ratios.get(self.max_round, 1.0))
        via_ratio = pred_final * ratio

        pred = ew * direct + (1 - ew) * via_ratio
        return max(1.0, self._denorm(pred, year))

    def predict_all_rounds(self, year: int, rounds: list[int],
                           w: float | None = None) -> dict[int, int]:
        """Return {round_no: predicted_close_rank} for each requested round."""
        return {r: int(round(self.predict_round(r, year, w=w))) for r in rounds}

    def get_round_ratios(self) -> dict[int, float]:
        """Return {round_no: mean(close_r / close_final)} across all training years."""
        return self.round_ratios.copy()


class _GPMLPSlotAdapter:
    """
    Makes GPMLPEnsemble look like a SlotModel / _GlobalSlotAdapter for predict.py.
    """

    def __init__(self, model: "GPMLPEnsemble", slot_key: tuple,
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
        gp_m = self._model._gp_slots.get(self._slot_key)
        if gp_m is not None:
            return gp_m.round_ratios.copy()
        max_med = self.round_medians.get(self.max_round, 0.0)
        if not max_med:
            return {}
        return {r: med / max_med for r, med in self.round_medians.items()}


class GPMLPEnsemble:
    """
    GP-MLP routing ensemble.

    For slots with >= MIN_YEARS_FOR_TREND years of history a per-slot GP RBF
    model is available.  At inference time the GP and MLP predictions are
    blended:

        pred = blend_alpha * GP_pred + (1 - blend_alpha) * MLP_pred

    For cold-start slots (not seen during training, or fewer than
    MIN_YEARS_FOR_TREND historical years) only the global MLP is used.

    Setting blend_alpha=1.0 gives pure-GP routing for known slots (MLP as
    cold-start fallback); blend_alpha=0.0 degenerates to the plain MLP.
    The default of 0.5 averages both signals for known slots.
    """

    def __init__(self, blend_alpha: float = 0.5):
        self.blend_alpha = blend_alpha
        self._gp_slots:  dict[tuple, SlotModel] = {}
        self._mlp:       "GlobalMLPModel | None" = None
        self.slot_stats: dict[tuple, dict] = {}

    def fit(self, df: pd.DataFrame,
            mlp_hidden: tuple[int, ...] = (256, 128, 64),
            mlp_dropout: float = 0.2,
            mlp_lr: float = 1e-3,
            mlp_batch_size: int = 4096,
            mlp_epochs: int = 200) -> None:
        from .mlp_model import GlobalMLPModel

        # Train GP RBF per-slot for slots with enough history
        n_gp = 0
        for key, grp in df.groupby(SLOT_COLS):
            if int(grp[COL_YEAR].nunique()) >= MIN_YEARS_FOR_TREND:
                m = SlotModel(trend_model="gp_rbf")
                m.fit(grp)
                self._gp_slots[tuple(key)] = m
                n_gp += 1
        print(f"  GPMLPEnsemble  |  GP slots={n_gp:,}")

        # Train global MLP on the full dataset
        self._mlp = GlobalMLPModel()
        self._mlp.fit(df, epochs=mlp_epochs, lr=mlp_lr,
                      batch_size=mlp_batch_size,
                      hidden=mlp_hidden, dropout=mlp_dropout)
        self.slot_stats = self._mlp.slot_stats

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        mlp_preds = self._mlp.predict_df(df)
        if not self._gp_slots or self.blend_alpha == 0.0:
            return mlp_preds

        preds    = mlp_preds.copy()
        df_r     = df.reset_index(drop=True)
        inst_arr = df_r[COL_INSTITUTE].values
        prog_arr = df_r[COL_PROGRAM].values
        quot_arr = df_r[COL_QUOTA].values
        st_arr   = df_r[COL_SEAT_TYPE].values
        gen_arr  = df_r[COL_GENDER].values
        et_arr   = df_r[COL_EXAM_TYPE].values
        rnd_arr  = df_r[COL_ROUND].values.astype(int)
        yr_arr   = df_r[COL_YEAR].values.astype(int)

        for i in range(len(df_r)):
            sk = (inst_arr[i], prog_arr[i], quot_arr[i],
                  st_arr[i],   gen_arr[i],  et_arr[i])
            gp_m = self._gp_slots.get(sk)
            if gp_m is not None:
                gp_pred = gp_m.predict_round(rnd_arr[i], yr_arr[i])
                preds[i] = (self.blend_alpha * gp_pred
                            + (1.0 - self.blend_alpha) * mlp_preds[i])
        return preds

    def predict_df_components(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (gp_preds, mlp_preds, gp_mask).

        gp_mask[i] is True where a per-slot GP model was available.
        For cold-start rows gp_preds[i] == mlp_preds[i].
        Caller can blend: alpha * gp_preds + (1 - alpha) * mlp_preds.
        """
        mlp_preds = self._mlp.predict_df(df)
        gp_preds  = mlp_preds.copy()
        gp_mask   = np.zeros(len(df), dtype=bool)

        if not self._gp_slots:
            return gp_preds, mlp_preds, gp_mask

        df_r     = df.reset_index(drop=True)
        inst_arr = df_r[COL_INSTITUTE].values
        prog_arr = df_r[COL_PROGRAM].values
        quot_arr = df_r[COL_QUOTA].values
        st_arr   = df_r[COL_SEAT_TYPE].values
        gen_arr  = df_r[COL_GENDER].values
        et_arr   = df_r[COL_EXAM_TYPE].values
        rnd_arr  = df_r[COL_ROUND].values.astype(int)
        yr_arr   = df_r[COL_YEAR].values.astype(int)

        for i in range(len(df_r)):
            sk = (inst_arr[i], prog_arr[i], quot_arr[i],
                  st_arr[i],   gen_arr[i],  et_arr[i])
            gp_m = self._gp_slots.get(sk)
            if gp_m is not None:
                gp_preds[i] = gp_m.predict_round(rnd_arr[i], yr_arr[i])
                gp_mask[i]  = True

        return gp_preds, mlp_preds, gp_mask

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

    def make_slot_adapter(self, slot_key: tuple) -> _GPMLPSlotAdapter:
        stats   = self.slot_stats.get(slot_key, {})
        max_r   = stats.get("max_round", 6)
        n_yrs   = stats.get("n_years",   0)
        medians = {r: v["median"]   for r, v in stats.items() if isinstance(r, int)}
        devs    = {r: v["abs_devs"] for r, v in stats.items() if isinstance(r, int)}
        return _GPMLPSlotAdapter(self, slot_key, max_r, n_yrs, medians, devs)


def train(csv_path: str, model_path: str = MODEL_PATH,
          trend_model: str = DEFAULT_TREND_MODEL,
          normalize: bool = False,
          tune_mlp: bool = False,
          mlp_hidden: tuple | None = None,
          mlp_dropout: float | None = None,
          blend_alpha: float = 0.5) -> dict:
    """
    tune_mlp : when True and trend_model in ("mlp", "mlp_ensemble"), run a
               random hyperparameter search before final training and use the
               best found config for the final model.
    """
    df = load(csv_path)
    print(f"Training on {len(df):,} rows  |  "
          f"{df[COL_YEAR].nunique()} years  |  "
          f"{df[COL_ROUND].nunique()} rounds  |  "
          f"trend={trend_model}  |  normalize={normalize}"
          + ("  |  tune_mlp=True" if tune_mlp else ""))

    mlp_cfg: dict = {}
    if trend_model in ("mlp", "mlp_ensemble"):
        from .mlp_model import GlobalMLPModel, tune_mlp_hyperparams
        if tune_mlp:
            print("\nRunning MLP hyperparameter search...")
            mlp_cfg = tune_mlp_hyperparams(df)
            print(f"Using tuned config: {mlp_cfg}\n")

    if trend_model == "mlp":
        gm = GlobalMLPModel()
        gm.fit(df, **mlp_cfg)
        slots = {key: gm.make_slot_adapter(key) for key in gm.slot_stats}
        model = {
            "slots":        slots,
            "slot_cols":    SLOT_COLS,
            "trend_model":  "mlp",
            "global_model": gm,
            "normalize":    False,
        }
        print(f"Trained GlobalMLPModel  ({len(slots):,} slot adapters) -> {model_path}")
    elif trend_model == "mlp_ensemble":
        gm = GPMLPEnsemble(blend_alpha=blend_alpha)
        gm.fit(df,
               mlp_hidden=mlp_cfg.get("hidden",     mlp_hidden  or (256, 128, 64)),
               mlp_dropout=mlp_cfg.get("dropout",   mlp_dropout or 0.2),
               mlp_lr=mlp_cfg.get("lr",             1e-3),
               mlp_batch_size=mlp_cfg.get("batch_size", 4096))
        slots = {key: gm.make_slot_adapter(key) for key in gm.slot_stats}
        model = {
            "slots":        slots,
            "slot_cols":    SLOT_COLS,
            "trend_model":  "mlp_ensemble",
            "global_model": gm,
            "normalize":    False,
        }
        print(f"Trained GPMLPEnsemble  ({len(slots):,} slot adapters) -> {model_path}")
    else:
        slots: dict[tuple, SlotModel] = {}
        for key, grp in df.groupby(SLOT_COLS):
            m = SlotModel(trend_model=trend_model, normalize=normalize)
            m.fit(grp)
            slots[tuple(key)] = m
        model = {
            "slots":      slots,
            "slot_cols":  SLOT_COLS,
            "trend_model": trend_model,
            "normalize":  normalize,
        }
        print(f"Trained {len(slots):,} slot models -> {model_path}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    train(csv_path)
