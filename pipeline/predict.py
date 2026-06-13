"""
Prediction engine.

Given a student profile, returns a ranked DataFrame of matching slots
with predicted closing ranks for every round (R1 … R6).

Output columns:
    Institute | Academic Program Name | Quota | Seat Type | Gender |
    R1 | R2 | R3 | R4 | R5 | R6 | Final Pred | Lower | Upper | Years | Seats | Category

Category (based on prediction interval vs student rank):
    safe   → rank ≤ lower bound of interval  (comfortably within range)
    match  → lower < rank ≤ Final Pred
    reach  → Final Pred < rank ≤ upper bound

Lower / Upper: per-slot prediction interval at the requested coverage level
    (default 90 %), derived from historical closing-rank variability.
    Slots with fewer than 2 data points fall back to ±20 % of Final Pred.

Seats column: current-year seat count from seat_matrix.csv (if available).
"""

import os
import pickle
import numpy as np
import pandas as pd
from .config import MODEL_PATH, PREDICT_YEAR, ALL_ROUNDS, CURRENT_ROUND_DATA

SEAT_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "seat_matrix.csv")


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_seat_matrix(path: str = SEAT_MATRIX_PATH) -> dict:
    """
    Load seat_matrix.csv and return a lookup dict:
        (institute, program, quota, seat_type, gender) -> seats (int)

    The seat matrix uses granular state names for NIT HS/OS quotas
    (e.g. "ANDHRA PRADESH" for home-state, "Other than ANDHRA PRADESH" for
    other-state). We aggregate these back to the coarse pipeline codes:
        AI              → "AI"
        <state name>    → "HS"   (home-state rows, not "Other than ...")
        Other than ...  → "OS"   (other-state rows)
        GO / JK / LA   → kept as-is
    Returns an empty dict if the file doesn't exist.
    """
    if not os.path.exists(path):
        return {}

    SPECIAL = {"AI", "GO", "JK", "LA"}

    df = pd.read_csv(path)
    df["_quota_norm"] = df["Quota"].apply(_coarse_quota)

    agg = (
        df.groupby(["Institute", "Program", "_quota_norm", "Seat Type", "Gender"],
                   sort=False)["Seats"]
        .sum()
        .reset_index()
    )

    return {
        (
            str(r["Institute"]).strip(),
            str(r["Program"]).strip(),
            str(r["_quota_norm"]).strip(),
            str(r["Seat Type"]).strip(),
            str(r["Gender"]).strip(),
        ): int(r["Seats"])
        for _, r in agg.iterrows()
    }


def load_actuals(path: str) -> dict:
    """
    Load actual closing ranks from a single round CSV.
    Returns {(institute, program, quota, seat_type, gender): closing_rank}.
    """
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    df["Closing Rank"] = pd.to_numeric(
        df["Closing Rank"].str.lstrip("P").str.strip(), errors="coerce"
    )
    df = df.dropna(subset=["Closing Rank"])
    result = {}
    for _, row in df.iterrows():
        key = (
            str(row["Institute"]).strip(),
            str(row["Academic Program Name"]).strip(),
            str(row["Quota"]).strip(),
            str(row["Seat Type"]).strip(),
            str(row["Gender"]).strip(),
        )
        result[key] = int(row["Closing Rank"])
    return result


def load_all_actuals(
    round_data: dict[int, str] | None = None,
) -> dict[int, dict]:
    """
    Load actuals for all configured rounds.

    Returns {round_num: {slot_key: closing_rank}}.
    Only rounds whose CSV file exists are included.
    """
    if round_data is None:
        round_data = CURRENT_ROUND_DATA
    return {r: data for r, path in round_data.items() if (data := load_actuals(path))}


def load_round1_actuals(path: str | None = None) -> dict:
    """Backward-compat wrapper: load R1 actuals from CURRENT_ROUND_DATA or a given path."""
    if path is None:
        path = CURRENT_ROUND_DATA.get(1, "")
    return load_actuals(path)


def evaluate_round(
    model: dict,
    actuals: dict,
    round_num: int,
    year: int = PREDICT_YEAR,
) -> pd.DataFrame:
    """
    Compare model's predictions for round_num against actual closing ranks.

    Returns DataFrame with Predicted_R{n}, Actual_R{n}, Error, Abs_Error, Pct_Error,
    sorted by Abs_Error descending.
    """
    slots = model["slots"]
    pred_col = f"Predicted_R{round_num}"
    actual_col = f"Actual_R{round_num}"
    rows = []
    for key, slot_model in slots.items():
        inst, prog, q, st, g, et = key
        actual = actuals.get((inst, prog, q, st, g))
        if actual is None:
            continue
        w = model.get("ensemble_weight")
        pred = slot_model.predict_round(round_num, year, w=w)
        error = int(round(pred)) - actual
        rows.append({
            "Institute":             inst,
            "Academic Program Name": prog,
            "Quota":                 q,
            "Seat Type":             st,
            "Gender":                g,
            "Exam Type":             et,
            pred_col:                int(round(pred)),
            actual_col:              actual,
            "Error":                 error,
            "Abs_Error":             abs(error),
            "Pct_Error":             round(100.0 * abs(error) / max(actual, 1), 1),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.sort_values("Abs_Error", ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def evaluate_round1(model: dict, round1_actuals: dict, year: int = PREDICT_YEAR) -> pd.DataFrame:
    """Backward-compat wrapper for evaluate_round(... round_num=1)."""
    return evaluate_round(model, round1_actuals, 1, year)


def _anchor_with_actuals(
    slot_model,
    round_preds: dict[int, int],
    actuals_by_round: dict[int, int],
    rounds: list[int],
) -> dict[int, int]:
    """
    Anchor all-round predictions using a weighted aggregate of observed actuals.

    For each observed round r with actual closing rank C*_r, independently estimate
    the final-round closing rank:
        F_r = C*_r / ρ̄_{s,r}
    Then combine with weights proportional to round number (later rounds are closer
    to final and have lower ratio variance, so r is a reasonable reliability proxy):
        F̂ = Σ(r × F_r) / Σ(r)
        anchored_Rk = F̂ × ρ̄_{s,k}

    With R1 only  → 100% R1 estimate.
    With R1 + R2  → R1 gets 33%, R2 gets 67%.
    With R1+R2+R3 → 17% / 33% / 50%.
    """
    ratios: dict[int, float] = {}
    if hasattr(slot_model, "get_round_ratios"):
        ratios = slot_model.get_round_ratios()

    if ratios:
        total_w = 0.0
        weighted_final = 0.0
        for r_obs, actual in actuals_by_round.items():
            r_ratio = ratios.get(r_obs)
            if r_ratio and r_ratio > 0:
                weighted_final += r_obs * (actual / r_ratio)
                total_w += r_obs
        if total_w > 0:
            estimated_final = weighted_final / total_w
            max_r = slot_model.max_round
            return {
                r: int(round(max(1.0, estimated_final * ratios.get(r, ratios.get(max_r, 1.0)))))
                for r in rounds
            }

    # Fallback: proportional scale from the latest observed round
    latest_r = max(actuals_by_round)
    model_pred = round_preds.get(latest_r)
    if not model_pred:
        return round_preds
    scale = actuals_by_round[latest_r] / model_pred
    return {r: int(round(max(1.0, p * scale))) for r, p in round_preds.items()}


def _coarse_quota(raw: str) -> str:
    """Map fine-grained seat-matrix quota strings to pipeline quota codes."""
    s = str(raw).strip()
    upper = s.upper()
    if upper in {"AI", "GO", "JK", "LA"}:
        return upper
    if s.lower().startswith("other than"):
        return "OS"
    # Remaining rows are state-specific home-state quotas → HS
    return "HS"


def predict(
    rank:            int,
    exam_type:       str,       # "advanced" | "mains"
    quota:           str,       # "AI" | "HS" | "OS" | ...
    seat_type:       str,       # "OPEN" | "OBC-NCL" | "SC" | "ST" | "EWS" | ...
    gender:          str,       # "Gender-Neutral" | "Female-only (including Supernumerary)"
    model:           dict | None = None,
    year:            int = PREDICT_YEAR,
    rounds:          list[int] = ALL_ROUNDS,
    include_reach:   bool = True,
    safe_threshold:  float = 0.80,   # fallback when interval unavailable
    reach_threshold: float = 1.20,   # fallback when interval unavailable
    seat_matrix:     dict | None = None,  # (inst,prog,quota,st,gender) -> seats
    coverage:        float = 0.90,   # prediction interval coverage level
    actual_rounds:   dict[int, dict] | None = None,  # {round_num: {slot_key: closing_rank}}
) -> pd.DataFrame:
    if model is None:
        model = load_model()

    # Load seat matrix lazily if not provided
    if seat_matrix is None:
        seat_matrix = load_seat_matrix()

    slots = model["slots"]

    results = []

    for key, slot_model in slots.items():
        inst, prog, q, st, g, et = key

        if et != exam_type or q != quota or st != seat_type or g != gender:
            continue

        # Predict all rounds (use per-model tuned weight if available)
        w = model.get("ensemble_weight")
        round_preds = slot_model.predict_all_rounds(year, rounds, w=w)

        # Multi-round anchor: collect observed actuals for this slot across all
        # available rounds, then use weighted-aggregate anchoring.
        # The interval half-width is preserved (uncertainty unchanged); only
        # the centre shifts to match the weighted estimate of the final close.
        anchored = False
        if actual_rounds:
            slot_actuals = {
                r: v
                for r, r_data in actual_rounds.items()
                if (v := r_data.get((inst, prog, q, st, g))) is not None
            }
            if slot_actuals:
                round_preds = _anchor_with_actuals(slot_model, round_preds, slot_actuals, rounds)
                anchored = True

        # Final round = highest round this slot was seen in *that is also in
        # the requested rounds list*.  Some old data has round 7 (JOSAA special
        # rounds pre-2018); predict_all_rounds only covers rounds 1-6, so
        # max_round=7 would leave round_preds.get(7) as None and the `or`
        # fallback would pick round 6. But predict_interval would still be
        # called with round 7, giving a mismatched interval center.
        final_r = slot_model.max_round
        if round_preds.get(final_r):
            interval_r = final_r
            pred_final = round_preds[final_r]
        else:
            interval_r = max(round_preds)
            pred_final = round_preds[interval_r]

        # Prediction interval for the final round.
        # When anchored, preserve the model's half-width but re-centre on the
        # anchored prediction (the level shift is real; uncertainty is not).
        has_intervals = hasattr(slot_model, "round_abs_deviations")
        if has_intervals:
            lower_raw, upper_raw = slot_model.predict_interval(interval_r, year, coverage)
            if anchored:
                half_w = (upper_raw - lower_raw) / 2.0
                lower  = max(1.0, pred_final - half_w)
                upper  = pred_final + half_w
            else:
                lower, upper = lower_raw, upper_raw
        else:
            # Old model pkl without interval support, fall back to fixed thresholds
            lower = safe_threshold * pred_final
            upper = reach_threshold * pred_final

        if rank <= lower:
            category = "safe"
        elif rank <= pred_final:
            category = "match"
        elif rank <= upper and include_reach:
            category = "reach"
        else:
            continue

        seats = seat_matrix.get((inst, prog, q, st, g))

        row = {
            "Institute":             inst,
            "Academic Program Name": prog,
            "Quota":                 q,
            "Seat Type":             st,
            "Gender":                g,
        }
        for r in rounds:
            row[f"R{r}"] = round_preds.get(r, "-")

        row["Final Pred"] = pred_final
        row["Lower"]      = int(round(lower))
        row["Upper"]      = int(round(upper))
        row["Years"]      = slot_model.n_years
        row["Seats"]      = seats
        row["Anchored"]   = anchored
        row["Category"]   = category
        results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    cat_order = {"safe": 0, "match": 1, "reach": 2}
    out["_order"] = out["Category"].map(cat_order)
    out.sort_values(["_order", "Final Pred"], inplace=True)
    out.drop(columns=["_order"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
