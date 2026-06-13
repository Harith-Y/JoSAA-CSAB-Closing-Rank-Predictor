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
from .config import MODEL_PATH, PREDICT_YEAR, ALL_ROUNDS

SEAT_MATRIX_PATH  = os.path.join(os.path.dirname(__file__), "..", "seat_matrix.csv")
ROUND1_2026_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "Round1-2026.csv")


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


def load_round1_actuals(path: str = ROUND1_2026_PATH) -> dict:
    """
    Load actual Round 1 2026 closing ranks.
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


def evaluate_round1(
    model: dict,
    round1_actuals: dict,
    year: int = PREDICT_YEAR,
) -> pd.DataFrame:
    """
    Compare model's R1 predictions against actual Round 1 closing ranks.

    Returns a DataFrame with columns:
        Institute | Academic Program Name | Quota | Seat Type | Gender |
        Predicted_R1 | Actual_R1 | Error | Abs_Error | Pct_Error
    sorted by Abs_Error descending.
    """
    slots = model["slots"]
    rows = []
    for key, slot_model in slots.items():
        inst, prog, q, st, g, et = key
        actual = round1_actuals.get((inst, prog, q, st, g))
        if actual is None:
            continue
        w = model.get("ensemble_weight")
        pred_r1 = slot_model.predict_round(1, year, w=w)
        error = int(round(pred_r1)) - actual
        rows.append({
            "Institute":             inst,
            "Academic Program Name": prog,
            "Quota":                 q,
            "Seat Type":             st,
            "Gender":                g,
            "Exam Type":             et,
            "Predicted_R1":          int(round(pred_r1)),
            "Actual_R1":             actual,
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


def _anchor_with_round1(
    slot_model,
    round_preds: dict[int, int],
    actual_r1: int,
    rounds: list[int],
) -> dict[int, int]:
    """
    Adjust per-round predictions using actual Round 1 as an anchor.

    Uses historical round ratios:  anchored_r[k] = actual_r1 * (ratio[k] / ratio[1])
    Falls back to simple proportional scaling if ratio[1] is unavailable.
    """
    ratios = {}
    if hasattr(slot_model, "get_round_ratios"):
        ratios = slot_model.get_round_ratios()

    r1_ratio = ratios.get(1)

    if r1_ratio and r1_ratio > 0:
        # Anchor via historical round ratios: estimated_final = actual_r1 / ratio[1]
        estimated_final = actual_r1 / r1_ratio
        anchored = {}
        max_r = slot_model.max_round
        for r in rounds:
            rk_ratio = ratios.get(r, ratios.get(max_r, 1.0))
            anchored[r] = int(round(max(1.0, estimated_final * rk_ratio)))
        return anchored

    # Fallback: scale all model predictions by (actual_r1 / model_pred_r1)
    model_r1 = round_preds.get(1)
    if not model_r1:
        return round_preds
    scale = actual_r1 / model_r1
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
    round1_actuals:  dict | None = None,  # (inst,prog,quota,st,gender) -> actual R1 rank
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

        # R1 anchor: if actual 2026 Round 1 data is available for this slot,
        # replace all round predictions with ratio-anchored estimates.
        # The interval half-width is preserved from the model (uncertainty is
        # the same); only the center shifts to match the actual R1 data.
        anchored = False
        actual_r1 = None
        if round1_actuals and 1 in rounds:
            actual_r1 = round1_actuals.get((inst, prog, q, st, g))
        if actual_r1:
            round_preds = _anchor_with_round1(slot_model, round_preds, actual_r1, rounds)
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
