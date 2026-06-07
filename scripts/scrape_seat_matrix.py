"""
JoSAA Seat Matrix Scraper
Saves to seat_matrix.csv with columns:
    Year, Institute, Program, Quota, Seat Type, Gender, Seats

URL: https://josaa.admissions.nic.in/applicant/seatmatrix/seatmatrixinfo.aspx

The GridView1 table has rows in groups of 3 per program x quota:
    Row A (Gender-Neutral): Institute | Program | Quota | "Gender-Neutral" | <10 seat-type counts> | Total | ...
    Row B: rowspan continuation cells (skipped)
    Row C (Female-only): "Female-only..." | <10 seat-type counts> | Total | ...

Seat-type column order in the table (indices 4..13 on row A, 1..10 on row C):
    OPEN | OPEN-PwD | GEN-EWS | GEN-EWS-PwD | SC | SC-PwD | ST | ST-PwD | OBC-NCL | OBC-NCL-PwD

Usage:
    python scripts/scrape_seat_matrix.py [--year YYYY]
"""

import argparse
import csv
import os
import time

from playwright.sync_api import sync_playwright

URL        = "https://josaa.admissions.nic.in/applicant/seatmatrix/seatmatrixinfo.aspx"
OUTPUT     = "data/seat_matrix.csv"
SUBMIT_BTN = "#ctl00_ContentPlaceHolder1_btnSubmit"
TABLE_ID   = "#GridView1"

# Seat-type columns in order of appearance in the table
# → mapped to the names used in the prediction pipeline
SEAT_TYPE_MAP = {
    "OPEN":         "OPEN",
    "OPEN-PwD":     "OPEN (PwD)",
    "GEN-EWS":      "EWS",
    "GEN-EWS-PwD":  "EWS (PwD)",
    "SC":           "SC",
    "SC-PwD":       "SC (PwD)",
    "ST":           "ST",
    "ST-PwD":       "ST (PwD)",
    "OBC-NCL":      "OBC-NCL",
    "OBC-NCL-PwD":  "OBC-NCL (PwD)",
}
ST_PIPELINE_NAMES = list(SEAT_TYPE_MAP.values())   # ordered

QUOTA_MAP = {
    "all india":   "AI",
    "home state":  "HS",
    "other state": "OS",
    "goa":         "GO",
    "jammu & kashmir": "JK",
    "jammu and kashmir": "JK",
    "ladakh":      "LA",
}

FIELDNAMES = ["Year", "Institute", "Program", "Quota", "Seat Type", "Gender", "Seats"]


def norm_quota(raw: str) -> str:
    r = raw.strip().lower()
    for k, v in QUOTA_MAP.items():
        if k in r:
            return v
    return raw.strip()


def parse_table(page, year: int) -> list[dict]:
    table = page.query_selector(TABLE_ID)
    if table is None:
        print("[warn] GridView1 not found")
        return []

    all_rows = table.query_selector_all("tr")
    print(f"  GridView1 has {len(all_rows)} rows")

    rows_out = []
    current_inst = current_prog = current_quota = ""

    # First 3 rows are header; data starts at index 3
    for tr in all_rows[3:]:
        cells = [td.inner_text().strip() for td in tr.query_selector_all("td")]
        if not cells:
            continue

        n = len(cells)

        # Gender-Neutral row: starts with Institute name, has ≥15 cells
        # cells: [inst, prog, quota, "Gender-Neutral", OPEN..OBC-NCL-PwD, Total, ...]
        if n >= 14 and cells[3].lower().startswith("gender"):
            current_inst  = cells[0] or current_inst
            current_prog  = cells[1] or current_prog
            current_quota = norm_quota(cells[2]) if cells[2] else current_quota
            gender = "Gender-Neutral"
            seat_values = cells[4:14]    # 10 seat-type counts
            for st_name, raw in zip(ST_PIPELINE_NAMES, seat_values):
                val = _parse_int(raw)
                if val is not None:
                    rows_out.append({
                        "Year": year, "Institute": current_inst,
                        "Program": current_prog, "Quota": current_quota,
                        "Seat Type": st_name, "Gender": gender, "Seats": val,
                    })

        # Female-only row: starts with "Female-only", has ~12 cells
        elif n >= 11 and "female" in cells[0].lower():
            gender = "Female-only (including Supernumerary)"
            seat_values = cells[1:11]    # 10 seat-type counts
            for st_name, raw in zip(ST_PIPELINE_NAMES, seat_values):
                val = _parse_int(raw)
                if val is not None:
                    rows_out.append({
                        "Year": year, "Institute": current_inst,
                        "Program": current_prog, "Quota": current_quota,
                        "Seat Type": st_name, "Gender": gender, "Seats": val,
                    })

        # else: rowspan continuation or summary row, then skip

    return rows_out


def _parse_int(raw: str) -> int | None:
    cleaned = raw.split("\t")[0].split("\n")[0].replace(",", "").strip()
    try:
        v = int(cleaned)
        return v if v > 0 else None
    except ValueError:
        return None


def scrape(year: int):
    if os.path.exists(OUTPUT):
        import pandas as pd
        existing = pd.read_csv(OUTPUT)
        if "Year" in existing.columns and year in existing["Year"].values:
            print(f"Year {year} already in {OUTPUT}, skipping.")
            return

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = browser.new_page()
        page.goto(URL)
        page.wait_for_load_state("networkidle")

        # Dropdowns already default to ALL (value='0'); just click Submit
        page.click(SUBMIT_BTN)
        page.wait_for_load_state("networkidle")
        time.sleep(1)

        print("Parsing seat matrix…")
        rows = parse_table(page, year)
        browser.close()

    if not rows:
        print("No data extracted.")
        return

    write_header = not os.path.exists(OUTPUT)
    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows):,} rows to {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()
    scrape(args.year)
