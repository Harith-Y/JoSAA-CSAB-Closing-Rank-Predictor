"""
JoSAA 2026 current-year round scraper.

Saves each round to data/RoundN-2026.csv with columns:
    Institute, Academic Program Name, Quota, Seat Type, Gender,
    Opening Rank, Closing Rank

Usage:
    python scripts/scrape_josaa_2026.py          # scrape all available rounds
    python scripts/scrape_josaa_2026.py --round 2  # scrape round 2 only
"""

import argparse
import csv
import os
import time

from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

URL      = "https://josaa.admissions.nic.in/Applicant/SeatAllotmentResult/currentorcr.aspx"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SEL_ROUND     = "#ctl00_ContentPlaceHolder1_ddlroundno"
SEL_INST_TYPE = "#ctl00_ContentPlaceHolder1_ddlInstype"
SEL_INST_NAME = "#ctl00_ContentPlaceHolder1_ddlInstitute"
SEL_BRANCH    = "#ctl00_ContentPlaceHolder1_ddlBranch"
SEL_SEAT_TYPE = "#ctl00_ContentPlaceHolder1_ddlSeattype"

EXPECTED_HEADERS = [
    "Institute", "Academic Program Name", "Quota",
    "Seat Type", "Gender", "Opening Rank", "Closing Rank",
]


def chosen_select(page, selector, value, wait_for_network=True):
    page.evaluate(
        """([sel, val]) => {
            const el = document.querySelector(sel);
            el.value = val;
            if (window.jQuery) {
                window.jQuery(el).trigger('change');
            } else {
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }""",
        [selector, value],
    )
    if wait_for_network:
        time.sleep(0.5)
        page.wait_for_load_state("networkidle")
        time.sleep(0.5)
    else:
        time.sleep(0.4)


def safe_get_options(page, selector):
    try:
        opts = []
        for opt in page.query_selector_all(f"{selector} option"):
            val  = (opt.get_attribute("value") or "").strip()
            text = opt.inner_text().strip()
            if val in ("", "0") or text.startswith("--") or text.lower().startswith("select"):
                continue
            opts.append((val, text))
        return opts
    except PlaywrightError:
        return []


def wait_for_options(page, selector, timeout=10):
    for _ in range(timeout * 2):
        opts = safe_get_options(page, selector)
        if opts:
            return opts
        time.sleep(0.5)
    return []


def pick_all(page, selector, wait_for_network=True):
    opts = safe_get_options(page, selector)
    if not opts:
        return None
    val = next((v for v, t in opts if t.strip().upper() == "ALL"), opts[0][0])
    chosen_select(page, selector, val, wait_for_network=wait_for_network)
    return val


def extract_table(page):
    try:
        page.wait_for_selector("table", timeout=40_000)
    except (PlaywrightTimeoutError, PlaywrightError):
        print(f"\n  [DEBUG] No <table> found. Page title: '{page.title()}' URL: {page.url}")
        return []
    try:
        tables = page.query_selector_all("table")
        if not tables:
            return []
        print(f"\n  [DEBUG] {len(tables)} table(s) found on page")
        best = max(tables, key=lambda t: len(t.query_selector_all("tr")))
        rows = []
        for tr in best.query_selector_all("tr"):
            cells = tr.query_selector_all("td, th")
            row = [c.inner_text().strip().replace("\n", " ").replace("\r", "") for c in cells]
            if any(cell.strip() for cell in row):
                rows.append(row)
        return rows
    except PlaywrightError:
        return []


def debug_selects(page):
    print("\n  [DEBUG] <select> elements on page:")
    for s in page.query_selector_all("select"):
        opts = [(o.get_attribute("value"), o.inner_text().strip())
                for o in s.query_selector_all("option")]
        print(f"    id='{s.get_attribute('id')}'  sample={opts[:3]}")


def scrape_round(page, round_val, round_num):
    out_path = os.path.join(DATA_DIR, f"Round{round_num}-2026.csv")

    print(f"  Selecting Round {round_num}...", end="", flush=True)
    chosen_select(page, SEL_ROUND, round_val)

    pick_all(page, SEL_INST_TYPE, wait_for_network=True)

    opts = wait_for_options(page, SEL_INST_NAME, timeout=8)
    if opts:
        pick_all(page, SEL_INST_NAME, wait_for_network=False)

    opts = wait_for_options(page, SEL_BRANCH, timeout=5)
    if opts:
        pick_all(page, SEL_BRANCH, wait_for_network=False)

    opts = wait_for_options(page, SEL_SEAT_TYPE, timeout=10)
    if not opts:
        print(" WARN: Seat Type dropdown empty - dumping selects for diagnosis:")
        debug_selects(page)
        return False

    pick_all(page, SEL_SEAT_TYPE, wait_for_network=False)
    time.sleep(0.5)

    btn = page.query_selector(
        "input[type='submit'], button[type='submit'], "
        "input[value='Submit'], input[value='submit']"
    )
    if not btn:
        print(" SUBMIT BUTTON NOT FOUND")
        debug_selects(page)
        return False

    btn.click()
    page.wait_for_load_state("networkidle")
    time.sleep(4)

    rows = extract_table(page)
    if len(rows) <= 1:
        print(" no data returned")
        screenshot = os.path.join(DATA_DIR, f"debug_round_scrape.png")
        try:
            page.screenshot(path=screenshot)
            print(f"  [DEBUG] Screenshot saved to {screenshot}")
        except Exception:
            pass
        return False

    header = [h.strip() for h in rows[0]]
    # Verify expected columns are present
    missing = [c for c in EXPECTED_HEADERS if c not in header]
    if missing:
        print(f" WARN: missing columns {missing} - got {header}")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows[1:])

    print(f" {len(rows) - 1} rows → {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None,
                        help="Specific round number to scrape (default: all available)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page    = browser.new_page()
        page.set_default_timeout(30_000)

        print("Opening JoSAA 2026 results page...")
        page.goto(URL, wait_until="networkidle")
        time.sleep(1)

        rounds = wait_for_options(page, SEL_ROUND, timeout=10)
        if not rounds:
            print("ERROR: Round dropdown not found. Dumping all selects:")
            debug_selects(page)
            browser.close()
            return

        print(f"Rounds available: {[r[1] for r in rounds]}")

        for round_val, round_label in rounds:
            # round_label is typically "1", "2", etc. or "Round 1", "Round 2"
            try:
                round_num = int("".join(filter(str.isdigit, round_label)))
            except ValueError:
                print(f"  Could not parse round number from '{round_label}', skipping.")
                continue

            if args.round is not None and round_num != args.round:
                continue

            out_path = os.path.join(DATA_DIR, f"Round{round_num}-2026.csv")
            if os.path.exists(out_path) and args.round is None:
                print(f"  Round {round_num}: {out_path} already exists, skipping.")
                continue

            try:
                ok = scrape_round(page, round_val, round_num)
                if ok:
                    print(f"\n  ✓ Round {round_num} saved. Add to config.py:")
                    print(f"    {round_num}: os.path.join(DATA_DIR, \"Round{round_num}-2026.csv\"),")
            except PlaywrightError as e:
                print(f"  ERROR on Round {round_num}: {e}")
                print("  Reloading page...")
                try:
                    page.goto(URL, wait_until="networkidle")
                    time.sleep(1)
                except Exception:
                    pass

            # Reload for clean form state before next round
            try:
                page.goto(URL, wait_until="networkidle")
                time.sleep(1)
            except PlaywrightError:
                break

        browser.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
