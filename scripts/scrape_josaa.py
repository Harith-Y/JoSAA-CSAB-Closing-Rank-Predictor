"""
JOSAA Opening & Closing Ranks Scraper
Extracts all years x all rounds with ALL filters, saves to josaa_ranks.csv

- Dropdowns use jQuery Chosen (native <select> is hidden); values set via JS.
- Resume-safe: skips (year, round) pairs already present in the output CSV.
- Navigation-safe: retries on execution-context-destroyed errors mid-postback.
"""

import csv
import os
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

URL = "https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx"
OUTPUT_FILE = "josaa_ranks.csv"

SEL_YEAR      = "#ctl00_ContentPlaceHolder1_ddlYear"
SEL_ROUND     = "#ctl00_ContentPlaceHolder1_ddlroundno"
SEL_INST_TYPE = "#ctl00_ContentPlaceHolder1_ddlInstype"
SEL_INST_NAME = "#ctl00_ContentPlaceHolder1_ddlInstitute"
SEL_BRANCH    = "#ctl00_ContentPlaceHolder1_ddlBranch"
SEL_SEAT_TYPE = "#ctl00_ContentPlaceHolder1_ddlSeatType"


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
    """Get options, returning [] if the page navigates mid-call."""
    try:
        opts = []
        for opt in page.query_selector_all(f"{selector} option"):
            val = (opt.get_attribute("value") or "").strip()
            text = opt.inner_text().strip()
            if val in ("", "0") or text.startswith("--") or text.lower().startswith("select"):
                continue
            opts.append((val, text))
        return opts
    except PlaywrightError:
        return []


def wait_for_options(page, selector, timeout=10):
    """Poll until dropdown has options or timeout (seconds) expires."""
    for _ in range(timeout * 2):
        opts = safe_get_options(page, selector)
        if opts:
            return opts
        time.sleep(0.5)
    return []


def pick_all(page, selector, wait_for_network=True):
    """Select the ALL option (or first available option) from a dropdown."""
    opts = safe_get_options(page, selector)
    if not opts:
        return None
    val = next((v for v, t in opts if t.strip().upper() == "ALL"), opts[0][0])
    chosen_select(page, selector, val, wait_for_network=wait_for_network)
    return val


def extract_table(page):
    try:
        page.wait_for_selector("table", timeout=20000)
    except (PlaywrightTimeoutError, PlaywrightError):
        return []
    try:
        tables = page.query_selector_all("table")
        if not tables:
            return []
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


def load_completed(output_file):
    """Return set of (year_label, round_label) already saved in the CSV."""
    done = set()
    if not os.path.exists(output_file):
        return done
    try:
        with open(output_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    done.add((row[0].strip(), row[1].strip()))
    except Exception:
        pass
    return done


def main():
    completed = load_completed(OUTPUT_FILE)
    if completed:
        print(f"Resuming - {len(completed)} (year, round) combinations already done.")

    # Append if resuming, otherwise write fresh
    file_mode = "a" if completed else "w"
    write_header = not completed

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_default_timeout(30000)

        print("Opening JOSAA page...")
        page.goto(URL, wait_until="networkidle")
        time.sleep(1)

        years = safe_get_options(page, SEL_YEAR)
        print(f"Years: {[y[1] for y in years]}")

        with open(OUTPUT_FILE, file_mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            for year_val, year_label in years:
                print(f"\n{'-'*60}")
                print(f"Year: {year_label}")

                chosen_select(page, SEL_YEAR, year_val)

                rounds = wait_for_options(page, SEL_ROUND, timeout=10)
                if not rounds:
                    print("  No rounds found, skipping.")
                    continue
                print(f"  Rounds: {[r[1] for r in rounds]}")

                for round_val, round_label in rounds:
                    if (year_label, round_label) in completed:
                        print(f"  Round {round_label} ... already done, skipping.")
                        continue

                    print(f"  Round {round_label} ... ", end="", flush=True)

                    try:
                        chosen_select(page, SEL_ROUND, round_val)

                        # Institute Type = ALL (required; cascades the other dropdowns)
                        pick_all(page, SEL_INST_TYPE, wait_for_network=True)

                        # Institute Name - wait for cascade then pick ALL
                        opts = wait_for_options(page, SEL_INST_NAME, timeout=8)
                        if opts:
                            pick_all(page, SEL_INST_NAME, wait_for_network=False)

                        # Academic Program
                        opts = wait_for_options(page, SEL_BRANCH, timeout=5)
                        if opts:
                            pick_all(page, SEL_BRANCH, wait_for_network=False)

                        # Seat Type
                        opts = wait_for_options(page, SEL_SEAT_TYPE, timeout=5)
                        if opts:
                            pick_all(page, SEL_SEAT_TYPE, wait_for_network=False)

                        time.sleep(0.5)

                        btn = page.query_selector(
                            "input[type='submit'], button[type='submit'], "
                            "input[value='Submit'], input[value='submit']"
                        )
                        if not btn:
                            print("SUBMIT BUTTON NOT FOUND")
                            continue

                        btn.click()
                        page.wait_for_load_state("networkidle")
                        time.sleep(2)

                        rows = extract_table(page)

                        if len(rows) <= 1:
                            print("no data")
                        else:
                            if write_header:
                                writer.writerow(["Year", "Round"] + rows[0])
                                write_header = False
                            for row in rows[1:]:
                                writer.writerow([year_label, round_label] + row)
                            f.flush()  # persist to disk immediately
                            print(f"{len(rows) - 1} rows saved")
                            completed.add((year_label, round_label))

                    except PlaywrightError as e:
                        print(f"ERROR (navigation/context): {e}")
                        print("  Reloading page and continuing...")
                        try:
                            page.goto(URL, wait_until="networkidle")
                            time.sleep(1)
                            chosen_select(page, SEL_YEAR, year_val)
                        except Exception:
                            pass
                        continue

                    # Reset to current year for next round
                    try:
                        chosen_select(page, SEL_YEAR, year_val)
                    except PlaywrightError:
                        page.goto(URL, wait_until="networkidle")
                        time.sleep(1)
                        chosen_select(page, SEL_YEAR, year_val)

        print(f"\nDone!  All data saved to: {OUTPUT_FILE}")
        browser.close()


if __name__ == "__main__":
    main()
