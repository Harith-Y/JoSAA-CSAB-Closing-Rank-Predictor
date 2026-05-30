"""
CSAB 2025 Current-Year Scraper
URL: https://admissions.nic.in/csabspl/Applicant/SeatAllotmentResult/CurrentORCR.aspx

Appends rows to csab_ranks.csv with Year=2025 and Special Round columns.
Resume-safe: skips rounds already present in the CSV.
"""

import csv
import os
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

URL         = "https://admissions.nic.in/csabspl/Applicant/SeatAllotmentResult/CurrentORCR.aspx"
OUTPUT_FILE = "csab_ranks.csv"
YEAR_LABEL  = "2025"

# Will be confirmed via debug_selects() if these IDs don't match
SEL_ROUND     = "#ctl00_ContentPlaceHolder1_ddlroundno"
SEL_INST_TYPE = "#ctl00_ContentPlaceHolder1_ddlInstype"
SEL_INST_NAME = "#ctl00_ContentPlaceHolder1_ddlInstitute"
SEL_BRANCH    = "#ctl00_ContentPlaceHolder1_ddlBranch"


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


def load_completed_rounds(output_file, year_label):
    done = set()
    if not os.path.exists(output_file):
        return done
    try:
        with open(output_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].strip() == year_label:
                    done.add(row[1].strip())
    except Exception:
        pass
    return done


def debug_selects(page):
    print("\n  [DEBUG] All <select> elements:")
    for s in page.query_selector_all("select"):
        opts = [(o.get_attribute("value"), o.inner_text().strip())
                for o in s.query_selector_all("option")]
        print(f"    id='{s.get_attribute('id')}'  opts={opts[:4]}")


def main():
    completed = load_completed_rounds(OUTPUT_FILE, YEAR_LABEL)
    if completed:
        print(f"Resuming - 2025 CSAB rounds already done: {sorted(completed)}")

    file_exists  = os.path.exists(OUTPUT_FILE)
    write_header = not file_exists

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page    = browser.new_page()
        page.set_default_timeout(30000)

        print("Opening CSAB 2025 page...")
        page.goto(URL, wait_until="networkidle")
        time.sleep(1)

        # Dump selects to confirm IDs
        debug_selects(page)

        rounds = wait_for_options(page, SEL_ROUND, timeout=10)
        if not rounds:
            print("Round dropdown not found - check debug output above.")
            browser.close()
            return

        print(f"Special Rounds for 2025: {[r[1] for r in rounds]}")

        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            for round_val, round_label in rounds:
                if round_label in completed:
                    print(f"  Round {round_label} ... already done, skipping.")
                    continue

                print(f"  Round {round_label} ... ", end="", flush=True)

                try:
                    chosen_select(page, SEL_ROUND, round_val)

                    pick_all(page, SEL_INST_TYPE, wait_for_network=True)

                    opts = wait_for_options(page, SEL_INST_NAME, timeout=8)
                    if opts:
                        pick_all(page, SEL_INST_NAME, wait_for_network=False)

                    opts = wait_for_options(page, SEL_BRANCH, timeout=5)
                    if opts:
                        pick_all(page, SEL_BRANCH, wait_for_network=False)

                    time.sleep(0.5)

                    btn = page.query_selector(
                        "input[type='submit'], button[type='submit'], "
                        "input[value='Submit'], input[value='submit']"
                    )
                    if not btn:
                        print("SUBMIT BUTTON NOT FOUND")
                        debug_selects(page)
                        continue

                    btn.click()
                    page.wait_for_load_state("networkidle")
                    time.sleep(2)

                    rows = extract_table(page)

                    if len(rows) <= 1:
                        print("no data")
                    else:
                        if write_header:
                            writer.writerow(["Year", "Special Round"] + rows[0])
                            write_header = False
                        for row in rows[1:]:
                            writer.writerow([YEAR_LABEL, round_label] + row)
                        f.flush()
                        print(f"{len(rows) - 1} rows saved")
                        completed.add(round_label)

                    # Reload for clean form state
                    page.goto(URL, wait_until="networkidle")
                    time.sleep(1)

                except PlaywrightError as e:
                    print(f"ERROR: {e} - reloading...")
                    try:
                        page.goto(URL, wait_until="networkidle")
                        time.sleep(1)
                    except Exception:
                        pass

        print(f"\nDone!  CSAB 2025 data appended to: {OUTPUT_FILE}")
        browser.close()


if __name__ == "__main__":
    main()
