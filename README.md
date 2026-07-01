# JoSAA / CSAB College Admission Predictor

Predicts JoSAA and CSAB closing ranks for the upcoming counselling year using an ensemble of per-slot year-trend and round-progression models trained on historical data (2016–2026).

## Overview

Each unique seat slot (institute × programme × quota × seat type × gender) is modelled as a two-dimensional time series over years and rounds. The system outputs a full **R1–R6 closing-rank trajectory** for the target year, not just a single-round estimate.

Key finding: JoSAA closing ranks are **mean-reverting**, not trending. The historical median outperforms all linear extrapolation methods by ~23% on average. A fixed-hyperparameter **GP RBF** model achieves the best average MAE (2,667 across 2022–2025) through Bayesian mean reversion. The **GP-MLP ensemble** (arch 512-256-128, dropout 0.15, α=0.5) achieves the best average MAE of **2,608** and is the deployed default for both JoSAA and CSAB (CSAB MAE 42,318, a 15% improvement over GP RBF alone).

## Project Structure

```
JOSAA/
├── scripts/
│   ├── scrape_josaa.py        # Historical JoSAA scraper (2016–2024, Playwright)
│   ├── scrape_josaa_2025.py   # JoSAA 2025 current-year scraper
│   ├── scrape_josaa_2026.py   # JoSAA 2026 current-year scraper (resume-safe, per-round)
│   ├── scrape_csab.py         # Historical CSAB scraper (2021–2024)
│   ├── scrape_csab_2025.py    # CSAB 2025 current-year scraper
│   ├── predict_cli.py         # CLI: train / backtest / tune / predict
│   └── compare_models.py      # Trend-model comparison framework
├── app.py                     # Streamlit multi-page app entry point
├── pages/
│   ├── 0_Predictor.py         # Main predictor UI
│   ├── 1_Historical_Data.py   # Historical rank browser
│   ├── 2_Slot_History.py      # Per-slot closing-rank history
│   └── 3_Model_Evaluation.py  # Live model accuracy vs announced actuals
├── data/
│   ├── josaa_ranks.csv        # JoSAA dataset (549,726 rows, 2016–2026 R1-R2)
│   ├── csab_ranks.csv         # CSAB dataset (~47k rows, 2021–2025)
│   ├── Round1-2026.csv        # JoSAA 2026 Round 1 actuals
│   └── Round2-2026.csv        # JoSAA 2026 Round 2 actuals
├── models/
│   ├── josaa_model.pkl        # Trained JoSAA slot models
│   └── csab_model.pkl         # Trained CSAB slot models
└── pipeline/
    ├── config.py              # Constants, hyperparameters, source configs
    ├── display.py             # Shared abbreviation helpers for UI pages
    ├── loader.py              # CSV loading, cleaning, quota normalisation
    ├── train.py               # SlotModel class and training loop
    ├── predict.py             # Eligibility filtering and per-round prediction
    └── evaluate.py            # Backtesting and ensemble weight tuning
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install playwright scikit-learn pandas numpy streamlit plotly
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu  # for MLP / GP-MLP ensemble
playwright install chromium      # only needed for scrapers
```

## Quickstart

### 1. Collect data (skip if CSVs already present)

```bash
# Historical archive (2016–2025)
python scripts/scrape_josaa.py && python scripts/scrape_josaa_2025.py
python scripts/scrape_csab.py  && python scripts/scrape_csab_2025.py

# Current-year actuals (2026 - run after each round is announced)
python scripts/scrape_josaa_2026.py          # all available rounds
python scripts/scrape_josaa_2026.py --round 3  # specific round only
```

Scraping takes ~4 hours for JoSAA and ~30 minutes for CSAB. The scrapers are resume-safe, a crash just requires a restart. After scraping a new round, add it to `CURRENT_ROUND_DATA` in `pipeline/config.py` and retrain.

### 2. Train

```bash
python scripts/predict_cli.py train                  # JoSAA (default)
python scripts/predict_cli.py train --source csab
```

### 3. Backtest

```bash
python scripts/predict_cli.py backtest --year 2024
python scripts/predict_cli.py backtest --source csab
```

### 4. Tune ensemble weight (optional)

```bash
# Inspect the w vs MAE table
python scripts/predict_cli.py tune --source josaa --val-year 2024

# Write the optimal w into the model pickle
python scripts/predict_cli.py tune --source josaa --val-year 2024 --save
```

### 5. Predict (CLI)

```bash
python scripts/predict_cli.py predict \
    --rank 5000 --exam mains \
    --quota AI --seat-type OPEN \
    --gender Gender-Neutral
```

```bash
# CSAB (shows disclaimer automatically)
python scripts/predict_cli.py predict --source csab \
    --rank 8000 --exam mains \
    --quota AI --seat-type OPEN \
    --gender Gender-Neutral
```

### 6. Web UI

```bash
streamlit run app.py
```

Opens in the browser. The app has four pages (sidebar navigation):

- **Predictor**: select source, exam type, rank, quota, seat type, and gender, then click **Predict**. Switch to the **Trajectory Plot** tab to compare predicted R1–R_max closing-rank trajectories for selected colleges.
- **Historical Data**: browse raw closing ranks by year, round, and filter.
- **Slot History**: per-slot closing-rank history with groundedness indicators.
- **Model Evaluation**: live accuracy of the deployed model against announced round actuals for the current counselling year (updates automatically as new rounds are added to `CURRENT_ROUND_DATA`).

Deployed Link : [Link](https://josaa-csab-closing-rank-predictor.streamlit.app)

## Prediction Categories

| Category | Condition |
|----------|-----------|
| Safe     | rank ≤ 0.80 × predicted close (JoSAA) / 0.60 × (CSAB) |
| Match    | 0.80 × pred < rank ≤ pred |
| Reach    | pred < rank ≤ 1.20 × pred (JoSAA) / 1.50 × (CSAB) |

CSAB thresholds are wider because CSAB MAE (~42,000 with the GP-MLP ensemble) is ~15× higher than JoSAA MAE (~2,880 with GP RBF), reflecting the inherent unpredictability of residual seat allocation.

## Trend Models

Pass `--trend-model <name>` to `train`, `backtest`, or `tune`.
Deployed defaults: **`mlp_ensemble`** for both JoSAA and CSAB.

| Model | Avg MAE 2022–2025 | Notes |
|-------|-------------------|-------|
| **GP-MLP ensemble** | **2,608** | Deployed default (both sources); arch 512-256-128, α=0.5; requires PyTorch |
| GP RBF | 2,667 | Best pure trend model; numpy-only; zero training time |
| Global MLP | 2,676 | Tied with GP RBF; handles cold-start slots; requires PyTorch |
| AR(1) | 2,827 | Explicit mean-reversion; zero dependencies; `O(n)` fit |
| AR(p) AICc | 3,153 | Higher-order AR; rarely justifies p>1 with ≤9 years |
| SVR RBF | 2,742 | Kernel decay → mean reversion; scikit-learn only |
| Median | 3,259 | Stable baseline; no extrapolation |
| Ridge | - | Slope shrinkage; better than OLS, worse than Median |
| Weighted OLS | - | Recent years weighted ~10× more |
| Theil–Sen | - | Robust to single-year outliers |
| OLS | - | Baseline linear extrapolation |
| SVR Linear | - | Similar to Ridge in this regime |

## Backtesting Results (JoSAA, w = 1.0)

| Model | 2022 | 2023 | 2024 | 2025 | Avg |
|-------|------|------|------|------|-----|
| **GP-MLP ensemble** | **2,618** | **2,310** | 2,645 | **2,858** | **2,608** |
| GP RBF | 2,731 | 2,432 | 2,624 | 2,880 | 2,667 |
| Global MLP | 2,676 | 2,437 | **2,597** | 2,992 | 2,676 |
| AR(1) | 3,189 | 2,441 | 2,765 | 2,913 | 2,827 |
| SVR RBF | 2,979 | 2,451 | 2,705 | 2,834 | 2,742 |
| Median | 3,586 | 3,015 | 3,174 | 3,262 | 3,259 |

CSAB overall MAE (2025, 4 training years): **42,318** (GP-MLP ensemble) vs 49,869 (GP RBF alone).

## Data Sources

- JoSAA archive: https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx
- CSAB archive: https://csab.nic.in/

## Paper

A research write-up covering the methodology, model comparisons, and evaluation results is in [`paper.pdf`](paper.pdf). The paper focuses on the statistical findings and system design; this README is the reference for installation, usage, and code structure.

ResearchGate Link : [Link](https://www.researchgate.net/publication/405481388_Predicting_JOSAA_College_Admissions_Using_an_Ensemble_of_Year-Trend_and_Round-Progression_Models_From_Automated_Data_Collection_to_Multi-Round_Forecasting)
