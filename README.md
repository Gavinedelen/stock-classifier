# Binary Stock Movement Classifier (10-Day Direction)

A binary classifier that predicts whether a stock’s **closing price will rise or fall 10 trading days ahead**.  
Built for learning and portfolio prototyping, with a focus on transparent features and reproducible evaluation.

---

## Project Overview
- **Universe:** 8 large-cap tickers across 4 sectors (Tech, Financials, Consumer Staples, Energy).  
  **Tickers:** AAPL, MSFT, GS, JPM, KO, PG, XOM, CVX
- **Period:** 2020-01-01 → 2024-01-01
- **Data Source:** Yahoo Finance via `yfinance`
- **Models tried:** Linear Regression (baseline), Random Forest (classifier), XGBoost (classifier)
- **Headline result:** XGBoost reached **~64% accuracy** on held-out data from the same tickers.

---

## Target Definition
For each trading day *t*, label the outcome for *t+10*:
- **Label = 1** if `Close[t+10] > Close[t]`
- **Label = 0** otherwise

---

## Features (10)
1. **MA_5** – 5-day rolling average of close  
2. **MA_10** – 10-day rolling average of close  
3. **Volatility_10** – 10-day rolling standard deviation of close  
4. **Momentum_10** – `Close[t] − Close[t−10]`  
5. **Daily_Return** – % change in close from *t−1* to *t*  
6. **RSI** – Relative Strength Index (momentum oscillator)  
7. **BB_Width** – Bollinger Band width (upper − lower)  
8. **Volume_Ratio_5** – `Volume[t] / mean(Volume[t−1…t−5])`  
9. **DayOfWeek** – Encoded trading day (Mon–Fri)  
10. **Month** – 1–12

*Why these?* Mix of **trend**, **momentum**, **volatility**, **flow/liquidity**, and **seasonality** to capture short-horizon edges.

---

## Data Preparation
1. Download OHLCV for the 8 tickers using `yfinance` over the stated period.  
2. Compute the 10 features above per ticker/date.  
3. Construct the binary 10-day-ahead target.  
4. Align the panel to avoid look-ahead leakage and ensure each row is a single (date, ticker) observation.

---

## Modeling & Evaluation
- **Models compared**
  - Linear Regression (sanity-check baseline even though the task is classification)
  - Random Forest Classifier
  - XGBoost Classifier
- **Split:** ~80% train / ~20% test by time/ticker (held-out data from the same tickers).  
- **Metrics:** Primary = **Accuracy**; also review per-ticker accuracy for stability.

## Results

**Test set:** 1,584 observations across 8 tickers  
**Class balance:** Up (1) = 892, Down (0) = 692 → **Baseline (always predict Up)** = **56.3%** accuracy

| Model               | Accuracy | Precision (Up) | Recall (Up) | F1 (Up) |
|---------------------|:--------:|:--------------:|:-----------:|:-------:|
| Logistic Regression |  0.520   |      0.55      |    0.81     |  0.66   |
| Random Forest       |  0.598   |      0.64      |    0.65     |  0.65   |
| XGBoost             |  **0.637** |    **0.65**    |  **0.76**   | **0.70** |

**Lift vs baseline:** Random Forest **+3.5 pts**, XGBoost **+7.4 pts** (absolute)

### Confusion Matrix — XGBoost (best model)
Actual ↓ / Predicted → | **Down (0)** | **Up (1)**
:--|--:|--:
**Down (0)** | 335 | 357
**Up (1)**   | 218 | 674

**Notes**
- Logistic Regression underperforms the “always up” baseline; tree-based models add meaningful signal, with **XGBoost the strongest**.
- Class 1 (Up) performance is the focus (directional prediction); XGBoost reaches **Precision 0.65 / Recall 0.76 / F1 0.70**.
- Next steps: calibrate class probability thresholds, add walk-forward splits, and evaluate simple trading rules (after costs) for economic relevance.


**Illustrative outcomes**
- Linear Regression: **~52%**  
- Random Forest: **~60%**  
- XGBoost: **~64%** (best overall)  
- Per-ticker range on the test set: **~50% (XOM)** to **~74% (PG)**

---

## Repository Structure
.
├─ notebooks/
│  └─ stock_classifier.ipynb        # main Colab/Notebook
├─ data/                            # optional placeholder; don't commit private data
├─ src/                             # optional helpers (features, eval)
├─ models/                          # optional saved artifacts
├─ requirements.txt                 # dependencies
└─ README.md

---

## Setup
**Option A: Google Colab (recommended)**  
Open `notebooks/stock_classifier.ipynb` in Colab and run all cells.

**Option B: Local**
    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    pip install -r requirements.txt

**requirements.txt (example)**
    pandas
    numpy
    scikit-learn
    xgboost
    yfinance
    matplotlib
    ta

---

## How to Reproduce
1) Open `notebooks/stock_classifier.ipynb` (Colab or local Jupyter).  
2) Run data download + feature engineering cells for the 8 tickers and full feature set.  
3) **Create the 10-day forward label (per ticker, no leakage):**
    
        # assumes one row per (Ticker, Date) with daily data
        df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)

        fwd_close = df.groupby("Ticker")["Close"].shift(-10)
        df["y"] = (fwd_close > df["Close"]).astype("Int8")  # nullable int during prep
        df = df.dropna(subset=["y"]).copy()
        df["y"] = df["y"].astype("int8")  # 0/1 for scikit-learn

4) Train and evaluate Linear Regression (baseline), Random Forest, and XGBoost models.  
5) Review overall accuracy and per-ticker breakdown; compare to the illustrative results in the overview.  
6) (Optional) Tune features/hyperparameters and re-run.

---

## Notes & Lessons Learned
- Very short horizons (e.g., 1-day ahead) were **too noisy**; **10-day** worked better.  
- Correct **alignment across tickers** and splits is crucial to avoid leakage.  
- Signal quality beats sheer number of features; focus on tractable, interpretable signals.

---

## Next Steps
- Use **walk-forward/rolling** time splits.  
- Add **probability thresholding** + simple trading rules to evaluate *economic* performance after costs.  
- Inspect **feature importance/SHAP** on XGBoost for interpretability.  
- Expand the **universe** and test on **out-of-sample tickers** for generalization.  
- Try **time-series CV**, **calibrated probabilities**, and/or **LightGBM/CatBoost**.

---

## Disclaimer
This project is for educational purposes only and is **not** investment advice.

