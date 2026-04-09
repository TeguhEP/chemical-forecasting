# Chemical Industry Forecasting Suite

A complete end-to-end data science project for **WTI crude oil price forecasting**
and **weekly demand prediction** for chemical industry procurement and supply chain planning.

**Procurement Dashboard**
[![Procurement Dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chemical-forecasting.streamlit.app/)

**Model Evaluation Dashboard**
[![Model Evaluation Dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chemical-forecasting-eval.streamlit.app/)

---

## Project Overview

Chemical manufacturers spend 40–70% of revenue on raw materials priced off crude oil.
This project builds a weekly forecasting system that answers two core business questions:

1. **Where will WTI crude oil be over the next 1–12 weeks?**
2. **What weekly demand volume should we plan for?**

### Results

| Model | Test MAPE | Notes |
|---|---|---|
| LightGBM | **1.89%** | Primary production model |
| Conformal Ensemble | 3.59% | Calibrated 90% PI ($15.80/bbl width) |
| Temporal Fusion Transformer | 15.49% | Limited by dataset size |
| Prophet | 18.22% | Interpretable statistical baseline |

Demand forecasting: **2.07% CV MAPE** (±0.18% across 5 temporal folds)

---

## Project Structure

chemical-forecasting/
├── config.py                    # Shared constants and paths
├── streamlit_app.py             # Model evaluation dashboard
├── procurement_dashboard.py     # Business procurement intelligence dashboard
├── requirements.txt             # Python dependencies
├── notebooks/
│   ├── 01_setup.ipynb           # Environment setup and data ingestion
│   ├── 02_eda.ipynb             # Exploratory data analysis
│   ├── 03_features.ipynb        # Feature engineering (243 features)
│   ├── 04_modelling.ipynb       # Model training and hyperparameter tuning
│   └── 05_evaluation.ipynb      # Evaluation, SHAP, residual diagnostics
├── data/
│   ├── raw/                     # Raw data (not tracked by git)
│   └── processed/               # Feature matrices (not tracked by git)
├── models/                      # Saved model artefacts (not tracked by git)
└── outputs/
├── figures/                 # All charts from notebooks
└── reports/                 # CSV evaluation reports


---

---

## Data Sources

All data is publicly available at no cost:

| Source | Data | Access |
|---|---|---|
| [FRED](https://fred.stlouisfed.org) | WTI price, macro indicators | fredapi (free key) |
| [World Bank](https://www.worldbank.org/en/research/commodity-markets) | Commodity prices | Direct download |
| [EIA](https://www.eia.gov/dnav) | Crude stocks, refinery utilisation | HTML scraping |
| [Kaggle M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy) | Weekly demand benchmark | Kaggle download |

---

## Methodology

### Feature Engineering
- **84 lag features** at lags 1, 2, 4, 8, 12, 26 weeks
- **114 rolling statistics** (mean, std, min, max over 4/8/12/26-week windows)
- **12 calendar features** including sine/cosine cyclical encoding
- **9 ratio and spread features** including crude/gas ratio and momentum signals

### Models
- **Prophet** — decomposable statistical baseline (Taylor & Letham, 2018)
- **LightGBM** — gradient boosted trees with Optuna hyperparameter tuning (Ke et al., 2017)
- **TFT** — attention-based deep learning via darts (Lim et al., 2021)
- **Conformal Ensemble** — inverse-MAPE weighted ensemble with conformal prediction intervals (Angelopoulos & Bates, 2021)

### Evaluation
- Strict temporal train-test split at January 1, 2023
- 5-fold time-series cross-validation (no data leakage)
- Metrics: MAPE, RMSE, MAE, MASE, Max Error, PI Coverage

---

## Streamlit Dashboards

### Model Evaluation Dashboard
```bash
streamlit run streamlit_app.py
```
Technical dashboard for data scientists: forecast explorer, model comparison,
residual diagnostics, SHAP explainability, demand forecasting, macro indicators.

### Procurement Intelligence Dashboard
```bash
streamlit run procurement_dashboard.py
```
Business dashboard for procurement teams: weekly BUY/WAIT/MONITOR signal,
price forecast with confidence range, supply indicators, commodity markets,
macro environment, model reliability.

---

## Setup and Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/chemical-forecasting.git
cd chemical-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up FRED API key
Create a file `.streamlit/secrets.toml`:
```toml
FRED_API_KEY = "your_fred_api_key_here"
```
Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

### 4. Download M5 data
Download manually from Kaggle and place `sales_train_evaluation.csv` and
`calendar.csv` in `data/raw/`.

### 5. Run notebooks in order
```bash
jupyter lab
```
Execute notebooks 01 through 05 sequentially.

### 6. Launch dashboard
```bash
streamlit run procurement_dashboard.py
```

---

## Key References

- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
- Taylor, S. J. & Letham, B. (2018). Forecasting at scale. *The American Statistician*.
- Lim, B. et al. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *IJF*.
- Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Angelopoulos, A. N. & Bates, S. (2021). A gentle introduction to conformal prediction. *arXiv*.
- Makridakis, S. et al. (2022). M5 accuracy competition: Results, findings, and conclusions. *IJF*.

---

## Author

**Praharadata**
Data Science Portfolio Project — April 2026

---

## License

MIT License — free to use, modify, and distribute with attribution.
