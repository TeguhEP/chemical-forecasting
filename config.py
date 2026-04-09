"""
config.py — Shared constants for the Chemical Industry Forecasting Suite.
Imported by all notebooks and streamlit_app.py.
"""

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE   = "2024-12-31"

# ── Target variables ──────────────────────────────────────────────────────────
PRICE_TARGET  = "crude_oil_wti"      # raw material price prediction target
DEMAND_TARGET = "chemical_production_index"  # demand forecasting target

# ── Forecast horizons (weeks) ─────────────────────────────────────────────────
HORIZONS = [4, 8, 12]
DEFAULT_HORIZON = 8

# ── Train / test split ────────────────────────────────────────────────────────
TEST_CUTOFF_DATE = "2023-01-01"   # everything after this is the holdout test set

# ── Raw data file paths ───────────────────────────────────────────────────────
RAW_FRED        = "data/raw/fred_prices.csv"
RAW_WORLDBANK   = "data/raw/worldbank_commodities.csv"
RAW_EIA         = "data/raw/eia_production.csv"
RAW_M5          = "data/raw/m5_demand.csv"

# ── Processed data file paths ─────────────────────────────────────────────────
FEATURES_TRAIN  = "data/processed/features_train.csv"
FEATURES_TEST   = "data/processed/features_test.csv"

# ── Model artefact paths ──────────────────────────────────────────────────────
MODEL_PROPHET   = "models/prophet_model.pkl"
MODEL_LGBM      = "models/lgbm_model.pkl"
MODEL_TFT       = "models/tft_model"
MODEL_ENSEMBLE  = "models/ensemble_weights.pkl"

# ── Output paths ──────────────────────────────────────────────────────────────
FIGURES_DIR     = "outputs/figures/"
REPORTS_DIR     = "outputs/reports/"

# ── FRED series IDs ───────────────────────────────────────────────────────────
# Full list: https://fred.stlouisfed.org
FRED_SERIES = {
    "crude_oil_wti":              "DCOILWTICO",   # WTI crude oil spot price (weekly)
    "natural_gas_henry_hub":      "DHHNGSP",      # Henry Hub natural gas spot price
    "ppi_chemicals":              "PCU325325",    # PPI: Chemical manufacturing
    "ppi_plastics":               "PCU326326",    # PPI: Plastics & rubber products
    "industrial_production_idx":  "INDPRO",       # Industrial production index
    "capacity_utilisation":       "TCU",          # Total industry capacity utilisation
    "us_gdp":                     "GDP",          # US GDP (quarterly)
    "unemployment_rate":          "UNRATE",       # US unemployment rate
}

# ── World Bank commodity indicator codes ──────────────────────────────────────
WB_INDICATORS = {
    "crude_oil_brent":   "CRUDE_BRENT",
    "natural_gas_us":    "NGAS_US",
    "coal_aus":          "COAL_AUS",
    "fertiliser_urea":   "UREA_E_US",
    "ammonia":           "PHOSROCK",
}

# ── EIA API settings ──────────────────────────────────────────────────────────
EIA_BASE_URL    = "https://api.eia.gov/v2/"
EIA_SERIES = {
    "us_chemical_production": "ELEC.PLANT.GEN.57890-NG-99.M",
    "refinery_utilisation":   "PET.WPULEUS2.W",
}

# ── Modelling constants ───────────────────────────────────────────────────────
RANDOM_SEED         = 42
CV_N_SPLITS         = 5         # number of time-series CV folds
PREDICTION_INTERVAL = 0.90      # target coverage for prediction intervals
LGBM_N_TRIALS       = 50        # Optuna tuning trials for LightGBM
