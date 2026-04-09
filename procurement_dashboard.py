"""
procurement_dashboard.py — Weekly Procurement Intelligence Dashboard
Run: streamlit run procurement_dashboard.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

st.set_page_config(
    page_title="Procurement Intelligence — Chemical Industry",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

st.markdown("""
<style>
    .stApp { background-color: #f7f9fc !important; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
    section[data-testid="stSidebar"] { background-color: #0d2137 !important; }
    section[data-testid="stSidebar"] * { color: #cce0f5 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stRadio label p { color: #cce0f5 !important; font-size: 0.95rem !important; }
    section[data-testid="stSidebar"] .stSlider label { color: #aac8e8 !important; font-size: 0.85rem !important; }
    .kpi-card { background: white; border: 1px solid #e2e8f0; border-radius: 0.6rem; padding: 1rem 1.2rem; text-align: center; }
    .kpi-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #0d2137; margin: 0.2rem 0; }
    .section-header { font-size: 1.1rem; font-weight: 700; color: #0d2137; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0; }
    .insight-green  { background:#f0faf5; border-left:4px solid #0a7c4e; padding:0.8rem 1rem; border-radius:0 0.4rem 0.4rem 0; margin:0.5rem 0; color:#1a1a1a; font-size:0.88rem; line-height:1.6; }
    .insight-orange { background:#fff8f0; border-left:4px solid #e06400; padding:0.8rem 1rem; border-radius:0 0.4rem 0.4rem 0; margin:0.5rem 0; color:#1a1a1a; font-size:0.88rem; line-height:1.6; }
    .insight-blue   { background:#f0f7ff; border-left:4px solid #1a5c99; padding:0.8rem 1rem; border-radius:0 0.4rem 0.4rem 0; margin:0.5rem 0; color:#1a1a1a; font-size:0.88rem; line-height:1.6; }
    .driver-up      { background:#fee2e2; color:#b91c1c; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; display:inline-block; margin:2px; }
    .driver-down    { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; display:inline-block; margin:2px; }
    .driver-neutral { background:#f1f5f9; color:#475569; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; display:inline-block; margin:2px; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

CHART_FONT = dict(family="Segoe UI, Arial, sans-serif", size=12, color="#1a1a1a")

def apply_theme(fig, height=380, title=None):
    upd = dict(height=height, font=CHART_FONT,
               paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
               margin=dict(l=65, r=30, t=55 if title else 30, b=50),
               hoverlabel=dict(bgcolor="#ffffff", font_size=12,
                               font_color="#1a1a1a", bordercolor="#cccccc"))
    if title:
        upd["title"] = dict(text=title,
                            font=dict(size=14, color="#0d2137",
                                      family="Segoe UI, Arial, sans-serif"),
                            x=0, xanchor="left")
    fig.update_layout(**upd)
    fig.update_xaxes(gridcolor="#f0f0f0", linecolor="#dddddd",
                     tickfont=dict(size=11, color="#444444"),
                     title_font=dict(size=12, color="#444444"))
    fig.update_yaxes(gridcolor="#f0f0f0", linecolor="#dddddd",
                     tickfont=dict(size=11, color="#444444"),
                     title_font=dict(size=12, color="#444444"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_all_data():
    def load(path):
        return pd.read_csv(os.path.join(ROOT, path),
                           index_col="date", parse_dates=True).sort_index()

    price_train = load("data/processed/features_train_price.csv")
    price_test  = load("data/processed/features_test_price.csv")

    # Raw data files — optional, may not exist on Streamlit Cloud
    # If missing, create minimal placeholders so the app still runs
    fred_raw = _safe_load(ROOT, "data/raw/fred_prices.csv")
    eia_raw  = _safe_load(ROOT, "data/raw/eia_production.csv")
    wb_raw   = _safe_load(ROOT, "data/raw/worldbank_commodities.csv")

    # SHAP values
    shap_df = None
    p = os.path.join(ROOT, "outputs/reports/shap_values.csv")
    if os.path.exists(p):
        shap_df = pd.read_csv(p, index_col="date", parse_dates=True).sort_index()

    return price_train, price_test, fred_raw, eia_raw, wb_raw, shap_df


def _safe_load(root, path):
    """Load a CSV if it exists, otherwise return an empty DataFrame."""
    full_path = os.path.join(root, path)
    if os.path.exists(full_path):
        return pd.read_csv(full_path, index_col="date",
                           parse_dates=True).sort_index().ffill()
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_models():
    lgbm     = joblib.load(os.path.join(ROOT, "models/lgbm_model.pkl"))
    ensemble = joblib.load(os.path.join(ROOT, "models/ensemble_weights.pkl"))
    return lgbm, ensemble


@st.cache_data(show_spinner=False)
def build_forecast_data(_lgbm, _price_train, _price_test):
    """
    Returns:
      hist_x, hist_y  — last 26 weeks actual prices (plain Python lists)
      fc_x, fc_y, fc_upper, fc_lower, fc_actual — test period (plain Python lists)
    All sorted ascending chronologically.
    """
    EXCL = ["log_crude_oil_wti", "crude_oil_wti", "crude_oil_avg"]
    FEAT = [c for c in _price_train.columns if c not in EXCL]

    X_test   = _price_test[FEAT]
    y_actual = np.exp(_price_test["log_crude_oil_wti"].values)
    y_pred   = np.exp(_lgbm["median"].predict(X_test))
    y_lower  = np.exp(_lgbm["lower"].predict(X_test))
    y_upper  = np.exp(_lgbm["upper"].predict(X_test))

    # Historical — last 26 weeks from training set
    n     = min(26, len(_price_train))
    h_y   = np.exp(_price_train["log_crude_oil_wti"].values[-n:])
    h_idx = _price_train.index[-n:]

    # Convert everything to plain Python — no pandas, no numpy arrays passed to Plotly
    hist_x    = [d.strftime("%Y-%m-%d") for d in h_idx]
    hist_y    = [float(v) for v in h_y]
    fc_x      = [d.strftime("%Y-%m-%d") for d in _price_test.index]
    fc_y      = [float(v) for v in y_pred]
    fc_upper  = [float(v) for v in y_upper]
    fc_lower  = [float(v) for v in y_lower]
    fc_actual = [float(v) for v in y_actual]

    return hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower, fc_actual


# ═══════════════════════════════════════════════════════════════════════════════
# BUSINESS LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signal(fc_y, fc_upper, fc_lower, horizon):
    hw      = min(horizon, len(fc_y))
    current = fc_y[0]
    target  = fc_y[hw - 1]
    lower   = fc_lower[hw - 1]
    upper   = fc_upper[hw - 1]
    pct     = (target - current) / current * 100
    pi_pct  = (upper - lower) / current * 100

    if pct > 3:
        signal, color = "BUY NOW", "#0a7c4e"
        rationale = (f"Price forecast rises {pct:+.1f}% over {hw} weeks "
                     f"(${current:.1f} → ${target:.1f}/bbl). "
                     "Lock in contracts now to reduce cost exposure.")
    elif pct < -3:
        signal, color = "WAIT", "#e06400"
        rationale = (f"Price forecast falls {pct:+.1f}% over {hw} weeks "
                     f"(${current:.1f} → ${target:.1f}/bbl). "
                     "Delay purchases — lower prices expected.")
    else:
        signal, color = "MONITOR", "#1a5c99"
        rationale = (f"Price forecast moves {pct:+.1f}% over {hw} weeks "
                     f"(${current:.1f} → ${target:.1f}/bbl). "
                     "No strong directional signal.")

    confidence = "High" if pi_pct < 5 else ("Medium" if pi_pct < 12 else "Low")
    return dict(signal=signal, color=color, pct=pct,
                current=current, target=target,
                lower=lower, upper=upper,
                confidence=confidence, rationale=rationale, horizon=hw)


def get_shap_drivers(shap_df, top_n=6):
    if shap_df is None or len(shap_df) == 0:
        return [], []
    row = shap_df.iloc[0]
    def clean(f):
        return (f.replace("log_crude_oil_wti_","WTI Price ")
                 .replace("log_natural_gas_henry_hub_","Nat Gas ")
                 .replace("log_crude_oil_avg_","Crude Avg ")
                 .replace("log_coal_australia_","Coal ")
                 .replace("log_fertiliser_urea_","Urea ")
                 .replace("crude_oil_stocks_","Crude Stocks ")
                 .replace("refinery_utilisation_","Refinery Util ")
                 .replace("natural_gas_storage_","Gas Storage ")
                 .replace("ppi_chemicals_","Chem PPI ")
                 .replace("price_pct_change_1w","Weekly Momentum")
                 .replace("price_pct_change_4w","4-Week Momentum")
                 .replace("price_spread_roll8","8-Week Price Spread")
                 .replace("crude_gas_ratio","Crude/Gas Ratio")
                 .replace("_lag1"," (last week)").replace("_lag2"," (2wk ago)")
                 .replace("_lag4"," (4wk ago)").replace("_roll4_mean"," (4wk avg)")
                 .replace("_roll4_min"," (4wk low)").replace("_roll4_max"," (4wk high)")
                 .replace("_roll8_mean"," (8wk avg)").replace("_","  ").strip().title())
    top  = row.abs().sort_values(ascending=False).head(top_n)
    up   = [(clean(k), float(row[k])) for k in top.index if row[k] > 0]
    down = [(clean(k), float(row[k])) for k in top.index if row[k] < 0]
    return up, down


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS — all inputs are plain Python lists of strings/floats
# ═══════════════════════════════════════════════════════════════════════════════

def make_forecast_chart(hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower, horizon, sig):
    hw = min(horizon, len(fc_x))
    # Slice to horizon
    fx = fc_x[:hw]
    fy = fc_y[:hw]
    fu = fc_upper[:hw]
    fl = fc_lower[:hw]

    fc_color = {"BUY NOW":"#0a7c4e","WAIT":"#e06400","MONITOR":"#1a5c99"}.get(
        sig["signal"], "#1a5c99")

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, mode="lines",
        line=dict(color="#334155", width=2.2),
        name="Historical WTI",
        hovertemplate="%{x}<br>Actual: $%{y:.2f}/bbl<extra></extra>",
    ))

    # Bridge
    fig.add_trace(go.Scatter(
        x=[hist_x[-1], fx[0]], y=[hist_y[-1], fy[0]],
        mode="lines", line=dict(color="#334155", width=2.2),
        showlegend=False, hoverinfo="skip",
    ))

    # PI band
    fig.add_trace(go.Scatter(
        x=fx + fx[::-1], y=fu + fl[::-1],
        fill="toself", fillcolor="rgba(29,100,200,0.13)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        name="90% Confidence Range", hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fx, y=fy, mode="lines+markers",
        line=dict(color=fc_color, width=2.5),
        marker=dict(size=7, color=fc_color, line=dict(color="white", width=1.5)),
        name=f"{hw}-Week Forecast",
        hovertemplate="%{x}<br>Forecast: $%{y:.2f}/bbl<extra></extra>",
    ))

    # PI dotted bounds
    for yv, lbl in [(fu, "Upper 90%"), (fl, "Lower 90%")]:
        fig.add_trace(go.Scatter(
            x=fx, y=yv, mode="lines",
            line=dict(color=fc_color, width=1.0, dash="dot"),
            showlegend=False,
            hovertemplate=f"{lbl}: $%{{y:.2f}}/bbl<extra></extra>",
        ))

    # Current marker
    fig.add_trace(go.Scatter(
        x=[fx[0]], y=[fy[0]], mode="markers",
        marker=dict(size=14, color="#0d2137", line=dict(color="white", width=2.5)),
        name=f"Current: ${fy[0]:.1f}/bbl",
        hovertemplate=f"Current: ${fy[0]:.2f}/bbl<extra></extra>",
    ))

    # Divider
    # Divider — use shape instead of add_vline (categorical axis incompatibility)
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=fx[0], x1=fx[0],
        y0=0, y1=1,
        line=dict(color="#94a3b8", width=1.2, dash="dash"),
    )
    fig.add_annotation(
        x=fx[0], xref="x",
        y=0.97, yref="paper",
        text="Forecast →",
        showarrow=False,
        font=dict(size=11, color="#64748b"),
        xanchor="left",
        bgcolor="rgba(255,255,255,0.7)",
    )

    # Target annotation
    arrow = "↑" if sig["pct"] > 0 else "↓"
    fig.add_annotation(
        x=fx[-1], y=fy[-1],
        text=f" {arrow} ${fy[-1]:.1f} ({sig['pct']:+.1f}%)",
        showarrow=True, arrowhead=2, arrowcolor=fc_color,
        font=dict(size=12, color=fc_color),
        ax=45, ay=-35,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor=fc_color, borderwidth=1, borderpad=4,
    )

    # x-axis: show history + forecast window, left-to-right
    all_x = hist_x + fx
    step  = max(1, len(all_x) // 10)
    fig.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=all_x,
            tickvals=all_x[::step],
            ticktext=all_x[::step],
            tickangle=-30,
        ),
        xaxis_title="",
        yaxis_title="WTI Crude Oil (USD/bbl)",
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor="#dddddd",
                    borderwidth=1, font=dict(size=11, color="#1a1a1a"),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_theme(fig, height=430,
                       title="WTI Crude Oil — Price Forecast with Confidence Range")


def make_backtest_chart(fc_x, fc_y, fc_upper, fc_lower, fc_actual):
    mape = float(np.mean(np.abs(
        (np.array(fc_actual) - np.array(fc_y)) / np.array(fc_actual)
    )) * 100)
    cov = float(np.mean(
        (np.array(fc_actual) >= np.array(fc_lower)) &
        (np.array(fc_actual) <= np.array(fc_upper))
    ) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc_x + fc_x[::-1], y=fc_upper + fc_lower[::-1],
        fill="toself", fillcolor="rgba(29,100,200,0.10)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        name="90% Confidence Range", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_actual, mode="lines",
        line=dict(color="#0d2137", width=2.2), name="Actual WTI",
        hovertemplate="%{x}<br>Actual: $%{y:.2f}/bbl<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_y, mode="lines",
        line=dict(color="#D94E2A", width=1.8),
        name=f"LightGBM Forecast (MAPE {mape:.2f}%)",
        hovertemplate="%{x}<br>Forecast: $%{y:.2f}/bbl<extra></extra>",
    ))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.99,
        text=f"<b>Backtest 2023–2024 | MAPE {mape:.2f}% | PI Coverage {cov:.0f}%</b>",
        showarrow=False, font=dict(size=11, color="#0d2137"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc",
        borderwidth=1, align="left",
    )
    step = max(1, len(fc_x) // 12)
    fig.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=fc_x,
                   tickvals=fc_x[::step], ticktext=fc_x[::step], tickangle=-30),
        xaxis_title="", yaxis_title="WTI Crude Oil (USD/bbl)",
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor="#dddddd",
                    borderwidth=1, font=dict(size=11, color="#1a1a1a"),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_theme(fig, height=400, title="Model Backtest — 2023–2024 Test Period")


def _line_chart(df, col, color, ylabel, title, fill=True, hline=None, yrange=None):
    if col not in df.columns:
        return None
    s     = df[col].dropna().iloc[-104:]
    dates = [d.strftime("%Y-%m-%d") for d in s.index]
    vals  = [float(v) for v in s.values]
    fig   = go.Figure()
    if hline:
        fig.add_hline(y=hline, line_dash="dash", line_color="#94a3b8",
                      annotation_text=f"  {hline}%",
                      annotation_font=dict(size=10, color="#94a3b8"))
    kw = dict(fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.08,)}",
              fill="tozeroy") if fill else {}
    fig.add_trace(go.Scatter(
        x=dates, y=vals, mode="lines",
        line=dict(color=color, width=2.0), **kw,
        hovertemplate=f"%{{x}}<br>{ylabel}: %{{y:.2f}}<extra></extra>",
    ))
    if yrange:
        fig.update_layout(yaxis=dict(range=yrange))
    fig.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=dates,
                   tickvals=dates[::10], ticktext=dates[::10], tickangle=-30),
        yaxis_title=ylabel, hovermode="x unified", showlegend=False,
    )
    return apply_theme(fig, height=260, title=title)


def make_inventory_chart(eia_raw):
    if "crude_oil_stocks" not in eia_raw.columns:
        return None
    s     = eia_raw["crude_oil_stocks"].dropna().iloc[-104:]
    dates = [d.strftime("%Y-%m-%d") for d in s.index]
    vals  = [float(v)/1e6 for v in s.values]
    ma4   = pd.Series(vals).rolling(4).mean().tolist()
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=vals, mode="lines",
        line=dict(color="#1a5c99", width=2.0), fill="tozeroy",
        fillcolor="rgba(26,92,153,0.07)", name="Crude Stocks (B bbl)",
        hovertemplate="%{x}<br>%{y:.3f}B bbl<extra></extra>"))
    fig.add_trace(go.Scatter(x=dates, y=ma4, mode="lines",
        line=dict(color="#e06400", width=1.5, dash="dot"), name="4-Week MA"))
    fig.update_layout(
        xaxis=dict(categoryorder="array", categoryarray=dates,
                   tickvals=dates[::10], ticktext=dates[::10], tickangle=-30),
        yaxis_title="Billion barrels", hovermode="x unified",
        legend=dict(font=dict(size=11, color="#1a1a1a"),
                    bgcolor="rgba(255,255,255,0.9)"))
    return apply_theme(fig, height=280, title="US Crude Oil Inventories (Weekly)")


def make_commodity_chart(fred_raw, wb_raw):
    palette = ["#1a5c99","#7F77DD","#e06400","#64748b"]
    config  = [("crude_oil_wti",fred_raw,"WTI Crude ($/bbl)"),
               ("natural_gas_henry_hub",fred_raw,"Nat Gas ($/MMBtu)"),
               ("crude_oil_avg",wb_raw,"Crude Avg ($/bbl)"),
               ("coal_australia",wb_raw,"Coal Australia ($/t)")]
    fig = go.Figure()
    for i,(col,df,lbl) in enumerate(config):
        if col not in df.columns:
            continue
        s     = df[col].dropna().iloc[-104:]
        dates = [d.strftime("%Y-%m-%d") for d in s.index]
        norm  = [float(v)/float(s.iloc[0])*100 for v in s.values]
        fig.add_trace(go.Scatter(x=dates, y=norm, mode="lines",
            line=dict(color=palette[i%len(palette)], width=1.8), name=lbl,
            hovertemplate=f"{lbl}<br>%{{x}}: %{{y:.1f}}<extra></extra>"))
    # pick dates from first valid series for axis
    try:
        ref_s = fred_raw["crude_oil_wti"].dropna().iloc[-104:]
        ref_dates = [d.strftime("%Y-%m-%d") for d in ref_s.index]
        fig.update_layout(xaxis=dict(categoryorder="array", categoryarray=ref_dates,
                                     tickvals=ref_dates[::10], ticktext=ref_dates[::10],
                                     tickangle=-30))
    except Exception:
        pass
    fig.add_hline(y=100, line_dash="dot", line_color="#cccccc",
                  annotation_text="  Base=100",
                  annotation_font=dict(size=10, color="#999999"))
    fig.update_layout(yaxis_title="Indexed (start=100)", hovermode="x unified",
                      legend=dict(font=dict(size=11,color="#1a1a1a"),
                                  bgcolor="rgba(255,255,255,0.9)",
                                  orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    return apply_theme(fig, height=320, title="Commodity Prices — 2-Year Indexed Comparison")


def make_macro_chart(fred_raw):
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["<b>Industrial Production Index</b>",
                        "<b>Capacity Utilisation (%)</b>",
                        "<b>Unemployment Rate (%)</b>",
                        "<b>Chemical PPI</b>"],
        vertical_spacing=0.18, horizontal_spacing=0.12)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#0d2137", family="Segoe UI, Arial, sans-serif")
    for col,row,cc,color in [("industrial_production_idx",1,1,"#1a5c99"),
                               ("capacity_utilisation",1,2,"#0a7c4e"),
                               ("unemployment_rate",2,1,"#e06400"),
                               ("ppi_chemicals",2,2,"#7F77DD")]:
        if col not in fred_raw.columns:
            continue
        s     = fred_raw[col].dropna().iloc[-104:]
        dates = [d.strftime("%Y-%m-%d") for d in s.index]
        vals  = [float(v) for v in s.values]
        fig.add_trace(go.Scatter(x=dates, y=vals, mode="lines",
            line=dict(color=color, width=1.8), showlegend=False), row=row, col=cc)
    fig.update_xaxes(gridcolor="#f0f0f0", linecolor="#dddddd",
                     tickfont=dict(size=10,color="#444444"), tickangle=-30)
    fig.update_yaxes(gridcolor="#f0f0f0", linecolor="#dddddd",
                     tickfont=dict(size=10,color="#444444"))
    fig.update_layout(height=450, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=CHART_FONT, margin=dict(l=60,r=30,t=60,b=50),
                      hovermode="x unified")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(fc_y, fc_actual):
    with st.sidebar:
        st.markdown("## 🛢️ Procurement\nIntelligence")
        st.markdown("<small style='color:#7da8cc'>Chemical Industry · Weekly Briefing</small>",
                    unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Navigation")
        page = st.radio("page", options=[
            "📋 Weekly Briefing","📈 Price Forecast","🏭 Supply Indicators",
            "🌍 Commodity Markets","📊 Macro Environment","✅ Model Reliability",
        ], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("#### Forecast Horizon")
        horizon = st.slider("Weeks ahead", 1, min(12, len(fc_y)), 4)
        st.markdown("---")
        st.markdown("#### Latest Prices")
        st.markdown(
            f"<div style='color:#7da8cc;font-size:0.75rem;'>1-week forecast</div>"
            f"<div style='color:white;font-size:1.4rem;font-weight:700;'>${fc_y[0]:.2f}/bbl</div>"
            f"<div style='color:#7da8cc;font-size:0.75rem;margin-top:0.5rem;'>Last actual</div>"
            f"<div style='color:#aac8e8;font-size:1.1rem;font-weight:600;'>${fc_actual[-1]:.2f}/bbl</div>",
            unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            "<small style='color:#4a7a9b'>Data: FRED · EIA · World Bank<br><br>"
            "Model: LightGBM + Conformal PI<br>"
            "Backtest MAPE: <b style='color:#7da8cc'>1.89%</b><br>"
            "PI Coverage: <b style='color:#7da8cc'>99%</b></small>",
            unsafe_allow_html=True)
    return page, horizon


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def render_weekly_briefing(hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower, sig, shap_df):
    st.markdown(
        "<h1 style='color:#0d2137;font-size:1.8rem;margin-bottom:0;'>Weekly Procurement Briefing</h1>"
        f"<p style='color:#64748b;font-size:0.9rem;margin-top:0.2rem;'>"
        f"WTI Crude Oil · {fc_x[0]} — {fc_x[sig['horizon']-1]}</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    bg    = sig["color"]
    arrow = "↑" if sig["pct"] > 0 else "↓"
    pc    = "#e06400" if sig["pct"] > 0 else "#0a7c4e"

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(
            f"<div style='background:{bg};color:white;padding:1rem;border-radius:0.6rem;text-align:center;'>"
            f"<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;opacity:0.85;'>Signal</div>"
            f"<div style='font-size:1.6rem;font-weight:800;margin:0.3rem 0;'>{sig['signal']}</div>"
            f"<div style='font-size:0.8rem;opacity:0.85;'>Confidence: {sig['confidence']}</div>"
            f"</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Current Forecast</div>"
                    f"<div class='kpi-value'>${sig['current']:.2f}</div>"
                    f"<div style='font-size:0.75rem;color:#64748b;'>USD/bbl</div></div>",
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{sig['horizon']}-Week Target</div>"
                    f"<div class='kpi-value'>${sig['target']:.2f}</div>"
                    f"<div style='font-size:0.75rem;color:{pc};font-weight:600;'>{arrow} {sig['pct']:+.1f}%</div></div>",
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>90% Price Range</div>"
                    f"<div class='kpi-value' style='font-size:1.2rem;'>${sig['lower']:.0f}–${sig['upper']:.0f}</div>"
                    f"<div style='font-size:0.75rem;color:#64748b;'>USD/bbl</div></div>",
                    unsafe_allow_html=True)
    with c5:
        st.markdown("<div class='kpi-card'><div class='kpi-label'>Model Accuracy</div>"
                    "<div class='kpi-value'>1.89%</div>"
                    "<div style='font-size:0.75rem;color:#64748b;'>Backtest MAPE</div></div>",
                    unsafe_allow_html=True)

    style_map = {"BUY NOW":"insight-green","WAIT":"insight-orange","MONITOR":"insight-blue"}
    st.markdown(
        f"<div class='{style_map.get(sig['signal'],'insight-blue')}'>"
        f"<b>Recommendation:</b> {sig['rationale']}</div>",
        unsafe_allow_html=True)

    st.markdown("---")
    st.plotly_chart(
        make_forecast_chart(hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower,
                            sig["horizon"], sig),
        use_container_width=True)

    st.markdown("<div class='section-header'>What is driving this forecast?</div>",
                unsafe_allow_html=True)
    if shap_df is not None:
        up, down = get_shap_drivers(shap_df)
        col_u, col_d = st.columns(2)
        with col_u:
            st.markdown("<b style='color:#b91c1c;'>🔴 Upward pressure</b>", unsafe_allow_html=True)
            if up:
                for name,_ in up:
                    st.markdown(f"<span class='driver-up'>↑ {name}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='driver-neutral'>None identified</span>", unsafe_allow_html=True)
        with col_d:
            st.markdown("<b style='color:#166534;'>🟢 Downward pressure</b>", unsafe_allow_html=True)
            if down:
                for name,_ in down:
                    st.markdown(f"<span class='driver-down'>↓ {name}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='driver-neutral'>None identified</span>", unsafe_allow_html=True)
        st.markdown("<div class='insight-blue'><b>How to read:</b> Red = factors pushing price up. "
                    "Green = factors pushing price down. Based on SHAP values from the LightGBM model.</div>",
                    unsafe_allow_html=True)
    else:
        st.info("SHAP driver analysis not available — run `05_evaluation.ipynb` first.")

    with st.expander("📋 View full forecast table"):
        hw  = sig["horizon"]
        tbl = pd.DataFrame({
            "Week of":      fc_x[:hw],
            "Forecast":     [f"${v:.2f}" for v in fc_y[:hw]],
            "Lower 90%":    [f"${v:.2f}" for v in fc_lower[:hw]],
            "Upper 90%":    [f"${v:.2f}" for v in fc_upper[:hw]],
            "vs. Current":  [f"{(v-fc_y[0])/fc_y[0]*100:+.1f}%" for v in fc_y[:hw]],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV",
                           tbl.to_csv(index=False).encode("utf-8"),
                           "wti_forecast.csv", "text/csv")


def render_price_forecast(hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower, horizon, sig):
    st.header("📈 Price Forecast")
    st.markdown("LightGBM model with 90% conformal prediction intervals. "
                "Backtest accuracy: **1.89% MAPE** over 2023–2024.")
    sig["horizon"] = horizon
    st.plotly_chart(
        make_forecast_chart(hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower, horizon, sig),
        use_container_width=True)
    st.subheader(f"Week-by-Week ({horizon} weeks)")
    base = fc_y[0]
    tbl = pd.DataFrame({
        "Week":       fc_x[:horizon],
        "Forecast":   [f"${v:.2f}" for v in fc_y[:horizon]],
        "Lower 90%":  [f"${v:.2f}" for v in fc_lower[:horizon]],
        "Upper 90%":  [f"${v:.2f}" for v in fc_upper[:horizon]],
        "vs. Now":    [f"{(v-base)/base*100:+.1f}%" for v in fc_y[:horizon]],
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)


def render_supply_indicators(eia_raw):
    st.header("🏭 Supply Indicators")
    st.markdown("Weekly EIA data on crude oil inventories, refinery utilisation, "
                "and natural gas storage — leading supply-side price signals.")
    inv = make_inventory_chart(eia_raw)
    if inv:
        st.plotly_chart(inv, use_container_width=True)
        st.markdown("<div class='insight-blue'><b>Inventory draws</b> signal tightening supply "
                    "and support prices. <b>Builds</b> above seasonal norms signal oversupply "
                    "and typically precede price declines by 2–4 weeks.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ref = _line_chart(eia_raw,"refinery_utilisation","#0a7c4e","Utilisation (%)","US Refinery Utilisation Rate",hline=80,yrange=[50,105])
        if ref:
            st.plotly_chart(ref, use_container_width=True)
            st.markdown("<div class='insight-green'>Above 90% → strong crude demand. Below 80% → weakness.</div>", unsafe_allow_html=True)
    with c2:
        gas = _line_chart(eia_raw,"natural_gas_storage","#7F77DD","Bcf","US Natural Gas Storage (Weekly)")
        if gas:
            st.plotly_chart(gas, use_container_width=True)
            st.markdown("<div class='insight-blue'>Below-average autumn/winter storage signals gas price upside.</div>", unsafe_allow_html=True)


def render_commodity_markets(fred_raw, wb_raw):
    st.header("🌍 Commodity Markets")
    st.markdown("Indexed commodity prices over the last 2 years (start = 100).")
    st.plotly_chart(make_commodity_chart(fred_raw, wb_raw), use_container_width=True)
    st.markdown("---")
    st.subheader("Current Prices")
    rows = []
    for lbl,col,df in [("WTI Crude Oil ($/bbl)","crude_oil_wti",fred_raw),
                        ("Natural Gas ($/MMBtu)","natural_gas_henry_hub",fred_raw),
                        ("Crude Avg ($/bbl)","crude_oil_avg",wb_raw),
                        ("Coal Australia ($/t)","coal_australia",wb_raw),
                        ("Fertiliser Urea ($/t)","fertiliser_urea",wb_raw),
                        ("Phosphate Rock ($/t)","phosphate_rock",wb_raw)]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 4:
            continue
        latest = float(s.iloc[-1]); prev = float(s.iloc[-2])
        chg = (latest-prev)/prev*100
        rows.append({"Commodity":lbl,"Latest":f"${latest:.2f}",
                     "Weekly Chg":f"{chg:+.1f}%",
                     "4-Week Avg":f"${float(s.iloc[-4:].mean()):.2f}",
                     "52W High":f"${float(s.iloc[-52:].max()):.2f}",
                     "52W Low":f"${float(s.iloc[-52:].min()):.2f}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_macro_environment(fred_raw):
    st.header("📊 Macro Environment")
    st.markdown("Key macroeconomic indicators used as model features.")
    st.plotly_chart(make_macro_chart(fred_raw), use_container_width=True)


def render_model_reliability(fc_x, fc_y, fc_upper, fc_lower, fc_actual):
    st.header("✅ Model Reliability")
    st.markdown("Backtest on 2023–2024 held-out test set.")
    st.plotly_chart(make_backtest_chart(fc_x, fc_y, fc_upper, fc_lower, fc_actual),
                    use_container_width=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MAPE","1.89%","vs 18% naive")
    c2.metric("RMSE","$1.73/bbl","weekly avg")
    c3.metric("Max Error","$4.81/bbl","worst week")
    c4.metric("PI Coverage","99%","target 90%")
    st.markdown("""
    <div class='insight-green'><b>What 1.89% MAPE means:</b> On a typical $80/bbl week,
    average error is $1.51/bbl. For 10,000 bbl/week, forecast cost uncertainty is ~$15,100/week
    — well within normal procurement tolerances.</div>
    <div class='insight-blue' style='margin-top:0.5rem;'><b>Confidence intervals</b> use
    conformal prediction — guaranteeing true prices fall inside the band ≥90% of the time.
    The 99% actual coverage means the model is slightly conservative — the right direction
    of error for risk-averse procurement teams.</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Known Limitations")
    st.markdown("""
    - **Geopolitical shocks** (war, sanctions) cannot be predicted from historical patterns
    - **OPEC+ decisions** can move prices within days — faster than the weekly forecast cadence
    - **Best horizon**: 1–4 weeks for operational decisions; 5–12 weeks for strategic planning
    - **Data lag**: FRED and EIA data is typically 1–5 days behind the actual event date
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    with st.spinner("Loading procurement intelligence..."):
        price_train, price_test, fred_raw, eia_raw, wb_raw, shap_df = load_all_data()
        lgbm, ensemble = load_models()

    with st.spinner("Running forecast model..."):
        hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower, fc_actual = build_forecast_data(
            lgbm, price_train, price_test)

    page, horizon = render_sidebar(fc_y, fc_actual)
    sig = compute_signal(fc_y, fc_upper, fc_lower, horizon)

    if   page == "📋 Weekly Briefing":
        render_weekly_briefing(hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower, sig, shap_df)
    elif page == "📈 Price Forecast":
        render_price_forecast(hist_x, hist_y, fc_x, fc_y, fc_upper, fc_lower, horizon, sig)
    elif page == "🏭 Supply Indicators":
        if eia_raw.empty:
            st.info("Supply indicator data is not available in this deployment. "
                    "Run the app locally with the full dataset to see EIA charts.")
        else:
            render_supply_indicators(eia_raw)

    elif page == "🌍 Commodity Markets":
        if fred_raw.empty and wb_raw.empty:
            st.info("Commodity market data is not available in this deployment. "
                    "Run the app locally with the full dataset to see commodity charts.")
        else:
            render_commodity_markets(fred_raw, wb_raw)

    elif page == "📊 Macro Environment":
        if fred_raw.empty:
            st.info("Macro indicator data is not available in this deployment. "
                    "Run the app locally with the full dataset to see macro charts.")
        else:
            render_macro_environment(fred_raw)


if __name__ == "__main__":
    main()
