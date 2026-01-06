import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from data_prep import (
    load_raw_data_heatmap,              
    build_trailing3m_flow_pct_table,    
)

st.set_page_config(page_title="ETF Flow Heatmaps", layout="wide")
st.title("Trailing 3-Month Flow % of AUM — Heatmaps")


ONEDRIVE_URL = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/EXAiBXWjq6ZMmkh17unVDIIBrLapRtA8PoYDycRTegM6EA?download=1"

DEFAULT_MIXED_ROWS = [
    "Lightly Leveraged Options Strategy",
    "Intl. Developed Equity",
    "Asset Allocation Portfolio",
    "Commodity",
    "Emerging Markets Equity",
    "Dividend/Income",
    "Options-Based Strategies",
    "Canada Equity",
    "Global Equity",
    "Canada Fixed Income",
    "Thematic",
    "U.S. Fixed Income",
    "U.S. Equity",
    "Crypto-Asset",
    "ESG",
    "Alternative Investment",
    "Preferred Share",
    "BetaPro",
    "Sector Equity",
]

fund_df, aum_df, flow_df = load_raw_data_heatmap(ONEDRIVE_URL)
flow_date_cols = [c for c in flow_df.columns if c != 'ETF']
available_dates = sorted(pd.to_datetime(flow_date_cols, errors='coerce').dropna(), reverse=True)
available_date_strs = [d.strftime('%Y-%m-%d') for d in available_dates]
all_categories = sorted(fund_df["Category"].dropna().unique().tolist())
with st.sidebar:
    st.header("Controls")
    asof = st.selectbox(
        "As-of month",
        options=available_date_strs,
        index=0
    )
    
    st.caption("This page will render: the mixed default table first, then one table per Category.")
    
    hide_empty = st.checkbox("Hide empty tables (no rows / all zeros)", value=True)

def style_heatmap(df: pd.DataFrame, *, q: float = 0.95, gamma: float = 0.6):
    """
    df: values are RATIOS (e.g., 0.012 = 1.2%)
    q:  quantile cap for robust scaling (0.95 = 95th percentile of |values|)
    gamma: 0<gamma<=1.0; lower -> more contrast for small/medium values
    """


    vals = df.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        vmax = 1.0
    else:
        # robust symmetric bound around 0
        vmax = float(np.nanpercentile(np.abs(finite), q * 100))
        if vmax == 0:
            vmax = float(np.nanmax(np.abs(finite)) or 1.0)

    # clip to robust range, then apply a power transform for nicer spread
    a = np.minimum(np.abs(vals), vmax) / vmax
    a = np.power(a, gamma)  # 0.6–0.8 is usually good
    sign = np.sign(vals)

    # build a 0..1 gradient map with 0.5 at zero (diverging)
    g = 0.5 + 0.5 * sign * a
    g = np.where(np.isfinite(g), g, 0.5)  # NaNs to neutral
    gmap = pd.DataFrame(g, index=df.index, columns=df.columns)

    return (
        df.style
          .format("{:.1%}")  # display ratios as percents
          .background_gradient(cmap="RdYlGn", gmap=gmap, axis=None)  # use custom map
          .set_properties(**{"text-align": "center"})
    )

st.subheader("Mixed Default Rows (no Category specified)")
try:
    df_mixed = build_trailing3m_flow_pct_table(
        fund_df, aum_df, flow_df,
        asof=asof,
        category=None,
        custom_rows=DEFAULT_MIXED_ROWS
    )


    
    if hide_empty and (df_mixed.empty or (df_mixed.fillna(0).to_numpy() == 0).all()):
        st.info("No data for the selected period.")
    else:
        st.dataframe(style_heatmap(df_mixed), use_container_width=True)
except Exception as e:
    st.error(f"Failed to build mixed table: {e}")

st.markdown("---")
all_categories =['Lightly Leveraged Options Strategy','Intl. Developed Equity','Asset Allocation Portfolio', 'Commodity', 'Emerging Markets Equity',
'Options-Based Strategies', 'Canada Equity', 'Global Equity', 'Thematic','U.S. Fixed Income', 'U.S. Equity','Canada Fixed Income', 'Sector Equity']
for cat in all_categories:
    st.subheader(cat)
    try:
        df_cat = build_trailing3m_flow_pct_table(
            fund_df, aum_df, flow_df,
            asof=asof,
            category=cat
        )
        if hide_empty and (df_cat.empty or (df_cat.fillna(0).to_numpy() == 0).all()):
            st.caption("No data (hidden).")
            continue
        st.dataframe(style_heatmap(df_cat), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to build table for '{cat}': {e}")
    st.markdown("---")
