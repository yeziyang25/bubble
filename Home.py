import streamlit as st
from data_prep import load_raw_data, load_raw_data_heatmap, process_data_for_date
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
st.set_page_config(
    page_title="ETF Flow & Tell",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f4f8 100%);
    }

    .global-x-main {
        font-size: 48px;
        font-weight: bold;
        color: #FF5722;
        font-family: 'Arial', 'Helvetica', sans-serif;
        letter-spacing: 3px;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .global-x-subtitle {
        font-size: 18px;
        color: #4682A9;
        font-family: 'Arial', 'Helvetica', sans-serif;
        margin: 5px 0 0 0;
        font-weight: normal;
    }
    h1 { 
        color: #00695C; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    h3 {
        color: #00695C;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 2px solid #FF5722;
        padding-bottom: 10px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #FF5722;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-left-color: #00695C;
    }
    
    .filters-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 30px;
        border-top: 3px solid #FF5722;
    }
    /* Multiselect improvements */
    .stMultiSelect > div > div > div {
        border-radius: 8px;
        border: 2px solid #BDC3C7;
        transition: border-color 0.3s ease;
    }
    .stMultiSelect > div > div > div:focus-within {
        border-color: #FF5722;
        box-shadow: 0 0 0 3px rgba(255, 87, 34, 0.1);
    }
    /* Selectbox improvements */
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 2px solid #BDC3C7;
        transition: border-color 0.3s ease;
    }
    .stSelectbox > div > div > div:focus-within {
        border-color: #FF5722;
        box-shadow: 0 0 0 3px rgba(255, 87, 34, 0.1);
    }
    /* Checkbox styling */
    .stCheckbox > label > div {
        color: #00695C;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def compute_trailing_12m_market_flows(
    process_fn,
    funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw,
    available_dates: list[pd.Timestamp],
    end_ts: pd.Timestamp
):
    # last 12 month-end dates up to end_ts
    months_asc = sorted([d for d in available_dates if d <= end_ts])
    last_12 = months_asc[-12:] if len(months_asc) >= 12 else months_asc

    labels = []
    totals_mn = []

    for d in last_12:
        d_str = d.strftime('%Y-%m-%d')
        df_m = process_fn(d_str, funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw)
        total = pd.to_numeric(df_m['Monthly Flow'], errors='coerce').sum()
        labels.append(d.strftime('%b %Y'))
        totals_mn.append(float(total) / 1e6)  # scale to millions

    return labels, totals_mn

onedrive_url = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/Eas53aR4lPlDn0ZlNHgX4ZABPDpH1Ign2mH4NGcJ0Hb80w?download=1"


funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw = load_raw_data(onedrive_url)
flow_date_cols = [c for c in flow_df_raw.columns if c != 'ETF']
available_dates = sorted(pd.to_datetime(flow_date_cols, errors='coerce').dropna(), reverse=True)
available_date_strs = [d.strftime('%Y-%m-%d') for d in available_dates]
selected_date = st.selectbox(
        "📅 Analysis Date",
        options=available_date_strs,
        index=0,
        help="Select the date for your analysis")
selected_ts = pd.to_datetime(selected_date)

df = process_data_for_date(selected_date, funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw)

labels_12m, totals_12m_mn = compute_trailing_12m_market_flows(
    process_data_for_date,
    funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw,
    available_dates, selected_ts
)


category_flow_summary = df.groupby("Category")[['Monthly Flow','YTD Flow']].sum()

category_flow_summary_sorted = category_flow_summary.sort_values(by='Monthly Flow')
top5_inflow = category_flow_summary_sorted.tail(5)
top5_outflow = category_flow_summary_sorted.head(5)

SCALE_DIVISOR = 1e6  
MONTHLY_COLOR_INFLOW = "#4472C4"  
MONTHLY_COLOR_OUTFLOW = "#ED7D31"  
YTD_COLOR = "#FFC000"       





def make_single_series_bar(x_labels, y_vals_mn, title: str, color: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_vals_mn,
        name="Net flow (mn)",
        marker_color=color,
        text=[f"{v:,.0f}" for v in y_vals_mn],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Flow: %{y:,.0f} mn<extra></extra>"
    ))
    fig.update_layout(
        title=title, title_x=0.5, barmode="group", bargap=0.3,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(
        title_text=None, tickangle=0,
        showline=True, linecolor="#444", showgrid=False,
        tickfont=dict(size=15)
    )
    fig.update_yaxes(
        title_text="Flow (Millions)", tickformat=",.0f",
        zeroline=True, zerolinecolor="#ddd",
        showline=False, linecolor="#444", showgrid=False
    )
    return fig
def make_bar_chart(df: pd.DataFrame, title: str, monthly_color: str, ytd_color: str):
    def _to_millions(vals):
        return [float(v) / 1e6 for v in vals]

    # helper to insert line breaks in labels
    def wrap_labels(labels, max_words=2):
        wrapped = []
        for lbl in labels:
            words = lbl.split()
            lines = []
            for i in range(0, len(words), max_words):
                lines.append(" ".join(words[i:i+max_words]))
            wrapped.append("<br>".join(lines))
        return wrapped

    categories = df.index.tolist()
    categories_wrapped = wrap_labels(categories, max_words=2)

    monthly_vals_mn = _to_millions(df['Monthly Flow'])
    ytd_vals_mn = _to_millions(df['YTD Flow'])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories_wrapped, y=monthly_vals_mn, name="1 mo flow (mn)",
        marker_color=monthly_color,
        text=[f"{v:,.1f}" for v in monthly_vals_mn], textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=categories_wrapped, y=ytd_vals_mn, name="ytd flow (mn)",
        marker_color=ytd_color,
        text=[f"{v:,.1f}" for v in ytd_vals_mn], textposition="outside"
    ))

    fig.update_layout(
        title=title, title_x=0.5, barmode="group", bargap=0.3,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(
        title_text=None, tickangle=0,  # keep horizontal now since we wrapped
        showline=True, linecolor="#444", showgrid=False,
        tickfont=dict(size=15)
    )
    fig.update_yaxes(
        title_text="Flow (Millions)", tickformat=",.0f", zeroline=True, zerolinecolor="#ddd",
        showline=False, linecolor="#444", showgrid=False
    )

    return fig

category_flow_summary_sorted = category_flow_summary.sort_values(by="Monthly Flow")
top5_inflow = category_flow_summary_sorted.tail(5)
top5_outflow = category_flow_summary_sorted.head(5)


fig_12m = make_single_series_bar(
    labels_12m,
    totals_12m_mn,
    "Past 12 Months — Total ETF Net Flow (Adjusted)",
    MONTHLY_COLOR_INFLOW
)
st.subheader("Past 12 Months — Total ETF Net Flow")
st.plotly_chart(fig_12m, use_container_width=True)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 5 Category Inflow")
    fig_inflow = make_bar_chart(top5_inflow, "Top 5 Category Inflow", MONTHLY_COLOR_INFLOW, YTD_COLOR)
    st.plotly_chart(fig_inflow, use_container_width=True)

with col2:
    st.subheader("Top 5 Category Outflow")
    fig_outflow = make_bar_chart(top5_outflow, "Top 5 Category Outflow", MONTHLY_COLOR_OUTFLOW, YTD_COLOR)
    st.plotly_chart(fig_outflow, use_container_width=True)



flow_long = flow_df_raw.melt(
    id_vars='ETF',
    value_vars=flow_date_cols,
    var_name='Date',
    value_name='Flow'
)
flow_long['Date'] = pd.to_datetime(flow_long['Date'], errors='coerce')
flow_long.dropna(subset=['Date', 'Flow'], inplace=True)
flow_long.sort_values(['ETF', 'Date'], inplace=True)

flow_dict = (
    flow_long
    .groupby('ETF')
    .apply(lambda g: dict(zip(g['Date'], g['Flow'])))
    .reset_index(name='Past Flows')     
)

df = df.merge(flow_dict, on='ETF', how='left')
df['Past Flows'] = df['Past Flows'].apply(lambda x: x if isinstance(x, dict) else {})


flow_lookup = dict(
    zip(flow_dict['ETF'].astype(str).str.strip(), flow_dict['Past Flows'])
)

u_mask = (
    df['ETF'].astype(str).str.endswith('(U)', na=False)
    & df['Past Flows'].apply(lambda d: isinstance(d, dict) and len(d) == 0)
)

def strip_u_suffix(etf: str) -> str:
    return re.sub(r'\(U\)$', '', str(etf)).strip()

def multiply_dict_values(d: dict, factor: float) -> dict:
    if not isinstance(d, dict) or not d:
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v) * factor if v is not None else v
        except (TypeError, ValueError):
            out[k] = v  
    return out

base_keys = df.loc[u_mask, 'ETF'].apply(strip_u_suffix)
base_hist = base_keys.map(flow_lookup)
imputed_hist = base_hist.apply(lambda d: multiply_dict_values(d, 2.0))
df.loc[u_mask, 'Past Flows'] = imputed_hist



def last_12m_flow_list(flow_map: dict, end_ts: pd.Timestamp) -> list[float]:
    """Return list of 12 monthly flows ending at end_ts (inclusive), in chronological order."""
    if not isinstance(flow_map, dict):
        return [0.0] * 12


    desired_months = [d for d in sorted(available_dates) if d <= end_ts][-12:]
    if len(desired_months) < 12:
        desired_months = [desired_months[0]] * (12 - len(desired_months)) + desired_months

    vals = [float(flow_map.get(m, 0.0)) for m in desired_months]
    return vals


def add_trend_and_scale(df_in: pd.DataFrame, normalize: bool = True, mark_symbol: str = "🔶") -> pd.DataFrame:
    """
    - Adds a brand mark (emoji) to Fund Name for Global X / BetaPro rows
    - Scales Flow/AUM to millions
    - Creates 'Flow Trend (12M)' (optionally min-max normalized)
    - Adds '_brand' boolean flag
    """
    out = df_in.copy()

    # 1) Brand mask
    out["_brand"] = out["Fund Name"].str.contains(r"(Global X|BetaPro)", case=False, na=False)

    # 2) Make Fund Name marking idempotent (strip any prior mark first)
    out["Fund Name"] = out["Fund Name"].astype(str).str.replace(r"^(🔶|🟠)\s+", "", regex=True)
    # Then prepend the orange symbol for brand rows
    out.loc[out["_brand"], "Fund Name"] = mark_symbol + " " + out.loc[out["_brand"], "Fund Name"]

    # 3) 12M trend (in millions, then optional 0–1 normalization)
    def _trend_list(row):
        flows_map = row.get("Past Flows", {})
        vals = [float(v) / 1e6 for v in last_12m_flow_list(flows_map, selected_ts)]
        if not normalize:
            return vals
        vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 0.0)
        if vmax == vmin:
            return [0.0 for _ in vals]
        return [(x - vmin) / (vmax - vmin) for x in vals]

    out["Flow Trend (12M)"] = out.apply(_trend_list, axis=1)

    # 4) Scale to millions
    out["Flow"] = out["Flow"].astype(float) / 1e6
    out["AUM"]  = out["AUM"].astype(float)  / 1e6

    return out

def style_brand_rows(df: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    mask = df.get("_brand", False)
    styles.loc[mask, :] = "color: #ED7D31; font-weight: 600;"
    return styles


top15_inflow  = df.nlargest(15, 'Monthly Flow')[['Fund Name','ETF','Monthly Flow','AUM','Past Flows']].rename(columns={'ETF':'Ticker','Monthly Flow':'Flow'})
top15_outflow = df.nsmallest(15, 'Monthly Flow')[['Fund Name','ETF','Monthly Flow','AUM','Past Flows']].rename(columns={'ETF':'Ticker','Monthly Flow':'Flow'})

df_small_aum = df[df['AUM'] < 1_000_000_000]
top15_inflow_small_aum  = df_small_aum.nlargest(15, 'Monthly Flow')[['Fund Name','ETF','Monthly Flow','AUM','Past Flows']].rename(columns={'ETF':'Ticker','Monthly Flow':'Flow'})
top15_outflow_small_aum = df_small_aum.nsmallest(15, 'Monthly Flow')[['Fund Name','ETF','Monthly Flow','AUM','Past Flows']].rename(columns={'ETF':'Ticker','Monthly Flow':'Flow'})

top15_inflow  = add_trend_and_scale(top15_inflow)
top15_outflow = add_trend_and_scale(top15_outflow)
top15_inflow_small_aum  = add_trend_and_scale(top15_inflow_small_aum)
top15_outflow_small_aum = add_trend_and_scale(top15_outflow_small_aum)



for dfx in (top15_inflow, top15_outflow, top15_inflow_small_aum, top15_outflow_small_aum):
    if 'Past Flows' in dfx.columns:
        dfx.drop(columns=['Past Flows'], inplace=True)


def render_table_with_spark(df_disp: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")

    # Ensure every row has a list-like sequence for the sparkline
    trend_series = df_disp["Flow Trend (12M)"].apply(
        lambda v: v if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 else [0.0]
    )

    # Safe global y-range for the sparkline column
    try:
        all_vals = np.concatenate([np.asarray(x, dtype=float) for x in trend_series])
        ymin = float(np.nanmin(all_vals)) if all_vals.size else 0.0
        ymax = float(np.nanmax(all_vals)) if all_vals.size else 1.0
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = 0.0, 1.0
    except Exception:
        ymin, ymax = 0.0, 1.0

    df_disp = df_disp.copy()
    df_disp["Flow Trend (12M)"] = trend_series
    df_show = df_disp.rename(columns={'Flow': 'Flow (M)', 'AUM': 'AUM (M)'}).copy()
    df_show['Flow (M)'] = df_show['Flow (M)'].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
    df_show['AUM (M)']  = df_show['AUM (M)'].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")

    st.dataframe(
        df_show,
        use_container_width=True,
        height=515,
        hide_index=True,
        column_config={
            # Now they are text columns (so they show commas)
            "Flow (M)": st.column_config.TextColumn(width="small"),
            "AUM (M)":  st.column_config.TextColumn(width="small"),
            "Flow Trend (12M)": st.column_config.LineChartColumn(
                label="Flow Trend (12M, norm)",
                y_min=ymin, y_max=ymax, width="small"
            ),
            "Ticker": st.column_config.TextColumn(width="small"),
            "Fund Name": st.column_config.TextColumn(width="large"),
        }
    )

st.subheader("Top ETF Inflows & Outflows")
c1, c2 = st.columns(2)
with c1:
    render_table_with_spark(top15_inflow[['Fund Name','Ticker','Flow','AUM','Flow Trend (12M)']], "Top 15 Inflow")
with c2:
    render_table_with_spark(top15_outflow[['Fund Name','Ticker','Flow','AUM','Flow Trend (12M)']], "Top 15 Outflow")

c3, c4 = st.columns(2)
with c3:
    render_table_with_spark(top15_inflow_small_aum[['Fund Name','Ticker','Flow','AUM','Flow Trend (12M)']], "Top 15 Inflow (AUM < $1B)")
with c4:
    render_table_with_spark(top15_outflow_small_aum[['Fund Name','Ticker','Flow','AUM','Flow Trend (12M)']], "Top 15 Outflow (AUM < $1B)")




brand_mask = df['Fund Name'].str.contains(r'(Global X|BetaPro)', case=False, na=False)
small_aum_mask = df['AUM'] < 1_000_000_000

gx_bp_small = df.loc[
    brand_mask & small_aum_mask,
    ['Fund Name', 'ETF', 'Monthly Flow', 'AUM', 'Past Flows']
].copy()

top10_brand_small_inflow = (
    gx_bp_small
    .nlargest(15, 'Monthly Flow')
    .rename(columns={'ETF': 'Ticker', 'Monthly Flow': 'Flow'})
)

top10_brand_small_inflow = add_trend_and_scale(top10_brand_small_inflow, normalize=True)


# (Inception year == selected_ts.year)
inception_dt = pd.to_datetime(df['Inception'], errors='coerce')
launched_this_year_mask = inception_dt.dt.year.eq(selected_ts.year)

ytd_new = df.loc[
    launched_this_year_mask,
    ['Fund Name', 'ETF', 'YTD Flow', 'AUM', 'Past Flows', 'Inception']
].copy()

top10_ytd_new = (
    ytd_new
    .nlargest(20, 'YTD Flow')
    .rename(columns={'ETF': 'Ticker', 'YTD Flow': 'Flow'})
)


top10_ytd_new = add_trend_and_scale(top10_ytd_new, normalize=True)
c3, c4 = st.columns(2)
with c3:
    render_table_with_spark(
        top10_brand_small_inflow[['Fund Name', 'Ticker', 'Flow', 'AUM', 'Flow Trend (12M)']],
        "Top 15 Monthly Inflow (< $1B) — Global X / BetaPro"
    )

with c4:
    render_table_with_spark(
        top10_ytd_new[['Fund Name', 'Ticker', 'Flow', 'AUM', 'Flow Trend (12M)']],
        f"Top 20 YTD Inflow — Launched in {selected_ts.year}"
    )