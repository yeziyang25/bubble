import streamlit as st
from data_prep import load_raw_data, process_data_for_date
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from config import apply_common_styling, render_header, render_metric_card, format_large_number
from analytics_utils import (
    calculate_market_concentration, 
    calculate_provider_market_share,
    create_provider_market_share_chart,
    create_concentration_chart,
    create_flow_momentum_indicator,
    create_flow_momentum_chart,
    calculate_category_performance_metrics
)

# Page config + styling
st.set_page_config(
    page_title="Canadian ETF Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply common styling
apply_common_styling()

# Render professional header
render_header(
    "Canadian ETF Market Intelligence Dashboard",
    "Comprehensive Analysis of ETF Flows, Performance, and Market Dynamics"
)

# Constants
SCALE_DIVISOR = 1e6
MONTHLY_COLOR_INFLOW = "#4472C4"
MONTHLY_COLOR_OUTFLOW = "#ED7D31"
YTD_COLOR = "#FFC000"

# Data loading
onedrive_url = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/Eas53aR4lPlDn0ZlNHgX4ZABPDpH1Ign2mH4NGcJ0Hb80w?download=1"
funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw = load_raw_data(onedrive_url)

# Available dates from flow sheet columns
flow_date_cols = [c for c in flow_df_raw.columns if c != "ETF"]
available_dates_desc = sorted(pd.to_datetime(flow_date_cols, errors="coerce").dropna(), reverse=True)
available_date_strs = [d.strftime("%Y-%m-%d") for d in available_dates_desc]

# Provider list
provider_col = funds_df_raw["ETF Provider"].fillna("Unknown").astype(str).str.strip()
provider_options = ["All (Industry)"] + sorted(provider_col.unique())

# Caching + helpers
@st.cache_data(show_spinner=False)
def process_cached(date_str: str) -> pd.DataFrame:
    return process_data_for_date(date_str, funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw)

# Lookup maps (for robustness if processed df is missing columns)
_ticker_key = funds_df_raw["Ticker"].astype(str).str.strip()

_provider_map = (
    funds_df_raw.assign(_t=_ticker_key)[["_t", "ETF Provider"]]
    .dropna(subset=["_t"])
    .set_index("_t")["ETF Provider"]
    .to_dict()
)

# Try common names for classification columns in your raw fund table
RAW_CAT_COL = "Category"
RAW_SEC_COL = "Secondary Category"

_category_map = {}
_secondary_map = {}
if RAW_CAT_COL in funds_df_raw.columns:
    _category_map = (
        funds_df_raw.assign(_t=_ticker_key)[["_t", RAW_CAT_COL]]
        .dropna(subset=["_t"])
        .set_index("_t")[RAW_CAT_COL]
        .to_dict()
    )
if RAW_SEC_COL in funds_df_raw.columns:
    _secondary_map = (
        funds_df_raw.assign(_t=_ticker_key)[["_t", RAW_SEC_COL]]
        .dropna(subset=["_t"])
        .set_index("_t")[RAW_SEC_COL]
        .to_dict()
    )

def ensure_provider_col(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    if "ETF Provider" not in out.columns:
        out["ETF Provider"] = out["ETF"].astype(str).str.strip().map(_provider_map)
    out["ETF Provider"] = out["ETF Provider"].fillna("Unknown").astype(str).str.strip()
    return out

def ensure_classification_cols(df_in: pd.DataFrame, cat_col="Category", sec_col="Secondary Category") -> pd.DataFrame:
    """
    Ensures Category / Secondary Category exist on processed df (maps from funds_df_raw if missing).
    """
    out = df_in.copy()

    if cat_col not in out.columns and _category_map:
        out[cat_col] = out["ETF"].astype(str).str.strip().map(_category_map)
    if sec_col not in out.columns and _secondary_map:
        out[sec_col] = out["ETF"].astype(str).str.strip().map(_secondary_map)

    if cat_col in out.columns:
        out[cat_col] = out[cat_col].fillna("Unknown").astype(str)
    if sec_col in out.columns:
        out[sec_col] = out[sec_col].fillna("Unknown").astype(str)

    return out

def apply_universe_filters(df_in: pd.DataFrame, category_sel=None, secondary_sel=None,
                          cat_col="Category", sec_col="Secondary Category") -> pd.DataFrame:
    out = df_in.copy()
    if category_sel and cat_col in out.columns:
        out = out[out[cat_col].astype(str).isin(category_sel)]
    if secondary_sel and sec_col in out.columns:
        out = out[out[sec_col].astype(str).isin(secondary_sel)]
    return out

# Trailing-series computations (now filter-aware)
def compute_trailing_13m_market_flows(end_ts: pd.Timestamp, category_sel=None, secondary_sel=None):
    """Industry total flows for past 13 month-ends up to end_ts (optionally filtered by Category/Secondary)."""
    months_asc = sorted([d for d in available_dates_desc if d <= end_ts])
    last_13 = months_asc[-13:] if len(months_asc) >= 13 else months_asc

    labels, totals_mn = [], []
    for d in last_13:
        d_str = d.strftime("%Y-%m-%d")
        df_m = process_cached(d_str)
        df_m = ensure_provider_col(df_m)
        df_m = ensure_classification_cols(df_m)
        df_m = apply_universe_filters(df_m, category_sel, secondary_sel)

        total = pd.to_numeric(df_m["Monthly Flow"], errors="coerce").sum()
        labels.append(d.strftime("%b %Y"))
        totals_mn.append(float(total) / 1e6)
    return labels, totals_mn

def compute_trailing_13m_gx_flows(end_ts: pd.Timestamp, category_sel=None, secondary_sel=None):
    """Global X total flows for past 13 month-ends up to end_ts (optionally filtered by Category/Secondary)."""
    months_asc = sorted([d for d in available_dates_desc if d <= end_ts])
    last_13 = months_asc[-13:] if len(months_asc) >= 13 else months_asc

    labels, totals_mn = [], []
    for d in last_13:
        d_str = d.strftime("%Y-%m-%d")
        df_m = process_cached(d_str)
        df_m = ensure_provider_col(df_m)
        df_m = ensure_classification_cols(df_m)
        df_m = apply_universe_filters(df_m, category_sel, secondary_sel)

        gx_mask = df_m["ETF Provider"].str.contains(r"(global x|betapro)", case=False, na=False)
        total = pd.to_numeric(df_m.loc[gx_mask, "Monthly Flow"], errors="coerce").sum()
        labels.append(d.strftime("%b %Y"))
        totals_mn.append(float(total) / 1e6)
    return labels, totals_mn

def compute_13m_provider_vs_gx_series(end_ts: pd.Timestamp, provider_choice: str | None,
                                     category_sel=None, secondary_sel=None):
    """
    If provider_choice is None:
      bars = Industry vs Global X
      line = Industry total AUM
    Else:
      bars = Provider vs Global X
      line = Provider total AUM
    All series are optionally filtered by Category/Secondary.
    """
    months_asc = sorted([d for d in available_dates_desc if d <= end_ts])
    last_13 = months_asc[-13:] if len(months_asc) >= 13 else months_asc

    if provider_choice and re.search(r"(global x|betapro)", provider_choice, flags=re.I):
        provider_choice = None

    labels = []
    scope_flow_mn, gx_flow_mn, scope_aum_bn = [], [], []

    for d in last_13:
        d_str = d.strftime("%Y-%m-%d")
        df_m = process_cached(d_str)
        df_m = ensure_provider_col(df_m)
        df_m = ensure_classification_cols(df_m)
        df_m = apply_universe_filters(df_m, category_sel, secondary_sel)

        if provider_choice is None:
            df_scope = df_m
            scope_name = "Industry"
        else:
            df_scope = df_m[df_m["ETF Provider"].eq(provider_choice)]
            scope_name = provider_choice

        gx_mask = df_m["ETF Provider"].str.contains(r"(global x|betapro)", case=False, na=False)
        df_gx = df_m[gx_mask]

        scope_flow = pd.to_numeric(df_scope.get("Monthly Flow", 0), errors="coerce").sum()
        gx_flow = pd.to_numeric(df_gx.get("Monthly Flow", 0), errors="coerce").sum()
        scope_aum = pd.to_numeric(df_scope.get("AUM", 0), errors="coerce").sum()

        labels.append(d.strftime("%b %Y"))
        scope_flow_mn.append(float(scope_flow) / 1e6)
        gx_flow_mn.append(float(gx_flow) / 1e6)
        scope_aum_bn.append(float(scope_aum) / 1e9)

    return labels, scope_flow_mn, gx_flow_mn, scope_aum_bn, scope_name

# Chart builders
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

    def wrap_labels(labels, max_words=2):
        wrapped = []
        for lbl in labels:
            words = str(lbl).split()
            lines = []
            for i in range(0, len(words), max_words):
                lines.append(" ".join(words[i:i+max_words]))
            wrapped.append("<br>".join(lines))
        return wrapped

    categories = df.index.tolist()
    categories_wrapped = wrap_labels(categories, max_words=2)

    monthly_vals_mn = _to_millions(df["Monthly Flow"]) if "Monthly Flow" in df.columns else []
    ytd_vals_mn = _to_millions(df["YTD Flow"]) if "YTD Flow" in df.columns else []

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
        title_text=None, tickangle=30,
        showline=True, linecolor="#444", showgrid=False,
        tickfont=dict(size=12)
    )
    fig.update_yaxes(
        title_text="Flow (Millions)", tickformat=",.0f", zeroline=True, zerolinecolor="#ddd",
        showline=False, linecolor="#444", showgrid=False
    )
    return fig

def make_provider_vs_gx_flow_with_aum_line(
    labels,
    scope_flow_mn,
    gx_flow_mn,
    scope_aum_bn,
    scope_name: str,
    scope_bar_color: str = "#4472C4",
    gx_bar_color: str = "#ED7D31",
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=labels, y=scope_flow_mn,
            name=f"{scope_name} Flow (mn)",
            marker_color=scope_bar_color,
            text=[f"{v:,.0f}" for v in scope_flow_mn],
            hovertemplate="<b>%{x}</b><br>Flow: %{y:,.0f} mn<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(
            x=labels, y=gx_flow_mn,
            name="Global X Flow (mn)",
            marker_color=gx_bar_color,
            text=[f"{v:,.0f}" for v in gx_flow_mn],
            hovertemplate="<b>%{x}</b><br>Global X Flow: %{y:,.0f} mn<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=labels, y=scope_aum_bn,
            name=f"{scope_name} AUM (bn)",
            mode="lines+markers",
            hovertemplate="<b>%{x}</b><br>AUM: %{y:,.1f} bn<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"{scope_name} vs Global X — Monthly Flows (Bars) + {scope_name} AUM (Line)",
        title_x=0.5,
        barmode="group",
        bargap=0.25,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=70, b=30),
    )

    fig.update_xaxes(showline=True, linecolor="#444", showgrid=False, tickangle=0)
    fig.update_yaxes(
        title_text="Flow (Millions)",
        tickformat=",.0f",
        zeroline=True,
        zerolinecolor="#ddd",
        showgrid=False,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="AUM (Billions)",
        tickformat=",.1f",
        showgrid=False,
        secondary_y=True,
    )
    return fig

# Filters section with improved layout
st.markdown("### 🎛️ Dashboard Controls")

filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 3])

with filter_col1:
    selected_date = st.selectbox(
        "📅 Analysis Date",
        options=available_date_strs,
        index=0,
        help="Select the date for your analysis"
    )
    selected_ts = pd.to_datetime(selected_date)

with filter_col2:
    selected_prov = st.selectbox(
        "🏢 ETF Provider",
        options=provider_options,
        index=0,
        help="All (Industry) shows the full market. Otherwise compares that provider vs Global X."
    )
    provider_choice = None if selected_prov == "All (Industry)" else selected_prov

with filter_col3:
    # Display data freshness
    st.markdown(f"""
    <div style='padding: 10px; background: #E8F5E9; border-radius: 8px; border-left: 4px solid #4CAF50;'>
        <div style='font-size: 12px; color: #666; margin-bottom: 5px;'>📊 Data as of</div>
        <div style='font-size: 18px; color: #00695C; font-weight: bold;'>{selected_ts.strftime('%B %d, %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

# Main dataframe for selected date
df = process_cached(selected_date)
df = ensure_provider_col(df)
df = ensure_classification_cols(df)

# NEW: Category + Secondary Category filters
CAT_COL = "Category"
SEC_COL = "Secondary Category"

cat_options = sorted(df[CAT_COL].dropna().astype(str).unique()) if CAT_COL in df.columns else []

cfa, cfb = st.columns(2)

with cfa:
    selected_categories = st.multiselect(
        "📂 Category",
        options=cat_options,
        default=[],
        help="Leave blank to include all categories."
    )

# Secondary options depend on Category selection (but still allow Secondary-only filtering)
if SEC_COL in df.columns:
    if selected_categories and CAT_COL in df.columns:
        sec_options = sorted(
            df.loc[df[CAT_COL].astype(str).isin(selected_categories), SEC_COL]
            .dropna().astype(str).unique()
        )
    else:
        sec_options = sorted(df[SEC_COL].dropna().astype(str).unique())
else:
    sec_options = []

with cfb:
    selected_secondary = st.multiselect(
        "🧩 Secondary Category",
        options=sec_options,
        default=[],
        help="You can filter by Secondary Category without selecting Category."
    )

# Apply universe filters to df for all tables / summaries below
df = apply_universe_filters(df, selected_categories, selected_secondary, CAT_COL, SEC_COL)
st.caption(f"Universe: {len(df):,} ETFs after filters")

# Provider vs GX chart (filtered universe)
labels_cmp, scope_flow_mn, gx_flow_mn, scope_aum_bn, scope_name = compute_13m_provider_vs_gx_series(
    end_ts=selected_ts,
    provider_choice=provider_choice,
    category_sel=selected_categories,
    secondary_sel=selected_secondary
)

fig_cmp = make_provider_vs_gx_flow_with_aum_line(
    labels_cmp,
    scope_flow_mn,
    gx_flow_mn,
    scope_aum_bn,
    scope_name=scope_name
)
st.subheader("Provider vs Global X — Monthly Flows (Bars) + AUM (Line)")
st.plotly_chart(fig_cmp, use_container_width=True)

# Industry trailing 13M bar (filtered universe)
labels_12m, totals_12m_mn = compute_trailing_13m_market_flows(
    selected_ts,
    category_sel=selected_categories,
    secondary_sel=selected_secondary
)
fig_12m = make_single_series_bar(
    labels_12m,
    totals_12m_mn,
    "Past 13 Months — Total ETF Net Flow (Adjusted)",
    MONTHLY_COLOR_INFLOW
)
st.subheader("Past 13 Months — Total ETF Net Flow")
st.plotly_chart(fig_12m, use_container_width=True)

# Global X trailing 13M bar (filtered universe)
labels_12m_gx, totals_12m_mn_gx = compute_trailing_13m_gx_flows(
    selected_ts,
    category_sel=selected_categories,
    secondary_sel=selected_secondary
)
fig_12m_GX = make_single_series_bar(
    labels_12m_gx,
    totals_12m_mn_gx,
    "Global X — Past 13 Months — Total ETF Net Flow (Adjusted)",
    MONTHLY_COLOR_OUTFLOW
)
st.subheader("Global X — Past 13 Months — Total ETF Net Flow")
st.plotly_chart(fig_12m_GX, use_container_width=True)

# If filters lead to empty df on selected date, avoid table errors
if df.empty:
    st.info("No ETFs match the selected Category/Secondary Category filters for this analysis date. Charts above still reflect the filtered universe across time.")
    st.stop()

# Category flow summaries (already filtered via df)
category_flow_summary = df.groupby("Category")[["Monthly Flow", "YTD Flow"]].sum()
category_flow_summary_sorted = category_flow_summary.sort_values(by="Monthly Flow")
top10_inflow = category_flow_summary_sorted.tail(10)
top10_outflow = category_flow_summary_sorted.head(10)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 5 Category Inflow")
    fig_inflow = make_bar_chart(top10_inflow, "Top 5 Category Inflow", MONTHLY_COLOR_INFLOW, YTD_COLOR)
    st.plotly_chart(fig_inflow, use_container_width=True)

with col2:
    st.subheader("Top 5 Category Outflow")
    fig_outflow = make_bar_chart(top10_outflow, "Top 5 Category Outflow", MONTHLY_COLOR_OUTFLOW, YTD_COLOR)
    st.plotly_chart(fig_outflow, use_container_width=True)

# ========== NEW: MARKET INTELLIGENCE SECTION ==========
st.markdown("---")
st.markdown("## 📊 Market Intelligence & Analytics")

# Row 1: Market Concentration and Provider Market Share
col_conc, col_provider = st.columns(2)

with col_conc:
    st.markdown("### Market Concentration Analysis")
    concentration_data = calculate_market_concentration(df, top_n=10)
    
    if concentration_data:
        # Display HHI metric
        hhi = concentration_data['hhi']
        hhi_interpretation = "Highly Concentrated" if hhi > 2500 else "Moderately Concentrated" if hhi > 1500 else "Competitive"
        
        col_hhi1, col_hhi2 = st.columns(2)
        with col_hhi1:
            st.markdown(
                render_metric_card(
                    "Top 10 Concentration",
                    f"{concentration_data['concentration_pct']:.1f}%"
                ),
                unsafe_allow_html=True
            )
        with col_hhi2:
            st.markdown(
                render_metric_card(
                    "HHI Score",
                    f"{hhi:.0f}",
                    delta=hhi_interpretation
                ),
                unsafe_allow_html=True
            )
        
        # Concentration chart
        fig_conc = create_concentration_chart(concentration_data)
        if fig_conc:
            st.plotly_chart(fig_conc, use_container_width=True)
    else:
        st.info("Insufficient data for concentration analysis")

with col_provider:
    st.markdown("### Provider Market Share")
    provider_stats = calculate_provider_market_share(df)
    
    if not provider_stats.empty:
        # Top 3 providers metrics
        col_p1, col_p2, col_p3 = st.columns(3)
        
        for idx, col in enumerate([col_p1, col_p2, col_p3]):
            if idx < len(provider_stats):
                provider_row = provider_stats.iloc[idx]
                with col:
                    st.markdown(
                        render_metric_card(
                            provider_row['Provider'],
                            f"{provider_row['Market Share %']:.1f}%",
                            delta=f"{provider_row['Number of ETFs']:.0f} ETFs"
                        ),
                        unsafe_allow_html=True
                    )
        
        # Provider pie chart
        fig_provider = create_provider_market_share_chart(provider_stats, top_n=8)
        if fig_provider:
            st.plotly_chart(fig_provider, use_container_width=True)
    else:
        st.info("Insufficient data for provider analysis")

# Row 2: Flow Momentum and Category Performance
st.markdown("---")
col_momentum, col_category_perf = st.columns(2)

with col_momentum:
    st.markdown("### Flow Momentum Analysis")
    momentum_data = create_flow_momentum_indicator(df, 'Monthly Flow')
    
    if momentum_data:
        # Display momentum metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown(
                render_metric_card(
                    "ETFs with Inflows",
                    f"{momentum_data['positive_count']:,}",
                    delta=f"{(momentum_data['positive_count']/(momentum_data['positive_count']+momentum_data['negative_count'])*100):.1f}%"
                ),
                unsafe_allow_html=True
            )
        
        with col_m2:
            st.markdown(
                render_metric_card(
                    "Total Inflows",
                    format_large_number(momentum_data['positive_sum'])
                ),
                unsafe_allow_html=True
            )
        
        with col_m3:
            st.markdown(
                render_metric_card(
                    "Net Flow",
                    format_large_number(momentum_data['net_flow']),
                    delta="Positive" if momentum_data['net_flow'] > 0 else "Negative"
                ),
                unsafe_allow_html=True
            )
        
        # Momentum chart
        fig_momentum = create_flow_momentum_chart(momentum_data)
        if fig_momentum:
            st.plotly_chart(fig_momentum, use_container_width=True)
    else:
        st.info("Insufficient data for momentum analysis")

with col_category_perf:
    st.markdown("### Top Categories by Performance")
    category_metrics = calculate_category_performance_metrics(df)
    
    if not category_metrics.empty:
        # Show top 10 categories by AUM
        top_categories = category_metrics.head(10)[
            ['Category', 'Total AUM', 'Number of ETFs', 'Monthly Flow %', 'Avg Performance']
        ].copy()
        
        # Format for display
        top_categories['Total AUM'] = top_categories['Total AUM'].apply(lambda x: format_large_number(x))
        top_categories['Monthly Flow %'] = top_categories['Monthly Flow %'].apply(lambda x: f"{x:.2f}%")
        top_categories['Avg Performance'] = top_categories['Avg Performance'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        top_categories.columns = ['Category', 'AUM', '# ETFs', 'Flow %', 'Perf %']
        
        st.dataframe(
            top_categories,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.info("Insufficient data for category performance analysis")

st.markdown("---")
# ========== END NEW SECTION ==========


# Build Past Flows dictionary per ETF (for sparklines)
flow_long = flow_df_raw.melt(
    id_vars="ETF",
    value_vars=flow_date_cols,
    var_name="Date",
    value_name="Flow"
)
flow_long["Date"] = pd.to_datetime(flow_long["Date"], errors="coerce")
flow_long.dropna(subset=["Date", "Flow"], inplace=True)
flow_long.sort_values(["ETF", "Date"], inplace=True)

flow_dict = (
    flow_long
    .groupby("ETF")
    .apply(lambda g: dict(zip(g["Date"], g["Flow"])))
    .reset_index(name="Past Flows")
)

df = df.merge(flow_dict, on="ETF", how="left")
df["Past Flows"] = df["Past Flows"].apply(lambda x: x if isinstance(x, dict) else {})

flow_lookup = dict(zip(flow_dict["ETF"].astype(str).str.strip(), flow_dict["Past Flows"]))

u_mask = (
    df["ETF"].astype(str).str.endswith("(U)", na=False)
    & df["Past Flows"].apply(lambda d: isinstance(d, dict) and len(d) == 0)
)

def strip_u_suffix(etf: str) -> str:
    return re.sub(r"\(U\)$", "", str(etf)).strip()

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

base_keys = df.loc[u_mask, "ETF"].apply(strip_u_suffix)
base_hist = base_keys.map(flow_lookup)
imputed_hist = base_hist.apply(lambda d: multiply_dict_values(d, 2.0))
df.loc[u_mask, "Past Flows"] = imputed_hist

def last_12m_flow_list(flow_map: dict, end_ts: pd.Timestamp) -> list[float]:
    """Return list of 12 monthly flows ending at end_ts (inclusive), in chronological order."""
    if not isinstance(flow_map, dict):
        return [0.0] * 12

    asc_dates = sorted(pd.to_datetime(flow_date_cols, errors="coerce").dropna())
    desired_months = [d for d in asc_dates if d <= end_ts][-12:]

    if len(desired_months) == 0:
        return [0.0] * 12

    if len(desired_months) < 12:
        desired_months = [desired_months[0]] * (12 - len(desired_months)) + desired_months

    return [float(flow_map.get(m, 0.0)) for m in desired_months]

def add_trend_and_scale(df_in: pd.DataFrame, normalize: bool = True, mark_symbol: str = "🔶") -> pd.DataFrame:
    """
    - Adds a brand mark (emoji) to Fund Name for Global X / BetaPro rows
    - Scales Flow/AUM to millions
    - Creates 'Flow Trend (12M)' (optionally min-max normalized)
    """
    out = df_in.copy()

    # Brand mask
    out["_brand"] = out["Fund Name"].str.contains(r"(Global X|BetaPro)", case=False, na=False)

    # Idempotent marking
    out["Fund Name"] = out["Fund Name"].astype(str).str.replace(r"^(🔶|🟠)\s+", "", regex=True)
    out.loc[out["_brand"], "Fund Name"] = mark_symbol + " " + out.loc[out["_brand"], "Fund Name"]

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

    # Scale to millions
    out["Flow"] = out["Flow"].astype(float) / 1e6
    out["AUM"] = out["AUM"].astype(float) / 1e6

    return out

def render_table_with_spark(df_disp: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")

    if df_disp.empty:
        st.dataframe(df_disp, use_container_width=True, hide_index=True)
        return

    trend_series = df_disp["Flow Trend (12M)"].apply(
        lambda v: v if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 else [0.0]
    )

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

    df_show = df_disp.rename(columns={"Flow": "Flow (M)", "AUM": "AUM (M)"}).copy()
    df_show["Flow (M)"] = df_show["Flow (M)"].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
    df_show["AUM (M)"] = df_show["AUM (M)"].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")

    st.dataframe(
        df_show,
        use_container_width=True,
        height=515,
        hide_index=True,
        column_config={
            "Flow (M)": st.column_config.TextColumn(width="small"),
            "AUM (M)": st.column_config.TextColumn(width="small"),
            "Flow Trend (12M)": st.column_config.LineChartColumn(
                label="Flow Trend (12M, norm)",
                y_min=ymin, y_max=ymax, width="small"
            ),
            "Ticker": st.column_config.TextColumn(width="small"),
            "Fund Name": st.column_config.TextColumn(width="large"),
        }
    )

# Top inflow/outflow tables w sparklines (filtered df)
top15_inflow = (
    df.nlargest(15, "Monthly Flow")[["Fund Name", "ETF", "Monthly Flow", "AUM", "Past Flows"]]
    .rename(columns={"ETF": "Ticker", "Monthly Flow": "Flow"})
)
top15_outflow = (
    df.nsmallest(15, "Monthly Flow")[["Fund Name", "ETF", "Monthly Flow", "AUM", "Past Flows"]]
    .rename(columns={"ETF": "Ticker", "Monthly Flow": "Flow"})
)

df_small_aum = df[df["AUM"] < 1_000_000_000]
top15_inflow_small_aum = (
    df_small_aum.nlargest(15, "Monthly Flow")[["Fund Name", "ETF", "Monthly Flow", "AUM", "Past Flows"]]
    .rename(columns={"ETF": "Ticker", "Monthly Flow": "Flow"})
)
top15_outflow_small_aum = (
    df_small_aum.nsmallest(15, "Monthly Flow")[["Fund Name", "ETF", "Monthly Flow", "AUM", "Past Flows"]]
    .rename(columns={"ETF": "Ticker", "Monthly Flow": "Flow"})
)

top15_inflow = add_trend_and_scale(top15_inflow)
top15_outflow = add_trend_and_scale(top15_outflow)
top15_inflow_small_aum = add_trend_and_scale(top15_inflow_small_aum)
top15_outflow_small_aum = add_trend_and_scale(top15_outflow_small_aum)

for dfx in (top15_inflow, top15_outflow, top15_inflow_small_aum, top15_outflow_small_aum):
    if "Past Flows" in dfx.columns:
        dfx.drop(columns=["Past Flows"], inplace=True)

st.subheader("Top ETF Inflows & Outflows")
c1, c2 = st.columns(2)
with c1:
    render_table_with_spark(
        top15_inflow[["Fund Name", "Ticker", "Flow", "AUM", "Flow Trend (12M)"]],
        "Top 15 Inflow"
    )
with c2:
    render_table_with_spark(
        top15_outflow[["Fund Name", "Ticker", "Flow", "AUM", "Flow Trend (12M)"]],
        "Top 15 Outflow"
    )

c3, c4 = st.columns(2)
with c3:
    render_table_with_spark(
        top15_inflow_small_aum[["Fund Name", "Ticker", "Flow", "AUM", "Flow Trend (12M)"]],
        "Top 15 Inflow (AUM < $1B)"
    )
with c4:
    render_table_with_spark(
        top15_outflow_small_aum[["Fund Name", "Ticker", "Flow", "AUM", "Flow Trend (12M)"]],
        "Top 15 Outflow (AUM < $1B)"
    )

# Global X tables + YTD new launches (all still filtered by Category/Secondary)
brand_mask = df["Fund Name"].str.contains(r"(Global X|BetaPro)", case=False, na=False)
small_aum_mask = df["AUM"] < 1_000_000_000

# Top 15 Monthly Inflow — Global X / BetaPro (ALL)
gx_bp_regular = df.loc[brand_mask, ["Fund Name", "ETF", "Monthly Flow", "AUM", "Past Flows", "Inception"]].copy()
top15_brand_inflow = (
    gx_bp_regular.nlargest(15, "Monthly Flow")
    .rename(columns={"ETF": "Ticker", "Monthly Flow": "Flow"})
)
top15_brand_inflow = add_trend_and_scale(top15_brand_inflow, normalize=True)

# Top 15 Monthly Inflow — Global X / BetaPro (AUM < 1B)
gx_bp_small = df.loc[
    brand_mask & small_aum_mask,
    ["Fund Name", "ETF", "Monthly Flow", "AUM", "Past Flows", "Inception"]
].copy()

top15_brand_small_inflow = (
    gx_bp_small.nlargest(15, "Monthly Flow")
    .rename(columns={"ETF": "Ticker", "Monthly Flow": "Flow"})
)
top15_brand_small_inflow = add_trend_and_scale(top15_brand_small_inflow, normalize=True)

# YTD new launches (launched in selected year)
inception_dt = pd.to_datetime(df["Inception"], errors="coerce")
launched_this_year_mask = inception_dt.dt.year.eq(selected_ts.year)

ytd_new = df.loc[
    launched_this_year_mask,
    ["Fund Name", "ETF", "YTD Flow", "AUM", "Past Flows", "Inception"]
].copy()

top20_ytd_new = (
    ytd_new.nlargest(20, "YTD Flow")
    .rename(columns={"ETF": "Ticker", "YTD Flow": "Flow"})
)
top20_ytd_new = add_trend_and_scale(top20_ytd_new, normalize=True)

# Display
c5, c6 = st.columns(2)
with c5:
    render_table_with_spark(
        top15_brand_small_inflow[["Fund Name", "Ticker", "Flow", "AUM", "Flow Trend (12M)"]],
        "Top 15 Monthly Inflow (< $1B) — Global X / BetaPro"
    )
with c6:
    render_table_with_spark(
        top15_brand_inflow[["Fund Name", "Ticker", "Flow", "AUM", "Flow Trend (12M)"]],
        "Top 15 Monthly Inflow — Global X / BetaPro"
    )

st.subheader(f"Top 20 YTD Inflow — Launched in {selected_ts.year}")
render_table_with_spark(
    top20_ytd_new[["Fund Name", "Ticker", "Flow", "AUM", "Flow Trend (12M)"]],
    f"Top 20 YTD Inflow — Launched in {selected_ts.year}"
)
