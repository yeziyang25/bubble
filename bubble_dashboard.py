from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import re
import requests
from io import BytesIO
st.set_page_config(
    page_title="ETF Bubble Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
    .stApp { background: #f9f9f9; }
    h1 { color: #2C3E50; }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 ETF Bubble Chart Dashboard")
st.markdown(
    """
    • **Step 1:** pick one or more *Category* values below.  
    • **Step 2:** then pick any relevant *Secondary Category* values.  
    """
)

onedrive_url = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/Eas53aR4lPlDn0ZlNHgX4ZABPDpH1Ign2mH4NGcJ0Hb80w?download=1"

# current_dir = Path(__file__).parent
# csv_path = current_dir /"etf_data/classification - data.xlsx"

@st.cache_data
def load_data(xlsx_path: str) -> pd.DataFrame:
    resp = requests.get(xlsx_path)
    resp.raise_for_status()
    excel_buffer = BytesIO(resp.content)

    funds_df = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="consolidated_10032022")
    aum_df   = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="aum")
    flow_df  = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="fund_flow")
    perf_df  = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="performance")

    
    def _col_to_str(col):
        return str(col)
    aum_df.columns   = aum_df.columns.map(_col_to_str)
    flow_df.columns  = flow_df.columns.map(_col_to_str)
    perf_df.columns  = perf_df.columns.map(_col_to_str)

    for df in (aum_df, flow_df, perf_df):
        df['ETF'] = (
            df['ETF']
            .astype(str)
            .str.replace(' CN Equity', '', regex=False)
            .str.strip()
        )

    aum_date_cols = [c for c in aum_df.columns if c != 'ETF']
    aum_date_cols_sorted = sorted(aum_date_cols, reverse=True)
    latest_col = aum_date_cols_sorted[0]
    prev_col   = aum_date_cols_sorted[1] if len(aum_date_cols_sorted) > 1 else None

    rename_map = {latest_col: 'AUM'}
    if prev_col is not None:
        rename_map[prev_col] = 'Prev AUM'
    aum_df = aum_df.rename(columns=rename_map)

    keep_aum = ['ETF', 'AUM']
    if prev_col is not None:
        keep_aum.append('Prev AUM')
    aum_df = aum_df[keep_aum].copy()

    aum_df['AUM'] = pd.to_numeric(aum_df['AUM'], errors='coerce')
    if prev_col is not None:
        aum_df['Prev AUM'] = pd.to_numeric(aum_df['Prev AUM'], errors='coerce')
    else:
        aum_df['Prev AUM'] = np.nan

    flow_date_cols = [c for c in flow_df.columns if c != 'ETF']
    flow_date_cols_sorted = sorted(flow_date_cols, reverse=True)
    ttm_cols = flow_date_cols_sorted[:12]

    af = aum_df.merge(flow_df, on='ETF', how='left')
    afm = af.merge(
        funds_df[['Ticker', 'Fund Name', 'Category', 'Secondary Category', 'Delisting Date', 'Indicator']],
        left_on='ETF',
        right_on='Ticker',
        how='left'
    )
    df = afm.merge(perf_df, on='ETF', how='left', suffixes=('', '_perf'))

    for col in ttm_cols:
        df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
    df['TTM Net Flow'] = df[ttm_cols].sum(axis=1)

    df['Monthly Flow'] = pd.to_numeric(df[flow_date_cols_sorted[0]].astype(str), errors='coerce')

   
    flow_dates = pd.to_datetime(flow_date_cols, errors='coerce')
    current_year = pd.Timestamp.now().year
    flow_year_map = dict(zip(flow_date_cols, flow_dates))
    ytd_cols = [col for col, dt in flow_year_map.items() if (not pd.isna(dt)) and dt.year == current_year]
    for col in ytd_cols:
        df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
    if ytd_cols:
        df['YTD Flow'] = df[ytd_cols].sum(axis=1)
    else:
        df['YTD Flow'] = 0.0

    df['Category'] = df['Category'].fillna('Unknown')
    df['Secondary Category'] = df['Secondary Category'].fillna('Unknown')

    perf_cols = [c for c in perf_df.columns if c != 'ETF']
    perf_cols_sorted = sorted(perf_cols, reverse=True)
    latest_perf_col = perf_cols_sorted[0]
    perf_df['Latest Performance'] = perf_df[latest_perf_col].astype(float)
    df = df.merge(
        perf_df[['ETF', 'Latest Performance']],
        on='ETF',
        how='left'
    )

    df['Lightly Leveraged Indicator'] = df['Fund Name'].str.contains(r"1\.25|1\.33", na=False)

    paired_rows = df[df['Indicator'] == 2].copy()
    single_rows = df[df['Indicator'] != 2].copy()

    def _combine_pair(group: pd.DataFrame) -> pd.Series:
        base_mask = group['ETF'].str.endswith('/B', na=False)
        if base_mask.any():
            base = group[base_mask].iloc[0]
        else:
            no_slash_mask = ~group['ETF'].str.contains(r'/.', na=False)
            if no_slash_mask.any():
                base = group[no_slash_mask].iloc[0]
            else:
                base = group.iloc[0]

        pair_key = re.sub(r'/.$', '', base['ETF'])
        if group['ETF'].str.endswith('/U', na=False).any():
            new_ticker = f"{pair_key}(U)"
        else:
            new_ticker = pair_key

        return pd.Series({
            'ETF': new_ticker,
            'Fund Name': base['Fund Name'],
            'Category': base['Category'],
            'Secondary Category': base['Secondary Category'],
            'Delisting Date': base['Delisting Date'],
            'Indicator': base['Indicator'],
            'AUM': base['AUM'],
            'Prev AUM': base['Prev AUM'],
            'Monthly Flow': base['Monthly Flow'],
            'TTM Net Flow': base['TTM Net Flow'],
            'YTD Flow': base['YTD Flow'],                     # preserve newly computed YTD Flow
            'Latest Performance': base['Latest Performance'],
            'Lightly Leveraged Indicator': bool(base['Lightly Leveraged Indicator'])
        })

    if not paired_rows.empty:
        paired_rows['PairKey'] = paired_rows['ETF'].str.replace(r'/.$', '', regex=True)
        combined_pairs = (
            paired_rows
            .groupby('PairKey', dropna=False)
            .apply(_combine_pair)
            .reset_index(drop=True)
        )
    else:
        combined_pairs = pd.DataFrame(columns=[
            'ETF', 'Fund Name', 'Category', 'Secondary Category', 'Delisting Date',
            'Indicator', 'AUM', 'Prev AUM', 'Monthly Flow', 'TTM Net Flow',
            'YTD Flow', 'Latest Performance', 'Lightly Leveraged Indicator'
        ])

    keep_cols = [
        'ETF', 'Fund Name', 'Category', 'Secondary Category', 'Delisting Date',
        'Indicator', 'AUM', 'Prev AUM', 'Monthly Flow', 'TTM Net Flow',
        'YTD Flow', 'Latest Performance', 'Lightly Leveraged Indicator'
    ]
    single_trimmed = single_rows[keep_cols].copy()
    final_df = pd.concat([single_trimmed, combined_pairs], ignore_index=True)

    return final_df


df = load_data(onedrive_url)

st.markdown("### Filters")
col1, col2, col3 = st.columns([2, 2, 1])

with col3:
    show_leveraged_only = st.checkbox("Show Lightly Leveraged Only")

filtered = df.copy()

if show_leveraged_only:
    filtered = filtered[filtered['Lightly Leveraged Indicator']]

with col1:
    all_cats = sorted(filtered['Category'].unique())
    selected_cats = st.multiselect(
        "Category",
        options=all_cats,
        default=[],
        help="Pick one or more categories (empty = no category filter unless secondary is chosen)",
        key="category_select" 
    )

with col2:
    if selected_cats:
        options = sorted(filtered[filtered['Category'].isin(selected_cats)]['Secondary Category'].unique())
    else:
        options = sorted(filtered['Secondary Category'].unique())
    selected_subcats = st.multiselect(
        "Secondary Category",
        options=options,
        default=[],
        help="Pick one or more sub-categories",
        key="secondary_category_select"  
    )

if selected_cats:
    filtered = filtered[filtered['Category'].isin(selected_cats)]
if selected_subcats:
    filtered = filtered[filtered['Secondary Category'].isin(selected_subcats)]

is_showing_all = not (selected_cats or selected_subcats or show_leveraged_only)
filtered = filtered.copy()
filtered['AUM'] = filtered['AUM'].fillna(0)
st.markdown("### Summary Statistics")
metric_cols = st.columns(5)

with metric_cols[0]:
    active_etfs = filtered[filtered['Delisting Date'] == 'Active']
    st.markdown(
        f"""<div class="metric-card">
            <h4>Number of ETFs</h4>
            <h2>{len(active_etfs):,}</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[1]:
    effective_aum = filtered['AUM'].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>Total AUM</h4>
            <h2>${effective_aum/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[2]:
    effective_monthly_flow = filtered['Monthly Flow'].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>Total Monthly Flow</h4>
            <h2>${effective_monthly_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[3]:
    effective_ttm_flow = filtered['TTM Net Flow'].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>Total TTM Flow</h4>
            <h2>${effective_ttm_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[4]:
    effective_ytd_flow = filtered['YTD Flow'].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>Total YTD Flow</h4>
            <h2>${effective_ytd_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )
st.markdown("### ETF Flows Bubble Chart")

def create_figure(data, show_text_labels=False):
    
    fig = px.scatter(
        data,
        x='TTM Net Flow',    
        y='Monthly Flow',  
        size='AUM',       
        color='Category',
        text='ETF' if show_text_labels else None,
        hover_name='ETF',
        hover_data={
            'Fund Name': True,
            'AUM': ':,.0f',
            'Monthly Flow': ':,.0f',
            'TTM Net Flow': ':,.0f',
            'Category': True,
            'Secondary Category': True,
        },
        labels={
            'TTM Net Flow': 'TTM Net Flows ($)',
            'Monthly Flow': 'Monthly Flows ($)',
            'AUM': 'Assets Under Management ($)'
        },
        size_max=80,
        template='plotly_white',
        height=700,
    )
    
    if show_text_labels:
        fig.update_traces(textposition='bottom center', textfont=dict(size=12, color='black'))
        fig.update_layout(showlegend=False)
    else:
        fig.update_layout(showlegend=True)
    
    fig.update_layout(
        title={
            'text': "ETF Flows vs. AUM Bubble Chart",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    return fig

if is_showing_all:
    fig = create_figure(filtered, show_text_labels=False)
elif show_leveraged_only:
    leveraged_data = filtered.copy()
    
    fig = px.scatter(
        leveraged_data,
        x='TTM Net Flow',    
        y='Monthly Flow',   
        size='AUM',        
        color='ETF',
        text='ETF',
        hover_name='ETF',
        hover_data={
            'Fund Name': True,
            'AUM': ':,.0f',
            'Monthly Flow': ':,.0f',
            'TTM Net Flow': ':,.0f',
            'Category': True,
            'Secondary Category': True,
        },
        labels={
            'TTM Net Flow': 'TTM Net Flows ($)',
            'Monthly Flow': 'Monthly Flows ($)',
            'AUM': 'Assets Under Management ($)'
        },
        size_max=80,
        template='plotly_white',
        height=700,
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition='bottom center', textfont=dict(size=12, color='black'))
    fig.update_layout(
        title={
            'text': "ETF Flows vs. AUM Bubble Chart",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(l=40, r=40, t=60, b=40)
    )
else:
    fig = px.scatter(
        filtered,
        x='TTM Net Flow',
        y='Monthly Flow',
        size='AUM',
        color='ETF',
        text='ETF',
        hover_name='ETF',
        hover_data={
            'Fund Name': True,
            'AUM': ':,.0f',
            'Monthly Flow': ':,.0f',
            'TTM Net Flow': ':,.0f',
            'Category': True,
            'Secondary Category': True,
        },
        labels={
            'TTM Net Flow': 'TTM Net Flows ($)',
            'Monthly Flow': 'Monthly Flows ($)',
            'AUM': 'Assets Under Management ($)'
        },
        size_max=80,
        template='plotly_white',
        height=700
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition='bottom center', textfont=dict(size=12, color='black'))
    
    fig.update_layout(
        title={
            'text': "ETF Flows vs. AUM Bubble Chart",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin=dict(l=40, r=40, t=60, b=40)
    )

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Secondary Category Analysis")

all_categories     = sorted(df['Category'].unique())
all_subcategories  = sorted(df['Secondary Category'].unique())

selected_categories    = st.multiselect(
    "Select Category(s) for Analysis",
    options=all_categories,
    default=[],
    help="Pick one or more primary Categories (leave empty to include all).",
    key="cat_analysis"
)

selected_subcategories = st.multiselect(
    "Select Secondary Category(s) for Analysis",
    options=all_subcategories,
    default=[],
    help="Pick one or more secondary Categories (leave empty to include all).",
    key="subcat_analysis"
)

analysis_df = df.copy()

if selected_categories:
    analysis_df = analysis_df[analysis_df['Category'].isin(selected_categories)]

if selected_subcategories:
    analysis_df = analysis_df[analysis_df['Secondary Category'].isin(selected_subcategories)]


if selected_subcategories:
    group_col = 'Category'
else:
    group_col = 'Secondary Category'

agg = (
    analysis_df
    .groupby(group_col)
    .agg({
        'AUM':       'sum',
        'Prev AUM':  'sum',
        'Monthly Flow':     'sum',
        'Latest Performance': 'mean'
    })
    .reset_index()
)

agg['Flow Percentage']     = (agg['Monthly Flow'] / agg['Prev AUM']) * 100
agg['Latest Performance'] *= 100

secondary_fig = px.scatter(
    agg,
    x='Latest Performance',
    y='Flow Percentage',
    size='AUM',
    color=group_col,
    text=group_col,
    hover_data={
        'AUM':               ':,.0f',
        'Prev AUM':          ':,.0f',
        'Monthly Flow':      ':,.0f',
        'Flow Percentage':   ':.2f',
        'Latest Performance':':.2f'
    },
    labels={
        'Latest Performance': 'Average Performance (%)',
        'Flow Percentage':    'Flow as % of Previous AUM',
        'AUM':                'Total AUM ($)'
    },
    size_max=60,
    template='plotly_white',
    width=1000,    
    height=800 ,  

)

secondary_fig.update_traces(
    textposition='top center',
    textfont=dict(size=15, color='black')
)

secondary_fig.update_layout(
    title={
        'text': f"Flow vs. Performance grouped by {group_col}",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    margin=dict(l=40, r=40, t=60, b=40),
    showlegend=False
)

st.plotly_chart(secondary_fig, use_container_width=True)

# Export filtered data as CSV for download
csv_data = filtered.to_csv(index=False).encode("utf-8")
st.markdown("### Export Data")
st.download_button(
    label="Download Filtered Dataset as CSV",
    data=csv_data,
    file_name="filtered_dataset.csv",
    mime="text/csv"
)
