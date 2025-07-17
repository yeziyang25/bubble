from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import re
import requests
from io import BytesIO
from datetime import datetime
from llm_api import ask_gemma  




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



onedrive_url = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/Eas53aR4lPlDn0ZlNHgX4ZABPDpH1Ign2mH4NGcJ0Hb80w?download=1"

@st.cache_data
def load_raw_data(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resp = requests.get(xlsx_path)
    resp.raise_for_status()
    excel_buffer = BytesIO(resp.content)

    funds_df = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="consolidated_10032022")
    aum_df   = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="aum")
    flow_df  = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="fund_flow")
    perf_df  = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="performance")

    def _col_to_str(col):
        if isinstance(col, datetime):
            return col.strftime('%Y-%m-%d')
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
    return funds_df, aum_df, flow_df, perf_df

def process_data_for_date(selected_date_str: str, funds_df: pd.DataFrame, aum_df: pd.DataFrame, flow_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
    selected_date = pd.to_datetime(selected_date_str)
    
    aum_df = aum_df.copy()
    flow_df = flow_df.copy()
    perf_df = perf_df.copy()

    aum_date_cols = [c for c in aum_df.columns if c != 'ETF']
    aum_date_cols_dt = pd.to_datetime(aum_date_cols, errors='coerce')
    valid_aum_dates = sorted([d for d in aum_date_cols_dt if pd.notna(d) and d <= selected_date], reverse=True)
    
    latest_col = valid_aum_dates[0].strftime('%Y-%m-%d') if valid_aum_dates else None
    prev_col = valid_aum_dates[1].strftime('%Y-%m-%d') if len(valid_aum_dates) > 1 else None

    rename_map = {}
    if latest_col:
        rename_map[latest_col] = 'AUM'
    if prev_col:
        rename_map[prev_col] = 'Prev AUM'
    
    aum_df_processed = aum_df.rename(columns=rename_map)
    
    keep_aum = ['ETF']
    if latest_col:
        keep_aum.append('AUM')
        aum_df_processed['AUM'] = pd.to_numeric(aum_df_processed['AUM'], errors='coerce')
    else:
        aum_df_processed['AUM'] = np.nan

    if prev_col:
        keep_aum.append('Prev AUM')
        aum_df_processed['Prev AUM'] = pd.to_numeric(aum_df_processed['Prev AUM'], errors='coerce')
    else:
        aum_df_processed['Prev AUM'] = np.nan
        
    aum_df_processed = aum_df_processed[keep_aum]

    flow_date_cols = [c for c in flow_df.columns if c != 'ETF']
    flow_date_cols_dt = pd.to_datetime(flow_date_cols, errors='coerce')
    
    ttm_end_date = selected_date
    ttm_start_date = ttm_end_date - pd.DateOffset(months=12)
    ttm_cols = [
        d.strftime('%Y-%m-%d') for d in flow_date_cols_dt if pd.notna(d) and ttm_start_date < d <= ttm_end_date
    ]
    for col in ttm_cols:
        flow_df[col] = pd.to_numeric(flow_df[col], errors='coerce')
    flow_df['TTM Net Flow'] = flow_df[ttm_cols].sum(axis=1) if ttm_cols else 0.0

    monthly_flow_col = selected_date.strftime('%Y-%m-%d')
    if monthly_flow_col in flow_df.columns:
        flow_df['Monthly Flow'] = pd.to_numeric(flow_df[monthly_flow_col], errors='coerce')
    else:
        flow_df['Monthly Flow'] = 0.0

    ytd_cols = [
        d.strftime('%Y-%m-%d') for d in flow_date_cols_dt if pd.notna(d) and d.year == selected_date.year and d <= selected_date
    ]
    for col in ytd_cols:
        flow_df[col] = pd.to_numeric(flow_df[col], errors='coerce')
    if ytd_cols:
        flow_df['YTD Flow'] = flow_df[ytd_cols].sum(axis=1)
    else:
        flow_df['YTD Flow'] = 0.0

    perf_cols = [c for c in perf_df.columns if c != 'ETF']
    perf_cols_dt = pd.to_datetime(perf_cols, errors='coerce')
    valid_perf_dates = sorted([d for d in perf_cols_dt if pd.notna(d) and d <= selected_date], reverse=True)
    
    if valid_perf_dates:
        latest_perf_col = valid_perf_dates[0].strftime('%Y-%m-%d')
        perf_df['Latest Performance'] = pd.to_numeric(perf_df[latest_perf_col], errors='coerce')
    else:
        perf_df['Latest Performance'] = np.nan

    af = aum_df_processed.merge(flow_df[['ETF', 'Monthly Flow', 'TTM Net Flow', 'YTD Flow']], on='ETF', how='left')
    afm = af.merge(
        funds_df[['Ticker', 'Fund Name','Inception', 'Category', 'Secondary Category', 'Delisting Date', 'Indicator']],
        left_on='ETF',
        right_on='Ticker',
        how='left'
    )
    df = afm.merge(perf_df[['ETF', 'Latest Performance']], on='ETF', how='left')

    df['Category'] = df['Category'].fillna('Unknown')
    df['Secondary Category'] = df['Secondary Category'].fillna('Unknown')
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
        if group['ETF'].str.contains('/U', na=False).any():
            new_ticker = f"{pair_key}(U)"
        else:
            new_ticker = pair_key

        return pd.Series({
            'ETF': new_ticker,
            'Fund Name': base['Fund Name'],
            'Category': base['Category'],
            'Secondary Category': base['Secondary Category'],
            'Delisting Date': base['Delisting Date'],
            'Inception': base['Inception'],
            'Indicator': base['Indicator'],
            'AUM': base['AUM'],
            'Prev AUM': base['Prev AUM'],
            'Monthly Flow': base['Monthly Flow'],
            'TTM Net Flow': base['TTM Net Flow'],
            'YTD Flow': base['YTD Flow'],
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
            'YTD Flow', 'Latest Performance', 'Lightly Leveraged Indicator', 'Inception'
        ])
    
    keep_cols = [
        'ETF', 'Fund Name', 'Category', 'Secondary Category', 'Delisting Date',
        'Indicator', 'AUM', 'Prev AUM', 'Monthly Flow', 'TTM Net Flow',
        'YTD Flow', 'Latest Performance', 'Lightly Leveraged Indicator','Inception'
    ]
    single_trimmed = single_rows[keep_cols].copy()
    final_df = pd.concat([single_trimmed, combined_pairs], ignore_index=True)

    return final_df

funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw = load_raw_data(onedrive_url)

flow_date_cols = [c for c in flow_df_raw.columns if c != 'ETF']
available_dates = sorted(pd.to_datetime(flow_date_cols, errors='coerce').dropna(), reverse=True)
available_date_strs = [d.strftime('%Y-%m-%d') for d in available_dates]

st.title("📊 ETF Bubble Chart Dashboard")
st.markdown(
    """
    • **Step 1:** pick a *Date* for analysis.
    • **Step 2:** pick one or more *Category* values below.  
    • **Step 3:** then pick any relevant *Secondary Category* values.  
    """
)

st.markdown("### Filters")
col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    selected_date = st.selectbox(
        "Select Analysis Date",
        options=available_date_strs,
        index=0,
        help="The dashboard will be run based on this date."
    )

df = process_data_for_date(selected_date, funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw)

with st.sidebar:
    st.header("🤖 GX Chat")
    # initialize history
    if "history" not in st.session_state:
        st.session_state.history = []
    # render messages
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    # input
    if user := st.chat_input("Ask me…"):
        st.session_state.history.append({"role":"user","content":user})
        with st.chat_message("assistant"):
            st.markdown("…")
        ans = ask_gemma(user, df,max_rows = 2000)
        st.session_state.history.append({"role":"assistant","content":ans})
        st.rerun()


with col4:
    show_leveraged_only = st.checkbox("Show Lightly Leveraged Only")

filtered = df.copy()
filtered = df[df['Inception'] <= selected_date].copy()

if show_leveraged_only:
    filtered = filtered[filtered['Lightly Leveraged Indicator']]

#multi
with col2:
    all_cats = sorted(filtered['Category'].unique())
    search_cat = st.text_input("Search for Categories", key="cat_search")
    if st.button("Select All Matching Categories", key="select_matching_cat"):
        matching = [cat for cat in all_cats if search_cat.lower() in cat.lower()]
        st.session_state["category_select"] = matching
    default_cats = st.session_state.get("category_select", [])
    selected_cats = st.multiselect(
        "Category",
        options=all_cats,
        default=default_cats,
        help="Pick one or more categories (empty = no category filter unless secondary is chosen)",
        key="category_select" 
    )

with col3:
    if selected_cats:
        options = sorted(filtered[filtered['Category'].isin(selected_cats)]['Secondary Category'].unique())
    else:
        options = sorted(filtered['Secondary Category'].unique())
    search_subcat = st.text_input("Search for Secondary Categories", key="subcat_search")
    if st.button("Select All Matching Secondary Categories", key="select_matching_subcat"):
        matching = [subcat for subcat in options if search_subcat.lower() in subcat.lower()]
        st.session_state["secondary_category_select"] = matching
    default_subcats = st.session_state.get("secondary_category_select", [])
    selected_subcats = st.multiselect(
        "Secondary Category",
        options=options,
        default=default_subcats,
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
    selected_date_dt = pd.to_datetime(selected_date)
    delisting_dates = pd.to_datetime(filtered['Delisting Date'], errors='coerce')
    

    active_etfs = filtered[
        (delisting_dates > selected_date_dt) | pd.isna(delisting_dates)
    ]
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

# Use the full dataframe for the selected date for this analysis section
analysis_base_df = df.copy()

all_categories     = sorted(analysis_base_df['Category'].unique())
all_subcategories  = sorted(analysis_base_df['Secondary Category'].unique())
analysis_search_cat = st.text_input("Search for Categories for Analysis", key="analysis_cat_search")
if st.button("Select All Matching Categories for Analysis", key="select_all_analysis_cat"):
    matching = [cat for cat in all_categories if analysis_search_cat.lower() in cat.lower()]
    st.session_state["cat_analysis"] = matching

default_analysis_cats = st.session_state.get("cat_analysis", [])
selected_categories    = st.multiselect(
    "Select Category(s) for Analysis",
    options=all_categories,
    default=[],
    help="Pick one or more primary Categories (leave empty to include all).",
    key="cat_analysis"
)



analysis_search_subcat = st.text_input("Search for Subcategories for Analysis", key="analysis_subcat_search")
if st.button("Select All Matching Subcategories for Analysis", key="select_all_analysis_subcat"):
    matching = [cat for cat in all_subcategories if analysis_search_subcat.lower() in cat.lower()]
    st.session_state["subcat_analysis"] = matching

default_analysis_subcats = st.session_state.get("subcat_analysis", [])
selected_subcategories = st.multiselect(
    "Select Secondary Category(s) for Analysis",
    options=all_subcategories,
    default=[],
    help="Pick one or more secondary Categories (leave empty to include all).",
    key="subcat_analysis"
)

analysis_df = analysis_base_df.copy()

if selected_categories:
    analysis_df = analysis_df[analysis_df['Category'].isin(selected_categories)]

if selected_subcategories:
    analysis_df = analysis_df[analysis_df['Secondary Category'].isin(selected_subcategories)]


if selected_subcategories:
    group_col = 'Category'
else:
    group_col = 'Secondary Category'

is_showing_all_sub = not (selected_categories or selected_subcategories or show_leveraged_only)

analysis_df = analysis_df[analysis_df['Prev AUM'].notna() & (analysis_df['Prev AUM'] != 0)]

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
if is_showing_all_sub:
    secondary_fig.update_traces(text=None)
    secondary_fig.update_layout(showlegend=True)
else:
    secondary_fig.update_traces(         
        textposition='top center',
        textfont=dict(size=15, color='black')
    )
    secondary_fig.update_layout(showlegend=False)

st.plotly_chart(secondary_fig, use_container_width=True)

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.markdown("### Export Data")
st.download_button(
    label="Download Filtered Dataset as CSV",
    data=csv_data,
    file_name=f"filtered_dataset_{selected_date}.csv",
    mime="text/csv"
)
#                 st.error(f"Oops, couldn’t get an answer: {e}")
#     else:
#         st.warning("Please type a question first.")
