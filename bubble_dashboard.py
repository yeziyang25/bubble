from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import re
import requests
from io import BytesIO
from datetime import datetime
from llm_api import ask_gemma, ask_gemma_with_context  




st.set_page_config(
    page_title="ETF Bubble Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 { 
        color: #2C3E50; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    h3 {
        color: #34495E;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 2px solid #3498DB;
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
        border-left: 4px solid #3498DB;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .step-instructions {
        background: linear-gradient(145deg, #E8F6F3 0%, #D5EFEB 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #27AE60;
        margin-bottom: 30px;
        font-size: 16px;
    }
    .filters-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 30px;
    }
    /* Multiselect improvements */
    .stMultiSelect > div > div > div {
        border-radius: 8px;
        border: 2px solid #BDC3C7;
        transition: border-color 0.3s ease;
    }
    .stMultiSelect > div > div > div:focus-within {
        border-color: #3498DB;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    /* Selectbox improvements */
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 2px solid #BDC3C7;
        transition: border-color 0.3s ease;
    }
    .stSelectbox > div > div > div:focus-within {
        border-color: #3498DB;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)



sample_data_path = "sample_data/etf_data.xlsx"

@st.cache_data
def load_raw_data(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    funds_df = pd.read_excel(xlsx_path, engine="openpyxl", sheet_name="consolidated_10032022")
    aum_df   = pd.read_excel(xlsx_path, engine="openpyxl", sheet_name="aum")
    flow_df  = pd.read_excel(xlsx_path, engine="openpyxl", sheet_name="fund_flow")
    perf_df  = pd.read_excel(xlsx_path, engine="openpyxl", sheet_name="performance")

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

funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw = load_raw_data(sample_data_path)

flow_date_cols = [c for c in flow_df_raw.columns if c != 'ETF']
available_dates = sorted(pd.to_datetime(flow_date_cols, errors='coerce').dropna(), reverse=True)
available_date_strs = [d.strftime('%Y-%m-%d') for d in available_dates]

st.title("📊 ETF Bubble Chart Dashboard")
st.markdown(
    """
    <div class="step-instructions">
    <strong>📋 How to Use This Dashboard:</strong><br/>
    <strong>Step 1:</strong> Select a <em>Date</em> for analysis<br/>
    <strong>Step 2:</strong> Choose one or more <em>Categories</em> using the search-enabled selector<br/>
    <strong>Step 3:</strong> Optionally select <em>Secondary Categories</em> to further refine your analysis
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="filters-container">', unsafe_allow_html=True)
st.markdown("### 🎛️ Filters")
col1, col2, col3, col4 = st.columns([2, 2.5, 2.5, 1.5])

with col1:
    selected_date = st.selectbox(
        "📅 Analysis Date",
        options=available_date_strs,
        index=0,
        help="Select the date for your analysis"
    )

df = process_data_for_date(selected_date, funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw)

with st.sidebar:
    st.markdown(
        """
        <div style="background: linear-gradient(145deg, #E8F4FD 0%, #D1E7DD 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #2C3E50; margin: 0;">🤖 GX Chat</h2>
            <p style="color: #6C757D; margin: 5px 0 0 0; font-size: 14px;">
                Ask questions about your ETF data
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Add context refresh button
    chat_col1, chat_col2 = st.columns([3, 1])
    with chat_col2:
        if st.button("🔄", help="Refresh data context", key="refresh_context"):
            if "conversation_history" in st.session_state:
                st.session_state.conversation_history = []
            st.rerun()
    
    # Initialize history and conversation history
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # render messages
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # input
    if user := st.chat_input("Ask me…"):
        st.session_state.history.append({"role":"user","content":user})
        with st.chat_message("assistant"):
            st.markdown("…")
        
        # Use context-aware function
        ans = ask_gemma(
            question=user, 
            df=df,
            max_rows=500  #subject to change. put as 500 to save some tokens limits
        )
        
        # Update conversation history for context
        st.session_state.conversation_history.append({"role": "user", "content": user})
        st.session_state.conversation_history.append({"role": "assistant", "content": ans})
        
        # Keep conversation history manageable (last 10 exchanges)
        if len(st.session_state.conversation_history) > 20:
            st.session_state.conversation_history = st.session_state.conversation_history[-20:]
        
        st.session_state.history.append({"role":"assistant","content":ans})
        st.rerun()


with col4:
    show_leveraged_only = st.checkbox("⚖️ Show Lightly Leveraged Only")

st.markdown('</div>', unsafe_allow_html=True)

filtered = df.copy()
filtered = df[df['Inception'] <= selected_date].copy()

if show_leveraged_only:
    filtered = filtered[filtered['Lightly Leveraged Indicator']]

with col2:
    all_cats = sorted(filtered['Category'].unique())
    selected_cats = st.multiselect(
        "📊 Category",
        options=all_cats,
        default=[],
        help="🔍 Start typing to search and select categories",
        key="category_select",
        placeholder="Search and select categories..."
    )

with col3:
    if selected_cats:
        options = sorted(filtered[filtered['Category'].isin(selected_cats)]['Secondary Category'].unique())
    else:
        options = sorted(filtered['Secondary Category'].unique())
    selected_subcats = st.multiselect(
        "🎯 Secondary Category",
        options=options,
        default=[],
        help="🔍 Start typing to search and select secondary categories",
        key="secondary_category_select",
        placeholder="Search and select secondary categories..."
    )

if selected_cats:
    filtered = filtered[filtered['Category'].isin(selected_cats)]
if selected_subcats:
    filtered = filtered[filtered['Secondary Category'].isin(selected_subcats)]

is_showing_all = not (selected_cats or selected_subcats or show_leveraged_only)
filtered = filtered.copy()
filtered['AUM'] = filtered['AUM'].fillna(0)
st.markdown("### 📈 Summary Statistics")
metric_cols = st.columns(5)

with metric_cols[0]:
    selected_date_dt = pd.to_datetime(selected_date)
    delisting_dates = pd.to_datetime(filtered['Delisting Date'], errors='coerce')
    

    active_etfs = filtered[
        (delisting_dates > selected_date_dt) | pd.isna(delisting_dates)
    ]
    st.markdown(
        f"""<div class="metric-card">
            <h4>📊 Number of ETFs</h4>
            <h2>{len(active_etfs):,}</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[1]:
    effective_aum = filtered['AUM'].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>💰 Total AUM</h4>
            <h2>${effective_aum/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[2]:
    effective_monthly_flow = filtered['Monthly Flow'].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>📅 Monthly Flow</h4>
            <h2>${effective_monthly_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[3]:
    effective_ttm_flow = filtered['TTM Net Flow'].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>📈 TTM Flow</h4>
            <h2>${effective_ttm_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[4]:
    effective_ytd_flow = filtered['YTD Flow'].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>📊 YTD Flow</h4>
            <h2>${effective_ytd_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )
st.markdown("### 🫧 ETF Flows Bubble Chart")

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

st.markdown("### 🔍 Secondary Category Analysis")

# Use the full dataframe for the selected date for this analysis section
analysis_base_df = df.copy()

analysis_col1, analysis_col2 = st.columns(2)

with analysis_col1:
    all_categories     = sorted(analysis_base_df['Category'].unique())
    selected_categories    = st.multiselect(
        "Select Category(s) for Analysis",
        options=all_categories,
        default=[],
        help="🔍 Start typing to search and select categories for analysis",
        key="cat_analysis",
        placeholder="Search and select categories..."
    )

with analysis_col2:
    all_subcategories  = sorted(analysis_base_df['Secondary Category'].unique())
    selected_subcategories = st.multiselect(
        "Select Secondary Category(s) for Analysis",
        options=all_subcategories,
        default=[],
        help="🔍 Start typing to search and select secondary categories for analysis",
        key="subcat_analysis",
        placeholder="Search and select secondary categories..."
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
st.markdown("### 💾 Export Data")
st.download_button(
    label="Download Filtered Dataset as CSV",
    data=csv_data,
    file_name=f"filtered_dataset_{selected_date}.csv",
    mime="text/csv"
)
#                 st.error(f"Oops, couldn’t get an answer: {e}")
#     else:
#         st.warning("Please type a question first.")
