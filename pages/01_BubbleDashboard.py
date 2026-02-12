from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from data_prep import load_raw_data, load_raw_data_heatmap, process_data_for_date
from llm_api import ask_gemma
from config import apply_common_styling, render_header, render_metric_card, format_large_number
from analytics_utils import create_category_sunburst, calculate_flow_consistency  




st.set_page_config(
    page_title="ETF Bubble Dashboard",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply common styling
apply_common_styling()

# Render professional header
render_header(
    "ETF Bubble Chart Visualization",
    "Interactive Analysis of ETF Flows, AUM, and Performance Relationships"
)



onedrive_url = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/Eas53aR4lPlDn0ZlNHgX4ZABPDpH1Ign2mH4NGcJ0Hb80w?download=1"


funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw = load_raw_data(onedrive_url)

flow_date_cols = [c for c in flow_df_raw.columns if c != 'ETF']
available_dates = sorted(pd.to_datetime(flow_date_cols, errors='coerce').dropna(), reverse=True)
available_date_strs = [d.strftime('%Y-%m-%d') for d in available_dates]

# Control Panel
st.markdown("### 🎛️ Dashboard Controls")


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
    st.header("🤖 GX Chat")
    
    api_token = st.text_input("Enter OpenAI API Token", type="password", key="api_token")
    if not api_token:
        st.warning("Please enter your API Token to enable chat functionality.")
    
    chat_col1, chat_col2 = st.columns([3, 1])
    with chat_col2:
        if st.button("🔄", help="Refresh data context", key="refresh_context"):
            if "conversation_history" in st.session_state:
                st.session_state.conversation_history = []
            st.rerun()
    
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if user := st.chat_input("Ask me…"):
        if not st.session_state.get("api_token"):
            st.error("API Token is required to use the chat functionality.")
        else:
            st.session_state.history.append({"role": "user", "content": user})
            with st.chat_message("assistant"):
                st.markdown("…")
            try:
                ans = ask_gemma(
                    question=user, 
                    df=df,
                    max_rows=2000,
                    token=st.session_state.get("api_token", "") or ""
                )
            except Exception as e:
                ans = ("It seems there was an issue with your API token. "
                       "Please check your token or try again with a valid one. You can access more information regarding the API token [here](https://platform.openai.com/docs/api-reference/authentication).")
            st.session_state.conversation_history.append({"role": "user", "content": user})
            st.session_state.conversation_history.append({"role": "assistant", "content": ans})
            
            if len(st.session_state.conversation_history) > 20:
                st.session_state.conversation_history = st.session_state.conversation_history[-20:]
            
            st.session_state.history.append({"role": "assistant", "content": ans})
            st.rerun()


with col4:
    show_leveraged_only = st.checkbox("⚖️ Show Lightly Leveraged Only", value=False, help="Filter to show only lightly leveraged ETFs (1.25x or 1.33x leverage).")

filtered = df.copy()
filtered = df[df['Inception'] <= selected_date].copy()

if show_leveraged_only:
    filtered = filtered[filtered['Lightly Leveraged Indicator']]

#multi
with col2:
    all_cats = sorted(filtered['Category'].unique())
    cat_search_col, cat_add_col = st.columns([4, 1])
    with cat_search_col:
        search_cat = st.text_input("🔍Search Category", key="cat_search")
    with cat_add_col:
        st.write("")
        st.write("")
        if st.button("➕All", key="add_matching_cat"):
            matching = [cat for cat in all_cats if search_cat.lower() in cat.lower()]
            current_selection = set(st.session_state.get("category_select", []))
            st.session_state["category_select"] = list(current_selection.union(matching))
    default_cats = st.session_state.get("category_select", [])
    selected_cats = st.multiselect(
        "📊 Category",
        options=all_cats,
        default=default_cats,
        help="Pick one or more categories (empty = no category filter unless secondary is chosen)",
        key="category_select",
        width=450
    )

with col3:
    if selected_cats:
        options = sorted(filtered[filtered['Category'].isin(selected_cats)]['Secondary Category'].unique())
    else:
        options = sorted(filtered['Secondary Category'].unique())
    subcat_search_col, subcat_add_col = st.columns([4, 1])
    with subcat_search_col:
        search_subcat = st.text_input("🔍Search Category", key="subcat_search")
    with subcat_add_col:
        st.write("")
        st.write("")
        if st.button("➕All", key="add_matching_subcat"):

            matching = [subcat for subcat in options if search_subcat.lower() in subcat.lower()]
            current_selection = set(st.session_state.get("secondary_category_select", []))
            st.session_state["secondary_category_select"] = list(current_selection.union(matching))
    default_subcats = st.session_state.get("secondary_category_select", [])
    selected_subcats = st.multiselect(
        "🎯 Secondary Category",
        options=options,
        default=default_subcats,
        help="Pick one or more sub-categories",
        key="secondary_category_select" ,
        width=450
    )

if selected_cats:
    filtered = filtered[filtered['Category'].isin(selected_cats)]
if selected_subcats:
    filtered = filtered[filtered['Secondary Category'].isin(selected_subcats)]

is_showing_all = not (selected_cats or selected_subcats or show_leveraged_only)
filtered = filtered.copy()
filtered['AUM'] = filtered['AUM'].fillna(0)
st.dataframe(filtered)
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
        fig.update_traces(textposition='bottom center', textfont=dict(size=20, color='black'))
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
    fig.update_traces(textposition='bottom center', textfont=dict(size=20, color='black'))
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
    fig.update_traces(textposition='bottom center', textfont=dict(size=20, color='black'))
    
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

col1_sec, col2_sec = st.columns(2)

with col1_sec:
    all_categories = sorted(analysis_base_df['Category'].unique())
    analysis_search_cat = st.text_input("🔍 Search Category for Analysis", key="analysis_cat_search")
    if st.button("➕ All Categories", key="select_all_analysis_cat"):
        matching = [cat for cat in all_categories if analysis_search_cat.lower() in cat.lower()]
        st.session_state["cat_analysis"] = matching
    default_analysis_cats = st.session_state.get("cat_analysis", [])
    selected_categories = st.multiselect(
        "Select Category(s) for Analysis",
        options=all_categories,
        default=default_analysis_cats,
        help="Pick one or more primary Categories (leave empty to include all).",
        key="cat_analysis"
    )

with col2_sec:
    all_subcategories = sorted(analysis_base_df['Secondary Category'].unique())
    analysis_search_subcat = st.text_input("🔍 Search Subcategory for Analysis", key="analysis_subcat_search")
    if st.button("➕ All Subcategories", key="select_all_analysis_subcat"):
        matching = [subcat for subcat in all_subcategories if analysis_search_subcat.lower() in subcat.lower()]
        st.session_state["subcat_analysis"] = matching
    default_analysis_subcats = st.session_state.get("subcat_analysis", [])
    selected_subcategories = st.multiselect(
        "Select Secondary Category(s) for Analysis",
        options=all_subcategories,
        default=default_analysis_subcats,
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
        group_col: True,
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
        textfont=dict(size=20, color='black')
    )
    secondary_fig.update_layout(showlegend=False)

st.plotly_chart(secondary_fig, use_container_width=True)

# ========== NEW: Additional Visualizations ==========
st.markdown("---")
st.markdown("## 📊 Advanced Visualizations")

col_sunburst, col_consistency = st.columns(2)

with col_sunburst:
    st.markdown("### Asset Allocation Breakdown")
    
    # Create sunburst chart
    fig_sunburst = create_category_sunburst(analysis_df)
    
    if fig_sunburst:
        st.plotly_chart(fig_sunburst, use_container_width=True)
    else:
        st.info("Insufficient data for sunburst visualization")

with col_consistency:
    st.markdown("### ETF Flow Consistency (6M)")
    
    # Calculate flow consistency
    consistency_df = calculate_flow_consistency(analysis_df, flow_df_raw, months=6)
    
    if not consistency_df.empty:
        # Show top 15 most consistent ETFs
        top_consistent = consistency_df.head(15)[
            ['ETF', 'Fund Name', 'Category', 'Consistency Score', 'Positive Months']
        ].copy()
        
        # Format for display
        top_consistent['Consistency Score'] = top_consistent['Consistency Score'].apply(lambda x: f"{x:.1f}%")
        top_consistent.columns = ['Ticker', 'Fund Name', 'Category', 'Consistency', 'Positive/6M']
        
        st.dataframe(
            top_consistent,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        st.caption("💡 Consistency Score = % of months with positive flows over last 6 months")
    else:
        st.info("Insufficient historical data for consistency analysis")


