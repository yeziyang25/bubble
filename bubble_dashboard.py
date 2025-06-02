from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

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

current_dir = Path(__file__).parent
aum_csv_path = current_dir /"etf_data/aum.csv"
flow_csv_path = current_dir / "etf_data/fund flow.csv"
funds_csv_path = current_dir / "etf_data/funds.csv"
perf_csv_path = current_dir / "etf_data/perf.csv"


@st.cache_data
def load_data(aum_path, flow_path, funds_path, perf_path):
    # Load existing data
    a = pd.read_csv(aum_path)
    f = pd.read_csv(flow_path)
    m = pd.read_csv(funds_path)
    p = pd.read_csv(perf_path)

    a['ETF'] = a['ETF'].str.replace(' CN Equity', '', regex=False)
    f['ETF'] = f['ETF'].str.replace(' CN Equity', '', regex=False)
    p['ETF'] = p['ETF'].str.replace(' CN Equity', '', regex=False)

    df = (
    a.merge(f, on='ETF', suffixes=('_aum', '_flow'))
        .merge(
            m[['Ticker', 'Fund Name', 'Category', 'Secondary Category']],
            left_on='ETF', right_on='Ticker', how='left'
        )
        .merge(p, on='ETF', how='left', suffixes=('', '_perf'))
    )
    aum_col   = [c for c in df if c.endswith('_aum')][0]
    flow_cols = [c for c in df if c.endswith('_flow')][:12]
    ttm_cols = flow_cols[-12:]
    df['TTM Net Flow'] = df[ttm_cols].sum(axis=1) 
    df['Monthly Flow'] = df[flow_cols[0]]
    df = df.rename(columns={aum_col: 'AUM'})

    df['Lightly Leveraged Indicator'] = df['Fund Name'].str.contains(r"1\.25|1\.33", na=False)

    df = df.dropna(subset=['AUM','Monthly Flow','TTM Net Flow'])
    df['Category']           = df['Category'].fillna('Unknown')
    df['Secondary Category'] = df['Secondary Category'].fillna('Unknown')

    perf_cols = [col for col in p.columns if col != 'ETF']
    df['Latest Performance'] = df[perf_cols[0]]

    return df

df = load_data(str(aum_csv_path), str(flow_csv_path), str(funds_csv_path), str(perf_csv_path))

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

st.markdown("### Summary Statistics")
metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown(
        f"""<div class="metric-card">
            <h4>Number of ETFs</h4>
            <h2>{len(filtered):,}</h2>
        </div>""", 
        unsafe_allow_html=True
    )
with metric_cols[1]:
    st.markdown(
        f"""<div class="metric-card">
            <h4>Total AUM</h4>
            <h2>${filtered['AUM'].sum()/1e9:.1f}B</h2>
        </div>""", 
        unsafe_allow_html=True
    )
with metric_cols[2]:
    st.markdown(
        f"""<div class="metric-card">
            <h4>Total Monthly Flow</h4>
            <h2>${filtered['Monthly Flow'].sum()/1e6:.1f}M</h2>
        </div>""", 
        unsafe_allow_html=True
    )
with metric_cols[3]:
    st.markdown(
        f"""<div class="metric-card">
            <h4>Total TTM Flow</h4>
            <h2>${filtered['TTM Net Flow'].sum()/1e9:.1f}B</h2>
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
    # Use create_figure for showing all with Category coloring
    fig = create_figure(filtered, show_text_labels=False)
elif show_leveraged_only:
    # Show leveraged ETFs with ETF-based coloring and labels
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

# Add Secondary Category Analysis Section
st.markdown("### Secondary Category Analysis")

# Category selector for the second chart
selected_category_for_analysis = st.selectbox(
    "Select Category for Analysis",
    options=sorted(df['Category'].unique()),
    key="category_analysis"
)

# Filter data for selected category
category_data = df[df['Category'] == selected_category_for_analysis]
category_data = category_data[["ETF","AUM","TTM Net Flow","Monthly Flow","Lightly Leveraged Indicator","Latest Performance","Secondary Category"]]
category_data['Latest Performance'] = category_data['Latest Performance'].str.rstrip('%').astype(float) / 100
# Aggregate by Secondary Category
secondary_analysis = category_data.groupby('Secondary Category').agg({
    'AUM': 'sum',
    'Monthly Flow': 'sum',
    'Latest Performance': 'mean'
}).reset_index()

# Calculate flow percentage
total_aum = category_data['AUM'].sum()
secondary_analysis['Flow Percentage'] = (secondary_analysis['Monthly Flow'] / total_aum) * 100

secondary_fig = px.scatter(
    secondary_analysis,
    x='Latest Performance',
    y='Flow Percentage',
    size='AUM',
    color='Secondary Category',
    text='Secondary Category',
    hover_data={
        'AUM': ':,.0f',
        'Flow Percentage': ':.2f',
        'Latest Performance': ':.2%',
    },
    labels={
        'Latest Performance': 'Average Performance (%)',
        'Flow Percentage': 'Flow as % of Previous AUM',
        'AUM': 'Total AUM ($)'
    },
    size_max=60,
    template='plotly_white',
    height=800
)

secondary_fig.update_traces(
    textposition='top center',
    textfont=dict(size=15, color='black')
)

secondary_fig.update_layout(
    title={
        'text': f"Flox v.s. Performance for {selected_category_for_analysis}",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    margin=dict(l=40, r=40, t=60, b=40),
    showlegend=False
)

st.plotly_chart(secondary_fig, use_container_width=True)