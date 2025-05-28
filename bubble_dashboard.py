import pandas as pd
import plotly.express as px
import streamlit as st

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

@st.cache_data
def load_data(aum_path, flow_path, funds_path):
    a = pd.read_csv(aum_path)
    f = pd.read_csv(flow_path)
    m = pd.read_csv(funds_path)

    a['ETF'] = a['ETF'].str.replace(' CN Equity', '', regex=False)
    f['ETF'] = f['ETF'].str.replace(' CN Equity', '', regex=False)

    df = (
        a.merge(f, on='ETF', suffixes=('_aum','_flow'))
         .merge(
             m[['Ticker','Category','Secondary Category']],
             left_on='ETF', right_on='Ticker', how='left'
         )
    )

    aum_col   = [c for c in df if c.endswith('_aum')][0]
    flow_cols = [c for c in df if c.endswith('_flow')]

    df['TTM Net Flow'] = df[flow_cols].sum(axis=1)
    df['Monthly Flow'] = df[flow_cols].iloc[:,0]
    df = df.rename(columns={aum_col: 'AUM'})

    df = df.dropna(subset=['AUM','Monthly Flow','TTM Net Flow'])
    df['Category']           = df['Category'].fillna('Unknown')
    df['Secondary Category'] = df['Secondary Category'].fillna('Unknown')

    return df[['ETF','AUM','Monthly Flow','TTM Net Flow','Category','Secondary Category']]

df = load_data('aum.csv', 'fund flow.csv', 'funds.csv')

st.markdown("### Filters")
col1, col2 = st.columns(2)

with col1:
    all_cats = sorted(df['Category'].unique())
    selected_cats = st.multiselect(
        "Category",
        options=all_cats,
        default=[],
        help="Pick one or more categories (empty = no category filter unless secondary is chosen)"
    )

with col2:
    if selected_cats:
        options = sorted(df[df['Category'].isin(selected_cats)]['Secondary Category'].unique())
    else:
        options = sorted(df['Secondary Category'].unique())
    selected_subcats = st.multiselect(
        "Secondary Category",
        options=options,
        default=[],
        help="Pick one or more sub-categories"
    )

if not selected_cats and not selected_subcats:
    filtered = df.iloc[0:0]
elif not selected_cats and selected_subcats:
    filtered = df[df['Secondary Category'].isin(selected_subcats)]
elif selected_cats and not selected_subcats:
    filtered = df[df['Category'].isin(selected_cats)]
else:
    filtered = df[
        df['Category'].isin(selected_cats) &
        df['Secondary Category'].isin(selected_subcats)
    ]

label_mode = st.radio(
    "Label bubbles by:",
    options=["Text under bubble", "Legend on side"],
    horizontal=True
)

st.markdown(f"**Showing {len(filtered)} ETFs** after filters.")

if label_mode == "Text under bubble":
    fig = px.scatter(
        filtered,
        x='TTM Net Flow',
        y='Monthly Flow',
        size='AUM',
        color='ETF',
        hover_name='ETF',
        text='ETF',
        labels={'TTM Net Flow':'TTM Net Flows','Monthly Flow':'Monthly Flows'},
        size_max=80,
        template='plotly_white',
        height=800,
        width=1400
    )
    fig.update_traces(
        textposition='bottom center',
        textfont=dict(size=12, color='black'),
        showlegend=False
    )
else:
    fig = px.scatter(
        filtered,
        x='TTM Net Flow',
        y='Monthly Flow',
        size='AUM',
        color='ETF',
        hover_name='ETF',
        text=None,
        labels={'TTM Net Flow':'TTM Net Flows','Monthly Flow':'Monthly Flows'},
        size_max=70,
        template='plotly_white',
        height=800,
        width=1400
    )
    fig.update_layout(showlegend=True)

fig.update_layout(
    title_text="ETF Flows vs. AUM Bubble Chart",
    title_x=0.5,
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "Data sources: `bloomberg`  \n"
    "Built with Streamlit & Plotly  •  © 2025"
)
