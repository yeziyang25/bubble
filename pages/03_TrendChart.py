from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

from data_prep import load_raw_data, load_raw_data_heatmap, process_data_for_date
from llm_api import ask_gemma
from config import apply_common_styling, render_header, render_metric_card, format_large_number
from analytics_utils import create_aum_growth_trajectory


st.set_page_config(
    page_title="ETF Trend Chart",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply common styling
apply_common_styling()

# Render professional header
render_header(
    "ETF Trend Analysis",
    "Track Monthly Flow Trends and AUM Growth Over Time"
)



onedrive_url = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/Eas53aR4lPlDn0ZlNHgX4ZABPDpH1Ign2mH4NGcJ0Hb80w?download=1"
funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw = load_raw_data(onedrive_url)

flow_date_cols = [c for c in flow_df_raw.columns if c != "ETF"]
available_dates = sorted(pd.to_datetime(flow_date_cols, errors="coerce").dropna(), reverse=True)
available_date_strs = [d.strftime("%Y-%m-%d") for d in available_dates]

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

selected_date_dt = pd.to_datetime(selected_date)

# Process universe for selected date (your existing function)
df = process_data_for_date(selected_date, funds_df_raw, aum_df_raw, flow_df_raw, perf_df_raw)



with col4:
    show_leveraged_only = st.checkbox(
        "⚖️ Show Lightly Leveraged Only",
        value=False,
        help="Filter to show only lightly leveraged ETFs (1.25x or 1.33x leverage)."
    )

filtered = df.copy()

# Ensure inception comparison is datetime-safe
filtered["Inception"] = pd.to_datetime(filtered["Inception"], errors="coerce")
filtered = filtered[filtered["Inception"].notna()]
filtered = filtered[filtered["Inception"] <= selected_date_dt].copy()

if show_leveraged_only:
    filtered = filtered[filtered["Lightly Leveraged Indicator"]]

# Category multiselect with search + add all matching
with col2:
    all_cats = sorted(filtered["Category"].dropna().unique())
    cat_search_col, cat_add_col = st.columns([4, 1])

    with cat_search_col:
        search_cat = st.text_input("🔍Search Category", key="cat_search")

    with cat_add_col:
        st.write("")
        st.write("")
        if st.button("➕All", key="add_matching_cat"):
            matching = [cat for cat in all_cats if search_cat.lower() in str(cat).lower()]
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

# Secondary category multiselect with search + add all matching
with col3:
    if selected_cats:
        options = sorted(filtered[filtered["Category"].isin(selected_cats)]["Secondary Category"].dropna().unique())
    else:
        options = sorted(filtered["Secondary Category"].dropna().unique())

    subcat_search_col, subcat_add_col = st.columns([4, 1])

    with subcat_search_col:
        search_subcat = st.text_input("🔍Search Category", key="subcat_search")

    with subcat_add_col:
        st.write("")
        st.write("")
        if st.button("➕All", key="add_matching_subcat"):
            matching = [subcat for subcat in options if search_subcat.lower() in str(subcat).lower()]
            current_selection = set(st.session_state.get("secondary_category_select", []))
            st.session_state["secondary_category_select"] = list(current_selection.union(matching))

    default_subcats = st.session_state.get("secondary_category_select", [])
    selected_subcats = st.multiselect(
        "🎯 Secondary Category",
        options=options,
        default=default_subcats,
        help="Pick one or more sub-categories",
        key="secondary_category_select",
        width=450  
    )

# Apply filters
if selected_cats:
    filtered = filtered[filtered["Category"].isin(selected_cats)]
if selected_subcats:
    filtered = filtered[filtered["Secondary Category"].isin(selected_subcats)]

is_showing_all = not (selected_cats or selected_subcats or show_leveraged_only)

filtered = filtered.copy()
filtered["AUM"] = pd.to_numeric(filtered["AUM"], errors="coerce").fillna(0)




# =========================
# Summary Statistics
# =========================
st.markdown("### Summary Statistics")
metric_cols = st.columns(5)

with metric_cols[0]:
    delisting_dates = pd.to_datetime(filtered["Delisting Date"], errors="coerce")
    active_etfs = filtered[(delisting_dates > selected_date_dt) | pd.isna(delisting_dates)]

    st.markdown(
        f"""<div class="metric-card">
            <h4>📊 Number of ETFs</h4>
            <h2>{len(active_etfs):,}</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[1]:
    effective_aum = filtered["AUM"].sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>💰 Total AUM</h4>
            <h2>${effective_aum/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[2]:
    effective_monthly_flow = pd.to_numeric(filtered["Monthly Flow"], errors="coerce").fillna(0).sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>📅 Monthly Flow</h4>
            <h2>${effective_monthly_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[3]:
    effective_ttm_flow = pd.to_numeric(filtered["TTM Net Flow"], errors="coerce").fillna(0).sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>📈 TTM Flow</h4>
            <h2>${effective_ttm_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )

with metric_cols[4]:
    effective_ytd_flow = pd.to_numeric(filtered["YTD Flow"], errors="coerce").fillna(0).sum()
    st.markdown(
        f"""<div class="metric-card">
            <h4>📊 YTD Flow</h4>
            <h2>${effective_ytd_flow/1e6:,.1f}M</h2>
        </div>""",
        unsafe_allow_html=True
    )


# =========================
# 12-Month Flow Trend Line Chart (Dynamic Grouping + Top-N default)
# =========================
st.markdown("### 📈 12-Month Monthly Flow Trend")

if filtered.empty:
    st.info("No ETFs match your current filters.")
else:
    # ---- Active ETFs only (same logic as KPIs) ----
    plot_universe = filtered.copy()
    dlist = pd.to_datetime(plot_universe["Delisting Date"], errors="coerce")
    plot_universe = plot_universe[(dlist > selected_date_dt) | dlist.isna()].copy()

    # ---- Choose grouping level dynamically ----
    if selected_subcats:
        group_field = "Secondary Category"
        
    else:
        group_field = "Category"
        

    # ---- Default behavior when NOTHING is selected ----
    # If no category & no subcategory chosen, do NOT show all categories.
    # Instead: show Top N categories by AUM (editable).
    topN_default = 8
    show_all_groups = False  # default OFF to avoid messy chart

    if (not selected_cats) and (not selected_subcats) and (group_field == "Category"):
        c1, c2 = st.columns([1.2, 2.8])
        with c1:
            topN = st.slider("Top Categories", 3, 20, topN_default, 1)
        with c2:
            show_all_groups = st.checkbox("Show all categories", value=False)

        if not show_all_groups:
            plot_universe["AUM"] = pd.to_numeric(plot_universe["AUM"], errors="coerce").fillna(0)
            top_groups = (
                plot_universe.dropna(subset=["Category"])
                            .groupby("Category")["AUM"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(topN)
                            .index
                            .tolist()
            )
            plot_universe = plot_universe[plot_universe["Category"].isin(top_groups)].copy()
            st.caption(f"Showing Top {len(top_groups)} categories by AUM (to reduce clutter).")

    # ---- Build mapping ETF -> group (Category or Secondary Category) ----
    etf_map = plot_universe[["ETF", group_field]].dropna(subset=["ETF", group_field]).drop_duplicates("ETF")

    if etf_map.empty:
        st.info("No data available for the current grouping after filters.")
    else:
        # ---- Find last 12 monthly flow columns <= selected_date_dt ----
        flow_col_info = []
        for c in flow_df_raw.columns:
            if c == "ETF":
                continue
            d = pd.to_datetime(c, errors="coerce")
            if pd.notna(d):
                flow_col_info.append((d, c))

        flow_col_info.sort(key=lambda x: x[0])
        past_12 = [t for t in flow_col_info if t[0] <= selected_date_dt][-12:]
        past_cols = [c for _, c in past_12]

        if len(past_cols) < 2:
            st.info("Not enough history to plot a 12-month trend for the selected date.")
        else:
            # ---- Pull flow history for ETFs in scope ----
            flow_hist = flow_df_raw.loc[
                flow_df_raw["ETF"].isin(etf_map["ETF"]),
                ["ETF"] + past_cols
            ].copy()

            # ---- Long format ----
            flow_long = flow_hist.melt(
                id_vars="ETF",
                var_name="Date",
                value_name="Monthly Flow"
            )
            flow_long["Date"] = pd.to_datetime(flow_long["Date"], errors="coerce")
            flow_long["Monthly Flow"] = pd.to_numeric(flow_long["Monthly Flow"], errors="coerce").fillna(0)

            # ---- Attach grouping + aggregate ----
            flow_long = flow_long.merge(etf_map, on="ETF", how="left")

            grouped = (
                flow_long.dropna(subset=[group_field, "Date"])
                         .groupby([group_field, "Date"], as_index=False)["Monthly Flow"]
                         .sum()
            )

            # ---- Plot ----
            fig_line = px.line(
                grouped.sort_values(["Date", group_field]),
                x="Date",
                y="Monthly Flow",
                color=group_field,
                template="plotly_white",
                height=550,
                labels={"Monthly Flow": "Monthly Net Flow ($)"},
            )

            fig_line.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=40, r=40, t=60, b=40),
            )

            st.plotly_chart(fig_line, use_container_width=True)

# =========================
# NEW: AUM Growth Trajectory for Top ETFs
# =========================
st.markdown("---")
st.markdown("### 📈 AUM Growth Trajectories")

# Allow user to select specific ETFs to track
top_etfs_by_aum = filtered.nlargest(20, 'AUM')['ETF'].tolist()

selected_etfs_for_trajectory = st.multiselect(
    "Select ETFs to Track (Top 20 by AUM shown)",
    options=top_etfs_by_aum,
    default=top_etfs_by_aum[:5] if len(top_etfs_by_aum) >= 5 else top_etfs_by_aum,
    help="Track AUM growth over the last 12 months for selected ETFs"
)

if selected_etfs_for_trajectory:
    fig_trajectory = create_aum_growth_trajectory(
        filtered,
        aum_df_raw,
        selected_etfs_for_trajectory,
        months_back=12
    )
    
    if fig_trajectory:
        st.plotly_chart(fig_trajectory, use_container_width=True)
    else:
        st.info("Insufficient historical AUM data for the selected ETFs")
else:
    st.info("Please select at least one ETF to view its AUM growth trajectory")

