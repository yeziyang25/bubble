import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime,timedelta
import requests
from io import BytesIO
#cur_dir = os.path.dirname(__file__)

#input config
input_path= "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/IQAP-oLdpRGhRrYhKkbmKaJ8AXbrUZIrS5NorYtmxHAqPgw?e=d2JfEm&download=1"
resp = requests.get(input_path)
resp.raise_for_status()
excel_buffer = BytesIO(resp.content)

 

st.set_page_config(page_title="Weekly Flow Report", layout="wide")

def top5_with_gx(df,provider_col="ETF Provider",gx_name="Global X Investments Canada Inc",flow_col="Weekly Flow"):
    out = df.copy()

    if provider_col not in out.columns:
        out = out.reset_index()

    top5 = out.head(5)
    if (top5[provider_col] == gx_name).any():
        return top5
    else:
        gx_row = out.loc[out[provider_col] == gx_name]
        if not gx_row.empty:
            return pd.concat([top5, gx_row], ignore_index=True)

def style_arrow(sym):
    if sym == "↑":
        return "color: #2e7d32; font-weight: 900; font-size: 18px;"
    if sym == "↓":
        return "color: #c62828; font-weight: 900; font-size: 18px;"
    return "color: #f4b400; font-weight: 900; font-size: 18px;"

def highlight_gx(val: str):
    if isinstance(val, str) and "global x" in val.lower():
        return "color: #E94D00; font-weight: 600;"
    return ""

def add_data_bars(styler, df_num, bar_cols, provider_col="ETF Provider", total_label="Grand Total"):
    disp = styler.data
    bar_cols = [c for c in bar_cols if c in disp.columns and c in df_num.columns]
    if not bar_cols:
        return styler

    mask_total = df_num[provider_col].astype(str).str.contains(total_label, case=False, na=False)
    scale_df = df_num.loc[~mask_total, bar_cols]

    max_abs = float(scale_df.abs().max().max())
    if not np.isfinite(max_abs) or max_abs == 0:
        return styler

    def css_for(v):
        if pd.isna(v):
            return ""
        pct = min(abs(v) / max_abs, 1.0) * 100
        if v >= 0:
            return f"background: linear-gradient(90deg, rgba(0,200,0,0.35) {pct}%, transparent {pct}%);"
        return f"background: linear-gradient(90deg, rgba(220,0,0,0.35) {pct}%, transparent {pct}%);"

    def style_df(_):
        styles = pd.DataFrame("", index=disp.index, columns=disp.columns)
        aligned = df_num.reindex(disp.index).reindex(columns=disp.columns)

        for c in bar_cols:
            styles[c] = aligned[c].apply(css_for)

        total_rows = aligned[provider_col].astype(str).str.contains(total_label, case=False, na=False)
        styles.loc[total_rows, bar_cols] = ""

        return styles

    return styler.apply(style_df, axis=None)

def provider_flow_tbl(df_raw: pd.DataFrame) -> pd.DataFrame:
    
    gx_name = "Global X Investments Canada Inc"
    provider_col = "ETF Provider"
    flow_col = "Weekly Flow"
    aum_2025_col = "AUM (2024)" # to change to 2025?
    cols_sum = ["AUM","Weekly Flow","Last Week","2 Weeks Prior Flow","3 Weeks Prior Flow","YTD Flow","Turnover",aum_2025_col]
    bar_cols = [flow_col, "Last Week", "2 Weeks Prior Flow", "3 Weeks Prior Flow"]
    ordered = ["Rank Change", provider_col, "Rank",
        "AUM", flow_col, "Last Week", "2 Weeks Prior Flow", "3 Weeks Prior Flow", "YTD Flow","Turnover",
        "% Weekly Segments Flow", "2025 Mkt. Share", "Cur. Mkt. Share", "Prior Rank"]

    g = df_raw.groupby(provider_col)[cols_sum].sum()
    g = g.sort_values(flow_col, ascending=False)

    #generate grand total line
    gt = pd.DataFrame([g.sum()], index=["Grand Total"])

    g["Rank"] = range(1, len(g) + 1)
    g["% Weekly Segments Flow"] = g[flow_col] / gt.loc["Grand Total", flow_col]
    g["2025 Mkt. Share"] = g[aum_2025_col] / gt.loc["Grand Total", aum_2025_col]
    g["Cur. Mkt. Share"] = g["AUM"] / gt.loc["Grand Total", "AUM"]
    gt = pd.DataFrame([g.sum()], index=["Grand Total"]) #redo gt for % columns

    g["Prior Rank"] = g["Last Week"].rank(ascending=False, method="min")
    g["Diff"] = g["Prior Rank"] - g["Rank"]
    g["Rank Change"] = np.select(
        [g["Diff"] > 0, g["Diff"] == 0, g["Diff"] < 0],
        ["↑", "→", "↓"],
        default="→")
    
    g = top5_with_gx(g, provider_col=provider_col, gx_name=gx_name, flow_col=flow_col)
    gt[provider_col] = "Grand Total"
    final_df = pd.concat([g, gt])
    final_df.loc[final_df.index == "Grand Total", provider_col] = "Grand Total*"
    
    df_num = final_df[ordered].copy()
    df_num = df_num.rename(columns={"Rank Change": "Δ"}) 

    
    #formatting columns
    final_df["Rank"] = final_df["Rank"].round(0).astype("Int64")
    final_df["Prior Rank"] = final_df["Prior Rank"].round(0).astype("Int64")
    final_df.loc["Grand Total", "Rank"]=None
    
    million_cols = ["AUM","Weekly Flow","Last Week","2 Weeks Prior Flow",
                    "3 Weeks Prior Flow","YTD Flow","Turnover"]
    final_df[million_cols] = (
        final_df[million_cols]
        .apply(pd.to_numeric, errors="coerce")
        .apply(lambda col: col.apply(lambda x: "" if pd.isna(x) else f"{x/1e6:,.0f} M")))
    
    perc_cols = ["% Weekly Segments Flow","2025 Mkt. Share","Cur. Mkt. Share"]
    final_df[perc_cols] = (final_df[perc_cols]
        .apply(pd.to_numeric, errors="coerce")
        .apply(lambda s: s.apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")))
    final_df = final_df.astype("string").fillna("")
    

    #rearranging columns in report format
    ordered = [c for c in ordered if c in final_df.columns]
    df_out = final_df[ordered]
    df_out.rename(columns={"Rank Change": "Δ"}, inplace=True)
    
    return df_out, df_num, bar_cols


def fund_flow_tbl(df_raw: pd.DataFrame) -> pd.DataFrame:
    flow_col = "Weekly Flow"
    cols = ["Fund Name","Ticker","AUM",flow_col,"Last Week","2 Weeks Prior Flow",
            "3 Weeks Prior Flow","YTD Flow","# of Trades","Turnover",
            "% of Weekly Segment Flow","% of YTD Segment Flow"]
    million_cols = ["AUM","Weekly Flow","Last Week","2 Weeks Prior Flow",
                    "3 Weeks Prior Flow","YTD Flow","Turnover"]
    perc_cols = ["% of Weekly Segment Flow","% of YTD Segment Flow"]
    df_ff = df_raw.reindex(columns=cols)

    gt = pd.DataFrame([df_ff.select_dtypes(include="number").sum()], index=["Grand Total"])

    df_ff = df_ff.sort_values(flow_col, ascending=False)
    df_ff["% of Weekly Segment Flow"] = df_ff[flow_col] / gt.loc["Grand Total", flow_col]
    df_ff["% of YTD Segment Flow"] = df_ff["YTD Flow"] / gt.loc["Grand Total", "YTD Flow"]

    #redo gt to sum the % columns
    gt = pd.DataFrame([df_ff.select_dtypes(include="number").sum()], index=["Grand Total"])
    
    #adding low aum fund tables (<500M)
    df_low_aum = df_ff[df_ff["AUM"]<500000000].copy()
    df_low_aum = df_low_aum.sort_values(flow_col, ascending=False)
    
    dfs = [df_ff, df_low_aum]
    for i in range(len(dfs)):
        df = dfs[i].head(10).copy()
        df = pd.concat([df,gt])
        df.loc[df.index == "Grand Total", "Fund Name"] = "Grand Total*"
    
        df[million_cols] = (
            df[million_cols]
            .apply(pd.to_numeric, errors="coerce")
            .apply(lambda col: col.apply(lambda x: "" if pd.isna(x) else f"{x/1e6:,.0f} M")))
    
        df[perc_cols] = (df[perc_cols]
                            .apply(pd.to_numeric, errors="coerce")
                            .apply(lambda s: s.apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")))
    
        df["# of Trades"] = df["# of Trades"].apply( lambda x: "" if pd.isna(x) else f"{x:,.0f}" )
        dfs[i] = df.fillna('')
        
        
    df_ff, df_low_aum = dfs
    return df_ff, df_low_aum


#"""-----------------IMPORT RAW DATA-----------------"""
df_raw = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="Weekly Consolidate")
df_raw = df_raw[(df_raw["Delisting Date"] == "Active") & (df_raw["Structure"] == "ETF")] #universal filter?
    

segments = {
    "Canadian Issuers Leagues Table - All Segments": df_raw[df_raw["Segment"] != "Other"],
    "Lightly Leveraged": df_raw[df_raw["Segment"] == "Lightly Leveraged"],
    "Premium Yield": df_raw[df_raw["Segment"] == "Premium Yield"],
    "Covered Call": df_raw[df_raw["Segment"] == "Covered Call"],
    "Asset Allocation": df_raw[df_raw["Segment"] == "Asset Allocation"],
    "Equity Essential": df_raw[df_raw["Segment"] == "Equity Essential"],
    "Canada Sector Equity": df_raw[df_raw["Segment"] == "Canada Sector Equity"],
    "Global Sector Equity": df_raw[df_raw["Segment"] == "Global Sector Equity"],
}

css_prov = """
<style>
/* Provider table row height */
.prov-table table td, .prov-table table th {
    font-size: 14px;
    padding: 8px 14px;     /* taller rows */
    line-height: 1.25;
    white-space: nowrap;
}
.prov-table table tr {
    height: 35px;           /* force row height */
}
</style>
"""
st.markdown(css_prov, unsafe_allow_html=True)

td = datetime.today().strftime("%B %d,%Y")
td_date = datetime.today().date()# if hasattr(td, "date") else td
weekday_mon1 = td_date - timedelta(days=td_date.weekday())
start = weekday_mon1 - timedelta(days=7)
end   = start + timedelta(days=4)

st.caption(f"{td}")
st.header("**Global X Weekly Flows Report**")
st.caption(f"For the week of {start.strftime('%B %d')} to {end.strftime('%B %d,%Y')}")
st.caption("**For Internal Use Only**")

for title, df_seg in segments.items():
    st.subheader(title)

    prov_disp, prov_num, bar_cols = provider_flow_tbl(df_seg)
    styled = (prov_disp.style
              .map(style_arrow, subset=["Δ"])
              .map(highlight_gx, subset=["ETF Provider"]))
    styled = add_data_bars(styled, prov_num, bar_cols)
    styled = styled.hide(axis="index")
    
    st.markdown(f'<div class="prov-table">{styled.to_html()}</div>', unsafe_allow_html=True)
    
    #st.markdown(styled.to_html(), unsafe_allow_html=True)
    #st.dataframe(styled, hide_index=True, use_container_width=True)

    fund_disp, f_low_aum = fund_flow_tbl(df_seg)
    if "All Segments" in title:
        st.caption("All Tickers")
        st.dataframe(fund_disp.style.map(highlight_gx, subset=["Fund Name"]),
                     height=422, hide_index=True, use_container_width=True)
        st.caption("")
        st.caption("Excluding tickers with AUM > 500M")
        st.dataframe(f_low_aum.style.map(highlight_gx, subset=["Fund Name"]),
                     height=422, hide_index=True, use_container_width=True)
    else:st.dataframe(fund_disp.style.map(highlight_gx, subset=["Fund Name"]),
                     height=422, hide_index=True, use_container_width=True)
    
    st.markdown("---")

st.caption("*Grand Total represents the full universe total; table rows display only the top 10 by weekly flow.")

