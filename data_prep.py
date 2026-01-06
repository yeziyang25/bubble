import pandas as pd
import numpy as np
import re
import requests
from io import BytesIO
from datetime import datetime
import streamlit as st
from pandas.tseries.offsets import MonthEnd
import datetime as _dt

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
        funds_df[['Ticker', 'Fund Name','Inception', 'Category', 'Secondary Category', 'Delisting Date', 'Indicator','ETF Provider']],
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
            'Lightly Leveraged Indicator': bool(base['Lightly Leveraged Indicator']),
            'ETF Provider': base['ETF Provider']
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
            'YTD Flow', 'Latest Performance', 'Lightly Leveraged Indicator', 'Inception','ETF Provider'
        ])
    
    keep_cols = [
        'ETF', 'Fund Name', 'Category', 'Secondary Category', 'Delisting Date',
        'Indicator', 'AUM', 'Prev AUM', 'Monthly Flow', 'TTM Net Flow',
        'YTD Flow', 'Latest Performance', 'Lightly Leveraged Indicator','Inception','ETF Provider'
    ]
    single_trimmed = single_rows[keep_cols].copy()
    final_df = pd.concat([single_trimmed, combined_pairs], ignore_index=True)

    return final_df



@st.cache_data
def load_raw_data_heatmap(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resp = requests.get(xlsx_path)
    resp.raise_for_status()
    excel_buffer = BytesIO(resp.content)

    funds_df = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="consolidated_10032022")
    aum_df   = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="aum")
    flow_df  = pd.read_excel(excel_buffer, engine="openpyxl", sheet_name="fund_flow")

    funds_df = funds_df.rename(columns={'Ticker': 'ETF'})
    def _norm_etf(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.replace(' CN Equity', '', regex=False).str.strip())

    for df in (funds_df, aum_df, flow_df):
        df['ETF'] = _norm_etf(df['ETF'])

    cat_lookup = funds_df.set_index('ETF')[['Category', 'Secondary Category']]

    aum_df  = aum_df.drop(columns=['Category', 'Secondary Category'], errors='ignore') \
                    .join(cat_lookup, on='ETF')

    flow_df = flow_df.drop(columns=['Category', 'Secondary Category'], errors='ignore') \
                     .join(cat_lookup, on='ETF')

    return funds_df, aum_df, flow_df



#heatmap data
DEFAULT_MIXED_ROWS = [
    "Single Stock",
    "Lightly Leveraged",
    "Intl. Developed Equity",
    "Asset Allocation Portfolio",
    "Commodity",
    "Emerging Markets Equity",
    "Dividend/Income",
    "Options-Based Strategies",
    "Canada Equity",
    "Global Equity",
    "Canada Fixed Income",
    "Thematic",
    "U.S. Fixed Income",
    "U.S. Equity",
    "Global Fixed Income",
    "High Yield Fixed Income",
    "Crypto-Asset",
    "ESG",
    "Alternative Investment",
    "Preferred Share",
    "BetaPro",
    "Sector Equity",
]

def build_trailing3m_flow_pct_table(
    class_df: pd.DataFrame,
    aum_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    asof: str,
    category: str | None = None,
    custom_rows: list[str] | None = None,
) -> pd.DataFrame:
    """
    If `category` is None and `custom_rows` is given (or defaults), build rows
    by *name matching* against either Category OR Secondary Category.
    If `category` is provided, rows are Secondary Categories within that category.
    """
    

    
    asof_dt = pd.to_datetime(asof) + MonthEnd(0)
    def month_cols(df):
        return [pd.Timestamp(c) for c in df.columns if isinstance(c, (pd.Timestamp, _dt.datetime))]
    
    months_all = sorted(set(month_cols(aum_df)).intersection(month_cols(flow_df)))
    months_all = [m for m in months_all if m <= asof_dt]
    if len(months_all) < 4:
        raise ValueError("Not enough historical months to compute trailing 3M flow % AUM.")
    months_12 = months_all[-13:]
    
    base_cols = ['ETF', 'Category', 'Secondary Category']
    earliest_eval_idx = months_all.index(months_12[0])

    


    needed_months = months_all[max(0, earliest_eval_idx-3):]
    aum = aum_df[base_cols + needed_months].copy()
    flow = flow_df[base_cols + needed_months].copy()

    for df in (aum, flow):
        df[needed_months] = df[needed_months].apply(pd.to_numeric, errors='coerce')


    bad_flow = flow[needed_months].dtypes[flow[needed_months].dtypes != 'float64']
    bad_aum  = aum[needed_months].dtypes[aum[needed_months].dtypes != 'float64']
    if len(bad_flow) or len(bad_aum):
        print("[warn] Non-float month columns after coercion:",
             {"flow": bad_flow.to_dict(), "aum": bad_aum.to_dict()})
    


    aum[needed_months] = aum[needed_months].fillna(0.0)
    flow[needed_months] = flow[needed_months].fillna(0.0)
    
    if category:
        aum = aum[aum['Category'] == category].copy()
        flow = flow[flow['Category'] == category].copy()
        row_labels = sorted(aum['Secondary Category'].dropna().unique().tolist())
        row_selector = {lbl: ( (aum['Secondary Category'] == lbl), (flow['Secondary Category'] == lbl) ) for lbl in row_labels}
    else:
        if custom_rows is None:
            custom_rows = DEFAULT_MIXED_ROWS
        row_labels = custom_rows
        row_selector = {}
        for lbl in row_labels:
            mask_a = (aum['Category'] == lbl) | (aum['Secondary Category'] == lbl)
            mask_f = (flow['Category'] == lbl) | (flow['Secondary Category'] == lbl)
            row_selector[lbl] = (mask_a, mask_f)
    
    out = pd.DataFrame(index=row_labels, columns=months_12, dtype=float)
    month_idx_map = {m: i for i, m in enumerate(needed_months)}
    
    for m in months_12:
        idx = month_idx_map[m]
        if idx < 3:
            continue
        m1, m2, m3 = needed_months[idx-1], needed_months[idx-2], needed_months[idx-3]
        
        flow_sum_col = (flow[m] + flow[m1] + flow[m2]).rename("flow_3m")
        aum_base_col = aum[m3].rename("aum_base")
        
        for lbl, (mask_a, mask_f) in row_selector.items():
            flow_3m = flow_sum_col[mask_f].sum()
            aum_base = aum_base_col[mask_a].sum()
            pct = (flow_3m / aum_base) if aum_base != 0 else 0.0
            out.loc[lbl, m] = pct
    
    out.columns = [m.strftime("%b %y") for m in out.columns]
    out = out.sort_values(by=out.columns[-1], ascending=False)

    return out
