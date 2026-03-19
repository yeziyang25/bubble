import pandas as pd
import sys
import streamlit as st
import numpy as np
import os
from datetime import datetime, date
import requests
from io import BytesIO
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

from xbbg import blp
from dateutil.relativedelta import relativedelta

config = { #"ticker":[leverage ratio, rebalance tolerance, underlying index]
    "BNKL":[1.25,0.005, "SOLCBEW Index"],
    "USSL":[1.25,0.005, "SPX Index"],
    "QQQL":[1.25,0.005, "NDX Index"],
    "EMML":[1.25,0.005, "MXEF Index"],
    "EAFL":[1.25,0.005, "MXEA Index"],
    "CANL":[1.25,0.005, "TX60AR Index"],
    "HEQL":[1.25,0.01, "HEQT CN Equity"],
    "FTSE CHINA 50":[3,0, "XIN0U Index"]
    }




def prep_fund_df(df, ticker, lev, tol, idx):
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.sort_values("Date").reset_index(drop = True)
        out = out.rename(columns={out.columns[1]:"price"})
        
        out.attrs["ticker"] = ticker
        out.attrs["leverage"] = lev
        out.attrs["tolerance"] = tol
        out.attrs["index"] = idx
        
    return out

def gen_df(df_in):
    df = df_in.copy().reset_index(drop=True)
    lev = df.attrs.get("leverage")
    tol = df.attrs.get("tolerance")
    
    new_cols=["lvr idx","idx exp of nav","idx lr","rb","exp after rb","idx perf","fund perf",\
              "idx rtn d","fund rtn d","eff lr d","idx rtn 5d","fund rtn 5d","eff lr 5d",\
                  "Year", "idx rolling max", "fund rolling max"]
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    #operate on 1st row
    fr = df.iloc[[0]].copy()
    fr["lvr idx"] = fr["price"]
    fr["idx exp of nav"] = fr["lvr idx"] * lev
    fr["idx lr"] = fr["idx exp of nav"]/fr["lvr idx"]
    fr_rb = (abs(fr["idx lr"][0] - lev) - tol) > 1e-5
    #fr_rb_num = (abs(fr["idx lr"][0] - lev) - tol)
    #print(f"{ticker} {lev} {tol} {fr_rb_num}")
    
    fr["rb"] = fr_rb
    fr["exp after rb"] = fr["idx exp of nav"]
    perf_mult = 1000/fr["price"][0]
    fr["idx perf"] = perf_mult * fr["price"][0]
    fr["fund perf"] = perf_mult * fr["lvr idx"][0]
    
    out = pd.concat([fr,df[1:]], ignore_index = True)
    
    n = len(out)
    for i in range (1,n):
        out.loc[i,"idx exp of nav"] =(out.loc[i,"price"]/out.loc[i-1,"price"])*out.loc[i-1,"exp after rb"]
        out.loc[i,"lvr idx"]=out.loc[i-1,"lvr idx"]+out.loc[i,"idx exp of nav"]-out.loc[i-1,"exp after rb"]
        out.loc[i,"idx lr"] = out.loc[i,"idx exp of nav"]/out.loc[i,"lvr idx"]
        rb_i=(abs(out.loc[i, "idx lr"] - lev) - tol) > 1e-5
        out.loc[i,"rb"]=rb_i
        out.loc[i,"exp after rb"]=(out.loc[i,"lvr idx"]*lev) if rb_i else out.loc[i,"idx exp of nav"]
        out.loc[i,"idx rtn d"]=out.loc[i,"price"]/out.loc[i-1,"price"]-1
        out.loc[i,"fund rtn d"]=out.loc[i,"lvr idx"]/out.loc[i-1,"lvr idx"]-1
        if i >= 5:
            out.loc[i,"idx rtn 5d"]=out.loc[i,"price"]/out.loc[i-5,"price"]-1
            out.loc[i,"fund rtn 5d"]=out.loc[i,"lvr idx"]/out.loc[i-5,"lvr idx"]-1
        
    out["idx perf"] = perf_mult * out["price"]
    out["fund perf"] = perf_mult * out["lvr idx"]
    out["eff lr d"] = out["fund rtn d"]/out["idx rtn d"]
    out["eff lr 5d"] = out["fund rtn 5d"]/out["idx rtn 5d"]
    out["Year"] = pd.to_datetime(out["Date"]).dt.year
    out["idx rolling max"] = out.groupby("Year")["price"].cummax()
    out["fund rolling max"] = out.groupby("Year")["lvr idx"].cummax()
    out["idx dd"] = (out["price"]-out["idx rolling max"])/out["idx rolling max"]
    out["fund dd"] =(out["lvr idx"]-out["fund rolling max"])/out["fund rolling max"]
    
    return out


def gen_time_series(df):
    fig = px.line(df, x="Date", y=["Index", ticker], title = f"{ticker}",
                  template = "plotly_white",color_discrete_sequence=["#FF5400", "#003B45"])
    return fig


def annualized_vol(df, date_col="Date", idx_col="idx rtn d", fund_col="fund rtn d", periods = 252):
    out = df[[date_col, idx_col, fund_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["Year"] = out[date_col].dt.year
    
    vol = out.groupby("Year").agg(
        Index=(idx_col, lambda s: s.dropna().std() * np.sqrt(periods)),
        Fund =(fund_col, lambda s: s.dropna().std() * np.sqrt(periods))
        #,Days =(idx_col, lambda s: s.dropna().shape[0])
        )
    return vol.reset_index()

def sum_dd(df):
    tmp = df[["Date","Year", "idx dd", "fund dd"]].copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    
    g = tmp.groupby("Year")
    out = pd.DataFrame({
        "Index Down": g["idx dd"].min(),
        "Fund Down":  g["fund dd"].min(),
    }).reset_index()

    return out

def annual_return(df, date_col="Date", idx_col="price", fund_col="lvr idx"):
    out = df[[date_col, idx_col, fund_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col)
    out["Year"] = out[date_col].dt.year

    yearly = out.groupby("Year").agg(
        Index_First=(idx_col, "first"),
        Index_Last=(idx_col, "last"),
        Fund_First=(fund_col, "first"),
        Fund_Last=(fund_col, "last"),
        Days=(idx_col, "count")
    ).reset_index()

    # default: use prior year's last close
    yearly["Index"] = yearly["Index_Last"] / yearly["Index_Last"].shift(1) - 1
    yearly["Fund"] = yearly["Fund_Last"] / yearly["Fund_Last"].shift(1) - 1

    # first year fallback: use first available close of that year
    yearly.loc[0, "Index"] = yearly.loc[0, "Index_Last"] / yearly.loc[0, "Index_First"] - 1
    yearly.loc[0, "Fund"] = yearly.loc[0, "Fund_Last"] / yearly.loc[0, "Fund_First"] - 1

    return yearly[["Year", "Index", "Fund"]]


if __name__ == "__main__":
    #dfs = pd.read_excel(wb_path, sheet_name=None)
    dfs_out = {}
    final_dfs = {}
    sum_dfs_vol={}
    sum_dfs_dd={}
    sum_dfs_rtn={}
    figs = {}
    bundle = {}
    
    perf_base = 1000
    #bktst_period = 5
    
    #Change Backtest Start Date Here
    startdate = datetime.strptime("2021-02-01", "%Y-%m-%d").date()
    
    startdate = st.sidebar.date_input(
        "Select backtest start date",
        value=date(2021, 2, 1)
    )

    st.write(startdate)
    today = date.today()
    days_diff = (today-startdate).days
    
    dfs = {}
    end = date.today()
    start = end - relativedelta(days=days_diff)
    
    for ticker, (lev, tol, idx) in config.items():
        dfs[ticker] = blp.bdh(idx, "PX_LAST", start_date=start, end_date=end, currency = "CAD")
        dfs[ticker] = dfs[ticker].reset_index()
        dfs[ticker].columns = ["Date", idx] 
    
    #populate data df
    for ticker, df in dfs.items():
        if ticker not in config:
            continue
        leverage, tol, idx = config[ticker]
        dfs_out[ticker] = prep_fund_df(df, ticker, leverage, tol, idx)
        final_dfs[ticker] = gen_df(dfs_out[ticker])
        final_dfs[ticker].rename\
            (columns={"idx perf": "Index", "fund perf": ticker}, inplace=True)
            
        sum_dfs_vol[ticker] = annualized_vol(final_dfs[ticker])
        sum_dfs_dd[ticker] = sum_dd(final_dfs[ticker])
        sum_dfs_rtn[ticker] = annual_return(final_dfs[ticker])
        
        
    #generate performance time series
    for ticker, df in final_dfs.items():
        figs[ticker] = gen_time_series(df)
    
    
    for ticker in final_dfs.keys():
        bundle[ticker] = {
            "df": final_dfs[ticker],
            "fig": figs.get(ticker),
            "sum_vol":sum_dfs_vol.get(ticker),
            "sum_dd":sum_dfs_dd.get(ticker),
            "sum_rtn":sum_dfs_rtn.get(ticker)
            }
        
    selected = st.sidebar.selectbox("Select Ticker", list(bundle.keys()))
    ticker = selected
    st.title(f"{ticker} Backtest")
    bktst_data = bundle[ticker]["df"]
    perf_fig = bundle[ticker]["fig"]
    st.plotly_chart(perf_fig)
    vol = bundle[ticker]["sum_vol"]
    dd = bundle[ticker]["sum_dd"]
    rtn = bundle[ticker]["sum_rtn"]
     
    perc_cols = ["Index Down", "Fund Down"]
    dd[perc_cols] = (dd[perc_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .apply(lambda s: s.apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")))
    
    perc_cols = ["Index", "Fund"]
    vol[perc_cols] = (vol[perc_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .apply(lambda s: s.apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")))
    rtn[perc_cols] = (rtn[perc_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .apply(lambda s: s.apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")))
    st.subheader("Max DrawDown")
    st.dataframe(dd, hide_index=True)
    st.subheader("Annualized Volatility")
    st.dataframe(vol, hide_index=True)
    st.subheader("Annualized Return")
    st.dataframe(rtn, hide_index=True)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        