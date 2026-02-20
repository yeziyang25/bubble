import pandas as pd
import datetime as dt
import sys
import numpy as np
import streamlit as st

sys.path.append('Z:\\ApolloGX')
import im_prod.std_lib.common as common
import im_prod.std_lib.data_library as data_library
import im_prod.std_lib.bloomberg_session as bloomberg_session

st.set_page_config(page_title="Daily BMO ETFs Dashboard", layout="wide")
executeDate = dt.datetime.now() + dt.timedelta(days=0)

def summarize_orders(live_orders):
    df = live_orders
    df = df.groupby(['ticker', 'order_type'])['pnu'].sum().reset_index()
    df = df[df['pnu'] != 0].reset_index(drop=True)

    df['Cash PNU'] = df.loc[df['order_type'] == 'Cash', 'pnu']
    df['In-Kind PNU'] = df.loc[df['order_type'] == 'In-Kind', 'pnu']
    df.fillna(0, inplace=True)
    df['Total PNU'] = df['Cash PNU'] + df['In-Kind PNU']

    df['transaction_value'] = df['pnu'] * df['ticker'].map(inav_dict) * df['ticker'].map(pnu_shares_dict)
    df['tnav (in MM)'] = df['ticker'].map(tnav_dict)
    df['new_tnav (in MM)'] = df['tnav (in MM)'] + df['transaction_value']

    df = df[['ticker', 'Total PNU', 'Cash PNU', 'In-Kind PNU', 'transaction_value', 'tnav (in MM)','new_tnav (in MM)']].sort_values(by=['ticker'])
    df_total = pd.DataFrame(data=[['Total', '', '', '', df['transaction_value'].sum(), 0, 0]], columns=df.columns)
    return pd.concat([df, df_total])


def format_order(df):
    df['transaction_value'] = np.where(df['transaction_value'] >= 0,
                                       df['transaction_value'].map('{:,.0f}'.format),
                                       abs(df['transaction_value']).map('({:,.0f})'.format))
    df['tnav (in MM)'] = np.where(df['tnav (in MM)'] == 0, '',
                                  np.where(df['tnav (in MM)'] >= 0,
                                           (df['tnav (in MM)'] / 1000000).map('{:,.1f}'.format),
                                           abs((df['tnav (in MM)'] / 1000000)).map('({:,.1f})'.format)))
    df['new_tnav (in MM)'] = np.where(df['new_tnav (in MM)'] == 0, '',
                                      np.where(df['new_tnav (in MM)'] >= 0,
                                               (df['new_tnav (in MM)'] / 1000000).map('{:,.1f}'.format),
                                               abs((df['new_tnav (in MM)'] / 1000000)).map('({:,.1f})'.format)))

    return df


if __name__ == "__main__":
    fund_list = ["ZEQT/T","ZWC/T","ZWB/T","ZWA/T","ZWEN/T","ZWHC/T","ZWGD/T",
                 "ZWT/T","ZWK/T","ZWU/T","ZWP/T","ZWE/T","ZWG/T","ZWH/T", "ZWS/T"]


    # convert to bbg tickers
    bbg_list_ticker = [item + " CN Equity" for item in fund_list]

    bdp = bloomberg_session.BDP_Session()
    data = bdp.bdp_request(bbg_list_ticker,["PX_VOLUME", "TURNOVER", "NUM_TRADES_RT", 'FUND_TOTAL_ASSETS', 'FUND_NET_ASSET_VAL',
                            'CURRENT_TRR_1D', 'PX_MID','CUR_MKT_CAP'])
    # update closing mid as nav
    timestamp_dict = bdp.unpact_dictionary(data, "timestamp")
    volume_dict = bdp.unpact_dictionary(data, "PX_VOLUME")
    turnover_dict = bdp.unpact_dictionary(data, "TURNOVER")
    trades_count_dict = bdp.unpact_dictionary(data, "NUM_TRADES_RT")
    aum_dict = bdp.unpact_dictionary(data, "CUR_MKT_CAP")
    nav_dict = bdp.unpact_dictionary(data, "FUND_NET_ASSET_VAL")
    #mkt_cap_dict = bdp.unpact_dictionary(data,'CUR_MKT_CAP')
    # trr_dict = bdp.unpact_dictionary(data, "CURRENT_TRR_1D")
    mid = bdp.unpact_dictionary(data, "PX_MID")  # 1D Total Return

    output = pd.DataFrame(bbg_list_ticker, columns=["ticker"])
    output["AUM ($)"] = output["ticker"].map(aum_dict)
    output["volume (# of shares)"] = output["ticker"].map(volume_dict)
    output["turnover today ($)"] = output["ticker"].map(turnover_dict)
    output["# of trades"] = output["ticker"].map(trades_count_dict)
    output["timestamp"] = output["ticker"].map(timestamp_dict)
    output["NAV"] = output["ticker"].map(nav_dict)
    # output["1 Day Return"] = output["ticker"].map(one_day_dict)
    output['EoD Mid'] = output['ticker'].map(mid)
    output['1 Day Return'] = ((output['EoD Mid'] - output['NAV']) / output['NAV']) * 100
    # output['1 Day Return'].fillna(0, inplace=True)
    output.sort_values(by='turnover today ($)', ascending=False, inplace=True)
    output = output[['ticker', 'AUM ($)', 'volume (# of shares)', 'turnover today ($)', '# of trades', 'NAV', '1 Day Return']]

    #output['AUM ($)'] = output['AUM ($)'] * 1000000


    output.fillna(0,inplace=True)


    output['AUM ($)'] = output['AUM ($)'].apply(lambda x: "{:,.0f}".format(x))
    output["NAV"] = output["NAV"].apply(lambda x: "{:,.2f}".format(x))
    output['1 Day Return'] = output['1 Day Return'].apply(lambda x: "{:.2f}%".format(x))
    output["volume (# of shares)"] = output["volume (# of shares)"].apply(lambda x: "{:,.0f}".format(x))
    output["turnover today ($)"] = output["turnover today ($)"].apply(lambda x: "{:,.0f}".format(x))
    
    rpt_date = executeDate.strftime("%Y-%m-%d")
    row_h = 35
    header_h = 37
    h = header_h + row_h * len(output)
    st.subheader(f"BMO ETFs Trading Dashboard {rpt_date}")
    st.dataframe(output, use_container_width=True, hide_index=True, height=h)
