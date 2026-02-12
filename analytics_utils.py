"""
Advanced analytics utilities for ETF Dashboard
Contains functions for market analysis, concentration metrics, and advanced calculations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def calculate_market_concentration(df, top_n=10):
    """
    Calculate market concentration metrics (Top N ETFs by AUM)
    Returns concentration percentage and Herfindahl-Hirschman Index (HHI)
    """
    if df.empty or 'AUM' not in df.columns:
        return None
    
    df_sorted = df.sort_values('AUM', ascending=False).copy()
    total_aum = df_sorted['AUM'].sum()
    
    if total_aum == 0:
        return None
    
    top_n_aum = df_sorted.head(top_n)['AUM'].sum()
    concentration_pct = (top_n_aum / total_aum) * 100
    
    # Calculate HHI (sum of squared market shares)
    market_shares = (df_sorted['AUM'] / total_aum) * 100
    hhi = (market_shares ** 2).sum()
    
    return {
        'top_n': top_n,
        'top_n_aum': top_n_aum,
        'total_aum': total_aum,
        'concentration_pct': concentration_pct,
        'hhi': hhi,
        'top_etfs': df_sorted.head(top_n)[['ETF', 'Fund Name', 'AUM', 'Category']].copy()
    }

def calculate_provider_market_share(df):
    """Calculate market share by ETF provider"""
    if df.empty or 'ETF Provider' not in df.columns or 'AUM' not in df.columns:
        return pd.DataFrame()
    
    provider_stats = df.groupby('ETF Provider').agg({
        'AUM': 'sum',
        'ETF': 'count',
        'Monthly Flow': 'sum',
        'YTD Flow': 'sum'
    }).reset_index()
    
    provider_stats.columns = ['Provider', 'Total AUM', 'Number of ETFs', 'Monthly Flow', 'YTD Flow']
    
    total_aum = provider_stats['Total AUM'].sum()
    if total_aum > 0:
        provider_stats['Market Share %'] = (provider_stats['Total AUM'] / total_aum) * 100
    else:
        provider_stats['Market Share %'] = 0
    
    provider_stats = provider_stats.sort_values('Total AUM', ascending=False)
    
    return provider_stats

def create_provider_market_share_chart(provider_stats, top_n=10):
    """Create a pie chart for provider market share"""
    if provider_stats.empty:
        return None
    
    top_providers = provider_stats.head(top_n).copy()
    
    # Add "Others" category if there are more providers
    if len(provider_stats) > top_n:
        others_aum = provider_stats.iloc[top_n:]['Total AUM'].sum()
        others_share = provider_stats.iloc[top_n:]['Market Share %'].sum()
        
        others_row = pd.DataFrame([{
            'Provider': 'Others',
            'Total AUM': others_aum,
            'Market Share %': others_share
        }])
        top_providers = pd.concat([top_providers, others_row], ignore_index=True)
    
    fig = px.pie(
        top_providers,
        values='Total AUM',
        names='Provider',
        title=f'Market Share by Provider (Top {top_n})',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>AUM: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def calculate_category_performance_metrics(df):
    """Calculate performance metrics by category"""
    if df.empty:
        return pd.DataFrame()
    
    category_metrics = df.groupby('Category').agg({
        'AUM': 'sum',
        'ETF': 'count',
        'Monthly Flow': 'sum',
        'YTD Flow': 'sum',
        'TTM Net Flow': 'sum',
        'Latest Performance': 'mean'
    }).reset_index()
    
    category_metrics.columns = [
        'Category', 'Total AUM', 'Number of ETFs', 
        'Monthly Flow', 'YTD Flow', 'TTM Flow', 'Avg Performance'
    ]
    
    # Calculate flow as % of AUM
    category_metrics['Monthly Flow %'] = (category_metrics['Monthly Flow'] / category_metrics['Total AUM']) * 100
    category_metrics['YTD Flow %'] = (category_metrics['YTD Flow'] / category_metrics['Total AUM']) * 100
    
    category_metrics = category_metrics.sort_values('Total AUM', ascending=False)
    
    return category_metrics

def create_concentration_chart(concentration_data):
    """Create a bar chart showing market concentration"""
    if not concentration_data or 'top_etfs' not in concentration_data:
        return None
    
    top_etfs = concentration_data['top_etfs'].copy()
    top_etfs['Market Share %'] = (top_etfs['AUM'] / concentration_data['total_aum']) * 100
    
    fig = px.bar(
        top_etfs,
        x='Market Share %',
        y='ETF',
        orientation='h',
        title=f'Top {concentration_data["top_n"]} ETFs by Market Share ({concentration_data["concentration_pct"]:.1f}% of total AUM)',
        color='Market Share %',
        color_continuous_scale='Blues',
        hover_data=['Fund Name', 'AUM', 'Category']
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>%{customdata[0]}<br>AUM: $%{customdata[1]:,.0f}<br>Market Share: %{x:.2f}%<extra></extra>'
    )
    
    return fig

def create_flow_momentum_indicator(df, flow_col='Monthly Flow'):
    """Create flow momentum indicators (positive/negative flow counts)"""
    if df.empty or flow_col not in df.columns:
        return None
    
    df_flow = df[df[flow_col].notna()].copy()
    
    positive_flow = df_flow[df_flow[flow_col] > 0]
    negative_flow = df_flow[df_flow[flow_col] < 0]
    
    momentum_data = {
        'positive_count': len(positive_flow),
        'negative_count': len(negative_flow),
        'positive_sum': positive_flow[flow_col].sum(),
        'negative_sum': abs(negative_flow[flow_col].sum()),
        'net_flow': df_flow[flow_col].sum()
    }
    
    return momentum_data

def create_flow_momentum_chart(momentum_data):
    """Create a chart showing flow momentum"""
    if not momentum_data:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Inflows',
        x=['Count', 'Amount ($M)'],
        y=[momentum_data['positive_count'], momentum_data['positive_sum'] / 1e6],
        marker_color='#4CAF50',
        text=[
            f"{momentum_data['positive_count']:,}",
            f"${momentum_data['positive_sum']/1e6:,.1f}M"
        ],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Outflows',
        x=['Count', 'Amount ($M)'],
        y=[momentum_data['negative_count'], momentum_data['negative_sum'] / 1e6],
        marker_color='#f44336',
        text=[
            f"{momentum_data['negative_count']:,}",
            f"${momentum_data['negative_sum']/1e6:,.1f}M"
        ],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Flow Momentum: Inflows vs Outflows',
        barmode='group',
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    
    return fig

def create_aum_growth_trajectory(df, aum_df_raw, selected_etfs, months_back=12):
    """Create AUM growth trajectories for selected ETFs"""
    if not selected_etfs or aum_df_raw.empty:
        return None
    
    # Get date columns
    date_cols = [c for c in aum_df_raw.columns if c != 'ETF']
    date_cols_dt = pd.to_datetime(date_cols, errors='coerce')
    valid_dates = sorted([d for d in date_cols_dt if pd.notna(d)], reverse=True)
    
    if len(valid_dates) < 2:
        return None
    
    # Get last N months
    last_n_months = valid_dates[:months_back]
    last_n_months.reverse()  # Chronological order
    
    # Filter for selected ETFs
    aum_subset = aum_df_raw[aum_df_raw['ETF'].isin(selected_etfs)].copy()
    
    # Build trajectory data
    trajectory_data = []
    for _, row in aum_subset.iterrows():
        etf = row['ETF']
        for date_dt in last_n_months:
            date_str = date_dt.strftime('%Y-%m-%d')
            if date_str in aum_df_raw.columns:
                aum_val = pd.to_numeric(row[date_str], errors='coerce')
                if pd.notna(aum_val):
                    trajectory_data.append({
                        'ETF': etf,
                        'Date': date_dt,
                        'AUM': aum_val
                    })
    
    if not trajectory_data:
        return None
    
    trajectory_df = pd.DataFrame(trajectory_data)
    
    fig = px.line(
        trajectory_df,
        x='Date',
        y='AUM',
        color='ETF',
        title=f'AUM Growth Trajectory (Last {months_back} Months)',
        template='plotly_white',
        height=500
    )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis_title='AUM ($)',
        xaxis_title='Date'
    )
    
    fig.update_traces(mode='lines+markers')
    
    return fig

def create_category_sunburst(df):
    """Create a sunburst chart for category/subcategory allocation"""
    if df.empty or 'Category' not in df.columns:
        return None
    
    # Prepare hierarchical data
    sunburst_data = []
    
    # Add root
    total_aum = df['AUM'].sum()
    
    # Add categories and subcategories
    for category in df['Category'].unique():
        cat_df = df[df['Category'] == category]
        cat_aum = cat_df['AUM'].sum()
        
        sunburst_data.append({
            'labels': category,
            'parents': '',
            'values': cat_aum
        })
        
        if 'Secondary Category' in df.columns:
            for subcat in cat_df['Secondary Category'].unique():
                subcat_aum = cat_df[cat_df['Secondary Category'] == subcat]['AUM'].sum()
                sunburst_data.append({
                    'labels': subcat,
                    'parents': category,
                    'values': subcat_aum
                })
    
    sunburst_df = pd.DataFrame(sunburst_data)
    
    fig = go.Figure(go.Sunburst(
        labels=sunburst_df['labels'],
        parents=sunburst_df['parents'],
        values=sunburst_df['values'],
        branchvalues='total',
        hovertemplate='<b>%{label}</b><br>AUM: $%{value:,.0f}<br>%{percentParent}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Asset Allocation by Category and Subcategory',
        margin=dict(l=20, r=20, t=60, b=20),
        height=600
    )
    
    return fig

def calculate_flow_consistency(df, flow_df_raw, months=6):
    """Calculate flow consistency score for ETFs (how often they have positive flows)"""
    if df.empty or flow_df_raw.empty:
        return pd.DataFrame()
    
    # Get last N month columns
    date_cols = [c for c in flow_df_raw.columns if c != 'ETF']
    date_cols_dt = pd.to_datetime(date_cols, errors='coerce')
    valid_dates = sorted([d for d in date_cols_dt if pd.notna(d)], reverse=True)[:months]
    valid_cols = [d.strftime('%Y-%m-%d') for d in valid_dates]
    
    consistency_data = []
    
    for _, row in df.iterrows():
        etf = row['ETF']
        flow_row = flow_df_raw[flow_df_raw['ETF'] == etf]
        
        if flow_row.empty:
            continue
        
        flow_values = []
        for col in valid_cols:
            if col in flow_df_raw.columns:
                val = pd.to_numeric(flow_row[col].iloc[0], errors='coerce')
                if pd.notna(val):
                    flow_values.append(val)
        
        if flow_values:
            positive_count = sum(1 for v in flow_values if v > 0)
            consistency_score = (positive_count / len(flow_values)) * 100
            avg_flow = np.mean(flow_values)
            
            consistency_data.append({
                'ETF': etf,
                'Fund Name': row.get('Fund Name', ''),
                'Category': row.get('Category', ''),
                'Consistency Score': consistency_score,
                'Avg Flow (6M)': avg_flow,
                'Positive Months': positive_count,
                'Total Months': len(flow_values)
            })
    
    if not consistency_data:
        return pd.DataFrame()
    
    consistency_df = pd.DataFrame(consistency_data)
    consistency_df = consistency_df.sort_values('Consistency Score', ascending=False)
    
    return consistency_df
