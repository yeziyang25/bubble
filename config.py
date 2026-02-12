"""
Configuration file for ETF Dashboard
Contains shared styling, constants, and utility functions
"""

import streamlit as st

# Brand Colors
PRIMARY_COLOR = "#FF5722"      # Orange
SECONDARY_COLOR = "#00695C"    # Teal
ACCENT_COLOR = "#4682A9"       # Blue
SUCCESS_COLOR = "#4CAF50"      # Green
WARNING_COLOR = "#FFC000"      # Yellow
DANGER_COLOR = "#ED7D31"       # Red

# Data Source
ONEDRIVE_URL = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/Eas53aR4lPlDn0ZlNHgX4ZABPDpH1Ign2mH4NGcJ0Hb80w?download=1"

# Common CSS Styling
COMMON_CSS = """
    <style>
    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f4f8 100%);
    }
    
    /* Header Styling */
    .dashboard-header {
        background: linear-gradient(135deg, #FF5722 0%, #E64A19 100%);
        padding: 20px 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .dashboard-title {
        font-size: 42px;
        font-weight: bold;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 2px;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .dashboard-subtitle {
        font-size: 16px;
        color: #FFE0B2;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 5px 0 0 0;
        font-weight: normal;
    }
    
    /* Navigation */
    .nav-link {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        background: white;
        color: #00695C;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .nav-link:hover {
        background: #FF5722;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }

    h1 { 
        color: #00695C; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #00695C;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 3px solid #FF5722;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    
    h3 {
        color: #00695C;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 2px solid #FF5722;
        padding-bottom: 10px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #FF5722;
        transition: transform 0.2s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-left-color: #00695C;
    }
    
    .metric-card h4 {
        color: #666;
        font-size: 14px;
        margin: 0 0 10px 0;
        font-weight: 600;
        border: none;
    }
    
    .metric-card h2 {
        color: #00695C;
        font-size: 32px;
        margin: 0;
        font-weight: bold;
        border: none;
    }
    
    .metric-card .delta {
        font-size: 14px;
        margin-top: 8px;
    }
    
    .delta-positive {
        color: #4CAF50;
    }
    
    .delta-negative {
        color: #f44336;
    }
    
    /* Filter Container */
    .filters-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 30px;
        border-top: 3px solid #FF5722;
    }
    
    /* Form Controls */
    .stMultiSelect > div > div > div {
        border-radius: 8px;
        border: 2px solid #BDC3C7;
        transition: border-color 0.3s ease;
    }
    
    .stMultiSelect > div > div > div:focus-within {
        border-color: #FF5722;
        box-shadow: 0 0 0 3px rgba(255, 87, 34, 0.1);
    }
    
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 2px solid #BDC3C7;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #FF5722;
        box-shadow: 0 0 0 3px rgba(255, 87, 34, 0.1);
    }
    
    .stCheckbox > label > div {
        color: #00695C;
    }
    
    /* Info Boxes */
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .success-box {
        background: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
    }
    
    /* Footer */
    .dashboard-footer {
        text-align: center;
        padding: 20px;
        margin-top: 50px;
        color: #666;
        border-top: 1px solid #ddd;
    }
    </style>
"""

def apply_common_styling():
    """Apply common CSS styling to the page"""
    st.markdown(COMMON_CSS, unsafe_allow_html=True)

def render_header(title, subtitle=None):
    """Render a professional dashboard header"""
    header_html = f"""
    <div class="dashboard-header">
        <div class="dashboard-title">{title}</div>
        {f'<div class="dashboard-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_metric_card(label, value, delta=None, delta_color="normal"):
    """Render a metric card with optional delta"""
    delta_class = ""
    if delta:
        if delta_color == "positive" or (delta_color == "normal" and isinstance(delta, (int, float)) and delta > 0):
            delta_class = "delta-positive"
        elif delta_color == "negative" or (delta_color == "normal" and isinstance(delta, (int, float)) and delta < 0):
            delta_class = "delta-negative"
        
        delta_html = f'<div class="delta {delta_class}">{"▲" if delta > 0 else "▼"} {abs(delta):,.2f}%</div>' if isinstance(delta, (int, float)) else f'<div class="delta">{delta}</div>'
    else:
        delta_html = ""
    
    card_html = f"""
    <div class="metric-card">
        <h4>{label}</h4>
        <h2>{value}</h2>
        {delta_html}
    </div>
    """
    return card_html

def format_large_number(num, decimals=1):
    """Format large numbers with K, M, B suffixes"""
    try:
        num = float(num)
        if abs(num) >= 1e9:
            return f"${num/1e9:,.{decimals}f}B"
        elif abs(num) >= 1e6:
            return f"${num/1e6:,.{decimals}f}M"
        elif abs(num) >= 1e3:
            return f"${num/1e3:,.{decimals}f}K"
        else:
            return f"${num:,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(num, decimals=2):
    """Format percentage values"""
    try:
        return f"{float(num):.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"
