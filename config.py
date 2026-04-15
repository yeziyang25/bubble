"""
Configuration file for ETF Dashboard
Contains shared styling, constants, and utility functions
"""

import streamlit as st

# Brand Colors
PRIMARY_COLOR = "#FF5722"      # Orange  (Global X brand)
SECONDARY_COLOR = "#00695C"    # Teal
ACCENT_COLOR = "#4682A9"       # Steel Blue
SUCCESS_COLOR = "#16A34A"      # Green
WARNING_COLOR = "#F59E0B"      # Amber
DANGER_COLOR = "#DC2626"       # Red
DARK_COLOR = "#1B2A3B"         # Dark Navy

# Data Source
ONEDRIVE_URL = "https://globalxcanada-my.sharepoint.com/:x:/g/personal/eden_ye_globalx_ca/Eas53aR4lPlDn0ZlNHgX4ZABPDpH1Ign2mH4NGcJ0Hb80w?download=1"

# Common CSS Styling
COMMON_CSS = """
    <style>
    /* ─── Base ────────────────────────────────────────────────────── */
    .stApp {
        background: #F8FAFD;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* ─── Dashboard Header ────────────────────────────────────────── */
    .dashboard-header {
        background: linear-gradient(135deg, #1B2A3B 0%, #2C3E50 60%, #FF5722 100%);
        padding: 22px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.18);
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .dashboard-title {
        font-size: 38px;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 1.5px;
        margin: 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.25);
    }

    .dashboard-subtitle {
        font-size: 15px;
        color: #CBD5E1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* ─── Typography ──────────────────────────────────────────────── */
    h1 {
        color: #1B2A3B;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        padding: 16px 0;
        margin-bottom: 24px;
    }

    h2 {
        color: #00695C;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 2px solid #FF5722;
        padding-bottom: 8px;
        margin-top: 28px;
    }

    h3 {
        color: #00695C;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 8px;
        margin-top: 32px;
        margin-bottom: 16px;
    }

    /* ─── Metric Cards ────────────────────────────────────────────── */
    .metric-card {
        background: #FFFFFF;
        padding: 20px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 3px solid #FF5722;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }

    .metric-card h4 {
        color: #64748B;
        font-size: 12px;
        margin: 0 0 8px 0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border: none;
    }

    .metric-card h2 {
        color: #1B2A3B;
        font-size: 28px;
        margin: 0;
        font-weight: 700;
        border: none;
        padding: 0;
    }

    .metric-card .delta {
        font-size: 13px;
        margin-top: 6px;
        font-weight: 500;
    }

    .delta-positive { color: #16A34A; }
    .delta-negative { color: #DC2626; }

    /* ─── Page Introduction Box ───────────────────────────────────── */
    .page-intro {
        background: #EFF6FF;
        border-left: 4px solid #4682A9;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        font-size: 14px;
        color: #334155;
        line-height: 1.7;
    }

    /* ─── Section Divider with Label ─────────────────────────────── */
    .section-divider {
        display: flex;
        align-items: center;
        margin: 32px 0 18px 0;
        gap: 12px;
    }

    .section-divider-label {
        color: #00695C;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Segoe UI', sans-serif;
        white-space: nowrap;
        background: #F8FAFD;
        padding: 0 8px;
    }

    .section-divider-line {
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, #FF5722 0%, #E2E8F0 100%);
        border-radius: 2px;
    }

    /* ─── Filter Container ────────────────────────────────────────── */
    .filters-container {
        background: #FFFFFF;
        padding: 22px 24px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 24px;
        border-top: 3px solid #FF5722;
    }

    /* ─── Form Controls ───────────────────────────────────────────── */
    .stMultiSelect > div > div > div {
        border-radius: 8px;
        border: 1.5px solid #CBD5E1;
        transition: border-color 0.2s ease;
    }

    .stMultiSelect > div > div > div:focus-within {
        border-color: #FF5722;
        box-shadow: 0 0 0 3px rgba(255, 87, 34, 0.1);
    }

    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1.5px solid #CBD5E1;
        transition: border-color 0.2s ease;
    }

    .stSelectbox > div > div > div:focus-within {
        border-color: #FF5722;
        box-shadow: 0 0 0 3px rgba(255, 87, 34, 0.1);
    }

    /* ─── Status / Info Boxes ─────────────────────────────────────── */
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #4682A9;
        padding: 14px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 14px;
    }

    .warning-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 14px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 14px;
    }

    .success-box {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 14px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 14px;
    }

    /* ─── Tables ──────────────────────────────────────────────────── */
    .dataframe { font-size: 13px; }

    /* ─── Navigation links ────────────────────────────────────────── */
    .nav-link {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px;
        background: white;
        color: #00695C;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    }

    .nav-link:hover {
        background: #FF5722;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    /* ─── Footer ──────────────────────────────────────────────────── */
    .dashboard-footer {
        text-align: center;
        padding: 20px;
        margin-top: 50px;
        color: #94A3B8;
        border-top: 1px solid #E2E8F0;
        font-size: 13px;
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


def render_page_intro(text: str):
    """Render a styled introductory paragraph at the top of a page."""
    st.markdown(f'<div class="page-intro">{text}</div>', unsafe_allow_html=True)


def render_section_divider(label: str):
    """Render a styled section divider with an orange-accented line."""
    st.markdown(
        f"""
        <div class="section-divider">
            <span class="section-divider-label">{label}</span>
            <div class="section-divider-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_glossary():
    """Expandable glossary of common ETF and analytics terms."""
    with st.expander("📖  Key Terms & Definitions — click to expand", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**AUM** — *Assets Under Management.*
The total market value of all assets held by an ETF. Larger AUM generally signals investor trust and liquidity.

---

**Monthly Flow** — *Net capital movement for the selected month.*
Positive = more money flowing in than out (inflows). Negative = more money leaving (outflows / redemptions).

---

**YTD Flow** — *Year-to-Date net flow.*
Cumulative net inflows or outflows since January 1 of the selected year.

---

**TTM** — *Trailing Twelve Months.*
A rolling 12-month window ending on the selected date, used to smooth out short-term noise.
""")
        with col2:
            st.markdown("""
**HHI** — *Herfindahl-Hirschman Index.*
Measures market concentration. Calculated as the sum of squared market-share percentages.
Below 1,500 = competitive · 1,500–2,500 = moderate · Above 2,500 = highly concentrated.

---

**ETF** — *Exchange-Traded Fund.*
A basket of securities (stocks, bonds, etc.) that trades on a stock exchange like a single share.

---

**Provider** — The asset management company that issues and manages the ETF (e.g., Global X, iShares, Vanguard).

---

**Net Flow** — Inflows minus outflows. Positive = investor demand growing; negative = net redemptions.
""")
        st.caption("Terms calculated using monthly data. Flows are in Canadian dollars unless noted.")