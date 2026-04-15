# Canadian ETF Intelligence Dashboard

A multi-page Streamlit application for visualising fund flows, AUM, and market dynamics across the Canadian ETF landscape.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How to Run](#how-to-run)
3. [Data Sources](#data-sources)
4. [Page-by-Page Guide](#page-by-page-guide)
5. [Key Utility Files](#key-utility-files)
6. [Styling & Theming](#styling--theming)
7. [How to Add a New Page](#how-to-add-a-new-page)
8. [Common Customisations](#common-customisations)

---

## Project Structure

```
bubble/
├── Home.py                   # Main flow analysis page (entry point)
├── pages/
│   ├── 01_BubbleDashboard.py # Bubble chart — flow vs AUM by category
│   ├── 02_HeatMap.py         # Flow % of AUM heatmaps over time
│   ├── 03_TrendChart.py      # 12-month flow trend lines by category
│   ├── 04_Flow.py            # Weekly flow report (provider + fund tables)
│   ├── 05_BMO_ETFs.py        # BMO ETF live trading metrics (Bloomberg)
│   ├── 06_Betapro_Backtest.py# Leveraged ETF backtest v1
│   └── 07_Backtest_V2.py     # Leveraged ETF backtest v2 (drawdown + return)
│
├── config.py                 # Shared colours, CSS, and UI utility functions
├── analytics_utils.py        # Market concentration, flow momentum, charts
├── data_prep.py              # Data loading and ETF data processing pipeline
├── llm_api.py                # OpenAI chat integration (sidebar Q&A)
│
├── .streamlit/
│   ├── config.toml           # App-wide theme (colours, font)
│   └── secrets.toml          # API keys and credentials (never commit this)
│
└── requirements.txt          # Python dependencies
```

> **Pages are automatically picked up by Streamlit.** Any `.py` file placed inside `pages/` becomes a sidebar navigation entry. The filename prefix (e.g. `01_`, `02_`) controls the sort order.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up secrets

Create `.streamlit/secrets.toml` (already gitignored). It should contain your OneDrive share token if the URL ever changes:

```toml
# .streamlit/secrets.toml
# No secrets are currently required at runtime —
# the OneDrive download URL is hardcoded in config.py.
# Add API keys here if needed in future.
```

### 3. Start the app

```bash
streamlit run Home.py
```

The app opens at `http://localhost:8501`.

---

## Data Sources

### Primary — OneDrive Excel file

All monthly ETF data is fetched from a shared OneDrive workbook at startup. The URL is defined in `config.py`:

```python
ONEDRIVE_URL = "https://globalxcanada-my.sharepoint.com/..."
```

The workbook has **four sheets**, each loaded into a separate DataFrame by `data_prep.load_raw_data()`:

| Sheet | DataFrame variable | Contents |
|---|---|---|
| `consolidated_10032022` | `funds_df_raw` | Fund metadata: ticker, name, provider, category, inception date |
| `aum` | `aum_df_raw` | Monthly AUM per ETF (columns = month-end dates) |
| `fund_flow` | `flow_df_raw` | Monthly net flow per ETF (columns = month-end dates) |
| `performance` | `perf_df_raw` | Monthly return per ETF (columns = month-end dates) |

**To change the data source**, update `ONEDRIVE_URL` in `config.py`. The sheet names are read positionally (index 0–3), so keep the sheet order the same or update the `sheet_name` arguments in `data_prep.load_raw_data()`.

### Secondary — Weekly flow Excel file

`04_Flow.py` fetches a separate weekly workbook (URL hardcoded at the top of that file). It expects a sheet named `Weekly Consolidate`.

### Bloomberg (pages 05–07 only)

Pages 05, 06, and 07 use `xbbg` (the Bloomberg Python API) and an internal library (`im_prod.std_lib`). These pages only work on machines with Bloomberg Terminal access. They will error on startup if Bloomberg is unavailable — all other pages are unaffected.

---

## Page-by-Page Guide

### Home.py — Flow Analysis Hub

The main page. Loads data for a selected month-end date and lets users filter by provider and ETF category.

**Key sections (top to bottom):**
1. **Filters** — Date, Provider, Category, Sub-Category
2. **Flow Trends** — 13-month bar charts for the industry, Global X, and provider comparisons
3. **Category Breakdown** — Top categories by inflow and outflow
4. **Market Intelligence** — HHI concentration score, provider market share pie, flow momentum
5. **ETF Tables** — Top 15 inflows/outflows with 12-month sparklines, YTD versions, Global X/BetaPro focus, new launches

**Where to make changes:**
- Chart colours → `MONTHLY_COLOR_INFLOW`, `MONTHLY_COLOR_OUTFLOW`, `YTD_COLOR` constants near the top of `Home.py`
- Chart logic → functions `make_single_series_bar`, `make_bar_chart`, `make_provider_vs_gx_flow_with_aum_line`
- Table columns shown → the `.rename()` and column selection in the `top15_*` blocks near the bottom

---

### 01_BubbleDashboard.py — Bubble Chart

Plots each ETF as a bubble: **X = TTM Net Flow**, **Y = Monthly Flow**, **size = AUM**, **colour = Category**.

Includes a secondary scatter (Flow % vs Performance) and a sidebar LLM chat (requires OpenAI token).

**To add a new axis option**, modify the `x`, `y`, or `size` arguments passed to `px.scatter`.

---

### 02_HeatMap.py — Flow % AUM Heatmaps

Shows trailing 3-month flow as a percentage of starting AUM for each category/sub-category combination. Uses a diverging red-yellow-green colour scale with power-transform normalisation.

**To add or reorder the default rows**, edit the `MIXED_DEFAULT_ROWS` list near the top of the file.

---

### 03_TrendChart.py — 12-Month Trend Lines

Line chart of monthly net flows for up to 20 categories over the past 12 months. Dynamically groups by Category or Sub-Category depending on which filters are active.

---

### 04_Flow.py — Weekly Flow Report

Provider and fund-level weekly trading tables, styled with green/red data bars. Loops over a `segments` dictionary — each key is a segment name (e.g. "Lightly Leveraged"), each value is a filtered DataFrame.

**To add a new segment**, append an entry to the `segments` dict:

```python
segments["My New Segment"] = df_weekly[df_weekly["Category"] == "My Category"]
```

---

### 05_BMO_ETFs.py — BMO Live Trading (Bloomberg only)

Fetches real-time Bloomberg fields (AUM, volume, NAV, 1-day return) for a fixed list of BMO ETF tickers. Requires Bloomberg Terminal + `xbbg`.

---

### 06_Betapro_Backtest.py / 07_Backtest_V2.py — Leveraged ETF Backtests

Both pages pull 5-year Bloomberg price history and simulate leveraged ETF mechanics (daily rebalance, drift tolerance). V2 adds drawdown and annualised return tables, and a user-selectable start date.

**To add a new ticker**, append to the `TICKER_CONFIG` dictionary:

```python
TICKER_CONFIG = {
    ...
    "MYETF": [1.25, 0.005, "UNDERLYING Index"],  # [leverage ratio, rebalance tolerance, Bloomberg index]
}
```

---

## Key Utility Files

### config.py

Central hub for **colours, CSS, and reusable UI components**. Import from here in any page:

```python
from config import (
    apply_common_styling,   # inject shared CSS — call once per page
    render_header,          # branded dark-navy + orange page header
    render_metric_card,     # KPI card HTML (label, value, optional delta)
    render_page_intro,      # blue-accented intro paragraph box
    render_section_divider, # orange-gradient section rule with label
    render_glossary,        # expandable ETF terms glossary
    format_large_number,    # e.g. 1_500_000 → "$1.5M"
    format_percentage,      # e.g. 0.1234 → "12.34%"
)
```

**Brand colours** are defined at the top of `config.py`:

```python
PRIMARY_COLOR  = "#FF5722"  # Orange  — Global X brand accent
SECONDARY_COLOR= "#00695C"  # Teal    — section headers
ACCENT_COLOR   = "#4682A9"  # Steel blue
DARK_COLOR     = "#1B2A3B"  # Navy    — page header background
SUCCESS_COLOR  = "#16A34A"  # Green   — positive flows
DANGER_COLOR   = "#DC2626"  # Red     — outflows / negative
```

---

### data_prep.py

Handles all data loading and transformation.

| Function | What it does |
|---|---|
| `load_raw_data(url)` | Downloads the OneDrive Excel file, returns 4 raw DataFrames. Cached with `@st.cache_data`. |
| `process_data_for_date(date_str, ...)` | Merges AUM + flow + performance for a given month-end date. Calculates Monthly Flow, YTD Flow, TTM Net Flow. Handles paired ETFs. |
| `load_raw_data_heatmap(url)` | Variant loader for the HeatMap page (different column pivoting). |
| `build_trailing3m_flow_pct_table(...)` | Builds the 3-month trailing flow % matrix used by HeatMap. |

**Paired ETF logic:** Some funds are issued in CAD (`XXX`) and USD-hedged (`XXX/U`) variants. `process_data_for_date` detects these pairs, keeps the USD version as `XXX(U)`, and combines the AUM/flows accordingly. The `Indicator` column value `2` marks a fund as part of a pair.

---

### analytics_utils.py

Advanced analytics called from `Home.py`.

| Function | Output |
|---|---|
| `calculate_market_concentration(df)` | HHI score + top-10 ETF AUM shares |
| `calculate_provider_market_share(df)` | Per-provider AUM and flow market share |
| `create_provider_market_share_chart(stats)` | Plotly pie chart (top 8 providers + "Others") |
| `create_concentration_chart(data)` | Horizontal bar chart of top ETFs by AUM share |
| `create_flow_momentum_indicator(df, col)` | Count and sum of positive vs negative flows |
| `create_flow_momentum_chart(data)` | Grouped bar chart of inflows vs outflows |

---

### llm_api.py

Wraps the OpenAI API for the sidebar chat on the Bubble Dashboard. The key function is `ask_gemma_with_context()`, which sends filtered ETF data on the first question and uses conversation history on follow-ups to minimise token usage.

To use the chat, users must paste an OpenAI API key into the sidebar text field at runtime. No key is stored in the app.

---

## Styling & Theming

The app uses a **two-layer** styling approach:

1. **`.streamlit/config.toml`** — Sets Streamlit's native theme (button colour, background, font). Changes here affect all native widgets (dropdowns, sliders, checkboxes) automatically.

2. **`config.py → COMMON_CSS`** — A `<style>` block injected via `st.markdown(..., unsafe_allow_html=True)`. Controls custom components (metric cards, section dividers, headers, intro boxes). Call `apply_common_styling()` at the top of any page to activate it.

**To change the brand colour globally**, update both:
- `PRIMARY_COLOR` in `config.py`
- `primaryColor` in `.streamlit/config.toml`

---

## How to Add a New Page

1. Create a new file in `pages/`, e.g. `08_MyNewPage.py`. The `08_` prefix sets its position in the sidebar.

2. Start with this boilerplate:

```python
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import apply_common_styling, render_header, render_page_intro
from data_prep import load_raw_data, process_data_for_date

st.set_page_config(page_title="My Page Title", layout="wide")
apply_common_styling()

# ── Data ──────────────────────────────────────────────────────────────────────
from config import ONEDRIVE_URL
funds_df, aum_df, flow_df, perf_df = load_raw_data(ONEDRIVE_URL)

# ── Header ────────────────────────────────────────────────────────────────────
render_header("My Page Title", "Short description shown under the title")
render_page_intro("One or two sentences explaining what this page shows.")

# ── Your content ──────────────────────────────────────────────────────────────
st.write("Hello, world!")
```

3. Streamlit picks up the file immediately — no registration required. Restart the dev server if it doesn't appear.

---

## Common Customisations

### Change which providers appear in the provider dropdown

In `Home.py`, the provider list is built from the raw funds sheet:

```python
provider_options = ["All (Industry)"] + sorted(provider_col.unique())
```

To exclude a provider, add a filter before `.unique()`:

```python
provider_options = ["All (Industry)"] + sorted(
    provider_col[~provider_col.str.contains("Exclude Me", case=False)].unique()
)
```

### Change the "Top N" table size

The tables show top 15 by default. Change the `15` in calls like `df.nlargest(15, "Monthly Flow")`.

### Change the AUM threshold for "small fund" tables

Currently `$1B`. Search for `1_000_000_000` in `Home.py` and update.

### Add a new segment to the Weekly Flow report

In `04_Flow.py`, find the `segments` dictionary and add:

```python
segments["New Segment Name"] = df_weekly[df_weekly["Category"] == "Your Category"]
```

### Update the OneDrive data source URL

Update the single constant in `config.py`:

```python
ONEDRIVE_URL = "https://your-new-sharepoint-url?download=1"
```

All pages that call `load_raw_data` will automatically use the new URL. The `?download=1` query parameter is required for SharePoint direct-download links.
