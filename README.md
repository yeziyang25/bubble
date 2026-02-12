# Canadian ETF Market Intelligence Dashboard

A comprehensive, professional analytics dashboard for analyzing the Canadian ETF landscape. Built with Streamlit and Plotly, this dashboard provides deep insights into ETF flows, performance, market concentration, and competitive dynamics.

## 🌟 Key Features

### Home Page - Market Intelligence
- **Professional Header & Navigation**: Clean, modern interface with consistent branding
- **Market Concentration Analysis**: 
  - Top 10 ETFs by market share visualization
  - Herfindahl-Hirschman Index (HHI) calculation for market competitiveness
  - Concentration percentage metrics
- **Provider Market Share Analysis**:
  - Interactive pie charts showing provider distribution
  - Market share percentages and ETF counts
  - Top provider performance metrics
- **Flow Momentum Indicators**:
  - Real-time tracking of inflows vs outflows
  - ETF count and dollar amount comparisons
  - Net flow calculations
- **Category Performance Metrics**:
  - Top categories by AUM with performance data
  - Flow percentages and ETF counts
  - Average performance tracking
- **13-Month Historical Flow Analysis**:
  - Industry-wide flow trends
  - Global X specific flow analysis
  - Provider vs Global X comparisons with AUM overlay
- **Top ETF Tables with Sparklines**:
  - Top 15 inflows/outflows by monthly flow
  - Top 15 for small-cap ETFs (AUM < $1B)
  - 12-month flow trend sparklines
  - Global X/BetaPro specific tables
  - YTD new launches performance

### Bubble Dashboard Page
- **Interactive Bubble Charts**:
  - Visualize ETF relationships between TTM flows, monthly flows, and AUM
  - Color-coded by category
  - Hover details with full fund information
- **Secondary Category Analysis**:
  - Performance vs flow percentage scatter plots
  - Dynamic grouping by category or subcategory
- **Asset Allocation Sunburst Chart**:
  - Hierarchical visualization of category and subcategory allocations
  - Interactive drill-down capabilities
- **Flow Consistency Analysis**:
  - 6-month flow consistency scores
  - Identifies ETFs with consistent positive flows
  - Ranking of most reliable performers
- **Advanced Filtering**:
  - Search and multi-select for categories
  - Secondary category filtering
  - Lightly leveraged ETF filter
  - Saved filter states
- **AI Chat Integration**:
  - Context-aware chat using OpenAI API
  - Smart data filtering based on questions
  - Token-optimized conversation history

### HeatMap Page
- **Trailing 3-Month Flow % Heatmaps**:
  - Color-coded visualization of flow trends
  - Category and subcategory breakdowns
  - Robust scaling with gamma transformation
  - Customizable time periods
- **Professional Color Gradients**:
  - Red-Yellow-Green diverging scale
  - Percentage-based formatting
  - Outlier-resistant scaling

### Trend Chart Page
- **12-Month Flow Trends**:
  - Dynamic category/subcategory grouping
  - Top N categories by AUM (configurable)
  - Monthly flow tracking over time
- **AUM Growth Trajectories**:
  - Track up to 20 top ETFs simultaneously
  - 12-month historical AUM data
  - Interactive line charts with markers
  - Customizable ETF selection
- **Smart Filtering**:
  - Automatic top category selection to reduce clutter
  - Option to show all categories
  - Real-time filter updates

## 📊 Analytics Capabilities

### Market Intelligence
- **Concentration Metrics**: HHI scores and top-N concentration analysis
- **Provider Analytics**: Market share, ETF counts, flow analysis
- **Flow Momentum**: Inflow/outflow tracking and net flow calculations
- **Category Performance**: AUM, flow %, and performance metrics by category

### Advanced Visualizations
- **Sunburst Charts**: Hierarchical asset allocation views
- **Flow Consistency Scores**: 6-month historical consistency tracking
- **AUM Growth Trajectories**: Multi-ETF comparison over time
- **Market Concentration Charts**: Top ETF market share visualization
- **Provider Pie Charts**: Market share distribution

### Data Processing
- **Smart ETF Pairing**: Automatic handling of USD/CAD pairs
- **Flow Imputation**: Intelligent flow estimation for U-class shares
- **12-Month Sparklines**: Embedded historical trend visualization
- **Category/Subcategory Mapping**: Robust classification handling

## 📁 Project Structure

```
bubble/
├── Home.py                    # Main dashboard page with market intelligence
├── config.py                  # Shared configuration, styling, and utilities
├── analytics_utils.py         # Advanced analytics and visualization functions
├── data_prep.py              # Data loading and processing functions
├── llm_api.py                # OpenAI API integration for chat
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── pages/
│   ├── 01_BubbleDashboard.py  # Bubble chart visualizations
│   ├── 02_HeatMap.py          # Flow heatmap analysis
│   └── 03_TrendChart.py       # Trend and trajectory analysis
└── .streamlit/
    └── secrets.toml           # API keys and secrets (not in repo)
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit 
- **Visualizations**: Plotly Express & Plotly Graph Objects
- **Data Processing**: Pandas, NumPy
- **Data Source**: Excel files from OneDrive (live data)
- **AI Integration**: OpenAI API for chat functionality
- **Styling**: Custom CSS with professional color scheme

## 🚀 Setup & Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/yeziyang25/bubble.git
   cd bubble
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure API keys** (optional, for chat functionality)
   - Create `.streamlit/secrets.toml` if using chat features
   - Add your OpenAI API key

4. **Run the application**
   ```sh
   streamlit run Home.py
   ```

5. **Access the dashboard**
   - The application will open automatically in your default browser
   - Navigate to `http://localhost:8501` if it doesn't open automatically

## 📖 How to Use the Dashboard

### Navigation
- **Home**: Market intelligence overview and comprehensive flow analysis
- **Bubble Dashboard**: Interactive scatter plots and asset allocation
- **HeatMap**: Historical flow percentage heatmaps
- **Trend Chart**: Time-series analysis and AUM growth tracking

### Key Workflows

#### 1. Market Overview Analysis
1. Select an analysis date from the dropdown
2. Choose a provider or view "All (Industry)"
3. Optionally filter by Category and/or Secondary Category
4. Review market concentration and provider share metrics
5. Analyze flow momentum (inflows vs outflows)

#### 2. ETF Selection & Research
1. Navigate to the Bubble Dashboard
2. Use category filters to narrow your focus
3. Identify outliers in the bubble chart (high flows, unusual positions)
4. Check flow consistency scores to find reliable performers
5. Use the sunburst chart for allocation insights

#### 3. Trend Analysis
1. Go to the Trend Chart page
2. Filter to specific categories or subcategories
3. Review 12-month flow trends
4. Select top ETFs to track their AUM growth
5. Compare trajectories across different funds

#### 4. Historical Pattern Recognition
1. Visit the HeatMap page
2. Select an as-of date
3. Observe flow patterns across categories and time
4. Identify consistent winners/losers
5. Spot trend reversals or accelerations

### Advanced Features

#### Category & Subcategory Filtering
- Use the search box to quickly find categories
- Click "➕All" to add all matching results
- Combine Category and Secondary Category filters for precision
- Filters persist across sessions via session state

#### Flow Consistency Analysis
- Consistency Score = % of months with positive flows
- Helps identify reliable, steady performers
- Useful for core portfolio selection
- Filters out volatile, unpredictable funds

#### Market Concentration Metrics
- **HHI < 1500**: Competitive market
- **HHI 1500-2500**: Moderately concentrated
- **HHI > 2500**: Highly concentrated
- Top 10 concentration shows market dominance

## 🎨 Design Philosophy

### Professional & Clean
- Modern gradient backgrounds
- Consistent color scheme (Orange primary, Teal secondary)
- Smooth transitions and hover effects
- Card-based layouts for metrics

### ETF Expert-Focused
- Industry-standard metrics (HHI, market share, flow %)
- Multi-timeframe analysis (monthly, YTD, TTM, trailing 13M)
- Provider-level comparisons
- Category/subcategory breakdowns

### Performance-Optimized
- Data caching for fast page loads
- Efficient computation of historical flows
- Robust handling of missing data
- Minimal API calls

## 📊 Data Sources & Updates

- **Primary Data**: OneDrive-hosted Excel file (live updates)
- **Sheets**: 
  - `consolidated_10032022`: Fund metadata
  - `aum`: Monthly AUM values
  - `fund_flow`: Monthly net flows
  - `performance`: Monthly performance data
- **Update Frequency**: As data source is updated
- **Historical Range**: Multi-year data available

## 🔧 Customization

### Styling
- Edit `config.py` to change colors, fonts, and styling
- Modify `COMMON_CSS` for global style changes
- Update `render_header()` for different branding

### Analytics
- Extend `analytics_utils.py` for new metrics
- Add new visualization functions
- Integrate additional data sources

### Pages
- Create new pages in the `pages/` directory
- Follow naming convention: `NN_PageName.py`
- Import from `config.py` and `analytics_utils.py` for consistency

## 🤝 Contributing

This dashboard is designed for continuous improvement. Potential enhancements:

- Export functionality (PDF, Excel)
- Custom date range comparisons
- Alerts for significant changes
- Risk-adjusted return metrics
- Correlation analysis
- Expense ratio comparisons

## 📝 Notes

- The dashboard requires internet access to fetch live data
- Chat functionality requires an OpenAI API key
- Some calculations may take a few seconds for large datasets
- Historical data availability varies by ETF inception date

---

## 📄 License & Credits

Built with Streamlit, Plotly, and Pandas.  
Data sourced from Bloomberg via Global X Canada.  
Created in 2025 for Canadian ETF market analysis.