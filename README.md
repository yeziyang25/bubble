# ETF Bubble Dashboard

This dashboard visualizes ETF data with bubble charts showing the relationship between flows and AUM.

## Project Structure

- `bubble_dashboard.py`: The main Streamlit app that generates the dashboard.
- `aum.csv`: CSV file containing AUM data.
- `fund flow.csv`: CSV file containing fund flow data.
- `funds.csv`: CSV file with additional fund information.
- `requirements.txt`: Contains the list of required packages:
  - pandas
  - plotly
  - streamlit

## Setup

1. **Clone this repository** and open it in Visual Studio Code.

2. **Install the required dependencies** by running:

    ```sh
    pip install -r requirements.txt
    ```

## Running the App

Start the application using Streamlit by running:

```sh
streamlit run bubble_dashboard.py
```

This will open the ETF Bubble Chart Dashboard in your default web browser.

## How It Works

1. The dashboard loads data from the CSV files using the `load_data` function.
2. Users can filter ETFs by category and secondary category.
3. The bubble chart is generated using Plotly based on user-selected filters:
   - **Label Options:** Choose between labels under the bubble or a side legend.
4. The chart displays key metrics like **TTM Net Flow**, **Monthly Flow**, and **AUM** for each ETF.

## Customization

You can customize the appearance and functionality by editing `bubble_dashboard.py`. The file contains inline comments to help guide customization related to:
- Data loading
- Filtering logic
- Chart appearance using Plotly section [`px.scatter`](https://plotly.com/python/plotly-express/).

## Data Sources

Data is sourced from Bloomberg. The dashboard is built with Streamlit & Plotly and was created in 2025.

---
Happy analyzing!