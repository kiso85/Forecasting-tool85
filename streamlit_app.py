import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EPSEVG Hourly Energy Forecast", layout="wide")

st.title("⚡ EPSEVG Hourly Energy Consumption Forecast (Prophet)")

# -----------------------------------
# File path setup
# -----------------------------------
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "energy_Via-Ag-36.csv"   # ✔ YOUR FILE

# -----------------------------------
# Load hourly dataset
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    # Detect date column
    date_col = [c for c in df.columns if "fecha" in c.lower() or "date" in c.lower()][0]
    energy_col = [c for c in df.columns if "energy" in c.lower() 
                  or "energ" in c.lower()][0]

    # Parse format: DD/MM/YYYY HH:MM
    df["ds"] = pd.to_datetime(df[date_col], format="%d/%m/%Y %H:%M", errors="coerce")
    df["y"] = df[energy_col].astype(float)

    # Clean + sort
    df = df[["ds", "y"]].dropna().sort_values("ds")

    return df

df = load_data()

# -----------------------------------
# Build holiday table (Spain + school)
# -----------------------------------
def make_holiday_df(start_year=2020, end_year=2025):
    holidays = []
    for year in range(start_year, end_year + 1):
        for d in ["01-01", "01-06", "05-01", "08-15",
                  "10-12", "11-01", "12-06", "12-08", "12-25"]:
            holidays.append({"holiday": "spain_public", "ds": f"{year}-{d}"})

        # School summer break (July–August)
        for m in [7, 8]:
            for day in range(1, 32):
                holidays.append({"holiday": "summer_break", "ds": f"{year}-{m:02d}-{day:02d}"})

    return pd.DataFrame(holidays)

holiday_df = make_holiday_df(df["ds"].dt.year.min(), df["ds"].dt.year.max() + 1)

# -----------------------------------
# Train Prophet model (Hourly)
# -----------------------------------
@st.cache_resource
def train_hourly_prophet(df, holidays):
    m = Prophet(
        daily_seasonality=True,      # captures hourly pattern inside a day
        weekly_seasonality=True,     # captures Mon–Sun pattern
        yearly_seasonality=True,     # long-term trend
        holidays=holidays,
        seasonality_mode="multiplicative"
    )

    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(df)
    return m

model = train_hourly_prophet(df, holiday_df)

# -----------------------------------
# Forecast horizon
# -----------------------------------
horizon_label = st.selectbox(
    "Select forecast horizon:",
    {
        "Next 24 hours": 24,
        "Next 48 hours": 48,
        "Next 7 days": 24 * 7,
        "Next 14 days": 24 * 14
    }
)

# Dropdown
horizon_label = st.selectbox(
    "Select forecast horizon:",
    list(horizon_options.keys())
)

# Convert label → integer hours
horizon_hours = horizon_options[horizon_label]

# -----------------------------------
# Generate forecast (hourly)
# -----------------------------------
future = model.make_future_dataframe(periods=horizon_hours, freq="H")
forecast = model.predict(future)

# -----------------------------------
# Plot forecast + history
# -----------------------------------
fig = px.line(
    forecast,
    x="ds",
    y="yhat",
    title=f"EPSEVG Hourly Energy Forecast ({horizon_hours} hours)",
    labels={"ds": "Date-Time", "yhat": "Predicted Energy (kWh)"}
)

fig.add_scatter(
    x=df["ds"],
    y=df["y"],
    name="Historical",
    line=dict(color="blue", width=1.3)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# Notes
# -----------------------------------
st.caption("""
This model learns from HOURLY energy data to accurately capture:
- Monday consumption spike
- Tuesday–Friday high and stable work cycle
- Saturday & Sunday low consumption
- Hourly patterns within each day (e.g., daytime vs nighttime)
- Spanish public holidays
- Summer school break (July–August)
""")
