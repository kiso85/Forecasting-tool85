import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EPSEVG Hourly Energy Forecast", layout="wide")

st.title("⚡ EPSEVG Hourly Energy Consumption Forecast (Prophet)")

DATA_DIR = Path(__file__).parent

# -----------------------------------
# Load hourly dataset
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "df_hourly_processed.csv")

    # Standardize column names
    date_col = [c for c in df.columns if "fecha" in c.lower() or "date" in c.lower()][0]
    energy_col = [c for c in df.columns if "energy" in c.lower() or "energ" in c.lower()][0]

    df["ds"] = pd.to_datetime(df[date_col], format="%d/%m/%Y %H:%M")
    df["y"] = df[energy_col].astype(float)

    df = df[["ds", "y"]].sort_values("ds")
    return df

df = load_data()

# -----------------------------------
# Build holiday table (Spain + school)
# -----------------------------------
def make_holiday_df(start_year=2020, end_year=2025):
    holidays = []
    for year in range(start_year, end_year + 1):
        for d in ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]:
            holidays.append({"holiday": "spain_public", "ds": f"{year}-{d}"})

        # July–August summer school break
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
        daily_seasonality=True,      # Hourly pattern within a day
        weekly_seasonality=True,     # Mon–Sun weekly cycle
        yearly_seasonality=True,     # Seasonal energy variations
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
horizon_option = st.selectbox(
    "Select forecast horizon:",
    {
        "Next 24 hours": 24,
        "Next 48 hours": 48,
        "Next 7 days": 24 * 7,
        "Next 14 days": 24 * 14
    }
)

horizon_hours = horizon_option

# -----------------------------------
# Generate hourly forecast
# -----------------------------------
future = model.make_future_dataframe(periods=horizon_hours, freq="H")
forecast = model.predict(future)

# -----------------------------------
# Plot
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
    mode="lines",
    line=dict(color="blue", width=1.5)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# Notes
# -----------------------------------
st.caption("""
This hourly forecast model uses **Facebook Prophet** to learn:
- Hour-of-day patterns (daily seasonality)
- Monday high consumption spike
- Tuesday–Friday stable high working cycle
- Saturday + Sunday low consumption
- Spanish national holidays
- Summer school break (July–August)

Because the model is trained on hourly data, the forecast reflects the true
hourly behaviour of the building.
""")
