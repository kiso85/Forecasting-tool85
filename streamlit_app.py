import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EPSEVG Hourly Energy Forecast", layout="wide")

st.title("⚡ EPSEVG Hourly Energy Forecast (Prophet)")

# ----------------------------------------------------------
# 1. File paths
# ----------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "energy_Via-Ag-36.csv"      # your file

# ----------------------------------------------------------
# 2. Load hourly energy dataset
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    # Auto-detect columns
    date_col = [c for c in df.columns if "fecha" in c.lower() or "date" in c.lower()][0]
    energy_col = [c for c in df.columns if "energy" in c.lower() or "energ" in c.lower()][0]

    # Parse datetime format: DD/MM/YYYY HH:MM
    df["ds"] = pd.to_datetime(df[date_col], format="%d/%m/%Y %H:%M", errors="coerce")
    df["y"] = df[energy_col].astype(float)

    df = df[["ds", "y"]].dropna().sort_values("ds")
    return df

df = load_data()

# ----------------------------------------------------------
# 3. Build Spain + School holiday table
# ----------------------------------------------------------
def make_holiday_df(start_year, end_year):
    holidays = []
    for year in range(start_year, end_year + 1):
        # Spain national holidays
        for d in ["01-01", "01-06", "05-01", "08-15",
                  "10-12", "11-01", "12-06", "12-08", "12-25"]:
            holidays.append({"holiday": "spain_public", "ds": f"{year}-{d}"})

        # School summer vacation
        for m in [7, 8]:
            for day in range(1, 32):
                holidays.append({"holiday": "summer_break", "ds": f"{year}-{m:02d}-{day:02d}"})

    return pd.DataFrame(holidays)

holiday_df = make_holiday_df(df["ds"].dt.year.min(), df["ds"].dt.year.max() + 1)

# ----------------------------------------------------------
# 4. Train Prophet hourly model
# ----------------------------------------------------------
@st.cache_resource
def train_model(df, holidays):
    model = Prophet(
        daily_seasonality=True,     # hourly inside day
        weekly_seasonality=True,    # Mon -> Sun
        yearly_seasonality=True,    # long-term
        holidays=holidays,
        seasonality_mode="multiplicative"
    )

    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(df)
    return model

model = train_model(df, holiday_df)

# ----------------------------------------------------------
# 5. Forecast Horizon  ( <-- FIXED DEFINITON HERE )
# ----------------------------------------------------------
horizon_options = {
    "Next 24 hours": 24,
    "Next 48 hours": 48,
    "Next 7 days": 24 * 7,
    "Next 14 days": 24 * 14
}

horizon_label = st.selectbox("Select forecast horizon:", list(horizon_options.keys()))
horizon_hours = horizon_options[horizon_label]  # Always integer

# ----------------------------------------------------------
# 6. Generate Forecast (Hourly)
# ----------------------------------------------------------
future = model.make_future_dataframe(periods=horizon_hours, freq="H")
forecast = model.predict(future)

# ----------------------------------------------------------
# 7. Plot forecast + historical
# ----------------------------------------------------------
fig = px.line(
    forecast,
    x="ds",
    y="yhat",
    title=f"EPSEVG Forecast for {horizon_hours} Hours",
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

# ----------------------------------------------------------
# 8. Footer
# ----------------------------------------------------------
st.caption("""
This hourly forecast is powered by Facebook Prophet.

The model learns:
- Monday morning consumption surge
- Tue–Fri stable high levels
- Saturday & Sunday sharp decrease
- Hour-of-day patterns (day vs night)
- Spain public holidays
- Summer school break (Jul–Aug)
""")
