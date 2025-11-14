# --------------------------------------------------------------------------
# |                   IMPORTAR LIBRERÍAS                                   |
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import requests
import os
import glob

# --------------------------------------------------------------------------
# |                   CONFIGURACIÓN DE LA PÁGINA                           |
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Consumo Energético con Prophet",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# |                   FUNCIONES DE CARGA DE DATOS                          |
# --------------------------------------------------------------------------

@st.cache_data
def load_asepeyo_energy_data(file_path):
    """Carga y procesa el archivo de consumo energético desde una ruta."""
    try:
        df = pd.read_csv(file_path, sep=',', decimal='.')
        if 'Fecha' not in df.columns or 'Energía activa (kWh)' not in df.columns:
            st.error(f"El archivo {file_path} debe contener 'Fecha' y 'Energía activa (kWh)'.")
            return pd.DataFrame()
            
        df.rename(columns={'Fecha': 'fecha', 'Energía activa (kWh)': 'consumo_kwh'}, inplace=True)
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return pd.DataFrame()


@st.cache_data
def get_weather_forecast(api_key, lat, lon):
    """Obtiene el pronóstico del tiempo diario desde la API de Meteosource."""
    BASE_URL = "https://www.meteosource.com/api/v1/free/point"
    params = {
        "lat": lat,
        "lon": lon,
        "sections": "daily",
        "units": "metric",
        "key": api_key
    }
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            daily_data = data.get('daily', {}).get('data', [])
            if not daily_data:
                st.warning("⚠️ La API no devolvió datos de pronóstico diario.")
                return pd.DataFrame()
            df = pd.DataFrame([{
                'fecha': day['day'],
                'temp_max_c': day['all_day']['temperature_max'],
                'temp_min_c': day['all_day']['temperature_min']
            } for day in daily_data])
            df['fecha'] = pd.to_datetime(df['fecha'])
            df['temp_avg_c'] = (df['temp_max_c'] + df['temp_min_c']) / 2
            return df
        else:
            st.error(f"Error en la API de Meteosource (Código {response.status_code})")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al conectar con la API del clima: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------------
# |                   BARRA LATERAL DE CONFIGURACIÓN                       |
# --------------------------------------------------------------------------

st.sidebar.title("⚙️ Configuración de la Predicción")
st.sidebar.markdown("---")

# Directorio de datos
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
st.sidebar.info(f"📁 Carpeta de datos: {DATA_DIR}")

# Selección de archivo
energy_pattern = os.path.join(DATA_DIR, "energy_*.csv")
energy_files = [os.path.basename(f) for f in glob.glob(energy_pattern)]
selected_energy_file = st.sidebar.selectbox("Selecciona archivo de consumo", energy_files) if energy_files else None

st.sidebar.markdown("---")

# Parámetros de API (opcional)
st.sidebar.header("🌤️ API Meteosource (opcional)")
api_key = st.sidebar.text_input("API Key de Meteosource", type="password")
lat = st.sidebar.text_input("Latitud", "40.4168")
lon = st.sidebar.text_input("Longitud", "-3.7038")

# Parámetros del modelo
st.sidebar.markdown("---")
st.sidebar.header("🧠 Parámetros del Modelo Prophet")
future_days = st.sidebar.slider("Días a predecir", 7, 90, 30)
future_hours = future_days * 24
include_holidays = st.sidebar.checkbox("Incluir festivos de España (ES)", value=True)

# --------------------------------------------------------------------------
# |                   CUERPO PRINCIPAL                                     |
# --------------------------------------------------------------------------

st.title("🎀 Predicción de Consumo Energético (Hourly) 🎀")
st.subheader("Modelo Prophet con resolución de 1 hora")
st.markdown("---")

if selected_energy_file:
    energy_path = os.path.join(DATA_DIR, selected_energy_file)
    df_energia = load_asepeyo_energy_data(energy_path)

    if not df_energia.empty:

        # ----------------------------------------------------------------------
        # 🔄 NUEVO: Convertir datos diarios a datos HORARIOS
        # ----------------------------------------------------------------------
        df_energia = df_energia.set_index("fecha").sort_index()

        # Resample HOURLY (Prophet necesita serie continua)
        df_hourly = df_energia['consumo_kwh'].resample("1H").interpolate()

        df_prophet = df_hourly.reset_index().rename(columns={"fecha": "ds", "consumo_kwh": "y"})

        # ----------------------------------------------------------------------
        # 🔒 NUEVO: Filtrar SOLO 2023-07-01 → 2024-07-01 para entrenar
        # ----------------------------------------------------------------------
        df_prophet = df_prophet[(df_prophet["ds"] >= "2023-07-01") & (df_prophet["ds"] < "2024-07-01")]

        # --- Crear y entrenar el modelo ---
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,   # 🔄 cambiar a True para patrones intradía
            changepoint_prior_scale=0.1
        )

        if include_holidays:
            try:
                model.add_country_holidays(country_name='ES')
                st.sidebar.success("✅ Festivos de España añadidos.")
            except Exception as e:
                st.sidebar.warning(f"No se pudieron añadir festivos: {e}")

        with st.spinner("Entrenando modelo Prophet (hourly)..."):
            model.fit(df_prophet)

        # ----------------------------------------------------------------------
        # ⏳ NUEVO: Predicción HORARIA
        # ----------------------------------------------------------------------
        future = model.make_future_dataframe(periods=future_hours, freq="H")
        forecast = model.predict(future)

        # --- Mostrar resultados ---
        st.subheader("📈 Predicción de Consumo Energético (HORARIA)")
        st.pyplot(model.plot(forecast))

        st.subheader("📊 Componentes del modelo")
        st.pyplot(model.plot_components(forecast))

        # --- Gráfico Interactivo ---
        st.subheader("📊 Gráfico Interactivo del Pronóstico (Hourly)")

        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_hours)
        forecast_display["Fecha"] = forecast_display["ds"]

        fig = px.line(
            forecast_display,
            x='Fecha',
            y='yhat',
            title="Predicción Horaria del Consumo (Próximas horas)",
            labels={'yhat': 'Consumo (kWh)'},
            color_discrete_sequence=['royalblue']
        )
        st.plotly_chart(fig, use_container_width=True)

        mostrar_tabla = st.checkbox("📋 Mostrar tabla de predicción detallada (hourly)")
        if mostrar_tabla:
            st.dataframe(forecast_display.round(2))

        # --- API del clima (opcional) ---
        if api_key:
            st.markdown("---")
            st.subheader("🌦️ Pronóstico del clima (Meteosource)")
            df_clima_futuro = get_weather_forecast(api_key, lat, lon)
            if not df_clima_futuro.empty:
                st.dataframe(df_clima_futuro)
                fig_temp = px.line(df_clima_futuro, x='fecha', y='temp_avg_c',
                                   title='Temperatura Promedio Prevista (°C)')
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.warning("No se pudo obtener datos del clima. Verifica tu API Key.")

    else:
        st.error("❌ No se pudieron cargar los datos de consumo.")
else:
    st.info("ℹ️ Selecciona un archivo CSV de energía en la barra lateral izquierda para comenzar.")
