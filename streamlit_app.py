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
include_holidays = st.sidebar.checkbox("Incluir festivos de España (ES)", value=True)

# --------------------------------------------------------------------------
# |                   CUERPO PRINCIPAL                                     |
# --------------------------------------------------------------------------

st.title("🎀 Predicción de Consumo Energético con Prophet")
st.subheader("Asepeyo")
st.markdown("---")
# ✅ Visual confirmation that this version of the app has loaded
st.success("✅ Accuracy test feature loaded successfully — you are running the latest version of the app.")

if selected_energy_file:
    energy_path = os.path.join(DATA_DIR, selected_energy_file)
    df_energia = load_asepeyo_energy_data(energy_path)

    if not df_energia.empty:
        

        # --- Preparar datos para Prophet ---
        df_prophet = df_energia.rename(columns={'fecha': 'ds', 'consumo_kwh': 'y'})
        df_prophet = df_prophet.groupby('ds')['y'].sum().reset_index()  # en caso de duplicados

        # --- Crear y entrenar el modelo ---
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1
        )

        if include_holidays:
            try:
                model.add_country_holidays(country_name='ES')
                st.sidebar.success("✅ Festivos de España añadidos.")
            except Exception as e:
                st.sidebar.warning(f"No se pudieron añadir festivos: {e}")

        with st.spinner("Entrenando modelo Prophet..."):
            model.fit(df_prophet)

        # --- Crear fechas futuras y predecir ---
        future = model.make_future_dataframe(periods=future_days)
        forecast = model.predict(future)

        st.warning("🔧 Accuracy test block loaded")
        # --------------------------------------------------------------------------
        # ✅ Optional Model Accuracy Test (Hold-out)
        # --------------------------------------------------------------------------
        st.markdown("---")
        check_accuracy = st.checkbox("🧪 Check Model Accuracy (Hold-out Test)")
        
        if check_accuracy:
            st.subheader("📊 Prophet Model Accuracy Test")
        
            # Split: last 90 days of data as test
            horizon_days = st.number_input("Days to hold out for testing", 30, 180, 90)
            cutoff_date = df_prophet['ds'].max() - pd.Timedelta(days=horizon_days)
            train = df_prophet[df_prophet['ds'] <= cutoff_date]
            test  = df_prophet[df_prophet['ds'] >  cutoff_date]
        
            st.write(f"Training data: {train['ds'].min().date()} → {train['ds'].max().date()}")
            st.write(f"Testing data: {test['ds'].min().date()} → {test['ds'].max().date()}")
        
            # Train Prophet on training data
            acc_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.1
            )
            acc_model.fit(train)
        
            # Predict for the test period
            future_test = acc_model.make_future_dataframe(periods=horizon_days, freq='D')
            forecast_test = acc_model.predict(future_test)
        
            # Compare only overlapping period
            pred = forecast_test[['ds', 'yhat']].merge(test, on='ds', how='inner')
            pred['squared_error'] = (pred['yhat'] - pred['y']) ** 2
        
            MSE  = np.mean(pred['squared_error'])
            RMSE = np.sqrt(MSE)
        
            st.success(f"✅ Mean Squared Error (MSE): **{MSE:.2f}**  Root MSE (RMSE): **{RMSE:.2f}**")
        
            # Plot predicted vs actual for test window
            import plotly.graph_objects as go
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=test['ds'], y=test['y'],
                mode='lines+markers', name='Actual', line=dict(color='black')
            ))
            fig_acc.add_trace(go.Scatter(
                x=pred['ds'], y=pred['yhat'],
                mode='lines+markers', name='Predicted', line=dict(color='royalblue')
            ))
            fig_acc.update_layout(
                title="Actual vs Predicted Energy Consumption (Validation Period)",
                xaxis_title="Date", yaxis_title="Consumption (kWh)",
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#E0E0E0"), yaxis=dict(gridcolor="#E0E0E0")
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        # --- Mostrar resultados ---
        st.subheader("📈 Predicción de Consumo Energético (Prophet)")
        st.pyplot(model.plot(forecast))
        
        st.subheader("📊 Componentes del modelo")
        st.pyplot(model.plot_components(forecast))
        
        # --- Gráfico Interactivo del Pronóstico ---
        st.subheader("📊 Gráfico Interactivo del Pronóstico")
        
        # 准备数据
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_days)
        forecast_display.rename(columns={
            'ds': 'Fecha',
            'yhat': 'Consumo_Predicho',
            'yhat_lower': 'Intervalo_Inferior',
            'yhat_upper': 'Intervalo_Superior'
        }, inplace=True)
        forecast_display['Fecha'] = forecast_display['Fecha'].dt.date  # 去掉时间部分
        
        # 折线图
        fig = px.line(
            forecast_display,
            x='Fecha',
            y='Consumo_Predicho',
            title="Predicción del Consumo Energético (Próximos Días)",
            labels={'Consumo_Predicho': 'Consumo (kWh)'},
            color_discrete_sequence=['royalblue']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ✅ 勾选框：控制是否显示预测表格
        mostrar_tabla = st.checkbox("📋 Mostrar tabla de predicción detallada")
        
        if mostrar_tabla:
            st.subheader("📋 Datos de Predicción (Resumen)")
            st.dataframe(forecast_display.round(2))


        # --- API del clima (opcional) ---
        if api_key:
            st.markdown("---")
            st.subheader("🌦️ Pronóstico del clima (Meteosource)")
            df_clima_futuro = get_weather_forecast(api_key, lat, lon)
            if not df_clima_futuro.empty:
                st.dataframe(df_clima_futuro)
                fig_temp = px.line(df_clima_futuro, x='fecha', y='temp_avg_c', title='Temperatura Promedio Prevista (°C)')
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.warning("No se pudo obtener datos del clima. Verifica tu API Key.")
    else:
        st.error("❌ No se pudieron cargar los datos de consumo.")
else:
    st.info("ℹ️ Selecciona un archivo CSV de energía en la barra lateral izquierda para comenzar.")
