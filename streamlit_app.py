# --------------------------------------------------------------------------
# |                        IMPORTAR LIBRERÍAS                             |
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# --------------------------------------------------------------------------
# |                     CONFIGURACIÓN DE LA PÁGINA                         |
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Consumo Energético con IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# |                         FUNCIONES DE CARGA DE DATOS                     |
# --------------------------------------------------------------------------

@st.cache_data
def load_asepeyo_energy_data(uploaded_file):
    """Carga y procesa el archivo de consumo energético de Asepeyo."""
    try:
        df = pd.read_csv(uploaded_file, sep=',', decimal='.')
        # Validar columnas esperadas
        if 'Fecha' not in df.columns or 'Energía activa (kWh)' not in df.columns:
            st.error("El archivo de consumo debe contener las columnas 'Fecha' y 'Energía activa (kWh)'.")
            return pd.DataFrame()
            
        df.rename(columns={'Fecha': 'fecha', 'Energía activa (kWh)': 'consumo_kwh'}, inplace=True)
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo de consumo: {e}")
        return pd.DataFrame()

@st.cache_data
def load_nasa_weather_data(uploaded_file):
    """Carga y procesa el archivo de clima histórico de NASA POWER."""
    try:
        # Encontrar la línea donde empiezan los datos reales
        lines = uploaded_file.getvalue().decode('utf-8').splitlines()
        start_row = 0
        for i, line in enumerate(lines):
            if line.strip() == "YEAR,MO,DY,HR,RH2M,T2M":
                start_row = i
                break
        
        uploaded_file.seek(0) # Resetear el puntero del archivo
        df = pd.read_csv(uploaded_file, skiprows=start_row)
        
        # Validar columnas
        expected_cols = ['YEAR', 'MO', 'DY', 'HR', 'T2M']
        if not all(col in df.columns for col in expected_cols):
            st.error("El archivo de clima de la NASA debe contener las columnas 'YEAR', 'MO', 'DY', 'HR', 'T2M'.")
            return pd.DataFrame()

        # Crear columna de fecha
        df['fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))
        df.rename(columns={'T2M': 'temperatura_c'}, inplace=True)
        
        # Reemplazar valores no válidos (-999) con el valor anterior
        df['temperatura_c'] = df['temperatura_c'].replace(-999, np.nan).ffill()
        
        return df[['fecha', 'temperatura_c']]
    except Exception as e:
        st.error(f"Error al procesar el archivo de clima de la NASA: {e}")
        return pd.DataFrame()

def crear_features_temporales(df):
    """Crea columnas de features basadas en la fecha."""
    df['hora'] = df['fecha'].dt.hour
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['es_finde'] = (df['dia_semana'] >= 5).astype(int)
    return df

# --------------------------------------------------------------------------
# |                      BARRA LATERAL (SIDEBAR)                          |
# --------------------------------------------------------------------------

# Asegurarse de que el logo no cause un error si no se encuentra
logo_path = "Logo_ASEPEYO.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)
else:
    st.sidebar.warning("Logo_ASEPEYO.png no encontrado.")

st.sidebar.title("Configuración de la Predicción")
st.sidebar.markdown("---")

st.sidebar.header("1. Carga de Datos")
st.sidebar.write("Sube los archivos CSV con los datos históricos y el pronóstico futuro.")

# --- Widgets para subir archivos ---
energy_file = st.sidebar.file_uploader("Archivo de Consumo (Asepeyo)", type="csv")
past_weather_file = st.sidebar.file_uploader("Archivo de Clima Histórico (NASA)", type="csv")
future_weather_file = st.sidebar.file_uploader("Pronóstico de Temperatura Futuro (CSV Diario)", type="csv", help="Debe tener columnas: 'fecha', 'temp_max_c', 'temp_min_c'")

st.sidebar.markdown("---")

# --- Variables adicionales ---
st.sidebar.header("2. Variables Adicionales")
ocupacion_media = st.sidebar.slider("Ocupación Media (%) del Centro", 0, 100, 80)

# --------------------------------------------------------------------------
# |                        CUERPO DE LA APLICACIÓN                          |
# --------------------------------------------------------------------------

st.title("🤖 Sistema de Inteligencia Energética con IA")
st.subheader("Herramienta de Predicción de Consumo para Instalaciones de Asepeyo")
st.markdown("---")

if energy_file and past_weather_file and future_weather_file:
    with st.spinner('Procesando datos y entrenando el modelo de IA...'):
        
        # --- Carga y preparación de datos ---
        df_energia = load_asepeyo_energy_data(energy_file)
        df_clima_pasado = load_nasa_weather_data(past_weather_file)
        
        try:
            df_clima_futuro = pd.read_csv(future_weather_file, parse_dates=['fecha'])
            if 'temp_max_c' not in df_clima_futuro.columns or 'temp_min_c' not in df_clima_futuro.columns:
                 st.error("El archivo de pronóstico futuro debe tener las columnas 'temp_max_c' y 'temp_min_c'.")
                 st.stop()
        except Exception as e:
            st.error(f"Error al leer el archivo de pronóstico futuro: {e}")
            st.stop()


        if not df_energia.empty and not df_clima_pasado.empty:
            # --- Unir datos históricos ---
            df_historico = pd.merge(df_energia, df_clima_pasado, on='fecha', how='inner')
            df_historico.dropna(inplace=True) # Eliminar filas donde no haya datos de clima o energía
            
            # --- Ingeniería de Features (Datos Históricos) ---
            df_historico = crear_features_temporales(df_historico)
            df_historico['ocupacion'] = ocupacion_media

            # --- Preparar datos futuros ---
            future_dates = pd.date_range(start=df_clima_futuro['fecha'].min(), end=df_clima_futuro['fecha'].max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='h')
            df_futuro = pd.DataFrame({'fecha': future_dates})
            df_futuro['fecha_dia'] = pd.to_datetime(df_futuro['fecha'].dt.date)
            df_clima_futuro['fecha_dia'] = pd.to_datetime(df_clima_futuro['fecha'].dt.date)
            df_futuro = pd.merge(df_futuro, df_clima_futuro[['fecha_dia', 'temp_min_c', 'temp_max_c']], on='fecha_dia', how='left')
            df_futuro.drop(columns=['fecha_dia'], inplace=True)
            
            df_futuro['temperatura_c'] = df_futuro['temp_min_c'] + (df_futuro['temp_max_c'] - df_futuro['temp_min_c']) * np.sin(np.pi * (df_futuro['fecha'].dt.hour - 6) / 12)
            df_futuro['temperatura_c'].fillna(method='ffill', inplace=True)
            
            df_futuro = crear_features_temporales(df_futuro)
            df_futuro['ocupacion'] = ocupacion_media
            
            # --- Entrenamiento del Modelo ---
            features = ['temperatura_c', 'hora', 'dia_semana', 'dia_mes', 'mes', 'es_finde', 'ocupacion']
            target = 'consumo_kwh'

            X = df_historico[features]
            y = df_historico[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            modelo.fit(X_train, y_train)

            # --- Realizar y mostrar predicciones ---
            X_futuro = df_futuro[features]
            df_futuro['consumo_predicho_kwh'] = modelo.predict(X_futuro)
            
            pred_test = modelo.predict(X_test)
            r2 = r2_score(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))

            st.success("✅ ¡Modelo entrenado y predicción completada!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Precisión del Modelo (R²)", f"{r2:.2f}", help="El R² indica qué tan bien las variables (temperatura, hora, etc.) explican la variación en el consumo. Un valor cercano a 1.0 es ideal.")
            col2.metric("Error Medio (RMSE)", f"{rmse:.2f} kWh", help="Indica la desviación promedio de las predicciones del modelo respecto a los valores reales. Un valor más bajo es mejor.")
            col3.metric("Consumo Total Predicho", f"{df_futuro['consumo_predicho_kwh'].sum():,.0f} kWh")
            st.markdown("---")

            st.subheader("Gráfico de la Predicción de Consumo Energético")
            
            df_historico_plot = df_historico[['fecha', 'consumo_kwh']].rename(columns={'consumo_kwh': 'Consumo'})
            df_historico_plot['Tipo'] = 'Histórico'

            df_futuro_plot = df_futuro[['fecha', 'consumo_predicho_kwh']].rename(columns={'consumo_predicho_kwh': 'Consumo'})
            df_futuro_plot['Tipo'] = 'Predicción'
            
            df_plot = pd.concat([df_historico_plot, df_futuro_plot])

            fig = px.line(df_plot, x='fecha', y='Consumo', color='Tipo', 
                          title='Consumo Histórico vs. Predicción Futura',
                          labels={'fecha': 'Fecha', 'Consumo': 'Consumo (kWh)'},
                          color_discrete_map={'Histórico': 'blue', 'Predicción': 'orange'})
            
            fig.add_vline(x=df_historico['fecha'].max(), line_width=2, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Datos Detallados de la Predicción")
            st.dataframe(df_futuro[['fecha', 'temperatura_c', 'consumo_predicho_kwh']].round(2))

else:
    st.info("ℹ️ **Para comenzar**, por favor sube los tres archivos de datos en la barra lateral izquierda.")
    st.markdown("""
    Esta herramienta utiliza los datos que proporcionas para:
    1.  **Aprender** los patrones de consumo de tu instalación usando los datos de Asepeyo y de la NASA.
    2.  **Entrenar** un modelo de Inteligencia Artificial (Machine Learning).
    3.  **Predecir** el consumo futuro basándose en el pronóstico meteorológico que subas.
    """)
