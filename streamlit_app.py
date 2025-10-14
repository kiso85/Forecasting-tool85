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
# |                         FUNCIONES PRINCIPALES                           |
# --------------------------------------------------------------------------

@st.cache_data
def generar_datos_ejemplo():
    """Crea DataFrames de ejemplo en memoria para que el usuario los descargue."""
    # Consumo Pasado
    dates_pasado = pd.to_datetime(pd.date_range(start="2023-09-01 00:00", end="2023-09-30 23:00", freq='h'))
    consumo_df = pd.DataFrame({'fecha': dates_pasado})
    consumo_df['consumo_kwh'] = np.random.normal(loc=150, scale=20, size=len(dates_pasado)) + \
                               15 * np.sin(2 * np.pi * consumo_df['fecha'].dt.hour / 24) + \
                               (consumo_df['fecha'].dt.dayofweek // 5) * 50 # Más consumo en días laborables
    
    # Clima Pasado
    clima_pasado_df = pd.DataFrame({'fecha': dates_pasado})
    clima_pasado_df['temperatura_c'] = np.random.normal(loc=25, scale=5, size=len(dates_pasado)) - \
                                      10 * np.cos(2 * np.pi * clima_pasado_df['fecha'].dt.hour / 24)

    # Clima Futuro (Diario)
    dates_futuro = pd.to_datetime(pd.date_range(start="2023-10-01", end="2023-10-07", freq='D'))
    clima_futuro_df = pd.DataFrame({'fecha': dates_futuro})
    clima_futuro_df['temp_max_c'] = np.random.uniform(low=22, high=28, size=len(dates_futuro))
    clima_futuro_df['temp_min_c'] = clima_futuro_df['temp_max_c'] - np.random.uniform(low=8, high=12, size=len(dates_futuro))

    return consumo_df, clima_pasado_df, clima_futuro_df

@st.cache_data
def to_csv(df):
    """Convierte un DataFrame a CSV en memoria para la descarga."""
    return df.to_csv(index=False).encode('utf-8')

def crear_features_temporales(df):
    """Crea columnas de features basadas en la fecha."""
    df['hora'] = df['fecha'].dt.hour
    df['dia_semana'] = df['fecha'].dt.dayofweek # Lunes=0, Domingo=6
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['es_finde'] = (df['dia_semana'] >= 5).astype(int) # 1 si es fin de semana, 0 si no
    return df

# --------------------------------------------------------------------------
# |                      BARRA LATERAL (SIDEBAR)                          |
# --------------------------------------------------------------------------

st.sidebar.image("Logo_ASEPEYO.png", width=200)
st.sidebar.title("Configuración de la Predicción")
st.sidebar.markdown("---")

st.sidebar.header("1. Carga de Datos")
st.sidebar.write("Sube tus archivos CSV o descarga los de ejemplo para ver cómo funciona.")

# --- Widgets para subir archivos ---
past_energy_file = st.sidebar.file_uploader("Consumo Energético Pasado (kWh)", type="csv")
past_temp_file = st.sidebar.file_uploader("Temperatura Histórica Horaria (°C)", type="csv")
future_temp_file = st.sidebar.file_uploader("Pronóstico de Temperatura Futuro (Diario)", type="csv")

st.sidebar.markdown("---")

# --- Descarga de archivos de ejemplo ---
st.sidebar.header("Archivos de Ejemplo")
ej_consumo, ej_clima_pasado, ej_clima_futuro = generar_datos_ejemplo()

st.sidebar.download_button(
    label="Descargar consumo_ejemplo.csv",
    data=to_csv(ej_consumo),
    file_name="consumo_ejemplo.csv",
    mime="text/csv",
)
st.sidebar.download_button(
    label="Descargar clima_pasado_ejemplo.csv",
    data=to_csv(ej_clima_pasado),
    file_name="clima_pasado_ejemplo.csv",
    mime="text/csv",
)
st.sidebar.download_button(
    label="Descargar clima_futuro_ejemplo.csv",
    data=to_csv(ej_clima_futuro),
    file_name="clima_futuro_ejemplo.csv",
    mime="text/csv",
)

st.sidebar.markdown("---")

# --- Variables adicionales ---
st.sidebar.header("2. Variables Adicionales")
st.sidebar.write("Añade variables que puedan influir en el consumo.")
ocupacion_media = st.sidebar.slider("Ocupación Media (%) del Centro", 0, 100, 80)
dias_festivos = st.sidebar.number_input("Nº de Días Festivos en el Período Futuro", min_value=0, max_value=7, value=1)


# --------------------------------------------------------------------------
# |                        CUERPO DE LA APLICACIÓN                          |
# --------------------------------------------------------------------------

st.title("🤖 Sistema de Inteligencia Energética con IA")
st.subheader("Herramienta de Predicción de Consumo para Instalaciones de Asepeyo")
st.markdown("---")

# --- Lógica principal ---
if past_energy_file and past_temp_file and future_temp_file:
    with st.spinner('Procesando datos y entrenando el modelo de IA...'):
        try:
            # --- Carga y preparación de datos ---
            df_energia = pd.read_csv(past_energy_file, parse_dates=['fecha'])
            df_clima_pasado = pd.read_csv(past_temp_file, parse_dates=['fecha'])
            df_clima_futuro = pd.read_csv(future_temp_file, parse_dates=['fecha'])

            # --- Unir datos históricos ---
            df_historico = pd.merge(df_energia, df_clima_pasado, on='fecha', how='inner')
            
            # --- Ingeniería de Features (Datos Históricos) ---
            df_historico = crear_features_temporales(df_historico)
            df_historico['ocupacion'] = ocupacion_media # Añadir variable externa

            # --- Preparar datos futuros ---
            # Crear un rango de fechas horario para el futuro
            future_dates = pd.date_range(start=df_clima_futuro['fecha'].min(), end=df_clima_futuro['fecha'].max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='h')
            df_futuro = pd.DataFrame({'fecha': future_dates})
            
            # Unir temperaturas diarias para poder interpolar
            df_futuro['fecha_dia'] = df_futuro['fecha'].dt.date
            df_clima_futuro['fecha_dia'] = df_clima_futuro['fecha'].dt.date
            df_futuro = pd.merge(df_futuro, df_clima_futuro[['fecha_dia', 'temp_min_c', 'temp_max_c']], on='fecha_dia', how='left')
            df_futuro.drop(columns=['fecha_dia'], inplace=True)
            
            # Interpolar temperatura para crear un perfil horario
            df_futuro['temperatura_c'] = df_futuro['temp_min_c'] + \
                (df_futuro['temp_max_c'] - df_futuro['temp_min_c']) * \
                np.sin(np.pi * (df_futuro['fecha'].dt.hour - 6) / 12) # Simulación de curva de temperatura
            df_futuro['temperatura_c'].fillna(method='ffill', inplace=True) # Rellenar posibles NaNs
            
            # --- Ingeniería de Features (Datos Futuros) ---
            df_futuro = crear_features_temporales(df_futuro)
            df_futuro['ocupacion'] = ocupacion_media # Añadir variable externa
            
            # --- Entrenamiento del Modelo de Machine Learning ---
            features = ['temperatura_c', 'hora', 'dia_semana', 'dia_mes', 'mes', 'es_finde', 'ocupacion']
            target = 'consumo_kwh'

            X = df_historico[features]
            y = df_historico[target]

            # Dividir en entrenamiento y prueba para evaluar el modelo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Usar RandomForest, un modelo versátil y potente
            modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            modelo.fit(X_train, y_train)

            # --- Realizar predicciones ---
            X_futuro = df_futuro[features]
            predicciones = modelo.predict(X_futuro)
            
            # Ajustar predicción por festivos (ejemplo simple: reducir consumo un 25%)
            dias_festivos_en_rango = df_futuro['fecha'].dt.normalize().isin(pd.to_datetime(['2023-10-06'])) # Ejemplo de festivo
            predicciones[dias_festivos_en_rango] *= (1 - (0.25 * (dias_festivos > 0)))


            df_futuro['consumo_predicho_kwh'] = predicciones

            # --- Evaluación del modelo (opcional pero recomendable) ---
            pred_test = modelo.predict(X_test)
            r2 = r2_score(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))

            # --- Mostrar resultados ---
            st.success("✅ ¡Modelo entrenado y predicción completada con éxito!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Precisión del Modelo (R²)", f"{r2:.2f}")
            col2.metric("Error Medio (RMSE)", f"{rmse:.2f} kWh")
            col3.metric("Consumo Total Predicho", f"{df_futuro['consumo_predicho_kwh'].sum():,.0f} kWh")
            st.markdown("---")

            st.subheader("Gráfico de la Predicción de Consumo Energético")

            # Combinar datos históricos y futuros para un gráfico completo
            df_historico_plot = df_historico[['fecha', 'consumo_kwh']].copy()
            df_historico_plot.rename(columns={'consumo_kwh': 'Consumo'}, inplace=True)
            df_historico_plot['Tipo'] = 'Histórico'

            df_futuro_plot = df_futuro[['fecha', 'consumo_predicho_kwh']].copy()
            df_futuro_plot.rename(columns={'consumo_predicho_kwh': 'Consumo'}, inplace=True)
            df_futuro_plot['Tipo'] = 'Predicción'
            
            df_plot = pd.concat([df_historico_plot, df_futuro_plot])


            fig = px.line(df_plot, x='fecha', y='Consumo', color='Tipo', 
                          title='Consumo Histórico vs. Predicción Futura',
                          labels={'fecha': 'Fecha', 'Consumo': 'Consumo (kWh)'},
                          color_discrete_map={'Histórico': 'blue', 'Predicción': 'orange'})
            
            # Añadir una línea vertical para separar el pasado del futuro
            fig.add_vline(x=df_historico['fecha'].max(), line_width=2, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)

            # Mostrar datos en una tabla
            st.subheader("Datos Detallados de la Predicción")
            st.dataframe(df_futuro[['fecha', 'temperatura_c', 'consumo_predicho_kwh']].round(2))

        except Exception as e:
            st.error(f"❌ Ha ocurrido un error al procesar los archivos: {e}")
            st.warning("Por favor, revisa que los archivos CSV tengan las columnas correctas ('fecha', 'consumo_kwh', 'temperatura_c', etc.) y el formato adecuado.")

else:
    st.info("ℹ️ **Para comenzar**, por favor sube los tres archivos de datos en la barra lateral izquierda.")
    st.markdown("""
    Esta herramienta utiliza los datos que proporcionas para:
    1.  **Aprender** los patrones de consumo de tu instalación usando datos históricos de energía y clima.
    2.  **Entrenar** un modelo de Inteligencia Artificial (Machine Learning).
    3.  **Predecir** el consumo futuro basándose en el pronóstico meteorológico.
    
    Esto permite superar las estimaciones generalizadas y avanzar hacia una gestión energética más inteligente y predictiva, tal como se describe en los objetivos del proyecto.
    """)
