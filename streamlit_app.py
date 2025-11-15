# --------------------------------------------------------------------------
# Streamlit app: Hourly Prophet with M-shaped intraday, Day-of-week heatmap
# - Trains on 2023-07-01 -> 2024-07-01 (hourly)
# - Removes / aggregates duplicate timestamps
# - Adds hourly regressors (hour_sin, hour_cos, is_weekend)
# - Keeps original UI and charts style, adds Day-of-Week x Hour heatmap and average-day profile
# --------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import glob

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Consumo Energético con Prophet (Hourly)",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# Data loader
# --------------------------------------------------------------------------
@st.cache_data
def load_asepeyo_energy_data(file_path):
    """Carga y procesa el archivo de consumo energético desde una ruta.
    Espera columnas: 'Fecha' (día/mes/año o iso), 'Energía activa (kWh)'.
    """
    try:
        df = pd.read_csv(file_path, sep=',', decimal='.')
        if 'Fecha' not in df.columns or 'Energía activa (kWh)' not in df.columns:
            st.error(f"El archivo {file_path} debe contener 'Fecha' y 'Energía activa (kWh)'.")
            return pd.DataFrame()

        df.rename(columns={'Fecha': 'fecha', 'Energía activa (kWh)': 'consumo_kwh'}, inplace=True)

        # Parse date (try dayfirst, then fallback)
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        if df['fecha'].isna().sum() > 0:
            # try ISO style
            df['fecha'] = pd.to_datetime(df['fecha'].astype(str), errors='coerce')

        df = df.dropna(subset=['fecha'])
        df = df.sort_values('fecha')
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
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración de la Predicción")
st.sidebar.markdown("---")

# Data directory
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
st.sidebar.info(f"📁 Carpeta de datos: {DATA_DIR}")

# File selector
energy_pattern = os.path.join(DATA_DIR, "energy_*.csv")
energy_files = [os.path.basename(f) for f in glob.glob(energy_pattern)]
selected_energy_file = st.sidebar.selectbox("Selecciona archivo de consumo", energy_files) if energy_files else None

st.sidebar.markdown("---")

# Weather API params (optional)
st.sidebar.header("🌤️ API Meteosource (opcional)")
api_key = st.sidebar.text_input("API Key de Meteosource", type="password")
lat = st.sidebar.text_input("Latitud", "40.4168")
lon = st.sidebar.text_input("Longitud", "-3.7038")

# Model params
st.sidebar.markdown("---")
st.sidebar.header("🧠 Parámetros del Modelo Prophet")
future_days = st.sidebar.slider("Días a predecir", 7, 90, 30)
future_hours = future_days * 24
include_holidays = st.sidebar.checkbox("Incluir festivos de España (ES)", value=True)

# Accuracy test options
st.sidebar.markdown("---")
check_accuracy = st.sidebar.checkbox("🧪 Check Model Accuracy (Hold-out Test)")
if check_accuracy:
    horizon_days = st.sidebar.number_input("Days to hold out for testing", 30, 180, 90)
    horizon_hours = int(horizon_days) * 24

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("🎀 Predicción de Consumo Energético (Hourly) 🎀")
st.subheader("Modelo de Series Temporales con Prophet — resolución horaria")
st.markdown("---")

if selected_energy_file:
    energy_path = os.path.join(DATA_DIR, selected_energy_file)
    df_energia = load_asepeyo_energy_data(energy_path)

    if not df_energia.empty:

        # ----------------------- Prepare hourly series -----------------------
        # set index to fecha (datetime)
        df_energia = df_energia.set_index('fecha')
        df_energia = df_energia.sort_index()

        # If duplicated timestamps exist, aggregate them (sum). This avoids resample errors.
        if df_energia.index.duplicated().any():
            st.warning(f"Se detectaron {df_energia.index.duplicated().sum()} timestamps duplicados. Se procederá a agrupar por timestamp (sum).")
            df_energia = df_energia.groupby(df_energia.index).agg({'consumo_kwh': 'sum'})

        # Resample hourly and interpolate missing hours
        df_hourly = df_energia['consumo_kwh'].resample('1H').mean().interpolate()

        # Build df_prophet: ds, y
        df_prophet = df_hourly.reset_index().rename(columns={'fecha': 'ds', 'consumo_kwh': 'y'})

        # ----------------------- Filter training window -----------------------
        train_start = pd.to_datetime('2023-07-01')
        train_end = pd.to_datetime('2024-07-01')  # exclusive
        df_prophet_train = df_prophet[(df_prophet['ds'] >= train_start) & (df_prophet['ds'] < train_end)].copy()

        if df_prophet_train.empty:
            st.error("No hay datos en el intervalo de entrenamiento 2023-07-01 → 2024-07-01. Verifica el CSV y las fechas.")
        else:
            st.write(f"Usando datos de entrenamiento: {df_prophet_train['ds'].min()} → {df_prophet_train['ds'].max()}")

            # ----------------------- Add regressors & features -----------------------
            df_prophet_train['hour'] = df_prophet_train['ds'].dt.hour
            df_prophet_train['is_weekend'] = (df_prophet_train['ds'].dt.dayofweek >= 5).astype(int)
            df_prophet_train['hour_sin'] = np.sin(2 * np.pi * df_prophet_train['hour'] / 24.0)
            df_prophet_train['hour_cos'] = np.cos(2 * np.pi * df_prophet_train['hour'] / 24.0)

            # ----------------------- Build Prophet -----------------------
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False    # 关闭默认 daily
            )
            
            # 添加自定义 M 型 daily seasonality
            m.add_seasonality(
                name='daily_m_shape',
                period=24,
                fourier_order=4
            )

            # add regressors - 修正：使用 m 而不是 model
            m.add_regressor('hour_sin')
            m.add_regressor('hour_cos')
            m.add_regressor('is_weekend')

            if include_holidays:
                try:
                    m.add_country_holidays(country_name='ES')  # 这里也要用 m
                    st.sidebar.success("✅ Festivos de España añadidos.")
                except Exception as e:
                    st.sidebar.warning(f"No se pudieron añadir festivos: {e}")

            with st.spinner("Entrenando Prophet (hourly) con regresores..."):
                m.fit(df_prophet_train)  # 这里也要用 m

            # ----------------------- Future dataframe and regressors -----------------------
            future = m.make_future_dataframe(periods=future_hours, freq='H')  # 这里也要用 m
            # ensure regressors on future
            future['hour'] = future['ds'].dt.hour
            future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
            future['hour_sin'] = np.sin(2 * np.pi * future['hour'] / 24.0)
            future['hour_cos'] = np.cos(2 * np.pi * future['hour'] / 24.0)

            forecast = m.predict(future)  # 这里也要用 m

            # ----------------------- Accuracy check (hourly hold-out if requested) -----------------------
            if check_accuracy:
                st.subheader("📊 Prophet Model Accuracy Test (Hourly)")
                # create hourly holdout from the end of training window
                # use last `horizon_hours` hours from the training window as test if available
                horizon_hours = min(int(horizon_days) * 24, len(df_prophet_train))
                cutoff = df_prophet_train['ds'].max() - pd.Timedelta(hours=horizon_hours)
                train_acc = df_prophet_train[df_prophet_train['ds'] <= cutoff]
                test_acc = df_prophet_train[df_prophet_train['ds'] > cutoff]

                st.write(f"Train for accuracy: {train_acc['ds'].min()} → {train_acc['ds'].max()}")
                st.write(f"Test for accuracy:  {test_acc['ds'].min()} → {test_acc['ds'].max()}")

                acc_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.1
                )
                acc_model.add_regressor('hour_sin')
                acc_model.add_regressor('hour_cos')
                acc_model.add_regressor('is_weekend')
                if include_holidays:
                    try:
                        acc_model.add_country_holidays(country_name='ES')
                    except Exception:
                        pass

                # fit
                acc_model.fit(train_acc)
                fut_test = acc_model.make_future_dataframe(periods=horizon_hours, freq='H')
                fut_test['hour'] = fut_test['ds'].dt.hour
                fut_test['is_weekend'] = (fut_test['ds'].dt.dayofweek >= 5).astype(int)
                fut_test['hour_sin'] = np.sin(2 * np.pi * fut_test['hour'] / 24.0)
                fut_test['hour_cos'] = np.cos(2 * np.pi * fut_test['hour'] / 24.0)

                fc_test = acc_model.predict(fut_test)

                # merge on ds where ds in test_acc
                merged = pd.merge(test_acc[['ds', 'y']], fc_test[['ds', 'yhat']], on='ds', how='inner')
                merged['squared_error'] = (merged['yhat'] - merged['y']) ** 2
                MSE = np.mean(merged['squared_error'])
                RMSE = np.sqrt(MSE)

                st.success(f"✅ Mean Squared Error (MSE): **{MSE:.2f}**  Root MSE (RMSE): **{RMSE:.2f}**")

                # plot
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], mode='lines', name='Actual', line=dict(color='black')))
                fig_acc.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat'], mode='lines', name='Predicted', line=dict(color='royalblue')))
                fig_acc.update_layout(title='Actual vs Predicted (hourly) — Validation Period', xaxis_title='Date', yaxis_title='kWh')
                st.plotly_chart(fig_acc, use_container_width=True)

            # ----------------------- Show results: Prophet plots -----------------------
            st.subheader("📈 Predicción de Consumo Energético (HORARIA)")
            
            # 使用Plotly创建交互式图表
            fig = go.Figure()
            
            # 添加实际值（黑色）
            fig.add_trace(go.Scatter(
                x=df_prophet_train['ds'],
                y=df_prophet_train['y'],
                mode='lines',
                name='Consumo Real',
                line=dict(color='black', width=1),
                opacity=0.7
            ))
            
            # 添加预测值（蓝色）
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Predicción',
                line=dict(color='blue', width=1.5)
            ))
            
            # 添加置信区间
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalo de Confianza',
                showlegend=True
            ))
            
            # 更新布局
            fig.update_layout(
                title='Predicción de Consumo Energético (HORARIA)',
                xaxis_title='Fecha',
                yaxis_title='Consumo (kWh)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

                        # ----------------------- Interactive forecast plot (hourly) -----------------------
            st.subheader("📊 Gráfico Interactivo del Pronóstico (Hourly) - Barras")
            
            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_hours).copy()
            forecast_display = forecast_display.rename(columns={
                'ds': 'Fecha',
                'yhat': 'Consumo_Predicho',
                'yhat_lower': 'Intervalo_Inferior',
                'yhat_upper': 'Intervalo_Superior'
            })
            
            # ------------------ 修复 1:00 凸起问题 + 强制非负 ------------------
            # 裁剪负值
            forecast_display['Consumo_Predicho'] = forecast_display['Consumo_Predicho'].clip(lower=0)
            
            # 删除午夜噪声前后的尖点（Prophet 常在 00:00–02:00 有异常）
            forecast_display['Consumo_Predicho_Suavizado'] = (
                forecast_display['Consumo_Predicho']
                .rolling(window=5, center=True, min_periods=1)
                .median()
            )
            
            # 添加日期时间信息用于更好的显示
            forecast_display['Dia'] = forecast_display['Fecha'].dt.date
            forecast_display['Hora'] = forecast_display['Fecha'].dt.hour
            forecast_display['Dia_Semana'] = forecast_display['Fecha'].dt.day_name()
            
            # 创建柱状图
            fig = px.bar(
                forecast_display,
                x='Fecha',
                y='Consumo_Predicho_Suavizado',
                title="Predicción Horaria del Consumo - Gráfico de Barras",
                labels={'Consumo_Predicho_Suavizado': 'Consumo (kWh)', 'Fecha': 'Fecha y Hora'},
                color='Consumo_Predicho_Suavizado',
                color_continuous_scale='blues'
            )
            
            # 自定义悬停信息
            fig.update_traces(
                hovertemplate=(
                    "<b>Fecha:</b> %{x}<br>" +
                    "<b>Hora:</b> %{customdata[0]}:00<br>" +
                    "<b>Consumo:</b> %{y:.2f} kWh<br>" +
                    "<b>Día:</b> %{customdata[1]}<extra></extra>"
                ),
                customdata=forecast_display[['Hora', 'Dia_Semana']]
            )
            
            # 更新布局
            fig.update_layout(
                xaxis_title="Fecha y Hora",
                yaxis_title="Consumo Predicho (kWh)",
                coloraxis_colorbar=dict(title="kWh"),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 可选：添加每日汇总视图
            st.subheader("📅 Vista Diaria Resumida")
            
            # 按日期分组计算每日总量
            daily_summary = forecast_display.groupby('Dia').agg({
                'Consumo_Predicho_Suavizado': 'sum',
                'Dia_Semana': 'first'
            }).reset_index()
            
            fig_daily = px.bar(
                daily_summary,
                x='Dia',
                y='Consumo_Predicho_Suavizado',
                title="Consumo Diario Predicho (Suma Horaria)",
                labels={'Consumo_Predicho_Suavizado': 'Consumo Total (kWh)', 'Dia': 'Fecha'},
                color='Consumo_Predicho_Suavizado',
                color_continuous_scale='viridis'
            )
            
            fig_daily.update_traces(
                hovertemplate=(
                    "<b>Fecha:</b> %{x}<br>" +
                    "<b>Día:</b> %{customdata}<br>" +
                    "<b>Consumo Total:</b> %{y:.2f} kWh<extra></extra>"
                ),
                customdata=daily_summary['Dia_Semana']
            )
            
            fig_daily.update_layout(showlegend=False)
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # ------------------ Mostrar tabla ------------------
            mostrar_tabla = st.checkbox("📋 Mostrar tabla de predicción detallada (hourly)")
            if mostrar_tabla:
                st.dataframe(forecast_display.round(2))
            
            # ------------------ Download forecast ------------------
            csv_fc = forecast_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar pronóstico hourly (CSV)",
                data=csv_fc,
                file_name='forecast_hourly.csv',
                mime='text/csv'
            )

            # ----------------------- Day of Week x Hour heatmap -----------------------
            st.subheader("📅 Day of Week × Hour — Mean Consumption (kWh)")
            hourly_df = df_hourly.reset_index().rename(columns={'fecha': 'ds', 'consumo_kwh': 'consumo_kwh'})
            hourly_df['hour'] = hourly_df['ds'].dt.hour
            hourly_df['dow'] = hourly_df['ds'].dt.dayofweek
            hourly_df['dow_name'] = hourly_df['ds'].dt.day_name()

            pivot = hourly_df.groupby(['dow', 'hour'])['consumo_kwh'].mean().reset_index()
            pivot['dow_name_short'] = pivot['dow'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
            matrix = pivot.pivot(index='dow_name_short', columns='hour', values='consumo_kwh').reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

            # If any NaNs, fill with 0 for display (or use interpolation)
            matrix_display = matrix.fillna(0)
            fig_dow = px.imshow(
                matrix_display.values,
                labels=dict(x="Hour", y="Day of Week", color="Mean kWh"),
                x=list(matrix_display.columns),
                y=list(matrix_display.index),
                aspect="auto",
                origin='lower'
            )
            fig_dow.update_layout(xaxis_nticks=24)
            st.plotly_chart(fig_dow, use_container_width=True)

            # ----------------------- Average day profile: historical vs forecast -----------------------
            st.subheader("📈 Average day profile (hourly) — historical vs forecast (mean per hour)")
            hist = df_prophet_train.copy()
            hist['hour'] = hist['ds'].dt.hour
            hist_mean = hist.groupby('hour')['y'].mean()

            fc = forecast.copy()
            fc['hour'] = fc['ds'].dt.hour
            fc_mean = fc.groupby('hour')['yhat'].mean()

            fig_mshape = go.Figure()
            fig_mshape.add_trace(go.Scatter(x=hist_mean.index, y=hist_mean.values, mode='lines+markers', name='Historical mean'))
            fig_mshape.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode='lines+markers', name='Forecast mean'))
            fig_mshape.update_layout(title="Average day profile (hourly)", xaxis_title="Hour (0-23)", yaxis_title="kWh")
            st.plotly_chart(fig_mshape, use_container_width=True)

            # ----------------------- Optional weather API -----------------------
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
