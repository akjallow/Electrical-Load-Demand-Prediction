# ====================================
# STREAMLIT SMART GRID FORECASTING APP
# ====================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import datetime
import pytz
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError

# ---------------------------
# Global Config
# ---------------------------
MODEL_PATH = "models/hybrid_cnn_gru.keras"
SCALER_PATH = "models/scalers.pkl"
DATA_PATH = "hourly data(2000-2023).csv"

CITY_TIMEZONES = {
    "Delhi": "Asia/Kolkata",
    "Ahmedabad": "Asia/Kolkata",
    "Mehsana": "Asia/Kolkata",
    "Mumbai": "Asia/Kolkata",
    "Bengaluru": "Asia/Kolkata"
}

# ---------------------------
# Load Model & Artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, custom_objects={"mse": MeanSquaredError()})
    with open(SCALER_PATH, "rb") as f:
        artifacts = pickle.load(f)
    return model, artifacts

model, artifacts = load_artifacts()
feature_scaler = artifacts["feature_scaler"]
demand_scaler = artifacts["demand_scaler"]
feature_cols = artifacts["feature_cols"]
SEQ_LEN = artifacts["seq_len"]
HORIZON = artifacts["horizon"]

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    col_names = [
        "timestamp", "day_of_week", "hour_of_day", "is_weekend",
        "temperature", "is_holiday", "solar_generation", "electricity_demand"
    ]
    df = pd.read_csv(DATA_PATH, names=col_names, header=0, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.ffill().dropna()
    return df

df = load_data()

# ---------------------------
# Weather API
# ---------------------------
API_KEY = "bd3506b4684ca28a123ba23f98615849"
def get_weather(city="Delhi"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        r = requests.get(url).json()
        return {
            "temperature": f"{r['main']['temp']} °C",
            "humidity": f"{r['main']['humidity']}%",
            "condition": r['weather'][0]['description'].title()
        }
    except Exception as e:
        return {"temperature": "N/A", "humidity": "N/A", "condition": f"Error: {e}"}

# ---------------------------
# Forecast Function
# ---------------------------
def forecast_load(city, horizon, start_hour, day_of_week, is_weekend, is_holiday):
    history = df[feature_cols].tail(SEQ_LEN).values
    weather = get_weather(city)

    latest = history[-1].copy()
    try:
        latest[1] = float(weather["temperature"].replace("°", ""))
    except:
        latest[1] = history[-1][1]

    latest[2] = day_of_week
    latest[3] = start_hour
    latest[4] = int(is_weekend)
    latest[5] = int(is_holiday)
    history[-1] = latest

    history_scaled = feature_scaler.transform(history).reshape(1, SEQ_LEN, len(feature_cols))
    y_pred_s = model.predict(history_scaled)
    y_pred = demand_scaler.inverse_transform(y_pred_s.reshape(-1,1)).reshape(-1)
    y_pred = y_pred[:horizon]

    # Localized time for selected city
    tz = pytz.timezone(CITY_TIMEZONES[city])
    start_time = datetime.datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    start_time = start_time.replace(hour=start_hour)

    future_hours = pd.date_range(start=start_time, periods=horizon, freq="H", tz=tz)
    df_pred = pd.DataFrame({"Time": future_hours, "Predicted Load (MW)": y_pred})

    # Last 24h actual demand for context
    past_hours = pd.date_range(end=start_time, periods=24, freq="H", tz=tz)
    df_hist = df.loc[df.index.tz_localize(tz).isin(past_hours), ["electricity_demand"]].reset_index()
    df_hist.rename(columns={"timestamp": "Time", "electricity_demand": "Actual Load (MW)"}, inplace=True)

    return df_pred, df_hist, weather

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Smart Grid Forecasting", layout="wide")
st.title("⚡ Smart Grid Load Forecasting Dashboard")
st.markdown("### Electricity Demand Prediction using Deep Learning")

# --- Sidebar Widgets ---
st.sidebar.header("⚙️ Forecast Settings")

city = st.sidebar.selectbox("🌍 Select City", list(CITY_TIMEZONES.keys()))
horizon = st.sidebar.slider("⏳ Forecast Horizon (hours)", 1, 24, 6)
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_of_week = st.sidebar.selectbox("📅 Day of Week", days)
hour_of_day = st.sidebar.selectbox("🕒 Start Hour of Day", list(range(24)))
is_weekend = st.sidebar.checkbox("Weekend?", value=False)
is_holiday = st.sidebar.checkbox("Holiday?", value=False)

# --- Weather (always visible) ---
st.subheader(f"🌤️ Current Weather in {city}")
weather = get_weather(city)
local_time = datetime.datetime.now(pytz.timezone(CITY_TIMEZONES[city]))
st.write(f"**Local Time:** {local_time.strftime('%Y-%m-%d %H:%M')}")
st.write(f"**Temperature:** {weather['temperature']}")
st.write(f"**Humidity:** {weather['humidity']}")
st.write(f"**Condition:** {weather['condition']}")

# --- Run Forecast ---
if st.sidebar.button("🔮 Run Forecast"):
    df_pred, df_hist, weather = forecast_load(city, horizon, hour_of_day, days.index(day_of_week), is_weekend, is_holiday)

    st.subheader(f"📊 Day-Ahead Forecast for {city}")

    # Show total & average demand
    total_demand = df_pred["Predicted Load (MW)"].sum()
    avg_demand = df_pred["Predicted Load (MW)"].mean()

    st.metric(label="Total Predicted Demand (MWh)", value=f"{total_demand:.2f}")
    st.metric(label="Average Hourly Demand (MW)", value=f"{avg_demand:.2f}")

    # Plot historical + forecast
    fig = go.Figure()

    # Historical load
    fig.add_trace(go.Scatter(
        x=df_hist["Time"], y=df_hist["Actual Load (MW)"],
        mode="lines+markers", name="Actual Demand (Last 24h)",
        line=dict(color="blue")
    ))

    # Forecast load
    # --- Plot forecast ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pred["Time"],
        y=df_pred["Predicted Load (MW)"],
        mode="lines",  # continuous line
        name="Forecast",
        line=dict(color="blue", width=3)  # solid blue line
    ))
    fig.add_trace(go.Scatter(
      x=df_pred["Time"],
      y=df_pred["Predicted Load (MW)"],
      mode="markers",
      name="Forecast Points",
      marker=dict(color="blue", size=8, symbol="circle")))

    fig.update_layout(
        title=f"Forecasted Electricity Demand ({city})",
        xaxis_title="Time",
        yaxis_title="Demand (MW)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show detailed tables
    st.write("### Historical Demand (Last 24h)")
    st.dataframe(df_hist)

    st.write("### Forecasted Demand (Next 24h)")
    st.dataframe(df_pred)
