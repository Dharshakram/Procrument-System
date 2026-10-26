import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import requests
from plyer import notification
import os
from dotenv import load_dotenv

st.set_page_config(page_title="PricePulse: Real-Time AI Procurement System", layout="wide")

@st.cache_data(ttl=30)
def fetch_csv_data(supplier_name: str):
    try:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
        base_price = np.random.uniform(80, 120)
        prices = base_price + np.random.randn(30).cumsum()
        df = pd.DataFrame({
            "timestamp": dates,
            "close": prices,
        })
        df["open"] = df["close"] + np.random.uniform(-2, 2, len(df))
        df["high"] = df["close"] + np.random.uniform(0, 3, len(df))
        df["low"] = df["close"] - np.random.uniform(0, 3, len(df))
        df["volume"] = np.random.randint(1000, 5000, len(df))
        return df.tail(30)
    except Exception as e:
        st.warning(f"Error generating data for {supplier_name}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=30)
def fetch_simulated_data(ticker: str, period_days: int = 30):
    np.random.seed(int(datetime.now().timestamp() // 60) + hash(ticker) % 1000)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=period_days, freq="D")
    prices = np.cumsum(np.random.randn(period_days)) + 100
    open_ = prices + np.random.uniform(-2, 2, period_days)
    high = prices + np.random.uniform(0, 3, period_days)
    low = prices - np.random.uniform(0, 3, period_days)
    volume = np.random.randint(1000, 5000, period_days)

    return pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume
    })


@st.cache_data(ttl=30)
def fetch_latest_price(ticker: str, name: str):
    df = fetch_csv_data(name) if name in ["Supplier 1", "Supplier 2", "Supplier 3"] else fetch_simulated_data(ticker)
    if df.empty or len(df) < 2:
        return {"price": None, "pct_change": 0, "df": df}
    last = df["close"].iloc[-1]
    prev = df["close"].iloc[-2]
    pct = ((last - prev) / prev) * 100 if prev else 0
    return {"price": round(float(last), 2), "pct_change": round(float(pct), 2), "df": df}


SUPPLIER_TICKERS = {f"Supplier {i}": f"SP{i}" for i in range(1, 13)}

st.sidebar.header("Controls")
selected_suppliers = st.sidebar.multiselect(
    "Select suppliers",
    options=list(SUPPLIER_TICKERS.keys()),
    default=["Supplier 1", "Supplier 2"]
)

custom_ticker = st.sidebar.text_input("Add custom ticker (optional)", value="")
custom_nickname = st.sidebar.text_input("Nickname for custom supplier", value="")

st.sidebar.subheader("Alerts")
alert_enable = st.sidebar.checkbox("Enable alerts", value=True)
alert_threshold = st.sidebar.number_input("Alert if price ≤", value=0.0, step=0.01)

st.title("PricePulse: Real-Time AI Procurement System")
st.write("The system facilitates real-time price monitoring, trend visualization, and automated alerts, providing a unified platform for proactive decision-making and market analysis.")

left_col, right_col = st.columns([2, 1])

tickers = []
supplier_to_name = {}
for name in selected_suppliers:
    ticker = SUPPLIER_TICKERS.get(name)
    if ticker:
        tickers.append(ticker)
        supplier_to_name[ticker] = name

if custom_ticker.strip():
    tickers.append(custom_ticker.strip().upper())
    supplier_to_name[custom_ticker.strip().upper()] = custom_nickname or custom_ticker.strip().upper()

with right_col:
    st.subheader("Current Snapshot")
    snapshot_rows = []
    prediction_rows = []

    for t in tickers:
        name = supplier_to_name.get(t, t)
        data = fetch_latest_price(t, name)
        df = data["df"]

        if data["price"] is not None:
            snapshot_rows.append({
                "ticker": t,
                "name": name,
                "price": data["price"],
                "% change": data["pct_change"]
            })

            if not df.empty and len(df) >= 5:
                try:
                    # Use ARIMA model for forecasting
                    model = ARIMA(df["close"], order=(2, 1, 2))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=1)
                    predicted = float(forecast.iloc[0])
                    direction = "Uptrend" if predicted > df["close"].iloc[-1] else "Downtrend"
                except Exception as e:
                    predicted = df["close"].iloc[-1]
                    direction = f"Model Error ({e})"

                prediction_rows.append({
                    "Supplier": name,
                    "Predicted Price": round(predicted, 2),
                    "Trend": direction
                })

    snapshot_df = pd.DataFrame(snapshot_rows)
    if not snapshot_df.empty:
        st.dataframe(snapshot_df.set_index("name"))

        if alert_enable:
            st.markdown("Alerts")
            alerted = False
            for _, row in snapshot_df.iterrows():
                price = row["price"]
                if alert_threshold and price <= alert_threshold:
                    alert_msg = f"{row['name']}: price {price} ≤ {alert_threshold}"
                    st.warning(f"⚠ {alert_msg}")
                    alerted = True
                    try:
                        notification.notify(
                            title="Alert: Price Below Threshold",
                            message=alert_msg,
                            timeout=5
                        )
                    except Exception as e:
                        st.info(f"(Notification not sent: {e})")
            if not alerted:
                st.success("Threshold is below - No Alert")

    if prediction_rows:
        st.markdown("---")
        st.subheader("Price Predictions (ARIMA Forecast)")
        pred_df = pd.DataFrame(prediction_rows)
        st.dataframe(pred_df.set_index("Supplier"))

with left_col:
    st.subheader("Trends and Forecast")
    combined_df = pd.DataFrame()
    for t in tickers:
        name = supplier_to_name.get(t, t)
        hist = fetch_csv_data(name) if name in ["Supplier 1", "Supplier 2", "Supplier 3"] else fetch_simulated_data(t)
        if not hist.empty:
            hist = hist.assign(ticker=t)
            hist["name"] = name
            hist["MA"] = hist["close"].rolling(window=14, min_periods=1).mean()
            combined_df = pd.concat([combined_df, hist])

    if not combined_df.empty:
        chosen = st.multiselect(
            "Pick suppliers to visualize",
            combined_df["name"].unique().tolist(),
            default=combined_df["name"].unique()[:2]
        )
        for name in chosen:
            plot_df = combined_df[combined_df["name"] == name]
            fig = px.line(plot_df, x="timestamp", y="close", title=f"Price Trend — {name}", markers=True)
            fig.add_scatter(
                x=plot_df["timestamp"],
                y=plot_df["MA"],
                mode="lines",
                name="14-Day MA",
                line=dict(dash="dot")
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("Procurement Signals — Price Change Overview")
st.write("The dashboard provides a comprehensive overview of recent price variations (%) across all selected suppliers, consolidated into a unified trend graph for clear comparison and analysis.")

if not combined_df.empty:
    latest_prices = combined_df.sort_values("timestamp").groupby("name").tail(2)
    daily_change = []
    for name, group in latest_prices.groupby("name"):
        if len(group) < 2:
            continue
        prev, curr = group.iloc[0]["close"], group.iloc[1]["close"]
        pct = ((curr - prev) / prev) * 100 if prev else 0
        daily_change.append({"Supplier": name, "Change in Price (%)": round(pct, 2)})

    if daily_change:
        trend_df = pd.DataFrame(daily_change)
        fig = px.bar(
            trend_df,
            x="Supplier",
            y="Change in Price (%)",
            color="Supplier",
            text="Change in Price (%)",
            title="Combined Procurement Signal — % Price Change",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(yaxis_title="Change in Price (%)", xaxis_title="Supplier", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to calculate daily change yet.")
else:
    st.warning("Please select suppliers and fetch data to view procurement signals.")

st.markdown("---")
st.header("Global Material Price Fluctuation")
st.write("The dashboard enables visualization of live INR/kg price trends across selected metals, consolidated into a unified line chart for seamless comparison and insightful analysis.")

load_dotenv()

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
CURRENCY = os.getenv("CURRENCY", "INR")
UNIT = os.getenv("UNIT", "kg")

@st.cache_data(ttl=3600)
def fetch_global_material_data():
    try:
        response = requests.get(
            API_URL,
            params={"api_key": API_KEY, "currency": CURRENCY, "unit": UNIT}
        )
        data = response.json()
        metals = data.get("metals", {})
        df = pd.DataFrame(list(metals.items()), columns=["Metal", "Price (INR/kg)"])
        df["Price (INR/kg)"] = df["Price (INR/kg)"].astype(float)
        df["timestamp"] = datetime.now()
        all_data = []
        for metal, price in metals.items():
            base = float(price)
            dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
            simulated = base + np.random.randn(30).cumsum()
            temp_df = pd.DataFrame({"timestamp": dates, "Price (INR/kg)": simulated, "Metal": metal})
            all_data.append(temp_df)
        return pd.concat(all_data)
    except Exception as e:
        st.error(f"Error fetching live metal prices: {e}")
        return pd.DataFrame()

df_global = fetch_global_material_data()

if not df_global.empty:
    selected_metals = st.multiselect(
        "Select metals to visualize:",
        options=df_global["Metal"].unique().tolist(),
        default=df_global["Metal"].unique()[:3]
    )

    if selected_metals:
        for metal in selected_metals:
            metal_df = df_global[df_global["Metal"] == metal]
            metal_df["MA"] = metal_df["Price (INR/kg)"].rolling(window=7, min_periods=1).mean()
            fig = px.line(metal_df, x="timestamp", y="Price (INR/kg)", title=f"Price Trend — {metal}", markers=True)
            fig.add_scatter(
                x=metal_df["timestamp"],
                y=metal_df["MA"],
                mode="lines",
                name="7-Day MA",
                line=dict(dash="dot")
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one metal to view the trend.")
else:
    st.warning("Unable to load live global material prices. Check API key or internet connection.")

st.markdown("---")
time.sleep(30)
st.rerun()