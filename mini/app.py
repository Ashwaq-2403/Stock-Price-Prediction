# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from online_lstm import build_online_lstm, update_online_lstm
from utils import create_sequences
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError

# 🎨 Streamlit Page Setup
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>📉 Stock Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: gray;'>Stock Price Forecasting using LSTM Deep Learning | Turning Data into Decisions</h5>", unsafe_allow_html=True)

# 🔧 Sidebar - Input Parameters
with st.sidebar:
    st.header("Input Parameters")
    symbol = st.text_input("Stock Ticker:", value="AAPL")
    
    st.subheader("Data Parameters")
    history_years = st.slider("Historical Data (Years):", 1, 10, 5)
    sequence_length = st.slider("Sequence Length:", 10, 100, 60)
    pred_days = st.slider("Prediction Days:", 1, 30, 13)

    st.subheader("Model Parameters")
    epochs = st.slider("Training Epochs:", 10, 100, 10)
    batch_size = st.slider("Batch Size:", 16, 128, 32)

    run = st.button("🧠 Generate Prediction")

# 📥 Data Fetch & Preprocessing
def load_data(symbol, years):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    df = yf.download(symbol, start=start, end=end)
    return df['Close'].values.reshape(-1, 1), df.index

if run:
    st.markdown("### 📊 Prediction Results")
    col1, col2 = st.columns([2, 3])

    with st.spinner("Processing and predicting..."):
        # Load frozen model and scaler
        frozen_model = load_model("frozen_lstm.h5", custom_objects={"mse": MeanSquaredError()})
        online_model = build_online_lstm(input_shape=(sequence_length, 1))
        scaler = joblib.load("scaler.save")

        raw_data, dates = load_data(symbol, history_years)
        if len(raw_data) <= sequence_length:
            st.error("Not enough data. Increase 'Historical Data (Years)'.")
        else:
            scaled_data = scaler.transform(raw_data)
            X_test, _ = create_sequences(scaled_data, seq_len=sequence_length)

            # Predictions
            frozen_pred = frozen_model.predict(X_test)
            online_model = update_online_lstm(online_model, raw_data, scaler)
            online_pred = online_model.predict(X_test)

            frozen_pred = scaler.inverse_transform(frozen_pred)
            online_pred = scaler.inverse_transform(online_pred)
            actual = raw_data[-len(frozen_pred):]

            # Forecast next `pred_days`
            forecast_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            future_preds = []
            for _ in range(pred_days):
                next_pred = online_model.predict(forecast_input)[0][0]
                future_preds.append(next_pred)
                forecast_input = np.append(forecast_input[:, 1:, :], [[[next_pred]]], axis=1)
            future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

            # 📅 Forecast dates
            future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=pred_days)

            # 🧾 Table Output
            with col1:
                pred_df = pd.DataFrame({
                    "Date": future_dates.strftime('%Y-%m-%d'),
                    "Predicted Price": future_preds.flatten()
                })
                st.dataframe(pred_df)

            # 📈 Visualization
            with col2:
                all_dates = list(dates[-len(frozen_pred):]) + list(future_dates)
                all_actual = list(raw_data[-len(frozen_pred):].flatten()) + [np.nan] * pred_days
                all_pred = list(online_pred.flatten()) + list(future_preds.flatten())

                plt.figure(figsize=(10, 4))
                plt.plot(dates, raw_data, label='Historical Prices', linewidth=1.5)
                plt.plot(all_dates, all_pred, 'r--', label='Predicted Prices', linewidth=1.5)
                plt.legend()
                plt.title(f"{symbol} Stock Price Prediction")
                plt.xlabel("Date")
                plt.ylabel("Price ($)")
                st.pyplot(plt)

            # 🔢 Key Metrics
            st.markdown("### 📌 Key Metrics")
            last_price = raw_data[-1][0]
            predicted_price = future_preds[-1][0]
            change = ((predicted_price - last_price) / last_price) * 100

            col3, col4, col5 = st.columns(3)
            col3.metric("Current Price", f"${last_price:.2f}")
            col4.metric("Predicted Price", f"${predicted_price:.2f}")
            col5.metric("Predicted Change", f"{change:.2f}%", delta_color="inverse")
