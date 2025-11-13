# frozen_lstm.py
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib
from utils import create_sequences

def fetch_data(symbol, start='2010-01-01', end='2023-01-01'):
    data = yf.download(symbol, start=start, end=end)
    return data['Close'].values.reshape(-1, 1)

def train_frozen_lstm(symbol='AAPL'):
    data = fetch_data(symbol)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32)

    model.save('frozen_lstm.h5')
    joblib.dump(scaler, 'scaler.save')
    print("[✓] Frozen LSTM trained and saved.")

if __name__ == "__main__":
    train_frozen_lstm()
