# online_lstm.py
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from utils import create_sequences

def build_online_lstm(input_shape=(60, 1)):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def update_online_lstm(model, data, scaler):
    scaled_data = scaler.transform(data)
    X, y = create_sequences(scaled_data)
    if len(X) == 0:
        return model
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)
    return model
