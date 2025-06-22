import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ticker = "TSLA"
data = yf.download(
    tickers=ticker, start="2014-01-01", end="2025-06-01", auto_adjust=False
)

from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

closing_prices = data[["Close"]]

train_data = closing_prices.loc["2014-01-01":"2023-05-31"]
test_data = closing_prices.loc["2023-06-01":"2025-05-31"]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

print("********* DATA HEAD ********")
print(data.head())
print("********* DATA TAIL ********")
print(data.tail())
print("********* TRAIN SHAPES ********")
print(f"TRAIN : {train_data.shape}, TEST : {test_data.shape}")


def create_dataset(data, step=1):
    X, Y = [], []

    for i in range(len(data) - step - 1):
        a = data[i : (i + step), 0]
        X.append(a)
        Y.append(data[i + step, 0])

    return np.array(X), np.array(Y)


step: int = 60

X_train, Y_train = create_dataset(scaled_train_data, step=step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # (2308, 60, 1)

X_test, Y_test = create_dataset(scaled_test_data, step=step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # (440, 60, 1)

model = Sequential()
model.add(Input(shape=(step, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

model.fit(x=X_train, y=Y_train, batch_size=32, epochs=50, verbose=1)

train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

plt.figure(figsize=(15, 8))
plt.plot(
    closing_prices.index,
    closing_prices["Close"],
    label="Original Actual Price (Full Data)",
    color="gray",
    alpha=0.7,
)
plt.plot(
    train_data.index[step + 1 : len(X_train) + step + 1],
    train_predict,
    label="Train Predicted Price",
    color="blue",
)
plt.plot(
    test_data.index[step + 1 : len(X_test) + step + 1],
    test_predict,
    label="Test Predicted Price",
    color="red",
)

last_60_days = closing_prices[-step:].values
last_60_days_scaled = scaler.transform(last_60_days)

X_last = []
X_last.append(last_60_days_scaled)

X_last = np.array(X_last)
X_last = np.reshape(X_last, (X_last.shape[0], X_last.shape[1], 1))

pred_price = []
for _ in range(60):
    pred = model.predict(X_last)
    pred_price.append(pred[0])
    X_last = np.append(X_last[:, 1:, :], [[pred[0]]], axis=1)

pred_price = scaler.inverse_transform(np.array(pred_price).reshape(-1, 1))

prediction_dates = pd.date_range(
    start=closing_prices.index[-1] + pd.Timedelta(days=1), periods=60
)

rmse = np.sqrt(
    mean_squared_error(Y_test, model.predict(X_test))
)
mae = mean_absolute_error(
    Y_test, model.predict(X_test)
)

"""
Test RMSE: 0.0297
Test MAE: 0.0207
"""
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

plt.plot(
    prediction_dates,
    pred_price,
    label="Future 60 Days Predicted Price",
    color="green",
    linestyle="--",
)
plt.title("Tesla Stock Price Prediction (Actual, Train/Test Predict, Future Predict)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()