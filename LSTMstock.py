# Improved LSTM Stock Price Prediction using TensorFlow and Keras
# This version includes better structure, validation, and an extended model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import yfinance as yf

# Load stock data (e.g., Apple Inc.)
ticker = 'AAPL'
df = yf.download(ticker, start='2012-01-01', end='2022-01-01')
data = df[['Close']]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Split data into train and test
look_back = 60
training_data_len = int(len(data_scaled) * 0.8)

train_data = data_scaled[:training_data_len]
test_data = data_scaled[training_data_len - look_back:]

# Prepare the data
def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(actual, label='Actual AAPL Price', color='blue')
plt.plot(predictions, label='Predicted AAPL Price', color='red')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
