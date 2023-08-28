"""
#PREDICTION DAY TO DAY


import sqlite3
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Fetch data from the database
query = "SELECT open, close, high, low, volume FROM SP500"
data = cursor.execute(query).fetchall()
data = np.array(data)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare data for LSTM
sequence_length = 10
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length, 2])  # Predicting 'high' price

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}')

# Save the model using pickle
with open('model_lstm_high.pickle', 'wb') as f:
    pickle.dump(model, f)

# Load the trained LSTM model from the pickle file
with open('model_lstm_high.pickle', 'rb') as f:
    model = pickle.load(f)

# Get the most recent sequence of data
latest_sequence = data_scaled[-sequence_length:]

# Reshape the sequence for prediction
latest_sequence = np.reshape(latest_sequence, (1, sequence_length, latest_sequence.shape[1]))

# Make predictions using the LSTM model
predicted_scaled_price = model.predict(latest_sequence)[0][0]

# Denormalize the predicted price to get the actual 'high' price
predicted_high_price = scaler.inverse_transform([[data_scaled[-1, 0], data_scaled[-1, 1],
                                                  predicted_scaled_price, data_scaled[-1, 3],
                                                  data_scaled[-1, 4]]])[0][2]

print("Predicted Next High Price:", predicted_high_price)


#PREDICCTION 5TH DAY


import sqlite3
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Fetch data from the database
query = "SELECT open, close, high, low, volume FROM SP500"
data = cursor.execute(query).fetchall()
data = np.array(data)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare data for LSTM
sequence_length = 4  # Number of previous days to consider
target_offset = 5    # Predicting the high price of the day n+5
X, y = [], []

for i in range(len(data_scaled) - sequence_length - target_offset + 1):
    X.append(data_scaled[i:i+sequence_length])
    y.append(data_scaled[i+sequence_length+target_offset-1, 2])  # Predicting 'high' price

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}')

# Get the most recent sequence of data
latest_sequence = data_scaled[-sequence_length:]

# Reshape the sequence for prediction
latest_sequence = np.reshape(latest_sequence, (1, sequence_length, latest_sequence.shape[1]))

# Make predictions using the LSTM model
predicted_scaled_price = model.predict(latest_sequence)[0][0]

# Denormalize the predicted price to get the actual 'high' price
predicted_high_price = scaler.inverse_transform([[data_scaled[-1, 0], data_scaled[-1, 1],
                                                  predicted_scaled_price, data_scaled[-1, 3],
                                                  data_scaled[-1, 4]]])[0][2]

print("Predicted High Price after 5 Days:", predicted_high_price)
"""

"""# PREDICTION MANY TO MANY
import numpy as np
import sqlite3
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Fetch data from the database
query = "SELECT date, open, close, high, low, volume FROM SP500"
data = cursor.execute(query).fetchall()
data = np.array(data)

# Normalize the data
data_min = np.min(data, axis=0)
data_max = np.max(data, axis=0)
normalized_data = (data - data_min) / (data_max - data_min)

# Prepare data for training
sequence_length = 4
num_features = 4  # High, Low, Close, Open

X = []
y = []

for i in range(len(normalized_data) - sequence_length - 1):
    X.append(normalized_data[i:i+sequence_length, :num_features])  # Exclude the last feature (volume)
    y.append(normalized_data[i+sequence_length, :num_features])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Build the model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, num_features)),
    Dense(num_features)  # Output layer for predicting the next day's prices
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=16)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Mean Squared Error:", loss)

# Make predictions
predicted_prices = model.predict(X_test)

# Denormalize the predictions
denormalized_predictions = predicted_prices * (data_max[:num_features] - data_min[:num_features]) + data_min[:num_features]

# Create a DataFrame for the results
result_columns = ['Predicted_High', 'Predicted_Low', 'Predicted_Close', 'Predicted_Open']
result_df = pd.DataFrame(denormalized_predictions, columns=result_columns)

# Add original features to the DataFrame
original_features = pd.DataFrame(X_test[:, :, :num_features].reshape(-1, num_features), columns=['Open', 'Close', 'High', 'Low'])
result_df = pd.concat([original_features, result_df], axis=1)

# Print the resulting DataFrame
print(result_df)




###WORKING MANY TO MANY 
import numpy as np
import sqlite3
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Fetch data from the database
query = "SELECT date, open, close, high, low, volume FROM SP500 ORDER BY date"
data = cursor.execute(query).fetchall()
data = np.array(data)

# Extract the date column and the numerical features
dates = data[:, 0]
numeric_data = data[:, 1:].astype(float)

# Normalize the numerical data
data_min = np.min(numeric_data, axis=0)
data_max = np.max(numeric_data, axis=0)
normalized_data = (numeric_data - data_min) / (data_max - data_min)

# Prepare data for training
sequence_length = 4
num_features = 4  # High, Low, Close, Open

X = []
y = []

for i in range(len(normalized_data) - sequence_length - 1):
    X.append(normalized_data[i:i+sequence_length, :num_features])
    y.append(normalized_data[i+sequence_length, :num_features])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Build the model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, num_features)),
    Dense(num_features)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=16)

# Make predictions
predicted_prices = model.predict(X_test)

# Denormalize the predictions
denormalized_predictions = predicted_prices * (data_max[:num_features] - data_min[:num_features]) + data_min[:num_features]

# Create a DataFrame for the results
result_columns = ['Predicted_High', 'Predicted_Low', 'Predicted_Close', 'Predicted_Open']
result_df = pd.DataFrame(denormalized_predictions, columns=result_columns)

# Add the 'date' column to the DataFrame
result_df['Date'] = dates[split_index + sequence_length : split_index + sequence_length + len(X_test)]

# Print the resulting DataFrame
print(result_df)

"""

"""import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Fetch data from the database
query = "SELECT date, open, close, high, low, volume FROM SP500 ORDER BY date"
data = cursor.execute(query).fetchall()
data = np.array(data)

# Extract the date column and the numerical features
dates = data[:, 0]
numeric_data = data[:, 1:].astype(float)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(numeric_data)

# Perform time series decomposition
series = pd.Series(data_scaled[:, 2], index=pd.to_datetime(dates))
result = seasonal_decompose(series, model='additive', period=30)  # Use 'additive' model for handling seasonality

# Print the trend and seasonality components
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(result.observed)
plt.title('Observed')
plt.subplot(3, 1, 2)
plt.plot(result.trend)
plt.title('Trend')
plt.subplot(3, 1, 3)
plt.plot(result.seasonal)
plt.title('Seasonal')
plt.tight_layout()
plt.show()

# Prepare data for LSTM (use the deseasonalized data)
deseasonalized_data = data_scaled[:, 2] - result.seasonal
sequence_length = 4  # Number of previous days to consider
target_offset = 5    # Predicting the high price of the day n+5
X, y = [], []

for i in range(len(deseasonalized_data) - sequence_length - target_offset + 1):
    X.append(deseasonalized_data[i:i+sequence_length])
    y.append(deseasonalized_data[i+sequence_length+target_offset-1])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}')

# Get the most recent sequence of data
latest_sequence = deseasonalized_data[-sequence_length:]

# Reshape the sequence for prediction
latest_sequence = np.reshape(latest_sequence, (1, sequence_length, 1))

# ...

# Make predictions using the LSTM model
predicted_deseasonalized_price = model.predict(latest_sequence)[0][0]

# Reseasonalize the predicted price
predicted_high_price_deseasonalized = predicted_deseasonalized_price + result.seasonal[-1]

# Denormalize the predicted deseasonalized price
predicted_high_price = scaler.inverse_transform([[data_scaled[-1, 0], data_scaled[-1, 1],
                                                  predicted_high_price_deseasonalized, data_scaled[-1, 3],
                                                  data_scaled[-1, 4]]])[0][2]

print("Predicted High Price after 5 Days:", predicted_high_price)

"""

import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Fetch data from the database
query = "SELECT date, open, close, high, low, volume FROM SP500 ORDER BY date"
data = cursor.execute(query).fetchall()
data = np.array(data)

# Extract the date column and the numerical features
dates = data[:, 0]
numeric_data = data[:, 1:].astype(float)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(numeric_data)

# Perform time series decomposition
series = pd.Series(data_scaled[:, 2], index=pd.to_datetime(dates))
result = seasonal_decompose(series, model='additive', period=30)  # Use 'additive' model for handling seasonality

# Print the trend and seasonality components
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(result.observed)
plt.title('Observed')
plt.subplot(3, 1, 2)
plt.plot(result.trend)
plt.title('Trend')
plt.subplot(3, 1, 3)
plt.plot(result.seasonal)
plt.title('Seasonal')
plt.tight_layout()
plt.show()

# Prepare data for LSTM (use the deseasonalized data)
deseasonalized_data = data_scaled[:, 2] - result.seasonal
sequence_length = 4  # Number of previous days to consider
target_offset = 5    # Predicting the high price of the day n+5
X, y = [], []

for i in range(len(deseasonalized_data) - sequence_length - target_offset + 1):
    X.append(deseasonalized_data[i:i+sequence_length])
    y.append(deseasonalized_data[i+sequence_length+target_offset-1])

X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}')

# Get the most recent sequence of data
latest_sequence = deseasonalized_data[-sequence_length:]

# Reshape the sequence for prediction
latest_sequence = np.reshape(latest_sequence, (1, sequence_length, 1))

# ...

# Make predictions using the LSTM model
predicted_deseasonalized_price = model.predict(latest_sequence)[0][0]

# Reseasonalize the predicted price
predicted_high_price_deseasonalized = predicted_deseasonalized_price + result.seasonal[-1]

# Denormalize the predicted deseasonalized price
predicted_high_price = scaler.inverse_transform([[data_scaled[-1, 0], data_scaled[-1, 1],
                                                  predicted_high_price_deseasonalized, data_scaled[-1, 3],
                                                  data_scaled[-1, 4]]])[0][2]

print("Predicted High Price after 5 Days:", predicted_high_price)

# Decompose the time series to extract residual component
plt.figure(figsize=(12, 4))
plt.plot(result.resid)
plt.title('Residual Component')
plt.tight_layout()
plt.show()
