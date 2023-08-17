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
