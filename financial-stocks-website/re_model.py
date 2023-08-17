import sqlite3
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Fetch data from the database
query = "SELECT open, close, high, low, volume FROM SP500"
cursor.execute(query)
data = cursor.fetchall()
data = np.array(data).astype(float)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Calculate price changes
price_changes = np.diff(data_scaled[:, 2])  # Using 'high' prices

# Define Q-learning parameters
num_states = 10
num_actions = 2  # Buy or sell
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration factor

# Initialize Q-table
q_table = np.zeros((num_states, num_actions))

# Define a function to discretize price changes into states
def discretize_state(price_change):
    return int(np.floor(price_change * num_states))

# Train the Q-learning agent
num_episodes = 5
for episode in range(num_episodes):
    state = 0  # Initial state
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(num_actions)  # Exploration
        else:
            action = np.argmax(q_table[state, :])  # Exploitation

        next_state = discretize_state(price_changes[state])
        reward = price_changes[state] * (action - 0.5) * 2  # Reward based on action and price change

        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        state = next_state

        if state == num_states - 1:
            done = True

# Save the Q-table using pickle
with open('q_table.pickle', 'wb') as f:
    pickle.dump(q_table, f)

# Now you can use the Q-table to predict the 'high' price for the next day
current_state = discretize_state(price_changes[-1])
predicted_action = np.argmax(q_table[current_state, :])
predicted_price_change = (predicted_action - 0.5) * 2
predicted_next_high = data_scaled[-1, 2] + predicted_price_change

# Denormalize the predicted price to get the actual 'high' price
predicted_next_high = scaler.inverse_transform([[data_scaled[-1, 0], data_scaled[-1, 1],
                                                 predicted_next_high, data_scaled[-1, 3],
                                                 data_scaled[-1, 4]]])[0][2]

print("Predicted Next High Price:", predicted_next_high)
