import yfinance as yf
import sqlite3
from datetime import datetime

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Define the ticker symbol
ticker = '^GSPC'

# Retrieve data from Yahoo Finance API
sp500_data = yf.download(ticker, start='2021-01-01', end='2022-01-01')

# Insert data into the SP500 table
for index, row in sp500_data.iterrows():
    cursor.execute('''
        INSERT INTO SP500 (ticker, date, open, close, high, low, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, index.date(), int(row['Open']), int(row['Close']), int(row['High']), int(row['Low']), int(row['Volume'])))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Data inserted successfully.")
