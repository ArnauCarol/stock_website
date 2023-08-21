import yfinance as yf
import sqlite3
from datetime import datetime
import numpy as np

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Define the dictionary of table names and tickers
table_TIKER = {'NASDQ': 'NQ=F', 'SP500': '^GSPC', 'IBEX35': '^IBEX', 'stocks':'MSFT'}

# Iterate through each table-ticker pair
for table_name, ticker in table_TIKER.items():
    # Retrieve new data from Yahoo Finance API
    new_data = yf.download(ticker, start='2005-01-01', end='2023-08-30')

    # Check for duplications and insert non-duplicate data
    for index, row in new_data.iterrows():
        existing_data = cursor.execute(f'''
            SELECT * FROM {table_name} WHERE ticker=? AND date=?
        ''', (ticker, index.date())).fetchone()

        if not existing_data:
            # Normalize volume data
            volume = row['Volume']
            min_volume = new_data['Volume'].min()
            max_volume = new_data['Volume'].max()
            normalized_volume = (volume - min_volume) / (max_volume - min_volume)

            cursor.execute(f'''
                INSERT INTO {table_name} (ticker, date, open, close, high, low, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (ticker, index.date(), int(row['Open']), int(row['Close']), int(row['High']), int(row['Low']), normalized_volume))
            print(f"Inserted data for {index.date()} in {table_name}")

# Commit changes and close the connection
conn.commit()
conn.close()

print("Data insertion completed.")

