import yfinance as yf
import sqlite3
import pandas as pd

# Define the stock ticker symbol (e.g., Microsoft: MSFT)
ticker_symbol = "MSFT"

# Create a Ticker object for the stock
stock = yf.Ticker(ticker_symbol)

# Access and display income statements

df1 = stock.quarterly_income_stmt
df2 = stock.income_stmt

# Merge the two DataFrames
income_statements = pd.concat([df1, df2], axis=1)


# Connect to the SQLite database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

## Iterate through the dates and insert data into the 'income' table
for date, income_statement in income_statements.items():
    # Convert the date to string format
    str_date = date.strftime('%Y-%m-%d')

    # Check if the data for the date already exists in the database
    existing_data = cursor.execute('SELECT * FROM income WHERE date=?', (str_date,)).fetchone()
    if existing_data:
        print(f"Data for {str_date} already exists, skipping...")
        continue

    # Create a data dictionary from the income statement data
    data = {}
    for column in income_statement.index:
        data[column] = [income_statement[column]]

    # Create a DataFrame from the data and set the index to the date
    df = pd.DataFrame(data, index=[date])
    df.index = pd.to_datetime(df.index)

    # Create a placeholder for the columns to insert
    placeholders = ', '.join(['?' for _ in df.columns])
    
    # Create the column names with double quotes for handling special characters
    column_names = ', '.join(['"' + col + '"' for col in df.columns])

    # Insert data into the 'income' table
    query = f"INSERT INTO income (ticker, date, {column_names}) VALUES (?, ?, {placeholders})"
    for _, row in df.iterrows():
        cursor.execute(query, ('MSFT', str_date, *row))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Data insertion completed.")