
import yfinance as yf
import sqlite3
import pandas as pd

# Define the stock ticker symbol (e.g., Microsoft: MSFT)
ticker_symbol = "MSFT"

# Create a Ticker object for the stock
stock = yf.Ticker(ticker_symbol)

# Access and merge income statements
df1 = stock.quarterly_income_stmt
df2 = stock.income_stmt
income_statements = pd.concat([df1, df2], axis=1)

# Connect to the SQLite database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# ...

## Iterate through the dates and insert data into the 'income' table
for date, income_statement in income_statements.items():
    # Convert the date to string format
    str_date = date.strftime('%Y-%m-%d')

    # Check if the data for the ticker and date already exist in the database
    existing_data = cursor.execute('SELECT * FROM INCOME WHERE ticker=? AND date=?', (ticker_symbol, str_date)).fetchone()
    if existing_data:
        print(f"Data for ticker '{ticker_symbol}' on {str_date} already exists, skipping...")
        continue

    # Create a data dictionary from the income statement data
    data = {}
    for column in income_statement.index:
        # Ensure column names match the SQL table column names
        sql_column_name = '"' + column.replace(" ", "_") + '"'
        data[sql_column_name] = [income_statement[column]]

    # Create a DataFrame from the data and set the index to the date
    df = pd.DataFrame(data, index=[date])
    df.index = pd.to_datetime(df.index)

    # Create a placeholder for the columns to insert
    placeholders = ', '.join(['?' for _ in df.columns])

    # Insert data into the 'income' table
    query = f"INSERT INTO INCOME (ticker, date, {', '.join(df.columns)}) VALUES (?, ?, {placeholders})"
    for _, row in df.iterrows():
        cursor.execute(query, (ticker_symbol, str_date, *row))

# ...


# Commit changes and close the connection
conn.commit()
conn.close()

print("Data insertion completed.")



################################################################################################################


# Access balance sheet data
balance_sheet = stock.balance_sheet
quarterly_balance_sheet = stock.quarterly_balance_sheet

# Merge the two DataFrames
all_balance_sheet = pd.concat([balance_sheet, quarterly_balance_sheet], axis=1)

# Connect to the SQLite database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

## Iterate through the dates and insert data into the 'income' table
for date, all_balance_sheet in all_balance_sheet.items():
    # Convert the date to string format
    str_date = date.strftime('%Y-%m-%d')

    # Check if the data for the ticker and date already exist in the database
    existing_data = cursor.execute('SELECT * FROM BALANCE WHERE ticker=? AND date=?', (ticker_symbol, str_date)).fetchone()
    if existing_data:
        print(f"Data for ticker '{ticker_symbol}' on {str_date} already exists, skipping...")
        continue

    # Create a data dictionary from the income statement data
    data = {}
    for column in all_balance_sheet.index:
        # Ensure column names match the SQL table column names
        sql_column_name = '"' + column.replace(" ", "_") + '"'
        data[sql_column_name] = [all_balance_sheet[column]]

    # Create a DataFrame from the data and set the index to the date
    df = pd.DataFrame(data, index=[date])
    df.index = pd.to_datetime(df.index)

    # Create a placeholder for the columns to insert
    placeholders = ', '.join(['?' for _ in df.columns])

    # Insert data into the 'income' table
    query = f"INSERT INTO BALANCE (ticker, date, {', '.join(df.columns)}) VALUES (?, ?, {placeholders})"
    for _, row in df.iterrows():
        cursor.execute(query, (ticker_symbol, str_date, *row))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Data insertion completed.")




################################################################################################################



# Access cash flow statements
df3 = stock.quarterly_cashflow
df4 = stock.cashflow
cashflow_statements = pd.concat([df3, df4], axis=1)

# Connect to the SQLite database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

## Iterate through the dates and insert data into the 'income' table
for date, cashflow_statements in cashflow_statements.items():
    # Convert the date to string format
    str_date = date.strftime('%Y-%m-%d')

    # Check if the data for the ticker and date already exist in the database
    existing_data = cursor.execute('SELECT * FROM CASH_FLOW WHERE ticker=? AND date=?', (ticker_symbol, str_date)).fetchone()
    if existing_data:
        print(f"Data for ticker '{ticker_symbol}' on {str_date} already exists, skipping...")
        continue

    # Create a data dictionary from the income statement data
    data = {}
    for column in cashflow_statements.index:
        # Ensure column names match the SQL table column names
        sql_column_name = '"' + column.replace(" ", "_") + '"'
        data[sql_column_name] = [cashflow_statements[column]]

    # Create a DataFrame from the data and set the index to the date
    df = pd.DataFrame(data, index=[date])
    df.index = pd.to_datetime(df.index)

    # Create a placeholder for the columns to insert
    placeholders = ', '.join(['?' for _ in df.columns])

    # Insert data into the 'income' table
    query = f"INSERT INTO CASH_FLOW (ticker, date, {', '.join(df.columns)}) VALUES (?, ?, {placeholders})"
    for _, row in df.iterrows():
        cursor.execute(query, (ticker_symbol, str_date, *row))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Cash flow data insertion completed.")

"""



import pandas as pd
import yfinance as yf

# Define the stock ticker symbol (e.g., Microsoft: MSFT)
ticker_symbol = "MSFT"

# Create a Ticker object for the stock
stock = yf.Ticker(ticker_symbol)

# Access financial data
income_statements = pd.concat([stock.income_stmt, stock.quarterly_income_stmt], axis=1)
balance_sheets = pd.concat([stock.balance_sheet, stock.quarterly_balance_sheet], axis=1)
cashflows = pd.concat([stock.cashflow, stock.quarterly_cashflow], axis=1)

# Print income statements data
print("Income Statements:")
print(income_statements)

# Print balance sheets data
print("Balance Sheets:")
print(balance_sheets)

# Print cashflows data
print("Cashflows:")
print(cashflows.index)
"""