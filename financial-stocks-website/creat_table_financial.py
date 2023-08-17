import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Create the 'income' table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS income (
        ticker TEXT,
        date DATE,
        "Tax Effect Of Unusual Items" REAL,
        "Tax Rate For Calcs" REAL,
        "Normalized EBITDA" REAL,
        "Total Unusual Items" REAL,
        "Total Unusual Items Excluding Goodwill" REAL,
        "Net Income From Continuing Operation Net Minority Interest" REAL,
        "Reconciled Depreciation" REAL,
        "Reconciled Cost Of Revenue" REAL,
        "EBIT" REAL,
        "Net Interest Income" REAL,
        "Interest Expense" REAL,
        "Interest Income" REAL,
        "Normalized Income" REAL,
        "Net Income From Continuing And Discontinued Operation" REAL,
        "Total Expenses" REAL,
        "Total Operating Income As Reported" REAL,
        "Diluted Average Shares" REAL,
        "Basic Average Shares" REAL,
        "Diluted EPS" REAL,
        "Basic EPS" REAL,
        "Diluted NI Availto Com Stockholders" REAL,
        "Net Income Common Stockholders" REAL,
        "Net Income" REAL,
        "Net Income Including Noncontrolling Interests" REAL,
        "Net Income Continuous Operations" REAL,
        "Tax Provision" REAL,
        "Pretax Income" REAL,
        "Other Income Expense" REAL,
        "Other Non Operating Income Expenses" REAL,
        "Special Income Charges" REAL,
        "Write Off" REAL,
        "Gain On Sale Of Security" REAL,
        "Net Non Operating Interest Income Expense" REAL,
        "Interest Expense Non Operating" REAL,
        "Interest Income Non Operating" REAL,
        "Operating Income" REAL,
        "Operating Expense" REAL,
        "Research And Development" REAL,
        "Selling General And Administration" REAL,
        "Selling And Marketing Expense" REAL,
        "General And Administrative Expense" REAL,
        "Other Gand A" REAL,
        "Gross Profit" REAL,
        "Cost Of Revenue" REAL,
        "Total Revenue" REAL,
        "Operating Revenue" REAL,
        PRIMARY KEY (date)
    )
''')

# Commit changes and close the connection
conn.commit()
conn.close()

print("Income table created.")
