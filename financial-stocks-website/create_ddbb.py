import sqlite3

# Connect to or create the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Define the table schema
table_schema = '''
    CREATE TABLE IF NOT EXISTS NASDQ (
        id INTEGER PRIMARY KEY,
        ticker TEXT,
        date DATE,
        open INT,
        close INT,
        high INT,
        low INT,
        volume INT
    )
'''

# Create the table
cursor.execute(table_schema)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database and table created successfully.")
