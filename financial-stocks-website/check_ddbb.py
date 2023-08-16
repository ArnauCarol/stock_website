import sqlite3

# Connect to the database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Execute a SELECT query to retrieve data from the table
query = "SELECT * FROM SP500"
cursor.execute(query)

# Fetch and print all rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
conn.close()
