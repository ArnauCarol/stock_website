import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('stock_data.db')
cursor = conn.cursor()

# Delete all data from the SP500 table
delete_query = "DELETE FROM SP500"
cursor.execute(delete_query)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("All data from the SP500 table has been deleted.")
