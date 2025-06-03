import sqlite3
from tabulate import tabulate  # for pretty table output, optional

# Connect to your SQLite DB (change the path accordingly)
conn = sqlite3.connect(r'C:\Users\sanjana\Desktop\curato\curato.db')
cursor = conn.cursor()

# 1. List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:")
for table_name in tables:
    print("-", table_name[0])

# Replace with your actual table name
table_to_show = tables[0][0]  # Just pick the first table for demo

print(f"\nShowing schema for table '{table_to_show}':")
cursor.execute(f"PRAGMA table_info({table_to_show})")
columns = cursor.fetchall()
for col in columns:
    print(f"Column: {col[1]}, Type: {col[2]}")

print(f"\nAll data from table '{table_to_show}':")
cursor.execute(f"SELECT * FROM {table_to_show}")
rows = cursor.fetchall()

# Print rows in a nice table format (requires tabulate)
try:
    print(tabulate(rows, headers=[col[1] for col in columns], tablefmt="grid"))
except ImportError:
    # If tabulate is not installed, print raw rows and headers
    print([col[1] for col in columns])
    for row in rows:
        print(row)

conn.close()
