from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

DATABASE = r'C:\Users\sanjana\Desktop\curato\curato.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # To access columns by name
    return conn

@app.route('/artworks', methods=['GET'])
def get_artworks():
    conn = get_db_connection()
    cursor = conn.execute('SELECT * FROM artworks')  # replace 'artworks' with your table name
    rows = cursor.fetchall()
    conn.close()

    artworks = [dict(row) for row in rows]
    return jsonify(artworks)

if __name__ == '__main__':
    app.run(debug=True)
