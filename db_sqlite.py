import sqlite3

def init_db():
    conn = sqlite3.connect("curato.db")
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artworks (
            filename TEXT PRIMARY KEY,
            style TEXT,
            tags TEXT,
            caption TEXT
        )
    """)

    # Add cloud_url column if not exists
    cursor.execute("PRAGMA table_info(artworks)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'cloud_url' not in columns:
        cursor.execute("ALTER TABLE artworks ADD COLUMN cloud_url TEXT")

    conn.commit()
    conn.close()

def save_metadata(filename, style, tags, caption, cloud_url):
    conn = sqlite3.connect("curato.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO artworks (filename, style, tags, caption, cloud_url)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, style, ",".join(tags), caption, cloud_url))
    conn.commit()
    conn.close()
