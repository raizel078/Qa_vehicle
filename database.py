import  sqlite3

conn = sqlite3.connect('database.db')

def create_table(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS detection (timestamp REAL, label TEXT)')
    conn.commit()

def insert_detection(conn, timestamp, label):
    conn.execute('INSERT INTO detection (timestamp, label) VALUES (?,?)', (timestamp, label))
    conn.commit()

def get_detection(conn):
    rows = conn.execute('SELECT timestamp, label FROM detection').fetchall()
    return rows