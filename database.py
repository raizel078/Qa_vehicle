import  sqlite3

from transformers.dynamic_module_utils import resolve_trust_remote_code

from Qa_vehicle.embder import timestamp
from Qa_vehicle.querry import result

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

def count_label(conn, target):
    result = conn.execute(
        'SELECT COUNT(*) FROM detection WHERE Label=?',(target,)
    ).fetchone()
    return result

def lookup(conn, target):
    result = conn.execute(
        "SELECT timestamp FROM detection WHERE label = ?", (target,)
    ).fetchall()
    return result