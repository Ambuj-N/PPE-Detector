import sqlite3
import datetime
from typing import List, Dict, Any

DB_NAME = 'ppe_violations.db'

def setup_database():
    """Initializes the SQLite database and creates the necessary tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print(f"Initializing database: {DB_NAME}")
                        
    # 1. Violations Log Table
    # Stores a record for every violation event detected
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT, 
            timestamp TEXT, 
            person_identifier TEXT, 
            missing_items TEXT,
            total_persons INTEGER,
            total_violators INTEGER
        )
    ''')
                        
    conn.commit()
    conn.close()
    print("Database setup complete.")


def log_violation_to_db(
    person_id: str, 
    missing_items: List[str], 
    person_count: int, 
    total_violators: int
):
    """Inserts a violation record into the SQLite database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        missing_str = ", ".join(missing_items)

        cursor.execute(
            """INSERT INTO violations 
               (timestamp, person_identifier, missing_items, total_persons, total_violators) 
               VALUES (?, ?, ?, ?, ?)""", 
            (timestamp, person_id, missing_str, person_count, total_violators)
        )
        conn.commit()
        conn.close()
        # print(f"Logged violation for {person_id}: {missing_str}") # Uncomment for debugging
    except Exception as e:
        print(f"Error logging to database: {e}")

def fetch_violation_logs() -> List[Dict[str, Any]]:
    """Fetches all violation records from the database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # This allows fetching rows as dictionaries
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM violations ORDER BY timestamp DESC")
    
    # Convert sqlite3.Row objects to standard dictionaries
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return logs

# Call the setup function when this module is imported (or call it once in app.py)
setup_database()