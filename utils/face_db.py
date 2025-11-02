import sqlite3
import datetime
import numpy as np
import io
from typing import Optional, List, Dict, Any, Tuple

DB_NAME = 'ppe_records.db'

def get_db_connection():
    """Returns a connection and cursor for the database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    """Initializes the SQLite database and creates the necessary tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Employee Table: Stores face encoding as a BLOB
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY, 
            name TEXT NOT NULL, 
            face_encoding BLOB NOT NULL 
        )
    ''')
                        
    # 2. Violations Log Table: Stores violation events
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            employee_id TEXT, 
            name TEXT, 
            timestamp TEXT, 
            missing_items TEXT,
            image_blob BLOB,
            -- FOREIGN KEY(employee_id) REFERENCES employees(employee_id) 
            -- Note: We intentionally avoid ON DELETE CASCADE here so violation 
            -- history remains even if an employee is deleted.
            FOREIGN KEY(employee_id) REFERENCES employees(employee_id)
        )
    ''')
                        
    conn.commit()
    conn.close()
    print("Face DB setup complete.")


def register_employee(employee_id: str, name: str, encoding: np.ndarray):
    """Registers a new employee with their face encoding."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Convert numpy array to bytes for storage (BLOB)
    encoding_bytes = encoding.astype(np.float64).tobytes()
    
    try:
        cursor.execute(
            """INSERT OR REPLACE INTO employees (employee_id, name, face_encoding) 
               VALUES (?, ?, ?)""", 
            (employee_id, name, encoding_bytes)
        )
        conn.commit()
    finally:
        conn.close()


def load_all_encodings() -> Dict[str, Tuple[str, np.ndarray]]:
    """Loads all known face encodings and names from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT employee_id, name, face_encoding FROM employees")
    rows = cursor.fetchall()
    conn.close()
    
    encodings = {}
    for row in rows:
        # Convert BLOB back to numpy array
        encoding = np.frombuffer(row['face_encoding'], dtype=np.float64)
        encodings[row['employee_id']] = (row['name'], encoding)
        
    return encodings


def log_violation(employee_id: Optional[str], name: Optional[str], missing_items: List[str], image_bytes: bytes):
    """Logs a single violation event."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    missing_str = ", ".join(missing_items)

    try:
        cursor.execute(
            """INSERT INTO violations 
               (employee_id, name, timestamp, missing_items, image_blob) 
               VALUES (?, ?, ?, ?, ?)""", 
            (employee_id, name, timestamp, missing_str, image_bytes)
        )
        conn.commit()
    finally:
        conn.close()


def get_violation_count(employee_id: str) -> int:
    """Gets the total number of violations for a specific employee ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT COUNT(*) FROM violations WHERE employee_id = ?", 
        (employee_id,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_recent_violations(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches the most recent violations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM violations ORDER BY timestamp DESC LIMIT ?", 
        (limit,)
    )
    
    # Process rows for the app.py display
    logs = []
    for row in cursor.fetchall():
        log = dict(row)
        log['missing_items'] = log['missing_items'].split(', ') if log['missing_items'] else []
        logs.append(log)

    conn.close()
    return logs

# --- NEW ADMIN FUNCTIONS ---

def delete_violation_entry(violation_id: int):
    """Deletes a single violation entry by its primary key ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM violations WHERE id = ?", 
            (violation_id,)
        )
        conn.commit()
    finally:
        conn.close()
        

def delete_employee(employee_id: str):
    """Deletes an employee record (name and face encoding) by their employee_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM employees WHERE employee_id = ?", 
            (employee_id,)
        )
        conn.commit()
    finally:
        conn.close()
        
# Execute the setup function on import
setup_database()
