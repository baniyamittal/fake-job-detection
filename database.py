import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "users.db"

def get_connection():
    return sqlite3.connect(DATABASE_PATH)

def create_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_title TEXT NOT NULL,
            prediction TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            decision_score REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.commit()
    connection.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email, password):
    connection = get_connection()
    cursor = connection.cursor()

    hashed_password = hash_password(password)

    try:
        cursor.execute(
            """
            INSERT INTO users (name, email, password, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                email,
                hashed_password,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )

        connection.commit()
        return True, "Registration successful."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    finally:
        connection.close()

def login_user(email, password):
    connection = get_connection()
    cursor = connection.cursor()

    hashed_password = hash_password(password)

    cursor.execute(
        """
        SELECT id, name, email
        FROM users
        WHERE email = ? AND password = ?
        """,
        (email, hashed_password)
    )

    user = cursor.fetchone()
    connection.close()

    if user:
        return True, user

    return False, None

def save_prediction(
    user_id,
    job_title,
    prediction,
    risk_level,
    decision_score
):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_history
        (
            user_id,
            job_title,
            prediction,
            risk_level,
            decision_score,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            job_title,
            prediction,
            risk_level,
            decision_score,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    connection.commit()
    connection.close()

def get_prediction_history(user_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            job_title,
            prediction,
            risk_level,
            decision_score,
            created_at
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,)
    )

    history = cursor.fetchall()
    connection.close()

    return history

def get_prediction_statistics(user_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM prediction_history
        WHERE user_id = ?
        """,
        (user_id,)
    )

    total = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM prediction_history
        WHERE user_id = ? AND prediction = 'Genuine'
        """,
        (user_id,)
    )

    genuine = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM prediction_history
        WHERE user_id = ? AND prediction = 'Fraudulent'
        """,
        (user_id,)
    )

    fraudulent = cursor.fetchone()[0]

    connection.close()

    fraud_rate = (fraudulent / total * 100) if total > 0 else 0

    return total, genuine, fraudulent, fraud_rate

create_database()