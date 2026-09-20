import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import numpy as np

# Database configuration
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "faces.db"

# Database connection manager
@contextmanager
def get_connection():

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# Initialize database
def initialize_database() -> None:
    
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                person_id TEXT NOT NULL UNIQUE,
                face_embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


# Embedding conversion
def embedding_to_blob(embedding: np.ndarray) -> bytes:

    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:

    return np.frombuffer(blob, dtype=np.float32)


# Generate next Person ID
def generate_person_id() -> str:

    with get_connection() as connection:

        rows = connection.execute(
            "SELECT person_id FROM persons"
        ).fetchall()

    highest_number = 0

    for row in rows:

        person_id = row["person_id"]

        if not person_id:
            continue

        match = re.fullmatch(
            r"(?:EMP|Person)(\d+)",
            person_id,
            re.IGNORECASE
        )

        if match:
            number = int(match.group(1))
            highest_number = max(highest_number, number)

    return f"Person{highest_number + 1:03d}"


# Add person
def add_person(name: str, embedding: np.ndarray) -> dict:

    name = name.strip()

    if not name:
        raise ValueError("Name cannot be empty.")

    if embedding is None:
        raise ValueError("Face embedding cannot be empty.")

    person_id = generate_person_id()

    created_at = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO persons (
                name,
                person_id,
                face_embedding,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (name, person_id, embedding_to_blob(embedding), created_at)
        )

        return {
            "id": cursor.lastrowid,
            "person_id": person_id
        }


# Get person by database ID
def get_person_by_id(person_db_id: int):

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT
                id,
                name,
                person_id,
                created_at
            FROM persons
            WHERE id = ?
            """,
            (person_db_id)
        ).fetchone()

    return dict(row) if row else None


# Get all people
def get_all_people() -> list[dict]:

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                id,
                name,
                person_id,
                created_at
            FROM persons
            ORDER BY id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


# Get all embeddings
def get_all_embeddings() -> list[dict]:

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                id,
                name,
                person_id,
                face_embedding,
                created_at
            FROM persons
            """
        ).fetchall()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "person_id": row["person_id"],
            "embedding": blob_to_embedding(row["face_embedding"]),
            "created_at": row["created_at"]
        }
        for row in rows
    ]


# Delete person
def delete_person(person_id: str) -> bool:

    with get_connection() as connection:

        cursor = connection.execute(
            "DELETE FROM persons WHERE person_id = ?",
            (person_id,)
        )

    return cursor.rowcount > 0


# Count people
def get_person_count() -> int:

    with get_connection() as connection:

        row = connection.execute(
            "SELECT COUNT(*) AS count FROM persons"
        ).fetchone()

    return int(row["count"])


# Initialize database
initialize_database()