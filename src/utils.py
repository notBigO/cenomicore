from datetime import datetime
from decimal import Decimal
import json
import logging
import os
import psycopg2
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from langdetect import detect
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "cenomi_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "your_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

def db_fetch_one(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG)  # type: ignore
        cur = conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    except Exception as e:
        logger.error(f"Database error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def db_fetch_all(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG)  # type: ignore
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def db_execute(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG)  # type: ignore
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Database error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def convert_to_json_safe(data):
    if isinstance(data, list):
        return [convert_to_json_safe(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_json_safe(value) for key, value in data.items()}
    elif isinstance(data, datetime):
        return str(data)
    else:
        return data

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def get_or_create_session(session_id: Optional[str], user_id: Optional[str], language: str) -> str:
    if session_id:
        db_execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
            (session_id,)
        )
        return session_id
    else:
        new_session_id = str(uuid.uuid4())
        user_id_clean = user_id[2:] if user_id and user_id.startswith(("t_", "c_")) else user_id
        db_execute(
            "INSERT INTO conversations (session_id, user_id, language, current_state) VALUES (%s, %s, %s, %s)",
            (new_session_id, user_id_clean, language, json.dumps({}))
        )
        return new_session_id

def add_message_to_conversation(session_id: str, role: str, content: str):
    db_execute(
        "INSERT INTO conversation_messages (session_id, role, content) VALUES (%s, %s, %s)",
        (session_id, role, content)
    )

def get_conversation_history(session_id: str, max_messages: int = 10) -> List[Message]:
    messages = db_fetch_all(
        "SELECT role, content, timestamp FROM conversation_messages "
        "WHERE session_id = %s ORDER BY timestamp DESC LIMIT %s",
        (session_id, max_messages)
    )
    return [
        Message(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
        for msg in reversed(messages)
    ]