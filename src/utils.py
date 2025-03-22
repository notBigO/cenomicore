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
import redis
import asyncpg
from asyncpg.pool import Pool

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

DB_CONFIG_ASYNC = {
    "database": os.getenv("DB_NAME", "cenomi_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "your_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

# Redis setup
REDIS_CLIENT = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Asyncpg pool
DB_POOL: Optional[Pool] = None

async def get_db_pool():
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = await asyncpg.create_pool(**DB_CONFIG_ASYNC)
    return DB_POOL

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

    def dict(self):
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

# Synchronous DB functions (for tenant compatibility)
def db_fetch_one(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
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
        conn = psycopg2.connect(**DB_CONFIG)
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
        conn = psycopg2.connect(**DB_CONFIG)
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

# Asynchronous DB functions
async def db_fetch_one_async(query: str, params: tuple = ()):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)
        if row:
            return dict(row)
        return None

async def db_fetch_all_async(query: str, params: tuple = ()):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

async def db_execute_async(query: str, params: tuple = ()):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(query, *params)

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

async def get_or_create_session(session_id: Optional[str], user_id: Optional[str], language: str) -> str:
    if session_id:
        await db_execute_async(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE session_id = $1",
            (session_id,)
        )
        return session_id
    else:
        new_session_id = str(uuid.uuid4())
        user_id_clean = user_id[2:] if user_id and user_id.startswith(("t_", "c_")) else user_id
        await db_execute_async(
            "INSERT INTO conversations (session_id, user_id, language, current_state) VALUES ($1, $2, $3, $4)",
            (new_session_id, user_id_clean, language, json.dumps({}))
        )
        return new_session_id

async def add_message_to_conversation(session_id: str, role: str, content: str):
    await db_execute_async(
        "INSERT INTO conversation_messages (session_id, role, content) VALUES ($1, $2, $3)",
        (session_id, role, content)
    )

async def get_conversation_history(session_id: str, max_messages: int = 10) -> List[Message]:
    cache_key = f"history:{session_id}"
    cached_history = REDIS_CLIENT.get(cache_key)
    if cached_history:
        return [Message(**msg) for msg in json.loads(cached_history)]
    
    messages = await db_fetch_all_async(
        "SELECT role, content, timestamp FROM conversation_messages "
        "WHERE session_id = $1 ORDER BY timestamp DESC LIMIT $2",
        (session_id, max_messages)
    )
    history = [
        Message(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
        for msg in reversed(messages)
    ]
    REDIS_CLIENT.set(cache_key, json.dumps([msg.dict() for msg in history]), ex=300)  # 5 min TTL
    return history