import asyncio
from datetime import datetime
from decimal import Decimal
import json
import logging
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from langdetect import detect
from dotenv import load_dotenv
import redis
import asyncpg
from asyncpg.pool import Pool
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
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

async def get_or_create_conversation(conversation_id: Optional[str], user_id: Optional[str], language: str) -> str:
    if conversation_id:
        await db_execute_async(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
            (conversation_id,)
        )
        return conversation_id
    else:
        new_conversation_id = str(uuid.uuid4())
        user_id_clean = user_id[2:] if user_id and user_id.startswith(("t_", "c_")) else user_id
        # Store language and state in meta_data as JSON
        meta_data = json.dumps({"language": language, "state": {}})
        await db_execute_async(
            "INSERT INTO conversations (id, user_id, meta_data) VALUES ($1, $2, $3)",
            (new_conversation_id, user_id_clean, meta_data)
        )
        return new_conversation_id

async def add_message_to_conversation(conversation_id: str, role: str, content: str):
    # Get the current max message_index
    max_index = await db_fetch_one_async(
        "SELECT COALESCE(MAX(message_index), -1) as max_idx FROM conversation_messages WHERE conversation_id = $1",
        (conversation_id,)
    )
    next_index = (max_index["max_idx"] + 1) if max_index else 0
    
    # Generate a unique ID for the message
    message_id = str(uuid.uuid4())
    
    await db_execute_async(
        "INSERT INTO conversation_messages (id, conversation_id, role, content, message_index) VALUES ($1, $2, $3, $4, $5)",
        (message_id, conversation_id, role, content, next_index)
    )

async def get_conversation_history(conversation_id: str, max_messages: int = 20) -> List[Message]:
    cache_key = f"history:{conversation_id}"
    cached_history = REDIS_CLIENT.get(cache_key)
    if cached_history:
        return [Message(**msg) for msg in json.loads(cached_history)]
    
    messages = await db_fetch_all_async(
        "SELECT role, content, created_at as timestamp FROM conversation_messages "
        "WHERE conversation_id = $1 ORDER BY message_index DESC LIMIT $2",
        (conversation_id, max_messages)
    )
    history = [
        Message(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
        for msg in reversed(messages)
    ]
    REDIS_CLIENT.set(cache_key, json.dumps([msg.dict() for msg in history], cls=DateTimeEncoder), ex=300)
    return history