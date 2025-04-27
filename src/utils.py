import asyncio
from datetime import datetime
from decimal import Decimal
import json
import logging
import os
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
from langdetect import detect
from dotenv import load_dotenv
import redis
import asyncpg
from asyncpg.pool import Pool
import uuid
import functools

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
    "port": int(os.getenv("DB_PORT", "5432"))  # Convert port to int
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
        # Check if the conversation exists
        existing_conversation = await db_fetch_one_async(
            "SELECT id FROM conversations WHERE id = $1",
            (conversation_id,)
        )
        if existing_conversation:
            # Update existing conversation
            await db_execute_async(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                (conversation_id,)
            )
            return conversation_id
        else:
            # Create new conversation if the provided ID doesn't exist
            logger.info(f"Conversation {conversation_id} not found, creating new conversation")
            return await create_new_conversation(user_id, language)
    else:
        # Create a new conversation
        return await create_new_conversation(user_id, language)

async def create_new_conversation(user_id: Optional[str], language: str) -> str:
    new_conversation_id = str(uuid.uuid4())
    user_id_clean = user_id[2:] if user_id and user_id.startswith(("t_", "c_")) else user_id
    # Store language and state in meta_data as JSON
    meta_data = json.dumps({"language": language, "state": {}})
    await db_execute_async(
        "INSERT INTO conversations (id, user_id, meta_data, created_at, updated_at) VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
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
        # Ensure properly decoded string for JSON loading
        if isinstance(cached_history, bytes):
            cached_history = cached_history.decode('utf-8')
        elif not isinstance(cached_history, str):
            cached_history = str(cached_history)
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

# Optimized history fetch with improved caching
async def get_history_cached(conversation_id: str, max_messages: int = 20) -> List[Dict[str, Any]]:
    """
    Optimized function to get conversation history with enhanced caching.
    Uses a TTL cache and batched fetching for better performance.
    """
    cache_key = f"history_opt:{conversation_id}"
    
    # Try to get from Redis cache first
    cached_history = REDIS_CLIENT.get(cache_key)
    if cached_history:
        # Ensure properly decoded string for JSON loading
        if isinstance(cached_history, bytes):
            cached_history = cached_history.decode('utf-8')
        elif not isinstance(cached_history, str):
            cached_history = str(cached_history)
        try:
            return json.loads(cached_history)
        except:
            # If cache parsing fails, proceed to fetch from database
            pass
    
    # Fetch from database using a more efficient query
    # Use a single query with ORDER BY and LIMIT
    query = """
    SELECT role, content 
    FROM conversation_messages 
    WHERE conversation_id = $1 
    ORDER BY message_index ASC 
    LIMIT $2
    """
    
    messages = await db_fetch_all_async(query, (conversation_id, max_messages))
    
    # Format messages for return
    history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    
    # Cache the result with a TTL
    try:
        REDIS_CLIENT.set(
            cache_key, 
            json.dumps(history),
            ex=600  # 10 minute cache
        )
    except Exception as e:
        logger.warning(f"Failed to cache conversation history: {e}")
    
    return history