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
import re
from urllib.parse import urlparse

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

# Redis setup with connection pooling
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', '6379'))
redis_url = os.getenv('RENDER_REDIS_INTERNAL_URI', None)

# Initialize Redis connection pool
if redis_url:
    try:
        REDIS_POOL = redis.ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=10
        )
        parsed_url = urlparse(redis_url)
        logger.info(f"Using Redis from RENDER_REDIS_INTERNAL_URI: {parsed_url.hostname}:{parsed_url.port}")
    except Exception as e:
        logger.warning(f"Failed to parse Redis URL: {e}, falling back to environment variables")
        REDIS_POOL = redis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=0,
            decode_responses=True,
            max_connections=10
        )
else:
    REDIS_POOL = redis.ConnectionPool(
        host=redis_host,
        port=redis_port,
        db=0,
        decode_responses=True,
        max_connections=10
    )

def get_redis_client():
    return redis.Redis(connection_pool=REDIS_POOL)

REDIS_CLIENT = get_redis_client()

# Cache TTLs
SHORT_CACHE_TTL = 60  # 1 minute
MEDIUM_CACHE_TTL = 300  # 5 minutes
LONG_CACHE_TTL = 3600  # 1 hour
EXTENDED_CACHE_TTL = 86400  # 24 hours

# In-memory LRU cache for frequently accessed data
# This reduces Redis network calls for hot data
MEMORY_CACHE = {}
MEMORY_CACHE_MAX_SIZE = 100
MEMORY_CACHE_TTL = 60  # 1 minute

def set_memory_cache(key, value, ttl=MEMORY_CACHE_TTL):
    """Set a value in the memory cache with expiration"""
    now = datetime.now().timestamp()
    MEMORY_CACHE[key] = (value, now + ttl)
    
    # Clean up cache if it's too large
    if len(MEMORY_CACHE) > MEMORY_CACHE_MAX_SIZE:
        # Remove expired items
        current_time = now
        expired_keys = [k for k, v in MEMORY_CACHE.items() if v[1] < current_time]
        for k in expired_keys:
            MEMORY_CACHE.pop(k, None)
        
        # If still too large, remove oldest items
        if len(MEMORY_CACHE) > MEMORY_CACHE_MAX_SIZE:
            items = sorted(MEMORY_CACHE.items(), key=lambda x: x[1][1])
            to_remove = items[:len(items) // 4]  # Remove 25% of oldest items
            for k, _ in to_remove:
                MEMORY_CACHE.pop(k, None)

def get_memory_cache(key):
    """Get a value from memory cache if it exists and hasn't expired"""
    if key in MEMORY_CACHE:
        value, expiry = MEMORY_CACHE[key]
        if datetime.now().timestamp() < expiry:
            return value
        else:
            MEMORY_CACHE.pop(key, None)
    return None

# Helper function to safely decode Redis responses
def safe_redis_decode(value):
    """Safely decode a Redis response to a string"""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8')
    elif not isinstance(value, str):
        return str(value)
    return value

# Helper function to handle Redis JSON serialization/deserialization
def redis_get_json(key):
    """Get a JSON value from Redis with safe decoding"""
    redis_client = get_redis_client()
    
    # Check memory cache first
    mem_cached = get_memory_cache(f"json:{key}")
    if mem_cached is not None:
        return mem_cached
    
    # Try Redis if not in memory cache
    value = redis_client.get(key)
    if value is None:
        return None
    
    try:
        value = safe_redis_decode(value)
        if value is None:
            return None
        result = json.loads(value)
        # Store in memory cache
        set_memory_cache(f"json:{key}", result)
        return result
    except (json.JSONDecodeError, TypeError):
        return None

def redis_set_json(key, value, ex=MEDIUM_CACHE_TTL):
    """Set a JSON value in Redis with expiry"""
    redis_client = get_redis_client()
    try:
        json_value = json.dumps(value)
        redis_client.set(key, json_value, ex=ex)
        # Update memory cache
        set_memory_cache(f"json:{key}", value)
        return True
    except Exception as e:
        logger.warning(f"Error setting Redis JSON value: {e}")
        return False

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
    Uses memory cache -> Redis cache -> database with efficient querying.
    """
    cache_key = f"history_opt:{conversation_id}"
    
    # Try memory cache first (fastest)
    mem_cached = get_memory_cache(cache_key)
    if mem_cached is not None:
        return mem_cached
    
    # Try Redis cache next
    cached_history = redis_get_json(cache_key)
    if cached_history:
        # Store in memory cache for future quick access
        set_memory_cache(cache_key, cached_history)
        return cached_history
    
    # Fetch from database using a more efficient query
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
    redis_set_json(cache_key, history, ex=LONG_CACHE_TTL)
    set_memory_cache(cache_key, history)
    
    return history

def strip_markdown(text: str) -> str:
    """Remove markdown formatting symbols from text for TTS"""
    # Replace bold text (**text**) with just the text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Replace italic text (*text*) with just the text
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    return text