import asyncio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from fastapi.responses import StreamingResponse
import base64
import requests
from utils import (
    detect_language, get_or_create_conversation, db_fetch_all_async, 
    get_conversation_history, add_message_to_conversation, db_fetch_one_async, 
    db_execute_async, DateTimeEncoder, logger, get_db_pool, REDIS_CLIENT,
    Message, get_history_cached, redis_get_json, redis_set_json, safe_redis_decode,
    SHORT_CACHE_TTL, MEDIUM_CACHE_TTL, LONG_CACHE_TTL, EXTENDED_CACHE_TTL,
    get_memory_cache, set_memory_cache, strip_markdown
)
from customer import CustomerState, customer_graph
from tenant import TenantState, tenant_graph
from typing import Optional, List, Dict, Any
from langsmith import Client
from langsmith import trace
import os
from customer import populate_knowledge_graph
import uuid
from io import BytesIO
import functools
import concurrent.futures

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable is not set")

# Thread pool for CPU-bound tasks
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    await get_db_pool()
    await populate_knowledge_graph()

# LangSmith setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "your_api_key_here")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "cenomi-bot")
langsmith_client = Client()

# Request and response models
class ChatRequest(BaseModel):
    text: Optional[str] = None
    audio: Optional[str] = None
    user_id: Optional[str] = None
    language: Optional[str] = None
    conversation_id: Optional[str] = None
    mall_id: Optional[int] = None
    include_tts: bool = False

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    audio_base64: Optional[str] = None
    recommendations: Optional[List[Dict[str, str]]] = None
    is_recommendation_format: bool = False
    follow_up_question: Optional[str] = None

class UpdateRequest(BaseModel):
    text: str
    user_id: str
    language: Optional[str] = None
    conversation_id: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speed: Optional[float] = None  # Optional speed parameter (0.7 to 1.2)

# Cache TTS responses with in-memory fast cache before Redis
TTS_CACHE = {}
TTS_CACHE_MAX_SIZE = 50

from pydub import AudioSegment
from io import BytesIO

async def generate_speech(text: str, language: str = "en", speed: Optional[float] = None) -> bytes:
    # Strip markdown symbols from text for TTS
    clean_text = strip_markdown(text)
    
    # Include speed in cache key to differentiate audio with different speeds
    cache_key = f"tts:{clean_text}:{language}:{speed or 'default'}"
    
    # Check memory cache first (fastest)
    mem_cached = get_memory_cache(cache_key)
    if mem_cached is not None:
        return mem_cached
    
    # Check Redis cache next
    redis_client = REDIS_CLIENT
    redis_cached = redis_client.get(cache_key)
    if redis_cached:
        # Cache hit - store in memory for future requests
        if isinstance(redis_cached, bytes):
            set_memory_cache(cache_key, redis_cached)
            return redis_cached
    
    # Cache miss - generate new audio
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default English voice (Rachel)
    voice_settings = {
        "stability": 0.5,
        "similarity_boost": 0.5,
    }
    
    if language == "ar":
        voice_id = "jsCqWAovK2LkecY7zXl4"  # Arabic voice (Salma, female)
        voice_settings = {
            "stability": 0.9,  # Increased for more consistency
            "similarity_boost": 0.9,  # Increased for more consistency
            "speed": speed if speed is not None else 0.7,  # Default to 0.8 for Arabic
            "seed": 42  # Fixed seed for deterministic output
        }
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "text": clean_text,
        "model_id": "eleven_monolingual_v1" if language == "en" else "eleven_multilingual_v2",
        "voice_settings": voice_settings
    }
    
    def _make_request(text_segment):
        data["text"] = text_segment
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    
    try:
        # Log details for debugging
        logger.info(f"Generating speech for language: {language}, speed: {voice_settings.get('speed', 'default')}, text length: {len(clean_text)}")
        
        if len(clean_text) > 4000:  # Updated threshold to 4000 characters
            segments = [clean_text[i:i+4000] for i in range(0, len(clean_text), 4000)]
            audio_segments = []
            for segment in segments:
                logger.info(f"Generating segment: {segment[:20]}..., speed: {voice_settings.get('speed', 'default')}")
                segment_audio_bytes = await asyncio.to_thread(_make_request, segment)
                segment_audio = AudioSegment.from_mp3(BytesIO(segment_audio_bytes))
                audio_segments.append(segment_audio)
            # Properly concatenate audio segments
            combined_audio = AudioSegment.empty()
            for segment in audio_segments:
                combined_audio += segment
            output = BytesIO()
            combined_audio.export(output, format="mp3")
            audio_data = output.getvalue()
        else:
            audio_data = await asyncio.to_thread(_make_request, clean_text)
        
        # Cache the result in both Redis and memory
        redis_client.set(cache_key, audio_data, ex=LONG_CACHE_TTL)  # 1 hour cache
        set_memory_cache(cache_key, audio_data, ttl=MEDIUM_CACHE_TTL)  # 5 minute memory cache
        
        return audio_data
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/tts")
async def tts(request: TTSRequest):
    try:
        # Validate speed if provided
        if request.speed is not None:
            if not (0.7 <= request.speed <= 1.2):
                raise HTTPException(status_code=400, detail="Speed must be between 0.7 and 1.2")
        
        audio_data = await generate_speech(request.text, request.language, speed=request.speed)
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        return {"audio_base64": audio_base64, "media_type": "audio/mpeg"}
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.post("/login")
async def login(request: LoginRequest):
    tenant = await db_fetch_one_async(
        "SELECT tenant_id FROM tenants WHERE email ILIKE $1 AND password = $2",
        (request.email, request.password)
    )
    if tenant:
        return {"user_id": f"t_{tenant['tenant_id']}"}
    
    customer = await db_fetch_one_async(
        "SELECT customer_id FROM customers WHERE email ILIKE $1 AND password = $2",
        (request.email, request.password)
    )
    if customer:
        return {"user_id": f"c_{customer['customer_id']}"}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

async def create_conversation(user_id: Optional[str], language: str) -> str:
    conversation_id = str(uuid.uuid4())
    user_id_clean = user_id[2:] if user_id and user_id.startswith(("t_", "c_")) else user_id
    meta_data = json.dumps({"language": language, "state": {}})
    
    await db_execute_async(
        "INSERT INTO conversations (id, user_id, meta_data, created_at, updated_at) "
        "VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (conversation_id, user_id_clean, meta_data)
    )
    
    return conversation_id

async def add_message(conversation_id: str, role: str, content: str) -> None:
    max_index = await db_fetch_one_async(
        "SELECT COALESCE(MAX(message_index), -1) as max_idx FROM conversation_messages "
        "WHERE conversation_id = $1",
        (conversation_id,)
    )
    next_index = (max_index["max_idx"] + 1) if max_index and max_index.get("max_idx") is not None else 0
    message_id = str(uuid.uuid4())
    
    await db_execute_async(
        "INSERT INTO conversation_messages (id, conversation_id, role, content, message_index, created_at) "
        "VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)",
        (message_id, conversation_id, role, content, next_index)
    )
    
    await asyncio.to_thread(REDIS_CLIENT.delete, f"history:{conversation_id}")

async def get_history(conversation_id: str, max_messages: int = 20) -> List[Dict[str, Any]]:
    cache_key = f"history:{conversation_id}"
    cached_history = REDIS_CLIENT.get(cache_key)
    if cached_history:
        # Handle potential type issues with Redis response
        if isinstance(cached_history, bytes):
            cached_history = cached_history.decode('utf-8')
        elif not isinstance(cached_history, str):
            # Convert to string if it's neither bytes nor string
            cached_history = str(cached_history)
        return json.loads(cached_history)
    
    messages = await db_fetch_all_async(
        "SELECT role, content, created_at as timestamp FROM conversation_messages "
        "WHERE conversation_id = $1 ORDER BY message_index ASC LIMIT $2",
        (conversation_id, max_messages)
    )
    
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]
    
    REDIS_CLIENT.set(
        cache_key, 
        json.dumps(history), 
        ex=300
    )
    
    return history

@app.post("/chat")
async def chat(request: ChatRequest):
    async def stream_response():
        text = request.text or ""
        cache_key = f"chat:{hash(json.dumps(request.dict(), sort_keys=True))}"
        cached_response = get_memory_cache(cache_key)
        if cached_response:
            yield cached_response["message"]
            return
        language = request.language
        if not language:
            def detect_lang():
                try:
                    return detect_language(text) or "en"
                except:
                    return "en"
            language = await asyncio.to_thread(detect_lang)
        tasks = [get_or_create_conversation(request.conversation_id, request.user_id, language)]
        conversation_id = await tasks[0]
        history_task = get_history_cached(conversation_id)
        conv_data_task = db_fetch_one_async(
            "SELECT meta_data FROM conversations WHERE id = $1",
            (conversation_id,)
        )
        history, conv_data = await asyncio.gather(history_task, conv_data_task)
        state_data = {}
        if conv_data and conv_data.get("meta_data"):
            try:
                meta_data = json.loads(conv_data["meta_data"])
                if "state" in meta_data:
                    state_data = meta_data["state"]
            except (json.JSONDecodeError, TypeError):
                pass
        for key in ['query', 'user_id', 'language', 'conversation_id', 'conversation_history', 'mall_id']:
            state_data.pop(key, None)
        state = CustomerState(
            query=text,
            user_id=request.user_id,
            language=language,
            conversation_id=conversation_id,
            conversation_history=history,
            mall_id=request.mall_id,
            **state_data
        )
        fallback_message = "I'm sorry, I'm having trouble understanding that right now. Could you try rephrasing your question?"
        try:
            # Timeout logic
            result = await asyncio.wait_for(customer_graph.ainvoke(state), timeout=10)
            response = result["response"]
        except asyncio.TimeoutError:
            response = fallback_message
        except Exception:
            response = fallback_message
        # Simulate streaming (token or chunk based)
        chunk_size = 20
        for i in range(0, len(response), chunk_size):
            yield response[i:i+chunk_size]
    return StreamingResponse(stream_response(), media_type="text/plain")

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    async def stream_response():
        fallback_message = "I'm sorry, I'm having trouble understanding that right now. Could you try rephrasing your question?"
        lang = request.language or "en"
        try:
            if not request.user_id.startswith("t_"):
                yield "Only tenants can perform updates"
                return
            tenant_id = int(request.user_id[2:])
            tenant = await db_fetch_one_async(
                "SELECT tenant_id FROM tenants WHERE tenant_id = $1",
                (tenant_id,)
            )
            if not tenant:
                yield "Invalid tenant ID"
                return
            conversation_id = await get_or_create_conversation(request.conversation_id, request.user_id, lang)
            history = await get_history(conversation_id)
            conv_state = await db_fetch_one_async(
                "SELECT meta_data FROM conversations WHERE id = $1",
                (conversation_id,)
            )
            state_dict = {}
            if conv_state and conv_state.get("meta_data"):
                try:
                    meta_data = json.loads(conv_state["meta_data"])
                    if "state" in meta_data:
                        state_dict = meta_data["state"]
                except (json.JSONDecodeError, ValueError):
                    pass
            if state_dict:
                try:
                    state = TenantState(**state_dict)
                    state.query = request.text
                    state.conversation_history = history
                except (json.JSONDecodeError, ValueError):
                    state = TenantState(
                        query=request.text,
                        user_id=request.user_id,
                        language=lang,
                        conversation_id=conversation_id,
                        conversation_history=history
                    )
            else:
                state = TenantState(
                    query=request.text,
                    user_id=request.user_id,
                    language=lang,
                    conversation_id=conversation_id,
                    conversation_history=history
                )
            try:
                result = await asyncio.wait_for(tenant_graph.ainvoke(state), timeout=5)
                response = result["response"]
            except asyncio.TimeoutError:
                response = fallback_message
            except Exception:
                response = fallback_message
            chunk_size = 20
            for i in range(0, len(response), chunk_size):
                yield response[i:i+chunk_size]
        except Exception:
            yield fallback_message
    return StreamingResponse(stream_response(), media_type="text/plain")

@app.get("/")
async def root():
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}

@app.get("/malls")
async def get_malls():
    # Use memory cache first
    mem_cached = get_memory_cache("malls:list")
    if mem_cached is not None:
        return mem_cached
    
    # Then try Redis cache
    cached_malls = redis_get_json("malls:list")
    if cached_malls:
        set_memory_cache("malls:list", cached_malls, ttl=LONG_CACHE_TTL)
        return cached_malls
    
    # Finally query database
    malls = await db_fetch_all_async("SELECT unique_property_id as mall_id, marketing_name as name_en FROM malls")
    result = [{"mall_id": str(mall["mall_id"]), "name_en": mall["name_en"]} for mall in malls]
    
    # Cache the result with a long TTL
    redis_set_json("malls:list", result, ex=EXTENDED_CACHE_TTL)  # Cache for 24 hours
    set_memory_cache("malls:list", result, ttl=LONG_CACHE_TTL)  # Cache in memory for 1 hour
    
    return result