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
    Message
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

async def generate_speech(text: str, language: str = "en") -> bytes:
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1" if language == "en" else "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/tts")
async def tts(request: TTSRequest):
    try:
        audio_data = await generate_speech(request.text, request.language)
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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        text = request.text or ""
        
        language = request.language or detect_language(text) or "en"
        conversation_id = await get_or_create_conversation(request.conversation_id, request.user_id, language)
        history = await get_history(conversation_id)

        conv_data = await db_fetch_one_async(
            "SELECT meta_data FROM conversations WHERE id = $1",
            (conversation_id,)
        )
        state_data = {}
        if conv_data and conv_data.get("meta_data"):
            meta_data = json.loads(conv_data["meta_data"])
            if "state" in meta_data:
                state_data = meta_data["state"]

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

        with trace(name="CustomerChat", inputs={"query": text, "user_id": request.user_id, "mall_id": request.mall_id}):
            result = await customer_graph.ainvoke(state)

        await add_message(conversation_id, "user", text)
        await add_message(conversation_id, "assistant", result["response"])

        updated_history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": result["response"]}
        ]
        result["conversation_history"] = updated_history

        meta_data = {"language": language, "state": result}
        await db_execute_async(
            "UPDATE conversations SET meta_data = $1 WHERE id = $2",
            (json.dumps(meta_data, cls=DateTimeEncoder), conversation_id)
        )

        audio_base64 = None
        if request.include_tts and result["response"]:
            audio_data = await generate_speech(result["response"], language)
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return ChatResponse(
            message=result["response"],
            conversation_id=conversation_id,
            audio_base64=audio_base64
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    logger.info(f"Received tenant update request: {request.text}, user_id: {request.user_id}")
    
    if not request.user_id.startswith("t_"):
        logger.error("Non-tenant user attempted update")
        raise HTTPException(status_code=403, detail="Only tenants can perform updates")
    
    tenant_id = int(request.user_id[2:])
    tenant = await db_fetch_one_async(
        "SELECT tenant_id FROM tenants WHERE tenant_id = $1",
        (tenant_id,)
    )
    if not tenant:
        logger.error(f"Invalid tenant ID: {tenant_id}")
        raise HTTPException(status_code=403, detail="Invalid tenant ID")
    
    lang = request.language or "en"
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
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid meta_data for conversation {conversation_id}: {e}")
    
    if state_dict:
        try:
            state = TenantState(**state_dict)
            state.query = request.text
            state.conversation_history = history
        except (json.JSONDecodeError, ValueError):
            logger.error(f"Invalid state data for conversation {conversation_id}, resetting to new state")
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

    logger.info(f"Invoking tenant graph with query: {request.text}")
    with trace(name="TenantUpdate", inputs={"query": request.text, "user_id": request.user_id}):
        result = await tenant_graph.ainvoke(state)
    
    meta_data = {"language": lang, "state": result}
    meta_data_json = json.dumps(meta_data, cls=DateTimeEncoder)
    await db_execute_async(
        "UPDATE conversations SET meta_data = $1 WHERE id = $2",
        (meta_data_json, conversation_id)
    )
    
    await add_message(conversation_id, "user", request.text)
    await add_message(conversation_id, "assistant", result["response"])

    await asyncio.to_thread(REDIS_CLIENT.delete, f"history:{conversation_id}")
    
    return {"message": result["response"], "conversation_id": conversation_id}

@app.get("/")
async def root():
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}

@app.get("/malls")
async def get_malls():
    malls = await db_fetch_all_async("SELECT unique_property_id as mall_id, marketing_name as name_en FROM malls")
    return [{"mall_id": str(mall["mall_id"]), "name_en": mall["name_en"]} for mall in malls]