import asyncio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from src.auth import verify_token
import json
from fastapi.responses import StreamingResponse, JSONResponse
import base64
import requests
from src.utils import (
    detect_language,
    get_or_create_conversation,
    db_fetch_all_async,
    get_conversation_history,
    add_message_to_conversation,
    db_fetch_one_async,
    db_execute_async,
    DateTimeEncoder,
    logger,
    get_db_pool,
    REDIS_CLIENT,
    Message,
    get_history_cached,
    redis_get_json,
    redis_set_json,
    safe_redis_decode,
    SHORT_CACHE_TTL,
    MEDIUM_CACHE_TTL,
    LONG_CACHE_TTL,
    EXTENDED_CACHE_TTL,
    get_memory_cache,
    set_memory_cache,
    strip_markdown,
)
from src.customer import CustomerState, customer_graph

# from src.tenant import TenantState, tenant_graph
from typing import Optional, List, Dict, Any
from langsmith import Client
from langsmith import trace
import os
from src.customer import populate_knowledge_graph
import uuid
from io import BytesIO
import functools
import concurrent.futures

# Create FastAPI app with custom documentation settings
app = FastAPI(
    title="Cenomi AI API",
    description="API for Cenomi AI Chatbot with enhanced conversational capabilities",
    version="1.0.0",
    docs_url="/docs",  # Enable automatic docs
    redoc_url="/redoc",  # Enable ReDoc
)

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
async def startup_event(token_payload: dict = Depends(verify_token)):
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

    class Config:
        schema_extra = {
            "example": {
                "text": "Where is the food court?",
                "user_id": "c_12345",
                "language": "en",
                "mall_id": 1,
                "include_tts": True,
            }
        }


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    audio_base64: Optional[str] = None
    recommendations: Optional[List[Dict[str, str]]] = None
    is_recommendation_format: bool = False
    follow_up_question: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "message": "The food court is located on the second floor, near the central atrium.",
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "recommendations": [
                    {"title": "View Food Court Map", "url": "/map/food-court"}
                ],
                "is_recommendation_format": True,
                "follow_up_question": "Would you like to know what restaurants are available there?",
            }
        }


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

    class Config:
        schema_extra = {
            "example": {
                "text": "Welcome to Cenomi Mall. How can I assist you today?",
                "language": "en",
                "speed": 1.0,
            }
        }


class TTSResponse(BaseModel):
    audio_base64: str
    media_type: str = "audio/mpeg"

    class Config:
        schema_extra = {
            "example": {
                "audio_base64": "base64_encoded_audio_data...",
                "media_type": "audio/mpeg",
            }
        }


class MallInfo(BaseModel):
    mall_id: str
    name_en: str

    class Config:
        schema_extra = {"example": {"mall_id": "1", "name_en": "Cenomi Mall Riyadh"}}


# Cache TTS responses with in-memory fast cache before Redis
TTS_CACHE = {}
TTS_CACHE_MAX_SIZE = 50

from pydub import AudioSegment
from io import BytesIO


async def generate_speech(
    text: str, language: str = "en", speed: Optional[float] = None
) -> bytes:
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
            "seed": 42,  # Fixed seed for deterministic output
        }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "text": clean_text,
        "model_id": (
            "eleven_monolingual_v1" if language == "en" else "eleven_multilingual_v2"
        ),
        "voice_settings": voice_settings,
    }

    def _make_request(text_segment):
        data["text"] = text_segment
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"TTS generation failed: {str(e)}"
            )

    try:
        # Log details for debugging
        logger.info(
            f"Generating speech for language: {language}, speed: {voice_settings.get('speed', 'default')}, text length: {len(clean_text)}"
        )

        if len(clean_text) > 4000:  # Updated threshold to 4000 characters
            segments = [
                clean_text[i : i + 4000] for i in range(0, len(clean_text), 4000)
            ]
            audio_segments = []
            for segment in segments:
                logger.info(
                    f"Generating segment: {segment[:20]}..., speed: {voice_settings.get('speed', 'default')}"
                )
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
        set_memory_cache(
            cache_key, audio_data, ttl=MEDIUM_CACHE_TTL
        )  # 5 minute memory cache

        return audio_data
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.post("/tts")
@app.post(
    "/tts",
    response_model=TTSResponse,
    tags=["Text-to-Speech"],
    summary="Convert text to speech",
    description="Converts the provided text to speech audio using ElevenLabs API. Supports English and Arabic languages.",
)
async def tts(request: TTSRequest, token_payload: dict = Depends(verify_token)):
    try:
        # Validate speed if provided
        if request.speed is not None:
            if not (0.7 <= request.speed <= 1.2):
                raise HTTPException(
                    status_code=400, detail="Speed must be between 0.7 and 1.2"
                )

        audio_data = await generate_speech(
            request.text, request.language, speed=request.speed
        )
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        return {"audio_base64": audio_base64, "media_type": "audio/mpeg"}
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.post("/login")
async def login(request: LoginRequest, token_payload: dict = Depends(verify_token)):
    tenant = await db_fetch_one_async(
        "SELECT tenant_id FROM tenants WHERE email ILIKE $1 AND password = $2",
        (request.email, request.password),
    )
    if tenant:
        return {"user_id": f"t_{tenant['tenant_id']}"}

    customer = await db_fetch_one_async(
        "SELECT customer_id FROM customers WHERE email ILIKE $1 AND password = $2",
        (request.email, request.password),
    )
    if customer:
        return {"user_id": f"c_{customer['customer_id']}"}

    raise HTTPException(status_code=401, detail="Invalid credentials")


async def create_conversation(user_id: Optional[str], language: str) -> str:
    conversation_id = str(uuid.uuid4())
    user_id_clean = (
        user_id[2:] if user_id and user_id.startswith(("t_", "c_")) else user_id
    )
    meta_data = json.dumps({"language": language, "state": {}})

    await db_execute_async(
        "INSERT INTO conversations (id, user_id, meta_data, created_at, updated_at) "
        "VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (conversation_id, user_id_clean, meta_data),
    )

    return conversation_id


async def add_message(conversation_id: str, role: str, content: str) -> None:
    max_index = await db_fetch_one_async(
        "SELECT COALESCE(MAX(message_index), -1) as max_idx FROM conversation_messages "
        "WHERE conversation_id = $1",
        (conversation_id,),
    )
    next_index = (
        (max_index["max_idx"] + 1)
        if max_index and max_index.get("max_idx") is not None
        else 0
    )
    message_id = str(uuid.uuid4())

    await db_execute_async(
        "INSERT INTO conversation_messages (id, conversation_id, role, content, message_index, created_at) "
        "VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)",
        (message_id, conversation_id, role, content, next_index),
    )

    await asyncio.to_thread(REDIS_CLIENT.delete, f"history:{conversation_id}")


async def get_history(
    conversation_id: str, max_messages: int = 20
) -> List[Dict[str, Any]]:
    cache_key = f"history:{conversation_id}"
    cached_history = REDIS_CLIENT.get(cache_key)
    if cached_history:
        # Handle potential type issues with Redis response
        if isinstance(cached_history, bytes):
            cached_history = cached_history.decode("utf-8")
        elif not isinstance(cached_history, str):
            # Convert to string if it's neither bytes nor string
            cached_history = str(cached_history)
        return json.loads(cached_history)

    messages = await db_fetch_all_async(
        "SELECT role, content, created_at as timestamp FROM conversation_messages "
        "WHERE conversation_id = $1 ORDER BY message_index ASC LIMIT $2",
        (conversation_id, max_messages),
    )

    history = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    REDIS_CLIENT.set(cache_key, json.dumps(history), ex=300)

    return history


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Process a chat message",
    description="Processes a user's chat message and returns an AI-generated response. Supports multiple languages and optional text-to-speech.",
)
async def chat(request: ChatRequest, token_payload: dict = Depends(verify_token)):
    try:
        text = request.text or ""

        # Log the request data
        logger.info(f"CHAT REQUEST: {json.dumps(request.dict(), default=str)}")

        # Use request-based caching for identical recent requests
        cache_key = f"chat:{hash(json.dumps(request.dict(), sort_keys=True))}"
        cached_response = get_memory_cache(cache_key)
        if cached_response:
            logger.info(
                f"CHAT RESPONSE (cached): {json.dumps(cached_response, default=str)}"
            )
            return ChatResponse(**cached_response)

        # Determine language asynchronously
        language = request.language
        if not language:

            def detect_lang():
                try:
                    return detect_language(text) or "en"
                except:
                    return "en"

            language = await asyncio.to_thread(detect_lang)

        # Log detected language
        logger.info(f"Detected language: {language}")

        # Create or get conversation and fetch initial data in parallel
        tasks = [
            get_or_create_conversation(
                request.conversation_id, request.user_id, language
            ),
        ]

        # Get the conversation ID first
        conversation_id = await tasks[0]

        # Now we can set up the history and metadata tasks
        history_task = get_history_cached(conversation_id)
        conv_data_task = db_fetch_one_async(
            "SELECT meta_data FROM conversations WHERE id = $1", (conversation_id,)
        )

        # Run these tasks in parallel
        history, conv_data = await asyncio.gather(history_task, conv_data_task)

        state_data = {}
        if conv_data and conv_data.get("meta_data"):
            try:
                meta_data = json.loads(conv_data["meta_data"])
                if "state" in meta_data:
                    state_data = meta_data["state"]
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error parsing meta_data JSON: {e}")

        # Remove keys that will be set directly
        for key in [
            "query",
            "user_id",
            "language",
            "conversation_id",
            "conversation_history",
            "mall_id",
        ]:
            state_data.pop(key, None)

        # Create customer state object
        state = CustomerState(
            query=text,
            user_id=request.user_id,
            language=language,
            conversation_id=conversation_id,
            conversation_history=history,
            mall_id=request.mall_id,
            **state_data,
        )

        # Process the request through the graph
        with trace(
            name="CustomerChat",
            inputs={
                "query": text,
                "user_id": request.user_id,
                "mall_id": request.mall_id,
            },
        ):
            result = await customer_graph.ainvoke(state)

            # Log response text
            logger.info(f"Response text: {result['response'][:50]}...")

            # If TTS is requested, start generating it with the appropriate speed
            if request.include_tts and result["response"]:
                tts_speed = 0.7 if language == "ar" else None
                tts_task = generate_speech(
                    result["response"], language, speed=tts_speed
                )
            else:
                tts_task = None

        # Prepare the response for caching
        response_data = {
            "message": result["response"],
            "conversation_id": conversation_id,
            "audio_base64": None,
        }

        # Add recommendation formatting if available
        if "response_format" in result and result["response_format"]:
            response_format = result["response_format"]
            if isinstance(response_format, dict):
                if "recommendations" in response_format:
                    response_data["recommendations"] = response_format[
                        "recommendations"
                    ]
                if "is_recommendation_format" in response_format:
                    response_data["is_recommendation_format"] = response_format[
                        "is_recommendation_format"
                    ]
                if "follow_up_question" in response_format:
                    response_data["follow_up_question"] = response_format[
                        "follow_up_question"
                    ]

        # Update the conversation with new messages and metadata
        tasks = [
            add_message(conversation_id, "user", text),
            add_message(conversation_id, "assistant", result["response"]),
        ]

        # Update history for metadata
        updated_history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": result["response"]},
        ]
        result["conversation_history"] = updated_history

        # Ensure response_format is serializable before storing in metadata
        if "response_format" in result and result["response_format"] is not None:
            if not isinstance(result["response_format"], dict):
                try:
                    result["response_format"] = dict(result["response_format"])
                except (TypeError, ValueError):
                    result["response_format"] = {
                        "response": result["response"],
                        "is_recommendation_format": False,
                    }

        # Update metadata
        meta_data = {"language": language, "state": result}
        update_meta_task = db_execute_async(
            "UPDATE conversations SET meta_data = $1 WHERE id = $2",
            (json.dumps(meta_data, cls=DateTimeEncoder), conversation_id),
        )

        tasks.append(update_meta_task)
        await asyncio.gather(*tasks)

        # Get TTS result if requested
        if tts_task:
            try:
                audio_data = await tts_task
                response_data["audio_base64"] = base64.b64encode(audio_data).decode(
                    "utf-8"
                )
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")

        # Cache the response for identical requests (short TTL)
        set_memory_cache(cache_key, response_data, ttl=SHORT_CACHE_TTL)

        # Log the response data (excluding large audio_base64 field)
        log_response = response_data.copy()
        if "audio_base64" in log_response:
            log_response["audio_base64"] = (
                "[TRUNCATED]" if log_response["audio_base64"] else None
            )
        logger.info(f"CHAT RESPONSE: {json.dumps(log_response, default=str)}")

        return ChatResponse(**response_data)
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")

        response_data = {
            "message": "I'm sorry, I'm having trouble understanding that right now. Could you try rephrasing your question?",
            "conversation_id": request.conversation_id or str(uuid.uuid4()),
            "recommendations": None,
            "is_recommendation_format": False,
            "follow_up_question": None,
            "audio_base64": None,
        }

        logger.error(f"CHAT ERROR DETAILS: {str(e)}")

        return ChatResponse(**response_data)


# @app.post("/tenant/update")
# async def tenant_update(request: UpdateRequest):
#     logger.info(f"Received tenant update request: {request.text}, user_id: {request.user_id}")

#     if not request.user_id.startswith("t_"):
#         logger.error("Non-tenant user attempted update")
#         raise HTTPException(status_code=403, detail="Only tenants can perform updates")

#     tenant_id = int(request.user_id[2:])
#     tenant = await db_fetch_one_async(
#         "SELECT tenant_id FROM tenants WHERE tenant_id = $1",
#         (tenant_id,)
#     )
#     if not tenant:
#         logger.error(f"Invalid tenant ID: {tenant_id}")
#         raise HTTPException(status_code=403, detail="Invalid tenant ID")

#     lang = request.language or "en"
#     conversation_id = await get_or_create_conversation(request.conversation_id, request.user_id, lang)
#     history = await get_history(conversation_id)

#     conv_state = await db_fetch_one_async(
#         "SELECT meta_data FROM conversations WHERE id = $1",
#         (conversation_id,)
#     )
#     state_dict = {}
#     if conv_state and conv_state.get("meta_data"):
#         try:
#             meta_data = json.loads(conv_state["meta_data"])
#             if "state" in meta_data:
#                 state_dict = meta_data["state"]
#         except (json.JSONDecodeError, ValueError) as e:
#             logger.error(f"Invalid meta_data for conversation {conversation_id}: {e}")

#     if state_dict:
#         try:
#             state = TenantState(**state_dict)
#             state.query = request.text
#             state.conversation_history = history
#         except (json.JSONDecodeError, ValueError):
#             logger.error(f"Invalid state data for conversation {conversation_id}, resetting to new state")
#             state = TenantState(
#                 query=request.text,
#                 user_id=request.user_id,
#                 language=lang,
#                 conversation_id=conversation_id,
#                 conversation_history=history
#             )
#     else:
#         state = TenantState(
#             query=request.text,
#             user_id=request.user_id,
#             language=lang,
#             conversation_id=conversation_id,
#             conversation_history=history
#         )

#     logger.info(f"Invoking tenant graph with query: {request.text}")
#     with trace(name="TenantUpdate", inputs={"query": request.text, "user_id": request.user_id}):
#         result = await tenant_graph.ainvoke(state)

#     meta_data = {"language": lang, "state": result}
#     meta_data_json = json.dumps(meta_data, cls=DateTimeEncoder)

#     # Perform these operations concurrently
#     tasks = [
#         db_execute_async(
#             "UPDATE conversations SET meta_data = $1 WHERE id = $2",
#             (meta_data_json, conversation_id)
#         ),
#         add_message(conversation_id, "user", request.text),
#         add_message(conversation_id, "assistant", result["response"]),
#     ]
#     await asyncio.gather(*tasks)

#     await asyncio.to_thread(REDIS_CLIENT.delete, f"history:{conversation_id}")

#     return {"message": result["response"], "conversation_id": conversation_id}


@app.get("/")
async def root(token_payload: dict = Depends(verify_token)):
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}


@app.get("/malls")
@app.get(
    "/malls",
    response_model=List[MallInfo],
    tags=["Malls"],
    summary="Get list of malls",
    description="Returns a list of all available malls with their IDs and names.",
)
async def get_malls(token_payload: dict = Depends(verify_token)):
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
    malls = await db_fetch_all_async(
        "SELECT unique_property_id as mall_id, marketing_name as name_en FROM malls"
    )
    result = [
        {"mall_id": str(mall["mall_id"]), "name_en": mall["name_en"]} for mall in malls
    ]

    # Cache the result with a long TTL
    redis_set_json("malls:list", result, ex=EXTENDED_CACHE_TTL)  # Cache for 24 hours
    set_memory_cache(
        "malls:list", result, ttl=LONG_CACHE_TTL
    )  # Cache in memory for 1 hour

    return result
