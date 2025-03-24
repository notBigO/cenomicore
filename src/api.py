import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from utils import detect_language, get_or_create_session, db_fetch_all_async, get_conversation_history, add_message_to_conversation, db_fetch_one_async, db_execute_async, DateTimeEncoder, logger, get_db_pool, REDIS_CLIENT
from customer import CustomerState, customer_graph
from tenant import TenantState, tenant_graph
from typing import Optional
from langsmith import Client
from langsmith import trace
import os
from customer import populate_knowledge_graph

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application...")
    # Pre-initialize the asyncpg pool
    await get_db_pool()
    await populate_knowledge_graph()

# LangSmith setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "your_api_key_here")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "cenomi-bot")
langsmith_client = Client()

class ChatRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    language: Optional[str] = None
    session_id: Optional[str] = None
    mall_id: Optional[int] = None

class ChatResponse(BaseModel):
    message: str
    session_id: str

class UpdateRequest(BaseModel):
    text: str
    user_id: str
    language: Optional[str] = None
    session_id: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    lang = request.language or detect_language(request.text)

    # Ensure session exists in the conversations table
    session_id = await get_or_create_session(request.session_id, request.user_id, lang)

    # Verify session exists in the database
    session_check = await db_fetch_one_async(
        "SELECT session_id FROM conversations WHERE session_id = $1",
        (session_id,)
    )
    if not session_check:
        await db_execute_async(
            "INSERT INTO conversations (session_id, user_id, language, current_state) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
            (session_id, request.user_id, lang, json.dumps({})),
        )
        logger.info(f"Created new session in conversations table: {session_id}")

    # Fetch conversation history
    conversation_history = await get_conversation_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_history]

    # Load or initialize state
    conv_state = await db_fetch_one_async(
        "SELECT current_state FROM conversations WHERE session_id = $1",
        (session_id,),
    )
    if conv_state and conv_state.get("current_state"):
        try:
            state_dict = json.loads(conv_state["current_state"])
            state_dict.setdefault("query", request.text)
            state_dict.setdefault("language", lang)
            state_dict.setdefault("session_id", session_id)
            state = CustomerState(**state_dict)
            state.query = request.text
            state.user_id = request.user_id
            state.conversation_history = history_dicts
            state.mall_id = request.mall_id
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid state data for session {session_id}, resetting to new state: {e}")
            state = CustomerState(
                query=request.text,
                user_id=request.user_id,
                language=lang,
                session_id=session_id,
                conversation_history=history_dicts,
                mall_id=request.mall_id,
            )
    else:
        state = CustomerState(
            query=request.text,
            user_id=request.user_id,
            language=lang,
            session_id=session_id,
            conversation_history=history_dicts,
            mall_id=request.mall_id,
        )

    # Process the request
    with trace(name="CustomerChat", inputs={"query": request.text, "user_id": request.user_id, "mall_id": request.mall_id}):
        result = await customer_graph.ainvoke(state)

    # Update conversation history in the result
    result["conversation_history"] = history_dicts + [
        {"role": "user", "content": request.text},
        {"role": "assistant", "content": result["response"]}
    ]

    # Save updated state
    state_json = json.dumps(result, cls=DateTimeEncoder)
    await db_execute_async(
        "UPDATE conversations SET current_state = $1 WHERE session_id = $2",
        (state_json, session_id),
    )

    # Add messages to conversation_messages table
    await add_message_to_conversation(session_id, "user", request.text)
    await add_message_to_conversation(session_id, "assistant", result["response"])

    await asyncio.to_thread(REDIS_CLIENT.delete, f"history:{session_id}")

    return ChatResponse(message=result["response"], session_id=session_id)

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    logger.info(f"Received tenant update request: {request.text}, user_id: {request.user_id}")
    
    if not request.user_id.startswith("t_"):
        logger.error("Non-tenant user attempted update")
        raise HTTPException(status_code=403, detail="Only tenants can perform updates")
    
    tenant_id = int(request.user_id[2:])  # Convert string to integer
    tenant = await db_fetch_one_async(
        "SELECT tenant_id FROM tenants WHERE tenant_id = $1",
        (tenant_id,)  # Pass as int, not str
    )
    if not tenant:
        logger.error(f"Invalid tenant ID: {tenant_id}")
        raise HTTPException(status_code=403, detail="Invalid tenant ID")
    
    lang = request.language or "en"
    logger.info(f"Getting or creating session for user_id: {request.user_id}, lang: {lang}")
    session_id = await get_or_create_session(request.session_id, request.user_id, lang)
    
    logger.info(f"Fetching conversation state for session_id: {session_id}")
    conv_state = await db_fetch_one_async(
        "SELECT current_state FROM conversations WHERE session_id = $1",
        (session_id,)
    )
    history = await get_conversation_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    
    if conv_state and conv_state.get("current_state"):
        try:
            state_dict = json.loads(conv_state["current_state"])
            state = TenantState(**state_dict)
            state.query = request.text
            state.conversation_history = history_dicts
        except (json.JSONDecodeError, ValueError):
            logger.error(f"Invalid state data for session {session_id}, resetting to new state")
            state = TenantState(
                query=request.text,
                user_id=request.user_id,
                language=lang,
                session_id=session_id,
                conversation_history=history_dicts
            )
    else:
        state = TenantState(
            query=request.text,
            user_id=request.user_id,
            language=lang,
            session_id=session_id,
            conversation_history=history_dicts
        )

    # Process the request
    logger.info(f"Invoking tenant graph with query: {request.text}")
    with trace(name="TenantUpdate", inputs={"query": request.text, "user_id": request.user_id}):
        result = await tenant_graph.ainvoke(state)
    
    logger.info(f"Tenant graph result: {result['response']}")
    state_json = json.dumps(result, cls=DateTimeEncoder)
    await db_execute_async(
        "UPDATE conversations SET current_state = $1 WHERE session_id = $2",
        (state_json, session_id)
    )
    
    await add_message_to_conversation(session_id, "user", request.text)
    await add_message_to_conversation(session_id, "assistant", result["response"])

    await asyncio.to_thread(REDIS_CLIENT.delete, f"history:{session_id}")
    
    return {"message": result["response"], "session_id": session_id}

@app.get("/")
async def root():
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}

@app.get("/malls")
async def get_malls():
    malls = await db_fetch_all_async("SELECT mall_id, name_en FROM malls")
    return [{"mall_id": str(mall["mall_id"]), "name_en": mall["name_en"]} for mall in malls]