from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from utils import detect_language, get_or_create_session, get_conversation_history, add_message_to_conversation, db_fetch_one, db_execute_async, DateTimeEncoder, logger
from customer import CustomerState, customer_graph
from tenant import TenantState, tenant_graph
from typing import Optional
from langsmith import Client
# Option 1: Try importing trace directly from langsmith
from langsmith import trace
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    tenant = db_fetch_one("SELECT tenant_id FROM tenants WHERE email ILIKE %s AND password = %s", (request.email, request.password))
    if tenant:
        return {"user_id": f"t_{tenant['tenant_id']}"}
    
    customer = db_fetch_one("SELECT customer_id FROM customers WHERE email ILIKE %s AND password = %s", (request.email, request.password))
    if customer:
        return {"user_id": f"c_{customer['customer_id']}"}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    lang = request.language or detect_language(request.text)
    session_id = await get_or_create_session(request.session_id, request.user_id, lang)
    conversation_history = await get_conversation_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_history]
    
    conv_state = db_fetch_one("SELECT current_state FROM conversations WHERE session_id = %s", (session_id,))
    if conv_state and conv_state.get("current_state"):
        try:
            state_dict = conv_state["current_state"]
            state = CustomerState(**state_dict)
            state.query = request.text
            state.conversation_history = history_dicts
        except (json.JSONDecodeError, ValueError):
            logger.error(f"Invalid state data for session {session_id}, resetting to new state")
            state = CustomerState(query=request.text, user_id=request.user_id, language=lang, session_id=session_id, conversation_history=history_dicts)
    else:
        state = CustomerState(query=request.text, user_id=request.user_id, language=lang, session_id=session_id, conversation_history=history_dicts)

    
    with trace(name="CustomerChat", inputs={"query": request.text, "user_id": request.user_id}):
        result = await customer_graph.ainvoke(state)
        
    
    state_json = json.dumps(result, cls=DateTimeEncoder)
    await db_execute_async(
        "UPDATE conversations SET current_state = $1 WHERE session_id = $2",
        (state_json, session_id)
    )
    
    await add_message_to_conversation(session_id, "user", request.text)
    await add_message_to_conversation(session_id, "assistant", result["response"])
    return ChatResponse(message=result["response"], session_id=session_id)

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    if not request.user_id.startswith("t_"):
        raise HTTPException(status_code=403, detail="Only tenants can perform updates")
    
    tenant_id = request.user_id[2:]
    tenant = db_fetch_one("SELECT tenant_id FROM tenants WHERE tenant_id = %s", (tenant_id,))
    if not tenant:
        raise HTTPException(status_code=403, detail="Invalid tenant ID")
    
    lang = request.language or "en"
    session_id = await get_or_create_session(request.session_id, request.user_id, lang)
    
    conv_state = db_fetch_one("SELECT current_state FROM conversations WHERE session_id = %s", (session_id,))
    history = await get_conversation_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    
    if conv_state and conv_state.get("current_state"):
        try:
            state_dict = conv_state["current_state"]
            state = TenantState(**state_dict)
            state.query = request.text
            state.conversation_history = history_dicts
        except (json.JSONDecodeError, ValueError):
            logger.error(f"Invalid state data for session {session_id}, resetting to new state")
            state = TenantState(query=request.text, user_id=request.user_id, language=lang, session_id=session_id, conversation_history=history_dicts)
    else:
        state = TenantState(query=request.text, user_id=request.user_id, language=lang, session_id=session_id, conversation_history=history_dicts) 

    # Using imported trace
    with trace(name="TenantUpdate", inputs={"query": request.text, "user_id": request.user_id}):
        result = tenant_graph.invoke(state)
    
    state_json = json.dumps(result, cls=DateTimeEncoder)
    await db_execute_async("UPDATE conversations SET current_state = %s WHERE session_id = %s", (state_json, session_id))
    
    await add_message_to_conversation(session_id, "user", request.text)
    await add_message_to_conversation(session_id, "assistant", result["response"])
    
    return {"message": result["response"], "session_id": session_id}

@app.get("/")
async def root():
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}