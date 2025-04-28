import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from utils import (
    detect_language, get_or_create_conversation, db_fetch_all_async, 
    get_conversation_history, add_message_to_conversation, db_fetch_one_async, 
    db_execute_async, DateTimeEncoder, logger, get_db_pool, REDIS_CLIENT,
    Message  # Make sure to import Message class
)
from customer import CustomerState, customer_graph
from tenant import TenantState, tenant_graph
from typing import Optional, List, Dict, Any
from langsmith import Client
from langsmith import trace
import os
from customer import populate_knowledge_graph
import uuid

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

# Request and response models with consistent naming
class ChatRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    language: Optional[str] = None
    conversation_id: Optional[str] = None  # Changed from session_id
    mall_id: Optional[int] = None

class ChatResponse(BaseModel):
    message: str
    conversation_id: str  # Changed from session_id

class UpdateRequest(BaseModel):
    text: str
    user_id: str
    language: Optional[str] = None
    conversation_id: Optional[str] = None  # Changed from session_id

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

# Create a new conversation record
async def create_conversation(user_id: Optional[str], language: str) -> str:
    conversation_id = str(uuid.uuid4())
    user_id_clean = user_id[2:] if user_id and user_id.startswith(("t_", "c_")) else user_id
    
    # Store language in meta_data JSON
    meta_data = json.dumps({"language": language, "state": {}})
    
    await db_execute_async(
        "INSERT INTO conversations (id, user_id, meta_data, created_at, updated_at) "
        "VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        (conversation_id, user_id_clean, meta_data)
    )
    
    return conversation_id

# Add a message to the conversation history
async def add_message(conversation_id: str, role: str, content: str) -> None:
    # Get the next message index
    max_index = await db_fetch_one_async(
        "SELECT COALESCE(MAX(message_index), -1) as max_idx FROM conversation_messages "
        "WHERE conversation_id = $1",
        (conversation_id,)
    )
    next_index = (max_index["max_idx"] + 1) if max_index and max_index.get("max_idx") is not None else 0
    
    # Create a unique message ID
    message_id = str(uuid.uuid4())
    
    # Insert the message
    await db_execute_async(
        "INSERT INTO conversation_messages (id, conversation_id, role, content, message_index, created_at) "
        "VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)",
        (message_id, conversation_id, role, content, next_index)
    )
    
    # Invalidate cache
    await asyncio.to_thread(REDIS_CLIENT.delete, f"history:{conversation_id}")

# Get conversation history
async def get_history(conversation_id: str, max_messages: int = 20) -> List[Dict[str, Any]]:
    # Check cache first
    cache_key = f"history:{conversation_id}"
    cached_history = REDIS_CLIENT.get(cache_key)
    if cached_history:
        return json.loads(cached_history)
    
    # Fetch from database
    messages = await db_fetch_all_async(
        "SELECT role, content, created_at as timestamp FROM conversation_messages "
        "WHERE conversation_id = $1 ORDER BY message_index ASC LIMIT $2",
        (conversation_id, max_messages)
    )
    
    # Format the messages - convert timestamp to string
    history = [
        {"role": msg["role"], "content": msg["content"]}  # Remove timestamp
        for msg in messages
    ]
    
    # Cache the result without timestamps
    REDIS_CLIENT.set(
        cache_key, 
        json.dumps(history), 
        ex=300
    )
    
    return history

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.text}")
        
        # Set language based on request or detect from text
        language = request.language or detect_language(request.text)
        logger.info(f"Language detected: {language}")
        
        # Get or create conversation using utility function
        conversation_id = await get_or_create_conversation(request.conversation_id, request.user_id, language)
        logger.info(f"Using conversation ID: {conversation_id}")
        
        # Get conversation history
        history = await get_history(conversation_id)
        logger.info(f"Retrieved {len(history)} history items")
        
        # Create a fresh CustomerState for each request to avoid carrying over stale data
        try:
            # Always create a fresh state without previous response or context data
            state = CustomerState(
                query=request.text,
                user_id=request.user_id,
                language=language,
                conversation_id=conversation_id,
                conversation_history=history,
                mall_id=request.mall_id
            )
            
            logger.info(f"Created new state with mall_id: {request.mall_id}, query: {request.text}")
            
            # Process the request through the graph
            with trace(name="CustomerChat", inputs={"query": request.text, "user_id": request.user_id, "mall_id": request.mall_id}):
                logger.info("Invoking customer graph")
                result = await customer_graph.ainvoke(state)
                logger.info(f"Graph execution completed, intent: {result.get('intent')}")
            
            # Check if response was properly generated
            if not result.get("response"):
                logger.error("No response was generated by the graph")
                result["response"] = "I'm having trouble understanding your request. Could you please try again?"
            else:
                logger.info(f"Response generated: {result['response'][:50]}{'...' if len(result['response']) > 50 else ''}")
            
            # Add the new messages to the history
            await add_message(conversation_id, "user", request.text)
            await add_message(conversation_id, "assistant", result["response"])
            logger.info("Added messages to conversation history")
            
            # Update conversation history in the result
            updated_history = history + [
                {"role": "user", "content": request.text},
                {"role": "assistant", "content": result["response"]}
            ]
            result["conversation_history"] = updated_history
            
            # Save minimal state to meta_data (avoid saving large context data)
            save_state = {
                "user_id": result.get("user_id"),
                "language": result.get("language"),
                "conversation_id": result.get("conversation_id"),
                "mall_id": result.get("mall_id"),
                "intent": result.get("intent")
            }
            
            meta_data = {"language": language, "state": save_state}
            await db_execute_async(
                "UPDATE conversations SET meta_data = $1 WHERE id = $2",
                (json.dumps(meta_data, cls=DateTimeEncoder), conversation_id)
            )
            logger.info("Updated conversation meta_data")
            
            # Clear Redis cache to ensure fresh responses
            await asyncio.to_thread(REDIS_CLIENT.delete, f"history:{conversation_id}")
            logger.info("Cleared Redis history cache")
            
            # Return the response
            return ChatResponse(
                message=result["response"], 
                conversation_id=conversation_id
            )
        
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    logger.info(f"Getting or creating conversation for user_id: {request.user_id}, lang: {lang}")
    
    # Get or create conversation using utility function
    conversation_id = await get_or_create_conversation(request.conversation_id, request.user_id, lang)
    
    logger.info(f"Fetching conversation state for conversation_id: {conversation_id}")
    conv_state = await db_fetch_one_async(
        "SELECT meta_data FROM conversations WHERE id = $1",
        (conversation_id,)
    )
    history = await get_history(conversation_id)
    
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

    # Process the request
    logger.info(f"Invoking tenant graph with query: {request.text}")
    with trace(name="TenantUpdate", inputs={"query": request.text, "user_id": request.user_id}):
        result = await tenant_graph.ainvoke(state)
    
    logger.info(f"Tenant graph result: {result['response']}")
    # Save updated state in meta_data
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