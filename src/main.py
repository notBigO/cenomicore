from fastapi import FastAPI, HTTPException
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import google.generativeai as genai
import psycopg2
import json
import logging
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel as PydanticBaseModel
from langdetect import detect
import uuid

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

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("cenomi")

# Sentence transformer for embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

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


def db_fetch_one(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return dict(zip(columns, row)) if row else None
    except Exception as e:
        logger.error(f"Database error: {e}")
        return None

def db_fetch_all(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []

def db_execute(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database error: {e}")
        return False

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

# Conversation memory functions
def get_or_create_session(session_id: Optional[str], user_id: Optional[str], language: str) -> str:
    if session_id:
        # Update the session's last activity time
        db_execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
            (session_id,)
        )
        return session_id
    else:
        # Create a new session
        new_session_id = str(uuid.uuid4())
        db_execute(
            "INSERT INTO conversations (session_id, user_id, language) VALUES (%s, %s, %s)",
            (new_session_id, user_id, language)
        )
        return new_session_id

def add_message_to_conversation(session_id: str, role: str, content: str):
    db_execute(
        "INSERT INTO conversation_messages (session_id, role, content) VALUES (%s, %s, %s)",
        (session_id, role, content)
    )

def get_conversation_history(session_id: str, max_messages: int = 10) -> List[Message]:
    messages = db_fetch_all(
        "SELECT role, content, timestamp FROM conversation_messages "
        "WHERE session_id = %s ORDER BY timestamp DESC LIMIT %s",
        (session_id, max_messages)
    )
    # Return in chronological order
    return [
        Message(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
        for msg in reversed(messages)
    ]

# Customer prompt template
customer_prompt = PromptTemplate(
    input_variables=["context", "query", "lang", "conversation_history"],
    template="""
    You are CenomiAI, a friendly and knowledgeable mall assistant. Respond conversationally in {lang}, using emojis 😊 to maintain a warm and engaging tone. 
    Your purpose is to assist customers with all mall-related inquiries, including stores, events, offers, loyalty programs, services, dining, navigation, and more. 
    Use the provided query, context, and conversation history to deliver accurate, detailed, and helpful responses. If the context lacks sufficient information, ask clarifying questions or offer further assistance while keeping the tone supportive.

    ### Key Guidelines for Responses:
    - **Store-Specific Queries**: 
      - Provide detailed information such as store name, exact location (e.g., floor, nearby landmarks), opening hours, contact details, and specific offerings (e.g., products, services, or amenities like cafes inside stores).
      - For questions like "Are there any [type] stores?", confirm their presence and list relevant examples with locations if available.

    - **General Store Category Queries**: 
      - Suggest stores based on categories (e.g., formal wear, toys, home decor) or customer needs (e.g., tailoring, plus-size clothing).
      - Offer multiple options when possible and tailor suggestions to specific preferences (e.g., budget, age group).

    - **Offers and Promotions**: 
      - Share details on current and past offers (even if expired), including discounts, bundle deals, or loyalty-specific promotions.
      - Specify applicable stores, product types, and conditions when available.

    - **Events**: 
      - Provide information on past, current, and upcoming events, including names, dates, times, locations, and descriptions.
      - Address queries about specific event types (e.g., workshops, kids' activities) or seasonal festivities.

    - **Mall Navigation and Amenities**: 
      - Offer clear directions to amenities (e.g., restrooms, ATMs, prayer rooms) or key areas (e.g., food court, parking).
      - Answer questions about mall policies (e.g., pets, smoking areas), accessibility, and safety features.

    - **Food and Dining**: 
      - Recommend dining options based on cuisine, dietary preferences (e.g., vegan, healthy), location (e.g., near cinema), or ambiance (e.g., family-friendly, outdoor seating).
      - Include details like operating hours, menu highlights, or specific dishes when relevant.

    - **Loyalty Programs**: 
      - If a `user_id` is provided, share personalized details (e.g., points balance, redemption options) using `customer_loyalty` and `loyalty_programs` data.
      - Explain program rules, tiers, earning methods, and terms (e.g., expiration, transfers) when asked.

    - **Personalized Recommendations**: 
      - Offer tailored suggestions based on interests (e.g., fashion, gifts), constraints (e.g., budget, time), or group needs (e.g., family-friendly activities).
      - For vague queries, ask clarifying questions or provide a variety of general options.

    - **Problem Solving and Assistance**: 
      - Guide customers through issues like lost items, complaints, or emergencies (e.g., lost child), providing actionable steps and contact details (e.g., security, lost and found).
      - Address safety concerns, accessibility needs, or mall policies with clear instructions.

    - **Product-Specific Queries**: 
      - Respond to questions about specific products, brands, or availability (e.g., "Does the Apple store have the iPhone 17?") with store names and details if known.

    - **Temporal Queries**: 
      - Provide information on mall hours, peak times, holiday schedules, or late-night shopping when requested.

    - **Open-Ended or Vague Queries**: 
      - For queries like "What's good here?", ask follow-up questions (e.g., "Are you looking for shopping, dining, or entertainment?") or offer a broad range of popular options.

    ### Instructions:
    - **Context Usage**: If the context contains relevant information, use it directly to craft your response. Quote specifics (e.g., store locations, event times) when possible.
    - **Conversation History**: Use the conversation history to maintain context. Reference previous questions and your answers when appropriate.
    - **Insufficient Context**: If the context is empty or lacks details, respond with, "I couldn't find that info right now 😞, but I'll keep looking! Can you give me more details to help me assist you better?"
    - **Tone**: Keep responses conversational, concise yet detailed, and customer-focused. Avoid technical jargon unless necessary.
    - **Read-Only**: This is a READ-ONLY chat. Do not offer to update information or suggest actions beyond providing assistance based on existing data.
    - **Continuity**: If the user is following up on a previous question with pronouns like "it", "they", "that store", etc., use conversation history to understand what they're referring to.

    ### Current Query:
    "{query}"

    ### Context:
    {context}
    
    ### Conversation History:
    {conversation_history}
    """
)

# LLM and chain for customer queries
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)
customer_chain = customer_prompt | llm | StrOutputParser()

# Customer state
class CustomerState(PydanticBaseModel):
    query: str
    user_id: Optional[str] = None
    language: str
    session_id: str
    conversation_history: List[Dict[str, str]] = []
    response: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None

# Customer workflow nodes
def retrieve_context(state: CustomerState) -> CustomerState:
    # Detect mall if mentioned
    mall_name = None
    if "nakheel mall" in state.query.lower():
        mall_name = "Nakheel Mall"
    filter = None
    if mall_name:
        mall = db_fetch_one("SELECT mall_id FROM malls WHERE name_en ILIKE %s", (mall_name,))
        if mall:
            filter = {"mall_id": mall["mall_id"]}
    
    # Check conversation history for entity references
    query = state.query
    history = state.conversation_history
    
    # Example of resolving references from history
    if history and any(word in query.lower() for word in ["it", "they", "that", "this", "there", "those"]):
        # Use the last assistant message as context
        last_exchanges = [msg for msg in history[-4:] if msg["role"] == "assistant"]
        if last_exchanges:
            query = f"{query} (This is a follow-up to: {last_exchanges[-1]['content']})"
    
    # Embed the query and search Pinecone
    query_vector = embeddings.embed_query(query)
    search_params = {"vector": query_vector, "top_k": 10, "include_metadata": True}
    if filter:
        search_params["filter"] = filter
    
    results = index.query(**search_params)
    docs = results["matches"] if "matches" in results else []
    
    context = {
        "stores": [], "offers": [], "events": [], "services": [], "amenities": [],
        "loyalty_programs": [], "customer_loyalty": []
    }
    for doc in docs:
        metadata = doc["metadata"]
        doc_type = metadata.get("type")
        if doc_type == "store":
            context["stores"].append({
                "name": metadata.get("name_en"),
                "location": metadata.get("location_en"),
                "category": metadata.get("category_en")
            })
        elif doc_type == "offer":
            context["offers"].append({
                "description": metadata.get("description_en"),
                "id": metadata.get("id")
            })
        elif doc_type == "event":
            context["events"].append({
                "name": metadata.get("name_en"),
                "date": metadata.get("start_time"),
                "location": metadata.get("location_en")
            })
        elif doc_type == "service":
            context["services"].append({
                "name": metadata.get("name_en"),
                "description": metadata.get("description_en")
            })
        elif doc_type == "amenity":
            context["amenities"].append({
                "name": metadata.get("name_en"),
                "location": metadata.get("location_en")
            })
    
    if state.user_id:
        loyalty = db_fetch_one(
            "SELECT cl.points_balance, lp.name_en, lp.description_en "
            "FROM customer_loyalty cl "
            "JOIN loyalty_programs lp ON cl.loyalty_id = lp.loyalty_id "
            "WHERE cl.customer_id = %s",
            (state.user_id,)
        )
        context["customer_loyalty"] = loyalty or {}
    
    state.context_data = context
    state.response = json.dumps(convert_to_json_safe(context))
    return state

def generate_response(state: CustomerState) -> CustomerState:
    # Format conversation history for prompt
    formatted_history = ""
    if state.conversation_history:
        formatted_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in state.conversation_history[-6:]  # Include last 6 messages
        ])
    
    response = customer_chain.invoke(
        {
            "context": state.response,
            "query": state.query,
            "lang": state.language,
            "conversation_history": formatted_history,
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
    )
    if any(keyword in state.query.lower() for keyword in ["add", "update", "delete"]):
        response = "This chat is read-only. Customers cannot make updates."
    state.response = response
    return state

# Customer workflow
customer_workflow = StateGraph(CustomerState)
customer_workflow.add_node("retrieve", retrieve_context)
customer_workflow.add_node("respond", generate_response)
customer_workflow.set_entry_point("retrieve")
customer_workflow.add_edge("retrieve", "respond")
customer_workflow.add_edge("respond", END)
customer_graph = customer_workflow.compile()

# Tenant update state
class TenantUpdateState(PydanticBaseModel):
    query: str
    user_id: str
    language: str
    session_id: str
    conversation_history: List[Dict[str, str]] = []
    intent: Optional[str] = None
    data: Dict[str, Any] = {}
    response: Optional[str] = None

# Tenant workflow nodes
def recognize_intent(state: TenantUpdateState) -> TenantUpdateState:
    # Check conversation history for context
    history = state.conversation_history
    query = state.query
    
    # If this is a follow-up question, add context from previous interactions
    if history and any(word in query.lower() for word in ["it", "this", "that", "them", "those"]):
        last_exchanges = [msg for msg in history[-4:]]
        if last_exchanges:
            context_messages = " ".join([msg["content"] for msg in last_exchanges])
            query = f"{query} (Based on previous context: {context_messages})"
    
    prompt = f"""
    You are CenomiAI, a tenant assistant. Parse the query into a JSON object for store/offer/product management:
    - Actions: 'read_store', 'add_offer', 'update_offer', 'delete_offer', 'update_store', 'update_product'
    - Include 'store', 'description', 'start_date', 'end_date', 'field', 'value' as needed.
    Query: "{query}"
    Return JSON wrapped in ```json``` markers.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt).text.strip()
    json_str = response.split("```json")[1].split("```")[0].strip()
    parsed = json.loads(json_str)
    state.intent = parsed.get("action", "general_query")
    state.data.update(parsed)
    return state

def process_update(state: TenantUpdateState) -> TenantUpdateState:
    stores = db_fetch_all("SELECT store_id, name_en FROM stores WHERE tenant_id = %s", (state.user_id,))
    store_name = state.data.get("store")
    store = next((s for s in stores if s["name_en"].lower() == store_name.lower()), None) if store_name else None
    
    if not store and "store" not in state.data:
        state.response = f"Please specify a store: {', '.join(s['name_en'] for s in stores)}."
        return state
    
    if store:
        state.data["store_id"] = store["store_id"]
        state.data["store_name"] = store["name_en"]

    if state.intent == "add_offer":
        desc = state.data.get("description")
        if not desc:
            state.response = "Please provide an offer description (e.g., '20% off laptops')."
            return state
        start_date = state.data.get("start_date", "CURRENT_DATE")
        end_date = state.data.get("end_date", "CURRENT_DATE + INTERVAL '7 days'")
        db_execute(
            "INSERT INTO offers (store_id, description_en, description_ar, start_date, end_date) VALUES (%s, %s, %s, %s, %s)",
            (state.data["store_id"], desc, desc, start_date, end_date)
        )
        offer_id = db_fetch_one("SELECT currval(pg_get_serial_sequence('offers', 'offer_id')) AS id")["id"]
        text = desc
        vector = embeddings.embed_query(text)
        index.upsert(vectors=[
            {"id": f"offer_{offer_id}_en", "values": vector, "metadata": {
                "type": "offer", "id": offer_id, "description_en": desc, "lang": "en"
            }}
        ])
        state.response = f"Added '{desc}' to {state.data['store_name']}!"

    elif state.intent == "update_offer":
        offers = db_fetch_all("SELECT offer_id, description_en FROM offers WHERE store_id = %s", (state.data["store_id"],))
        offer = next((o for o in offers if state.data.get("description", "").lower() in o["description_en"].lower()), None)
        if not offer:
            state.response = f"No offer found matching '{state.data.get('description')}'."
        else:
            value = state.data.get("value")
            db_execute(
                "UPDATE offers SET description_en = %s, description_ar = %s WHERE offer_id = %s",
                (value, value, offer["offer_id"])
            )
            text = value
            vector = embeddings.embed_query(text)
            index.upsert(vectors=[
                {"id": f"offer_{offer['offer_id']}_en", "values": vector, "metadata": {
                    "type": "offer", "id": offer["offer_id"], "description_en": value, "lang": "en"
                }}
            ])
            state.response = f"Updated offer to '{value}'!"

    elif state.intent == "delete_offer":
        offers = db_fetch_all("SELECT offer_id, description_en FROM offers WHERE store_id = %s", (state.data["store_id"],))
        offer = next((o for o in offers if state.data.get("description", "").lower() in o["description_en"].lower()), None)
        if not offer:
            state.response = f"No offer found matching '{state.data.get('description')}'."
        else:
            db_execute("DELETE FROM offers WHERE offer_id = %s", (offer["offer_id"],))
            index.delete(ids=[f"offer_{offer['offer_id']}_en"])
            state.response = f"Deleted offer '{offer['description_en']}'!"
    
    elif state.intent == "read_store":
        # Get store details for reference
        store_details = db_fetch_one(
            "SELECT s.*, m.name_en as mall_name FROM stores s "
            "JOIN malls m ON s.mall_id = m.mall_id "
            "WHERE s.store_id = %s", 
            (state.data["store_id"],)
        )
        offers = db_fetch_all(
            "SELECT * FROM offers WHERE store_id = %s AND end_date >= CURRENT_DATE",
            (state.data["store_id"],)
        )
        state.response = f"Store: {store_details['name_en']} in {store_details['mall_name']}\n\nActive offers: " + \
            (', '.join([o['description_en'] for o in offers]) if offers else "None")
    
    else:
        state.response = "I'm not sure what you want to do. You can add, update, or delete offers for your stores."
    
    return state

# Tenant workflow
tenant_workflow = StateGraph(TenantUpdateState)
tenant_workflow.add_node("recognize_intent", recognize_intent)
tenant_workflow.add_node("process", process_update)
tenant_workflow.set_entry_point("recognize_intent")
tenant_workflow.add_edge("recognize_intent", "process")
tenant_workflow.add_edge("process", END)
tenant_graph = tenant_workflow.compile()

# Endpoints
@app.post("/login")
async def login(request: LoginRequest):
    tenant = db_fetch_one(
        "SELECT tenant_id, email, password FROM tenants WHERE email ILIKE %s",
        (request.email,)
    )
    if not tenant or tenant["password"] != request.password:  # Use hashing in production
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": str(tenant["tenant_id"]), "role": "tenant"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Detect language if not provided
    lang = request.language or detect_language(request.text)
    
    # Get or create session
    session_id = get_or_create_session(request.session_id, request.user_id, lang)
    
    # Get conversation history
    conversation_history = get_conversation_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_history]
    
    # Process the query
    state = CustomerState(
        query=request.text, 
        user_id=request.user_id, 
        language=lang,
        session_id=session_id,
        conversation_history=history_dicts
    )
    result = customer_graph.invoke(state)
    
    # Save the conversation messages
    add_message_to_conversation(session_id, "user", request.text)
    add_message_to_conversation(session_id, "assistant", result["response"])
    
    return ChatResponse(message=result["response"], session_id=session_id)

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    tenant = db_fetch_one("SELECT tenant_id FROM tenants WHERE tenant_id = %s", (request.user_id,))
    if not tenant:
        raise HTTPException(status_code=403, detail="Only tenants can update")
    
    # Detect language if not provided
    lang = request.language or "en"
    
    # Get or create session
    session_id = get_or_create_session(request.session_id, request.user_id, lang)
    
    # Get conversation history
    conversation_history = get_conversation_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_history]
    
    # Process the query
    state = TenantUpdateState(
        query=request.text, 
        user_id=request.user_id, 
        language=lang,
        session_id=session_id,
        conversation_history=history_dicts
    )
    result = tenant_graph.invoke(state)
    
    # Save the conversation messages
    add_message_to_conversation(session_id, "user", request.text)
    add_message_to_conversation(session_id, "assistant", result["response"])
    
    return {"message": result["response"], "session_id": session_id}

@app.get("/")
async def root():
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}

