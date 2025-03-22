from decimal import Decimal
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
genai.configure(api_key=GEMINI_API_KEY) # type: ignore

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        conn = psycopg2.connect(**DB_CONFIG) # type: ignore
        cur = conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    except Exception as e:
        logger.error(f"Database error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def db_fetch_all(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG) # type: ignore
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def db_execute(query: str, params: tuple = ()):
    try:
        conn = psycopg2.connect(**DB_CONFIG) # type: ignore
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Database error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

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

def get_or_create_session(session_id: Optional[str], user_id: Optional[str], language: str) -> str:
    if session_id:
        db_execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
            (session_id,)
        )
        return session_id
    else:
        new_session_id = str(uuid.uuid4())
        user_id_clean = user_id[2:] if user_id and user_id.startswith(("t_", "c_")) else user_id
        db_execute(
            "INSERT INTO conversations (session_id, user_id, language, current_state) VALUES (%s, %s, %s, %s)",
            (new_session_id, user_id_clean, language, json.dumps({}))
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
    return [
        Message(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
        for msg in reversed(messages)
    ]

# Customer prompt template (unchanged)
customer_prompt = PromptTemplate(
    input_variables=["context", "query", "lang", "conversation_history"],
    template="""
    You are CenomiAI, a friendly and highly knowledgeable mall assistant designed to enhance the shopping experience. Respond in {lang} with a warm, conversational tone, using emojis 😊 to keep it engaging. Your sole purpose is to assist with mall-related inquiries—stores, products, dining, services, amenities, events, offers, navigation, and more—while staying strictly within the mall context. Use the query, context, and conversation history to provide accurate, detailed, and tailored responses. If unsure or lacking info, ask clarifying questions or suggest helpful next steps while maintaining a supportive vibe.

    Response Guidelines:
    - Store Queries: 
      - Share specifics: store name, exact location (floor, nearby landmarks), hours, and offerings (e.g., products, brands, in-store cafes).
      - For category queries (e.g., "clothing stores"), list relevant options with locations and tailor to preferences (e.g., budget, style).

    - Product Queries: 
      - Address availability, brands, or types (e.g., "Does Zara have dresses?") with store suggestions and details if known.
      - If vague (e.g., "I need a gift"), ask about recipient or budget, then recommend stores or items.

    - Dining: 
      - Suggest options by cuisine, vibe (e.g., quick bites, date-night spots), or dietary needs (e.g., vegan, kid-friendly), including locations and highlights.

    - Services & Amenities: 
      - Provide directions to restrooms, ATMs, parking, play areas, etc., with practical details (e.g., "wheelchairs at info desk, Level 1").
      - Explain policies (e.g., Wi-Fi access, pet rules) or accessibility features.

    - Offers & Events: 
      - Detail current promotions (e.g., discounts, BOGO) or events (e.g., date, time, location), even suggesting upcoming ones if relevant.
      - For vague queries (e.g., "What’s happening?"), highlight popular options or ask for preferences.

    - Navigation: 
      - Offer clear, concise directions (e.g., "Food court’s on Level 2, left of the escalators") based on assumed or stated location.
      - Handle vague requests (e.g., "I’m lost") by asking for nearby landmarks or suggesting the info desk.

    - Vague or Emotional Queries: 
      - Interpret intent creatively: "I’m bored" → entertainment options; "I’m on a date" → romantic dining or activities; "I’m with kids" → family-friendly spots.
      - Ask follow-ups if needed (e.g., "What do you feel like doing? Shopping, eating, or fun?").

    - Personalization: 
      - Use `user_id` for loyalty details (points, perks) if provided, otherwise explain generic program benefits.
      - Tailor suggestions to context (e.g., time-sensitive events, group dynamics).

    - Limitations: 
      - If context lacks details, say: "I’m digging for that info 😅! Can you tell me more (e.g., which store or area) to help me out?"
      - Politely deflect non-mall topics: "I’m all about the mall—ask me anything from stores to events!"

    Instructions:
    - Context: Leverage provided data (e.g., store locations, event times) for precision. If empty, rely on general mall knowledge or seek clarification.
    - History: Reference past exchanges for continuity (e.g., "You asked about shoes earlier—want directions to Foot Locker?").
    - Tone: Be concise yet rich in detail, avoiding jargon. Sound like a friend who knows the mall inside out.
    - Read-Only: Don’t offer to change data—just inform and assist based on what’s available.
    - Out-of-Scope: For non-mall queries, gently redirect: "I’m your mall expert—got any questions about here?"
    - Continuity: If the user is following up on a previous question with pronouns like "it", "they", "that store", etc., use conversation history to understand what they're referring to.

    Current Query:
    "{query}"

    Context:
    {context}
    
    Conversation History:
    {conversation_history}
    """
)

# LLM and chain for customer queries
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
customer_chain = customer_prompt | llm | StrOutputParser()

# Customer state and workflow (unchanged)
class CustomerState(PydanticBaseModel):
    query: str
    user_id: Optional[str] = None
    language: str
    session_id: str
    conversation_history: List[Dict[str, str]] = []
    response: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None

def retrieve_context(state: CustomerState) -> CustomerState:
    mall_name = None
    query_lower = state.query.lower()
    possible_malls = db_fetch_all("SELECT name_en, mall_id FROM malls")
    for mall in possible_malls:
        if mall["name_en"].lower() in query_lower:
            mall_name = mall["name_en"]
            break
    
    filter = None
    if mall_name:
        mall = db_fetch_one("SELECT mall_id FROM malls WHERE name_en ILIKE %s", (mall_name,))
        if mall:
            filter = {"mall_id": mall["mall_id"]}

    query = state.query
    history = state.conversation_history
    if history and any(word in query_lower for word in ["it", "they", "that", "this", "there", "those"]):
        last_exchanges = [msg for msg in history[-4:] if msg["role"] == "assistant"]
        if last_exchanges:
            query = f"{query} (Referring to: {last_exchanges[-1]['content']})"
    
    query_vector = embeddings.embed_query(f"mall {query}")
    search_params = {"vector": query_vector, "top_k": 15, "include_metadata": True}
    if filter:
        search_params["filter"] = filter
    
    results = index.query(**search_params)
    docs = results["matches"] if "matches" in results else []
    
    context = {
        "stores": [],
        "offers": [],
        "events": [],
        "services": [],
        "amenities": [],
        "products": [],
        "loyalty_programs": [],
        "customer_loyalty": []
    }
    store_ids = set()
    
    for doc in docs:
        metadata = doc["metadata"]
        doc_type = metadata.get("type")
        if doc_type == "store":
            store = {
                "name": metadata.get("name_en"),
                "location": metadata.get("location_en"),
                "category": metadata.get("category_en")
            }
            context["stores"].append(store)
        elif doc_type == "offer":
            offer = {
                "description": metadata.get("description_en"),
                "id": metadata.get("id"),
                "store_id": metadata.get("store_id"),
                "store_name": metadata.get("store_name"),
                "location_en": metadata.get("location_en")
            }
            context["offers"].append(offer)
            if metadata.get("store_id"):
                store_ids.add(metadata["store_id"])
        elif doc_type == "event":
            event = {
                "name": metadata.get("name_en"),
                "date": metadata.get("start_time"),
                "location": metadata.get("location_en")
            }
            context["events"].append(event)
        elif doc_type == "service":
            service = {
                "name": metadata.get("name_en"),
                "description": metadata.get("description_en")
            }
            context["services"].append(service)
        elif doc_type == "amenity":
            amenity = {
                "name": metadata.get("name_en"),
                "location": metadata.get("location_en")
            }
            context["amenities"].append(amenity)
        elif doc_type == "product":
            product = {
                "name": metadata.get("name_en"),
                "description": metadata.get("description_en"),
                "price": metadata.get("price"),
                "currency": metadata.get("currency"),
                "store_id": metadata.get("store_id"),
                "store_name": metadata.get("store_name"),
                "location_en": metadata.get("location_en")
            }
            context["products"].append(product)
            if metadata.get("store_id"):
                store_ids.add(metadata["store_id"])
    
    # Fetch store details for products/offers missing them (e.g., older data)
    if store_ids:
        stores = db_fetch_all(
            "SELECT store_id, name_en, location_en, category_en FROM stores WHERE store_id IN %s",
            (tuple(store_ids),)
        )
        store_map = {s["store_id"]: s for s in stores}
        for product in context["products"]:
            if not product.get("store_name") and product.get("store_id") in store_map:
                store = store_map[product["store_id"]]
                product["store_name"] = store["name_en"]
                product["location_en"] = store["location_en"]
        for offer in context["offers"]:
            if not offer.get("store_name") and offer.get("store_id") in store_map:
                store = store_map[offer["store_id"]]
                offer["store_name"] = store["name_en"]
                offer["location_en"] = store["location_en"]

    if state.user_id and state.user_id.startswith("c_"):
        customer_id = state.user_id[2:]  # Strip 'c_' prefix
        customer = db_fetch_one("SELECT customer_id FROM customers WHERE customer_id = %s", (customer_id,))
        if customer:
            loyalty = db_fetch_one(
                "SELECT cl.points_balance, lp.name_en, lp.description_en "
                "FROM customer_loyalty cl "
                "JOIN loyalty_programs lp ON cl.loyalty_id = lp.loyalty_id "
                "WHERE cl.customer_id = %s",
                (customer_id,)
            )
            context["customer_loyalty"] = loyalty or {}
    
    state.context_data = context
    state.response = json.dumps(convert_to_json_safe(context))
    return state

def generate_response(state: CustomerState) -> CustomerState:
    formatted_history = ""
    if state.conversation_history:
        formatted_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in state.conversation_history[-6:]
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
    state.response = response
    return state

customer_workflow = StateGraph(CustomerState)
customer_workflow.add_node("retrieve", retrieve_context)
customer_workflow.add_node("respond", generate_response)
customer_workflow.set_entry_point("retrieve")
customer_workflow.add_edge("retrieve", "respond")
customer_workflow.add_edge("respond", END)
customer_graph = customer_workflow.compile()

class TenantState(BaseModel):
    query: str
    user_id: str
    language: str = "en"
    session_id: str
    conversation_history: List[Dict[str, str]] = []
    entity_type: Optional[str] = None
    action: Optional[str] = None
    collected_data: Dict[str, Any] = {}
    current_step: Optional[str] = None
    store_name: Optional[str] = None
    response: Optional[str] = None
    offer_list: Optional[List[Dict[str, Any]]] = None

# Updated intent prompt to include products
intent_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    You are CenomiAI's tenant assistant. Parse this query to determine the tenant's intent based on the current query and conversation history.

    Valid entities: store, offer, product
    Valid actions: create, update, delete, list

    Query: "{query}"

    Previous conversation:
    {conversation_history}

    Instructions:
    - Use the conversation history to resolve vague terms like "it", "that", or "the product/offer".
    - If the query mentions "update" or "change" and follows a prior product/offer creation or mention, assume it refers to modifying an existing entity.
    - If the query is ambiguous but follows a completed action (e.g., product creation), start fresh unless explicitly continuing.
    - Extract any specific details (e.g., name, description, store name) into collected_data.

    Return a JSON object with:
    - entity_type: What they’re working with (store, offer, product)
    - action: What they want to do (create, update, delete, list)
    - collected_data: Any details provided (e.g., "name": "Blue Shirt", "description": "20% off", "store": "Zara")

    Return only valid JSON in triple backticks.
    """
)

def analyze_intent(state: TenantState) -> TenantState:
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.conversation_history[-4:]])
    intent_chain = intent_prompt | llm | StrOutputParser()
    intent_result = intent_chain.invoke({"query": state.query, "conversation_history": formatted_history})
    start = intent_result.find("```json") + 7
    end = intent_result.rfind("```")
    intent_data = json.loads(intent_result[start:end].strip())
    
    logger.info(f"Intent parsed: {intent_data}")
    
    if not state.current_step:
        state.collected_data = {}
    
    state.entity_type = intent_data.get("entity_type")
    state.action = intent_data.get("action")
    state.collected_data.update(intent_data.get("collected_data", {}))
    
    tenant_id = state.user_id[2:] if state.user_id.startswith("t_") else state.user_id
    user_stores = db_fetch_all("SELECT name_en FROM stores WHERE tenant_id = %s", (tenant_id,))
    if "store" in state.collected_data:
        requested_store = state.collected_data["store"].lower()
        matching_store = next((s for s in user_stores if s["name_en"].lower() == requested_store), None)
        if matching_store:
            state.store_name = matching_store["name_en"]
        else:
            state.response = f"I couldn’t find '{requested_store}'. Your stores: {', '.join([s['name_en'] for s in user_stores])}"
            state.current_step = "select_store"
            return state
    elif len(user_stores) == 1:
        state.store_name = user_stores[0]["name_en"]
        logger.info(f"Default store set to {state.store_name}")
    elif len(user_stores) > 1 and not state.store_name:
        state.current_step = "select_store"
    
    return state

def prompt_for_missing_info(state: TenantState) -> TenantState:
    tenant_id = state.user_id[2:] if state.user_id.startswith("t_") else state.user_id
    user_stores = db_fetch_all("SELECT name_en FROM stores WHERE tenant_id = %s", (tenant_id,))
    
    if state.current_step == "select_store" or (len(user_stores) > 1 and not state.store_name):
        if not user_stores:
            state.response = "You don’t have any stores yet. Contact mall management to get started!"
            return state
        store_list = "\n".join([f"{i+1}) {s['name_en']}" for i, s in enumerate(user_stores)])
        state.response = f"You’ve got multiple stores! Which one?\n{store_list}\nType the number!"
        state.current_step = "select_store"
        return state
    
    if not state.store_name and len(user_stores) == 1:
        state.store_name = user_stores[0]["name_en"]
        logger.info(f"Auto-set store_name to {state.store_name}")
    
    if state.entity_type == "offer":
        if state.action == "create":
            if "description" not in state.collected_data:
                state.response = f"What’s the offer for {state.store_name}? (e.g., '20% off summer clothes')"
                state.current_step = "description"
            elif "start_date" not in state.collected_data:
                state.response = "When should it start? (e.g., 'today' or '2025-04-01')"
                state.current_step = "start_date"
            elif "end_date" not in state.collected_data:
                state.response = "When should it end? (e.g., '2025-04-30')"
                state.current_step = "end_date"
            else:
                execute_operation(state)
        elif state.action == "update":
            if "description" not in state.collected_data:
                offers = db_fetch_all(
                    "SELECT description_en, start_date, end_date FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                    (state.store_name, tenant_id)
                )
                if not offers:
                    state.response = f"No offers found for {state.store_name}. Want to add one?"
                    state.entity_type = None
                    state.action = None
                    return state
                state.offer_list = offers
                offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(offers)])
                state.response = f"Which offer to update in {state.store_name}?\n{offer_list}\nType the number!"
                state.current_step = "select_offer"
            elif "update_field" not in state.collected_data:
                state.response = f"What do you want to change for '{state.collected_data['description']}'?\n1) Description\n2) Start Date\n3) End Date\nType the number!"
                state.current_step = "update_field"
            elif state.collected_data["update_field"] == "1" and "new_description" not in state.collected_data:
                state.response = f"What’s the new description for '{state.collected_data['description']}'?"
                state.current_step = "new_description"
            elif state.collected_data["update_field"] == "2" and "new_start_date" not in state.collected_data:
                state.response = f"What’s the new start date for '{state.collected_data['description']}'? (e.g., '2025-04-01')"
                state.current_step = "new_start_date"
            elif state.collected_data["update_field"] == "3" and "new_end_date" not in state.collected_data:
                state.response = f"What’s the new end date for '{state.collected_data['description']}'? (e.g., '2025-04-30')"
                state.current_step = "new_end_date"
            else:
                execute_operation(state)
        elif state.action == "delete":
            if "description" not in state.collected_data:
                offers = db_fetch_all(
                    "SELECT description_en, start_date, end_date FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                    (state.store_name, tenant_id)
                )
                if not offers:
                    state.response = f"No offers found for {state.store_name}. Want to add one?"
                    state.entity_type = None
                    state.action = None
                    return state
                state.offer_list = offers
                offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(offers)])
                state.response = f"Which offer to remove from {state.store_name}?\n{offer_list}\nType the number!"
                state.current_step = "select_offer"
            else:
                execute_operation(state)
        elif state.action == "list":
            offers = db_fetch_all(
                "SELECT description_en, start_date, end_date FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                (state.store_name, tenant_id)
            )
            if not offers:
                state.response = f"No offers in {state.store_name} yet. Want to add one?"
            else:
                offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(offers)])
                state.response = f"Here are your offers for {state.store_mode}:\n{offer_list}\nAnything else?"
    
    elif state.entity_type == "product":
        if state.action == "create":
            if "name" not in state.collected_data:
                state.response = f"What’s the product name for {state.store_name}? (e.g., 'Blue Shirt')"
                state.current_step = "name"
            elif "description" not in state.collected_data:
                state.response = f"What’s the description for '{state.collected_data['name']}'? (e.g., 'Cotton, size M')"
                state.current_step = "description"
            elif "price" not in state.collected_data:
                state.response = f"How much does '{state.collected_data['name']}' cost? (e.g., '50')"
                state.current_step = "price"
            elif "currency" not in state.collected_data:
                state.response = f"What’s the currency for '{state.collected_data['name']}'? (e.g., 'SAR')"
                state.current_step = "currency"
            else:
                execute_operation(state)
        elif state.action == "update":
            if "name" not in state.collected_data:
                products = db_fetch_all(
                    "SELECT name_en, description_en, price, currency FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                    (state.store_name, tenant_id)
                )
                if not products:
                    state.response = f"No products found for {state.store_name}. Want to add one?"
                    state.entity_type = None
                    state.action = None
                    return state
                state.offer_list = products
                product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(products)])
                state.response = f"Which product to update in {state.store_name}?\n{product_list}\nType the number!"
                state.current_step = "select_product"
            elif "update_field" not in state.collected_data:
                state.response = f"What do you want to change for '{state.collected_data['name']}'?\n1) Name\n2) Description\n3) Price\n4) Currency\nType the number!"
                state.current_step = "update_field"
            elif state.collected_data["update_field"] == "1" and "new_name" not in state.collected_data:
                state.response = f"What’s the new name for '{state.collected_data['name']}'?"
                state.current_step = "new_name"
            elif state.collected_data["update_field"] == "2" and "new_description" not in state.collected_data:
                state.response = f"What’s the new description for '{state.collected_data['name']}'?"
                state.current_step = "new_description"
            elif state.collected_data["update_field"] == "3" and "new_price" not in state.collected_data:
                state.response = f"What’s the new price for '{state.collected_data['name']}'? (e.g., '60')"
                state.current_step = "new_price"
            elif state.collected_data["update_field"] == "4" and "new_currency" not in state.collected_data:
                state.response = f"What’s the new currency for '{state.collected_data['name']}'? (e.g., 'USD')"
                state.current_step = "new_currency"
            else:
                execute_operation(state)
        elif state.action == "delete":
            if "name" not in state.collected_data:
                products = db_fetch_all(
                    "SELECT name_en, description_en, price, currency FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                    (state.store_name, tenant_id)
                )
                if not products:
                    state.response = f"No products found for {state.store_name}. Want to add one?"
                    state.entity_type = None
                    state.action = None
                    return state
                state.offer_list = products
                product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(products)])
                state.response = f"Which product to remove from {state.store_name}?\n{product_list}\nType the number!"
                state.current_step = "select_product"
            else:
                execute_operation(state)
        elif state.action == "list":
            products = db_fetch_all(
                "SELECT name_en, description_en, price, currency FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                (state.store_name, tenant_id)
            )
            if not products:
                state.response = f"No products in {state.store_name} yet. Want to add one?"
            else:
                product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(products)])
                state.response = f"Here are your products for {state.store_name}:\n{product_list}\nAnything else?"
    
    return state

def process_input(state: TenantState) -> TenantState:
    if not state.current_step:
        return analyze_intent(state)
    
    tenant_id = state.user_id[2:] if state.user_id.startswith("t_") else state.user_id
    user_stores = db_fetch_all("SELECT name_en FROM stores WHERE tenant_id = %s", (tenant_id,))
    
    if state.current_step == "select_store":
        try:
            choice = int(state.query.strip()) - 1
            if 0 <= choice < len(user_stores):
                state.store_name = user_stores[choice]["name_en"]
                state.current_step = None
                logger.info(f"Store selected: {state.store_name}")
            else:
                store_list = "\n".join([f"{i+1}) {s['name_en']}" for i, s in enumerate(user_stores)])
                state.response = f"Invalid option! Pick a number:\n{store_list}"
                return state
        except ValueError:
            store_list = "\n".join([f"{i+1}) {s['name_en']}" for i, s in enumerate(user_stores)])
            state.response = f"Please type a number! Here are your stores:\n{store_list}"
            return state
    
    elif state.current_step == "select_offer":
        try:
            choice = int(state.query.strip()) - 1
            if 0 <= choice < len(state.offer_list):
                state.collected_data["description"] = state.offer_list[choice]["description_en"]
                state.current_step = None
                state.offer_list = None
            else:
                offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(state.offer_list)])
                state.response = f"Invalid option! Pick a number:\n{offer_list}"
                return state
        except ValueError:
            offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(state.offer_list)])
            state.response = f"Please type a number! Options:\n{offer_list}"
            return state
    
    elif state.current_step == "select_product":
        try:
            choice = int(state.query.strip()) - 1
            if 0 <= choice < len(state.offer_list):
                state.collected_data["name"] = state.offer_list[choice]["name_en"]
                state.current_step = None
                state.offer_list = None
            else:
                product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(state.offer_list)])
                state.response = f"Invalid option! Pick a number:\n{product_list}"
                return state
        except ValueError:
            product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(state.offer_list)])
            state.response = f"Please type a number! Options:\n{product_list}"
            return state
    
    elif state.current_step == "update_field":
        try:
            choice = int(state.query.strip())
            if state.entity_type == "offer" and choice in [1, 2, 3]:
                state.collected_data["update_field"] = str(choice)
                state.current_step = None
            elif state.entity_type == "product" and choice in [1, 2, 3, 4]:
                state.collected_data["update_field"] = str(choice)
                state.current_step = None
            else:
                if state.entity_type == "offer":
                    state.response = "Invalid option! Choose: 1) Description, 2) Start Date, 3) End Date"
                else:
                    state.response = "Invalid option! Choose: 1) Name, 2) Description, 3) Price, 4) Currency"
                return state
        except ValueError:
            if state.entity_type == "offer":
                state.response = "Please type a number! 1) Description, 2) Start Date, 3) End Date"
            else:
                state.response = "Please type a number! 1) Name, 2) Description, 3) Price, 4) Currency"
            return state
    
    elif state.current_step in ["description", "new_description", "start_date", "end_date", "new_start_date", "new_end_date", 
                                "name", "new_name", "price", "new_price", "currency", "new_currency"]:
        state.collected_data[state.current_step] = state.query.strip()
        state.current_step = None
    
    return prompt_for_missing_info(state)


def execute_operation(state: TenantState) -> None:
    tenant_id = state.user_id[2:] if state.user_id.startswith("t_") else state.user_id
    store = db_fetch_one(
        "SELECT store_id, name_en, location_en FROM stores WHERE name_en = %s AND tenant_id = %s",
        (state.store_name, tenant_id)
    )
    if not store:
        state.response = f"I couldn’t find {state.store_name} in your stores."
        return
    
    store_id = store["store_id"]
    store_name = store["name_en"]
    location_en = store["location_en"]
    logger.info(f"Executing {state.action} on {state.entity_type} for store_id: {store_id}")
    
    if state.entity_type == "offer":
        if state.action == "create":
            description = state.collected_data["description"]
            start_date = state.collected_data["start_date"] if state.collected_data["start_date"] != "today" else datetime.now().strftime("%Y-%m-%d")
            end_date = state.collected_data["end_date"]
            db_execute(
                "INSERT INTO offers (store_id, description_en, description_ar, start_date, end_date) VALUES (%s, %s, %s, %s, %s)",
                (store_id, description, description, start_date, end_date)
            )
            offer = db_fetch_one("SELECT offer_id FROM offers WHERE store_id = %s AND description_en = %s", (store_id, description))
            if offer:
                offer_id = offer["offer_id"]
                vector = embeddings.embed_query(description)
                index.upsert(vectors=[{
                    "id": f"offer_{offer_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "offer",
                        "id": offer_id,
                        "description_en": description,
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                state.response = f"Added '{description}' to {state.store_name} from {start_date} to {end_date}. Anything else? 😊"
            else:
                state.response = f"Failed to add '{description}'. Try again or contact support."
        elif state.action == "update":
            old_desc = state.collected_data["description"]
            update_field = state.collected_data["update_field"]
            offer = db_fetch_one(
                "SELECT offer_id FROM offers WHERE store_id = %s AND description_en = %s",
                (store_id, old_desc)
            )
            if not offer:
                state.response = f"Couldn’t find '{old_desc}' in {state.store_name}. Want to list offers?"
                return
            offer_id = offer["offer_id"]
            if update_field == "1":
                new_desc = state.collected_data["new_description"]
                db_execute(
                    "UPDATE offers SET description_en = %s, description_ar = %s WHERE offer_id = %s",
                    (new_desc, new_desc, offer_id)
                )
                vector = embeddings.embed_query(new_desc)
                index.upsert(vectors=[{
                    "id": f"offer_{offer_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "offer",
                        "id": offer_id,
                        "description_en": new_desc,
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                state.response = f"Updated '{old_desc}' to '{new_desc}' in {state.store_name}. Anything else? 😊"
            elif update_field == "2":
                new_start_date = state.collected_data["new_start_date"]
                db_execute(
                    "UPDATE offers SET start_date = %s WHERE offer_id = %s",
                    (new_start_date, offer_id)
                )
                state.response = f"Updated '{old_desc}' start date to {new_start_date} in {state.store_name}. Anything else? 😊"
            elif update_field == "3":
                new_end_date = state.collected_data["new_end_date"]
                db_execute(
                    "UPDATE offers SET end_date = %s WHERE offer_id = %s",
                    (new_end_date, offer_id)
                )
                state.response = f"Updated '{old_desc}' end date to {new_end_date} in {state.store_name}. Anything else? 😊"
        elif state.action == "delete":
            description = state.collected_data["description"]
            offer = db_fetch_one(
                "SELECT offer_id FROM offers WHERE store_id = %s AND description_en = %s",
                (store_id, description)
            )
            if offer:
                offer_id = offer["offer_id"]
                db_execute("DELETE FROM offers WHERE offer_id = %s", (offer_id,))
                index.delete(ids=[f"offer_{offer_id}_en"])
                state.response = f"Removed '{description}' from {state.store_name}. Anything else? 😊"
            else:
                state.response = f"Couldn’t find '{description}' in {state.store_name}. Want to list offers?"
    
    elif state.entity_type == "product":
        if state.action == "create":
            name = state.collected_data["name"]
            description = state.collected_data.get("description")
            price = float(state.collected_data["price"])
            currency = state.collected_data["currency"]
            db_execute(
                "INSERT INTO products (store_id, name_en, name_ar, description_en, description_ar, price, currency) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (store_id, name, name, description, description, price, currency)
            )
            product = db_fetch_one("SELECT product_id FROM products WHERE store_id = %s AND name_en = %s", (store_id, name))
            if product:
                product_id = product["product_id"]
                vector = embeddings.embed_query(f"{name} {description or ''}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "name_en": name,
                        "description_en": description or "",
                        "price": price,
                        "currency": currency,
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                state.response = f"Added '{name}' ({description or 'No description'}) to {state.store_name} for {price} {currency}. Anything else? 😊"
            else:
                state.response = f"Failed to add '{name}'. Try again or contact support."
        elif state.action == "update":
            old_name = state.collected_data["name"]
            update_field = state.collected_data["update_field"]
            product = db_fetch_one(
                "SELECT product_id FROM products WHERE store_id = %s AND name_en = %s",
                (store_id, old_name)
            )
            if not product:
                state.response = f"Couldn’t find '{old_name}' in {state.store_name}. Want to list products?"
                return
            product_id = product["product_id"]
            if update_field == "1":
                new_name = state.collected_data["new_name"]
                db_execute(
                    "UPDATE products SET name_en = %s, name_ar = %s WHERE product_id = %s",
                    (new_name, new_name, product_id)
                )
                vector = embeddings.embed_query(f"{new_name} {state.collected_data.get('description', '')}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "name_en": new_name,
                        "description_en": state.collected_data.get("description", ""),
                        "price": float(state.collected_data.get("price", 0)),
                        "currency": state.collected_data.get("currency", ""),
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                state.response = f"Updated '{old_name}' to '{new_name}' in {state.store_name}. Anything else? 😊"
            elif update_field == "2":
                new_desc = state.collected_data["new_description"]
                db_execute(
                    "UPDATE products SET description_en = %s, description_ar = %s WHERE product_id = %s",
                    (new_desc, new_desc, product_id)
                )
                vector = embeddings.embed_query(f"{old_name} {new_desc}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "name_en": old_name,
                        "description_en": new_desc,
                        "price": float(state.collected_data.get("price", 0)),
                        "currency": state.collected_data.get("currency", ""),
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                state.response = f"Updated '{old_name}' description to '{new_desc}' in {state.store_name}. Anything else? 😊"
            elif update_field == "3":
                new_price = float(state.collected_data["new_price"])
                db_execute(
                    "UPDATE products SET price = %s WHERE product_id = %s",
                    (new_price, product_id)
                )
                vector = embeddings.embed_query(f"{old_name} {state.collected_data.get('description', '')}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "name_en": old_name,
                        "description_en": state.collected_data.get("description", ""),
                        "price": new_price,
                        "currency": state.collected_data.get("currency", ""),
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                state.response = f"Updated '{old_name}' price to {new_price} in {state.store_name}. Anything else? 😊"
            elif update_field == "4":
                new_currency = state.collected_data["new_currency"]
                db_execute(
                    "UPDATE products SET currency = %s WHERE product_id = %s",
                    (new_currency, product_id)
                )
                vector = embeddings.embed_query(f"{old_name} {state.collected_data.get('description', '')}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "name_en": old_name,
                        "description_en": state.collected_data.get("description", ""),
                        "price": float(state.collected_data.get("price", 0)),
                        "currency": new_currency,
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                state.response = f"Updated '{old_name}' currency to {new_currency} in {state.store_name}. Anything else? 😊"
        elif state.action == "delete":
            name = state.collected_data["name"]
            product = db_fetch_one(
                "SELECT product_id FROM products WHERE store_id = %s AND name_en = %s",
                (store_id, name)
            )
            if product:
                product_id = product["product_id"]
                db_execute("DELETE FROM products WHERE product_id = %s", (product_id,))
                index.delete(ids=[f"product_{product_id}_en"])
                state.response = f"Removed '{name}' from {state.store_name}. Anything else? 😊"
            else:
                state.response = f"Couldn’t find '{name}' in {state.store_name}. Want to list products?"
    
    state.entity_type = None
    state.action = None
    state.collected_data = {}
    state.current_step = None
    state.offer_list = None

tenant_prompt = PromptTemplate(
    input_variables=["message", "conversation_history", "entity_type", "action"],
    template="""
    You are CenomiAI, a helpful assistant for mall tenants. Format the following system message into a natural, conversational response:
    
    System message: {message}
    
    Current context:
    - Entity: {entity_type}
    - Action: {action}
    
    Previous conversation:
    {conversation_history}
    
    Make the response friendly and professional. Use emoji occasionally to add warmth. Focus on helping the tenant manage their store information efficiently.
    """
)

def tenant_recognize_intent(state: TenantState) -> TenantState:
    conversation_history = get_conversation_history(state.session_id)
    state.conversation_history = [{"role": msg.role, "content": msg.content} for msg in conversation_history]
    
    if not state.current_step:
        state = analyze_intent(state)
        state = prompt_for_missing_info(state)
    else:
        state = process_input(state)
    
    if state.response:
        tenant_chain = tenant_prompt | llm | StrOutputParser()
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.conversation_history[-4:]])
        state.response = tenant_chain.invoke({
            "message": state.response,
            "conversation_history": formatted_history,
            "entity_type": state.entity_type or "unknown",
            "action": state.action or "unknown"
        })
    else:
        state.response = "I’m not sure what you want to do. You can add, update, or remove offers or products—just let me know!"
    return state

tenant_workflow = StateGraph(TenantState)
tenant_workflow.add_node("process", tenant_recognize_intent)
tenant_workflow.set_entry_point("process")
tenant_workflow.add_edge("process", END)
tenant_graph = tenant_workflow.compile()

@app.post("/login")
async def login(request: LoginRequest):
    # Check tenants first
    tenant = db_fetch_one("SELECT tenant_id FROM tenants WHERE email ILIKE %s AND password = %s", (request.email, request.password))
    if tenant:
        return {"user_id": f"t_{tenant['tenant_id']}"}
    
    # Check customers
    customer = db_fetch_one("SELECT customer_id FROM customers WHERE email ILIKE %s AND password = %s", (request.email, request.password))
    if customer:
        return {"user_id": f"c_{customer['customer_id']}"}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    lang = request.language or detect_language(request.text)
    session_id = get_or_create_session(request.session_id, request.user_id, lang)
    conversation_history = get_conversation_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_history]
    
    # Load existing state or create new
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

    result = customer_graph.invoke(state)
    
    state_json = json.dumps(result, cls=DateTimeEncoder)
    db_execute("UPDATE conversations SET current_state = %s WHERE session_id = %s", (state_json, session_id))
    
    add_message_to_conversation(session_id, "user", request.text)
    add_message_to_conversation(session_id, "assistant", result["response"])
    return ChatResponse(message=result["response"], session_id=session_id)

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    if not request.user_id.startswith("t_"):
        raise HTTPException(status_code=403, detail="Only tenants can perform updates")
    
    tenant_id = request.user_id[2:]  # Strip 't_' prefix
    tenant = db_fetch_one("SELECT tenant_id FROM tenants WHERE tenant_id = %s", (tenant_id,))
    if not tenant:
        raise HTTPException(status_code=403, detail="Invalid tenant ID")
    
    lang = request.language or "en"
    session_id = get_or_create_session(request.session_id, request.user_id, lang)
    
    conv_state = db_fetch_one("SELECT current_state FROM conversations WHERE session_id = %s", (session_id,))
    history = get_conversation_history(session_id)
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

    result = tenant_graph.invoke(state)
    
    state_json = json.dumps(result, cls=DateTimeEncoder)
    db_execute("UPDATE conversations SET current_state = %s WHERE session_id = %s", (state_json, session_id))
    
    add_message_to_conversation(session_id, "user", request.text)
    add_message_to_conversation(session_id, "assistant", result["response"])
    
    return {"message": result["response"], "session_id": session_id}

@app.get("/")
async def root():
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}