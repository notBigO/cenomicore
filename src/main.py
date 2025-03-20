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

def get_or_create_session(session_id: Optional[str], user_id: Optional[str], language: str) -> str:
    if session_id:
        db_execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE session_id = %s",
            (session_id,)
        )
        return session_id
    else:
        new_session_id = str(uuid.uuid4())
        db_execute(
            "INSERT INTO conversations (session_id, user_id, language, current_state) VALUES (%s, %s, %s, %s)",
            (new_session_id, user_id, language, json.dumps({}))
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
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
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
    mall_name = None
    if "nakheel mall" in state.query.lower():
        mall_name = "Nakheel Mall"
    filter = None
    if mall_name:
        mall = db_fetch_one("SELECT mall_id FROM malls WHERE name_en ILIKE %s", (mall_name,))
        if mall:
            filter = {"mall_id": mall["mall_id"]}
    
    query = state.query
    history = state.conversation_history
    
    if history and any(word in query.lower() for word in ["it", "they", "that", "this", "there", "those"]):
        last_exchanges = [msg for msg in history[-4:] if msg["role"] == "assistant"]
        if last_exchanges:
            query = f"{query} (This is a follow-up to: {last_exchanges[-1]['content']})"
    
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

class TenantState(BaseModel):
    """State for tenant conversation flow"""
    query: str
    user_id: str
    language: str = "en"
    session_id: str
    conversation_history: List[Dict[str, str]] = []
    entity_type: Optional[str] = None  # e.g., offer, product
    action: Optional[str] = None      # e.g., create, update, delete, list
    collected_data: Dict[str, Any] = {}  # Natural language details (e.g., description)
    current_step: Optional[str] = None   # Tracks what we’re asking for (e.g., store, description)
    store_name: Optional[str] = None     # Store name, not ID
    response: Optional[str] = None

intent_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    You are CenomiAI's tenant assistant. Parse this query to understand what the tenant wants to do.
    
    Valid entities: store, offer, product
    Valid actions: create, update, delete, list
    
    Query: "{query}"
    
    Previous conversation:
    {conversation_history}
    
    Return a JSON object with:
    - entity_type: What they’re working with (store, offer, product)
    - action: What they want to do (create, update, delete, list)
    - collected_data: Any details they’ve provided (e.g., "description": "20% off summer sale", "store": "Fashion Hub")
    
    Use context from the conversation history to resolve vague terms like "it" or "that." If a store, offer, or product is mentioned vaguely, infer it from the history.
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
    
    state.entity_type = intent_data.get("entity_type")
    state.action = intent_data.get("action")
    state.collected_data.update(intent_data.get("collected_data", {}))
    
    # Infer store automatically
    user_stores = db_fetch_all("SELECT name_en FROM stores WHERE tenant_id = %s", (state.user_id,))
    if "store" in state.collected_data:
        state.store_name = state.collected_data["store"]
    elif len(user_stores) == 1:
        state.store_name = user_stores[0]["name_en"]
    elif len(user_stores) > 1 and not state.store_name:
        state.current_step = "select_store"
    
    return state

def prompt_for_missing_info(state: TenantState) -> TenantState:
    user_stores = db_fetch_all("SELECT name_en FROM stores WHERE tenant_id = %s", (state.user_id,))
    
    # Step 1: Handle store selection
    if state.current_step == "select_store" or (len(user_stores) > 1 and not state.store_name):
        if not user_stores:
            state.response = "Looks like you don’t have any stores yet. Contact mall management to get started!"
            return state
        store_list = "\n".join([f"- {store['name_en']}" for store in user_stores])
        state.response = f"Hi! You’ve got a few stores. Which one should I use for this?\n{store_list}\nJust tell me the name!"
        state.current_step = "select_store"
        return state
    
    # Step 2: Collect entity details
    if state.entity_type == "offer":
        if state.action == "create":
            if "description" not in state.collected_data:
                state.response = "Awesome! What’s this offer about? (e.g., '20% off summer clothes')"
                state.current_step = "description"
            elif "start_date" not in state.collected_data:
                state.response = "When should this offer start? You can say 'today' or something like '2025-04-01'."
                state.current_step = "start_date"
            elif "end_date" not in state.collected_data:
                state.response = "And when should it end? (e.g., '2025-04-30')"
                state.current_step = "end_date"
            else:
                execute_operation(state)
        elif state.action == "update":
            if "description" not in state.collected_data:
                offers = db_fetch_all(
                    "SELECT description_en FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                    (state.store_name, state.user_id)
                )
                offer_list = "\n".join([f"- {o['description_en']}" for o in offers]) if offers else "None yet!"
                state.response = f"Which offer do you want to change in {state.store_name}?\n{offer_list}\nTell me the one you mean!"
                state.current_step = "description"
            elif "new_description" not in state.collected_data:
                state.response = f"Okay, updating '{state.collected_data['description']}'. What should the new offer say?"
                state.current_step = "new_description"
            else:
                execute_operation(state)
        elif state.action == "delete":
            if "description" not in state.collected_data:
                offers = db_fetch_all(
                    "SELECT description_en FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                    (state.store_name, state.user_id)
                )
                offer_list = "\n".join([f"- {o['description_en']}" for o in offers]) if offers else "None yet!"
                state.response = f"Which offer should I remove from {state.store_name}?\n{offer_list}\nJust say the one you want gone!"
                state.current_step = "description"
            else:
                execute_operation(state)
        elif state.action == "list":
            offers = db_fetch_all(
                "SELECT description_en, start_date, end_date FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                (state.store_name, state.user_id)
            )
            if not offers:
                state.response = f"You don’t have any offers in {state.store_name} yet. Want to add one?"
            else:
                offer_list = "\n".join([f"- {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for o in offers])
                state.response = f"Here are your offers for {state.store_name}:\n{offer_list}"
    
    elif state.entity_type == "product":
        if state.action == "create":
            if "name" not in state.collected_data:
                state.response = "Cool! What’s the product called? (e.g., 'Blue T-Shirt')"
                state.current_step = "name"
            elif "description" not in state.collected_data:
                state.response = f"Nice! What’s '{state.collected_data['name']}' about? (e.g., 'Cotton, size M')"
                state.current_step = "description"
            elif "price" not in state.collected_data:
                state.response = "How much does it cost? (e.g., '50')"
                state.current_step = "price"
            elif "currency" not in state.collected_data:
                state.response = "What currency? (e.g., 'SAR' or 'USD')"
                state.current_step = "currency"
            else:
                execute_operation(state)
        elif state.action == "update":
            if "name" not in state.collected_data:
                products = db_fetch_all(
                    "SELECT name_en FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                    (state.store_name, state.user_id)
                )
                product_list = "\n".join([f"- {p['name_en']}" for p in products]) if products else "None yet!"
                state.response = f"Which product do you want to change in {state.store_name}?\n{product_list}\nTell me the name!"
                state.current_step = "name"
            elif "new_description" not in state.collected_data:
                state.response = f"Okay, updating '{state.collected_data['name']}'. What should the new description be?"
                state.current_step = "new_description"
            else:
                execute_operation(state)
        elif state.action == "delete":
            if "name" not in state.collected_data:
                products = db_fetch_all(
                    "SELECT name_en FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                    (state.store_name, state.user_id)
                )
                product_list = "\n".join([f"- {p['name_en']}" for p in products]) if products else "None yet!"
                state.response = f"Which product should I remove from {state.store_name}?\n{product_list}\nJust say the name!"
                state.current_step = "name"
            else:
                execute_operation(state)
        elif state.action == "list":
            products = db_fetch_all(
                "SELECT name_en, description_en, price, currency FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s)",
                (state.store_name, state.user_id)
            )
            if not products:
                state.response = f"You don’t have any products in {state.store_name} yet. Want to add one?"
            else:
                product_list = "\n".join([f"- {p['name_en']}: {p['description_en']} ({p['price']} {p['currency']})" for p in products])
                state.response = f"Here are your products for {state.store_name}:\n{product_list}"
    
    return state

def process_input(state: TenantState) -> TenantState:
    if not state.current_step:
        return analyze_intent(state)
    
    if state.current_step == "select_store":
        user_stores = db_fetch_all("SELECT name_en FROM stores WHERE tenant_id = %s", (state.user_id,))
        store_name = state.query.strip().lower()
        matching_store = next((s for s in user_stores if s["name_en"].lower() == store_name), None)
        if matching_store:
            state.store_name = matching_store["name_en"]
            state.current_step = None
        else:
            store_list = "\n".join([f"- {s['name_en']}" for s in user_stores])
            state.response = f"Hmm, I didn’t find that store. Pick one of yours:\n{store_list}"
            return state
    
    elif state.current_step in ["description", "new_description", "start_date", "end_date", "name", "price", "currency"]:
        state.collected_data[state.current_step] = state.query.strip()
        state.current_step = None
    
    return prompt_for_missing_info(state)

def execute_operation(state: TenantState) -> None:
    store_id = db_fetch_one(
        "SELECT store_id FROM stores WHERE name_en = %s AND tenant_id = %s",
        (state.store_name, state.user_id)
    )["store_id"]
    
    if state.entity_type == "offer":
        if state.action == "create":
            description = state.collected_data["description"]
            start_date = state.collected_data["start_date"] if state.collected_data["start_date"] != "today" else datetime.now().strftime("%Y-%m-%d")
            end_date = state.collected_data["end_date"]
            db_execute(
                "INSERT INTO offers (store_id, description_en, description_ar, start_date, end_date) VALUES (%s, %s, %s, %s, %s)",
                (store_id, description, description, start_date, end_date)
            )
            offer_id = db_fetch_one("SELECT currval(pg_get_serial_sequence('offers', 'offer_id')) AS id")["id"]
            vector = embeddings.embed_query(description)
            index.upsert(vectors=[{
                "id": f"offer_{offer_id}_en",
                "values": vector,
                "metadata": {"type": "offer", "id": offer_id, "description_en": description, "lang": "en"}
            }])
            state.response = f"Done! Added '{description}' to {state.store_name} from {start_date} to {end_date}. Anything else you’d like to do? 😊"
        elif state.action == "update":
            old_desc = state.collected_data["description"]
            new_desc = state.collected_data["new_description"]
            db_execute(
                "UPDATE offers SET description_en = %s, description_ar = %s WHERE store_id = %s AND description_en = %s",
                (new_desc, new_desc, store_id, old_desc)
            )
            offer_id = db_fetch_one(
                "SELECT offer_id FROM offers WHERE store_id = %s AND description_en = %s",
                (store_id, new_desc)
            )["offer_id"]
            vector = embeddings.embed_query(new_desc)
            index.upsert(vectors=[{
                "id": f"offer_{offer_id}_en",
                "values": vector,
                "metadata": {"type": "offer", "id": offer_id, "description_en": new_desc, "lang": "en"}
            }])
            state.response = f"Updated! '{old_desc}' is now '{new_desc}' in {state.store_name}."
        elif state.action == "delete":
            description = state.collected_data["description"]
            offer = db_fetch_one(
                "SELECT offer_id FROM offers WHERE store_id = %s AND description_en = %s",
                (store_id, description)
            )
            if offer:
                offer_id = offer["offer_id"]
                db_execute("DELETE FROM offers WHERE store_id = %s AND description_en = %s", (store_id, description))
                index.delete(ids=[f"offer_{offer_id}_en"])
                state.response = f"Poof! '{description}' is gone from {state.store_name}."
            else:
                state.response = f"I couldn’t find '{description}' in {state.store_name}. Want to list your offers to check?"
    
    elif state.entity_type == "product":
        if state.action == "create":
            name = state.collected_data["name"]
            description = state.collected_data["description"]
            price = state.collected_data["price"]
            currency = state.collected_data.get("currency", "SAR")
            db_execute(
                "INSERT INTO products (store_id, name_en, name_ar, description_en, description_ar, price, currency) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (store_id, name, name, description, description, price, currency)
            )
            product_id = db_fetch_one("SELECT currval(pg_get_serial_sequence('products', 'product_id')) AS id")["id"]
            vector = embeddings.embed_query(f"{name} {description}")
            index.upsert(vectors=[{
                "id": f"product_{product_id}_en",
                "values": vector,
                "metadata": {"type": "product", "id": product_id, "name_en": name, "description_en": description, "price": price, "currency": currency, "lang": "en"}
            }])
            state.response = f"Done! Added '{name}' ({description}) to {state.store_name} for {price} {currency}. Anything else? 😊"
        elif state.action == "update":
            name = state.collected_data["name"]
            new_desc = state.collected_data["new_description"]
            db_execute(
                "UPDATE products SET description_en = %s, description_ar = %s WHERE store_id = %s AND name_en = %s",
                (new_desc, new_desc, store_id, name)
            )
            product = db_fetch_one(
                "SELECT product_id FROM products WHERE store_id = %s AND name_en = %s",
                (store_id, name)
            )
            product_id = product["product_id"]
            vector = embeddings.embed_query(f"{name} {new_desc}")
            index.upsert(vectors=[{
                "id": f"product_{product_id}_en",
                "values": vector,
                "metadata": {"type": "product", "id": product_id, "name_en": name, "description_en": new_desc, "lang": "en"}
            }])
            state.response = f"Updated! '{name}' now has description '{new_desc}' in {state.store_name}."
        elif state.action == "delete":
            name = state.collected_data["name"]
            product = db_fetch_one(
                "SELECT product_id FROM products WHERE store_id = %s AND name_en = %s",
                (store_id, name)
            )
            if product:
                product_id = product["product_id"]
                db_execute("DELETE FROM products WHERE store_id = %s AND name_en = %s", (store_id, name))
                index.delete(ids=[f"product_{product_id}_en"])
                state.response = f"Poof! '{name}' is gone from {state.store_name}."
            else:
                state.response = f"I couldn’t find '{name}' in {state.store_name}. Want to list your products to check?"
    
    # Reset state
    state.entity_type = None
    state.action = None
    state.collected_data = {}
    state.current_step = None

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
    tenant = db_fetch_one("SELECT tenant_id, email, password FROM tenants WHERE email ILIKE %s", (request.email,))
    if not tenant or tenant["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": str(tenant["tenant_id"]), "role": "tenant"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    lang = request.language or detect_language(request.text)
    session_id = get_or_create_session(request.session_id, request.user_id, lang)
    conversation_history = get_conversation_history(session_id)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_history]
    state = CustomerState(query=request.text, user_id=request.user_id, language=lang, session_id=session_id, conversation_history=history_dicts)
    result = customer_graph.invoke(state)
    add_message_to_conversation(session_id, "user", request.text)
    add_message_to_conversation(session_id, "assistant", result["response"])
    return ChatResponse(message=result["response"], session_id=session_id)

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    tenant = db_fetch_one("SELECT tenant_id FROM tenants WHERE tenant_id = %s", (request.user_id,))
    if not tenant:
        raise HTTPException(status_code=403, detail="Only tenants can update")
    
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
    
    state_json = json.dumps(result)
    db_execute("UPDATE conversations SET current_state = %s WHERE session_id = %s", (state_json, session_id))
    
    add_message_to_conversation(session_id, "user", request.text)
    add_message_to_conversation(session_id, "assistant", result["response"])
    
    return {"message": result["response"], "session_id": session_id}

@app.get("/")
async def root():
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}