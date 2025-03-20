from fastapi import FastAPI, HTTPException
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from typing import Optional, Dict, Any
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

# Request models
class ChatRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    language: Optional[str] = None

class UpdateRequest(BaseModel):
    text: str
    user_id: str
    language: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

# Database helper functions
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
    except Exception as e:
        logger.error(f"Database error: {e}")

def convert_to_json_safe(data):
    if isinstance(data, list):
        return [convert_to_json_safe(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_json_safe(value) for key, value in data.items()}
    elif isinstance(data, datetime):
        return str(data)
    else:
        return data

# Language detection
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

# Customer prompt template
customer_prompt = PromptTemplate(
    input_variables=["context", "query", "lang"],
    template="""
    You are CenomiAI, a friendly mall assistant.
    Respond conversationally in {lang} with emojis 😊.
    Use the query and context to answer accurately and helpfully:
    - For stores: List name, location, and category if available. If asked "Are there any [type] stores," confirm their presence and provide examples if found.
    - For events: Include name, date, time, location even if they are already completed.
    - For offers: Provide all details, even if expired.
    - For loyalty: Use customer_loyalty and loyalty_programs data.
    - For services/amenities: Detail what’s available.
    If the context has relevant info, use it directly. If empty or insufficient, say, "I couldn’t find that info right now 😞, but I’ll keep looking! Can you give me more details?"
    This is READ-ONLY. No updates allowed.
    
    Query: "{query}"
    Context: {context}
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
    response: Optional[str] = None

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
    
    # Embed the query and search Pinecone
    query_vector = embeddings.embed_query(state.query)
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
    
    state.response = json.dumps(convert_to_json_safe(context))
    return state

def generate_response(state: CustomerState) -> CustomerState:
    response = customer_chain.invoke(
        {
            "context": state.response,
            "query": state.query,
            "lang": state.language,
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
    intent: Optional[str] = None
    data: Dict[str, Any] = {}
    response: Optional[str] = None

# Tenant workflow nodes
def recognize_intent(state: TenantUpdateState) -> TenantUpdateState:
    prompt = f"""
    You are CenomiAI, a tenant assistant. Parse the query into a JSON object for store/offer/product management:
    - Actions: 'read_store', 'add_offer', 'update_offer', 'delete_offer', 'update_store', 'update_product'
    - Include 'store', 'description', 'start_date', 'end_date', 'field', 'value' as needed.
    Query: "{state.query}"
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

@app.post("/chat")
async def chat(request: ChatRequest):
    lang = request.language or detect_language(request.text)
    state = CustomerState(query=request.text, user_id=request.user_id, language=lang)
    result = customer_graph.invoke(state)
    return {"message": result["response"]}

@app.post("/tenant/update")
async def tenant_update(request: UpdateRequest):
    tenant = db_fetch_one("SELECT tenant_id FROM tenants WHERE tenant_id = %s", (request.user_id,))
    if not tenant:
        raise HTTPException(status_code=403, detail="Only tenants can update")
    lang = request.language or "en"
    state = TenantUpdateState(query=request.text, user_id=request.user_id, language=lang)
    result = tenant_graph.invoke(state)
    return {"message": result["response"]}

@app.get("/")
async def root():
    return {"message": "Cenomi Chatbot with Gemini is up and running!"}