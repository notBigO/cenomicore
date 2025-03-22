from typing import Optional, List, Dict, Any
from pydantic import BaseModel as PydanticBaseModel
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from datetime import datetime
import json
import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from utils import db_fetch_all, db_fetch_one, convert_to_json_safe, DateTimeEncoder, logger

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("cenomi")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

# Customer prompt template
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

customer_chain = customer_prompt | llm | StrOutputParser()

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
        customer_id = state.user_id[2:]
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