from typing import Optional, List, Dict, Any
from pydantic import BaseModel as PydanticBaseModel
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from datetime import datetime
import json
import os
import asyncio
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from utils import db_fetch_all_async, db_fetch_one_async, convert_to_json_safe, DateTimeEncoder, REDIS_CLIENT, logger
import networkx as nx
import spacy

# Load spaCy NLP model for store name and category extraction
nlp = spacy.load("en_core_web_sm")

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("cenomicore")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

# Knowledge graph for relationships
knowledge_graph = nx.Graph()

async def populate_knowledge_graph():
    stores = await db_fetch_all_async("SELECT store_id, name_en, mall_id FROM stores")
    products = await db_fetch_all_async("SELECT product_id, name_en, store_id FROM products")
    offers = await db_fetch_all_async("SELECT offer_id, description_en, store_id FROM offers")
    for store in stores:
        knowledge_graph.add_node(f"store_{store['store_id']}", type="store", name=store["name_en"], mall_id=store["mall_id"])
    for product in products:
        knowledge_graph.add_node(f"product_{product['product_id']}", type="product", name=product["name_en"])
        knowledge_graph.add_edge(f"store_{product['store_id']}", f"product_{product['product_id']}")
    for offer in offers:
        knowledge_graph.add_node(f"offer_{offer['offer_id']}", type="offer", description=offer["description_en"])
        knowledge_graph.add_edge(f"store_{offer['store_id']}", f"offer_{offer['offer_id']}")

# Intent Classification Prompt
intent_classification_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    You are CenomiAI, a mall assistant. Parse this query to determine the user's intent based on the query and conversation history.

    Valid entities: store, offer, product, event, service, amenity
    Valid actions: info, navigate, recommend, list

    Query: "{query}"

    Previous conversation:
    {conversation_history}

    Instructions:
    - Use the conversation history to resolve vague terms like "it", "that", or "the store" to specific entities mentioned earlier.
    - If the query is a follow-up (e.g., "Where is it?"), link it to the most recent entity from history.
    - For broad queries (e.g., "What’s good here?"), assume 'recommend' or 'list' based on context.
    - Extract specific details (e.g., store name) into collected_data.

    Return a JSON object with:
    - entity_type: What they’re asking about (store, offer, product, etc.)
    - action: What they want (info, navigate, recommend, list)
    - collected_data: Any details provided (e.g., "name": "Tiffany & Co.")

    Example output:
    ```json
    {{"entity_type": "product", "action": "info", "collected_data": {{"name": "wedding ring"}}}}
    """
)
intent_chain = intent_classification_prompt | llm | StrOutputParser()

# Customer Response Prompt
customer_prompt = PromptTemplate(
    input_variables=["context", "query", "lang", "conversation_history", "mall_name", "resolved_entity"],
    template="""
    You are CenomiAI, a friendly, proactive, and highly knowledgeable assistant for {mall_name} mall. 
    
    # Core Identity
    - Respond in {lang} with a warm, conversational tone
    - Use appropriate emojis 😊 to keep interaction engaging without overusing them
    - Your purpose is to be the definitive source of information about {mall_name} mall
    
    # Context Awareness
    - Always reference the most recent conversation history to maintain continuity: {conversation_history}
    - If a resolved entity is provided (e.g., "{resolved_entity}"), treat it as the subject of the query unless contradicted
    - When users refer to something previously mentioned ("it", "that store", "those products"), connect back to the resolved entity or specific items from earlier in the conversation
    - If the user asks follow-up questions, ensure your answers build on previous exchanges rather than starting fresh
    - For multi-part questions, address each component thoroughly
    
    # Response Guidelines
    
    ## Store Information
    - Provide specific details: exact location (floor, section), operating hours, contact information
    - Include relevant category and description of what the store offers
    - If the user asks about a store not mentioned in context, acknowledge this and suggest similar stores in {mall_name}
    
    ## Product Queries (including shopping lists)
    - For each item requested, match to specific stores that carry it in {mall_name}
    - Include product details: price, availability, features, and store location
    - For lists, organize recommendations by store location to create an efficient shopping route
    - Structure as a clear, numbered list when responding to multiple items
    
    ## Dining Recommendations
    - Suggest restaurants based on cuisine type, price range, dietary requirements, or ambiance
    - Include location details, specialty dishes, and current promotions
    - For families, highlight kid-friendly options and special menus
    - Mention seating availability (food court vs. sit-down restaurant)
    
    ## Offers & Events
    - Highlight current promotions with specific details (discount amounts, conditions, end dates)
    - Connect offers to user's interests based on conversation history
    - For events, include dates, times, locations, and any registration requirements
    - Personalize recommendations based on previous interactions
    - If a specific store is mentioned or implied (e.g., "they"), list its offers.
    - If no offers exist for that store, say so gracefully and suggest offers from similar stores by category (e.g., fashion, electronics).
    
    ## Navigation Assistance
    - Provide clear, step-by-step directions within {mall_name}
    - Reference landmarks and store names as navigation points
    - Mention transportation options (elevators, escalators, walking distances)
    - If starting point isn't specified, provide directions from main entrance or information desk
    
    ## Vague/Open-Ended Queries
    - For broad requests ("What's good here?", "I'm so bored", "I'm so hungry"), propose a structured plan with multiple options
    - Segment recommendations by categories (shopping, dining, entertainment)
    - Ground suggestions in user's previous interests if available from conversation history
    - Present a clear, actionable itinerary that covers different areas of the mall
    
    ## Personalization
    - Remember and reference previous interactions within the same session
    
    # Special Handling Instructions
    
    - For complex queries, break down information into digestible sections
    - If information is not available in context, clearly state this and provide the most relevant alternative from {mall_name}
    - Always prioritize accuracy over completeness - if uncertain about details, acknowledge limitations
    - Maintain consistent personality throughout the conversation, building rapport over multiple exchanges
    - For time-sensitive queries, prioritize current events and ongoing promotions
    
    # Contextual Information Processing
    
    Carefully analyze the provided context about {mall_name}:
    {context}
    
    Review the full conversation history to maintain continuity:
    {conversation_history}
    
    Now respond to the current query with complete, helpful information:
    "{query}"
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
    intent: Optional[str] = None
    initial_context: Optional[List[Dict[str, Any]]] = None
    suggestions: Optional[List[str]] = None
    mall_id: Optional[int] = None

async def classify_intent(state: CustomerState) -> CustomerState:
    formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-6:]]) if state.conversation_history else "No prior conversation."
    intent_result = await intent_chain.ainvoke({"query": state.query, "conversation_history": formatted_history})
    logger.info(f"Raw intent_result: '{intent_result}'")
    
    cleaned_result = intent_result.strip()
    if cleaned_result.startswith("```json") and cleaned_result.endswith("```"):
        cleaned_result = cleaned_result[7:-3].strip()
    elif cleaned_result.startswith("```") and cleaned_result.endswith("```"):
        cleaned_result = cleaned_result[3:-3].strip()
    
    try:
        intent_json = json.loads(cleaned_result)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse intent_result as JSON: '{cleaned_result}', Error: {e}")
        state.intent = "other_info"
        state.response = "I’m having trouble understanding that 😅. Could you clarify what you’re asking about?"
        return state

    state.intent = f"{intent_json['entity_type']}_{intent_json['action']}"
    state.context_data = state.context_data or {}
    if intent_json["collected_data"].get("name"):
        state.context_data["resolved_entity"] = intent_json["collected_data"]["name"]
    if state.intent.startswith("other_"):
        state.response = "I’m not sure what you mean 😅. Could you tell me more?"
    logger.info(f"Classified intent: {state.intent}, Resolved entity: {state.context_data.get('resolved_entity')}")
    return state

async def initial_retrieval(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you’re asking about. Please select a mall first! 😊"
        return state

    cache_key = f"initial_context:{state.query}:{state.intent}:{state.user_id or 'anon'}:{state.mall_id}"
    cached_context = REDIS_CLIENT.get(cache_key)
    if cached_context:
        state.initial_context = json.loads(cached_context)
        return state

    query_items = [item.strip() for item in state.query.split("\n") if item.strip()] if "\n" in state.query else [state.query]
    intent_prefixes = {
        "store_info": "store",
        "store_navigate": "store",
        "store_recommend": "store",
        "store_list": "store",
        "product_info": "product",
        "product_recommend": "product",
        "product_list": "product",
        "offer_info": "offer",
        "offer_recommend": "offer",
        "offer_list": "offer",
        "event_info": "event",
        "event_recommend": "event",
        "event_list": "event",
        "service_info": "service",
        "service_recommend": "service",
        "service_list": "service",
        "amenity_info": "amenity",
        "amenity_navigate": "amenity",
        "amenity_list": "amenity",
    }
    entity_type = state.intent.split("_")[0]
    query_prefix = intent_prefixes.get(state.intent, entity_type)

    # Extract store name or category
    store_name, category = None, None
    all_stores = await db_fetch_all_async("SELECT name_en, category_en FROM stores WHERE mall_id = $1", (state.mall_id,))
    store_names = [s["name_en"].lower() for s in all_stores]
    categories = set(s["category_en"].lower() for s in all_stores)
    for item in query_items:
        doc = nlp(item.lower())
        for ent in doc.ents:
            if ent.text in store_names:
                store_name = ent.text
                break
        if not store_name:
            for name in store_names:
                if name in item.lower():
                    store_name = name
                    break
        if not store_name:
            for cat in categories:
                if cat in item.lower():
                    category = cat
                    break
        if store_name or category:
            break

    # Build query vector
    if store_name:
        query_vectors = [embeddings.embed_query(f"{query_prefix} {store_name}")]
    elif category:
        query_vectors = [embeddings.embed_query(f"{query_prefix} {category} stores")]
    else:
        query_vectors = [embeddings.embed_query(f"{query_prefix} {item}") for item in query_items]

    avg_vector = [sum(v[i] for v in query_vectors) / len(query_vectors) for i in range(len(query_vectors[0]))]

    # Enhance with history
    if state.conversation_history and any(word in state.query.lower() for word in ["they", "it", "that", "this", "there", "those"]):
        last_response = next((msg["content"] for msg in reversed(state.conversation_history[-6:]) if msg["role"] == "assistant"), "")
        if last_response:
            history_vector = embeddings.embed_query(last_response)
            avg_vector = [(a + h) / 2 for a, h in zip(avg_vector, history_vector)]

    # Pinecone query
    filter = {"mall_id": state.mall_id}
    results = await asyncio.to_thread(index.query, vector=avg_vector, top_k=25, include_metadata=True, filter=filter)
    state.initial_context = [{"id": doc["id"], "score": float(doc["score"]), "metadata": doc["metadata"]} for doc in results.get("matches", [])]
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    return state

async def refine_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you’re asking about. Please select a mall first! 😊"
        return state

    context = {
        "stores": [],
        "offers": [],
        "events": [],
        "products": [],
        "services": [],
        "amenities": [],
        "mall_name": "",
    }
    mall = await db_fetch_one_async("SELECT name_en FROM malls WHERE mall_id = $1", (state.mall_id,))
    context["mall_name"] = mall["name_en"] if mall else "Unknown Mall"

    # Fetch offers explicitly for offer_info intent
    if state.intent.startswith("offer_"):
        offers = await db_fetch_all_async(
            "SELECT o.offer_id, o.description_en, o.store_id "
            "FROM offers o "
            "JOIN stores s ON o.store_id = s.store_id "
            "WHERE s.mall_id = $1",
            (state.mall_id,)
        )
        for offer in offers:
            store = await db_fetch_one_async(
                "SELECT name_en, location_en FROM stores WHERE store_id = $1 AND mall_id = $2",
                (offer["store_id"], state.mall_id)
            )
            context["offers"].append({
                "id": offer["offer_id"],
                "description": offer["description_en"],
                "store_id": offer["store_id"],
                "store_name": store["name_en"] if store else "Unknown Store",
                "location_en": store["location_en"] if store else "Unknown Location",
            })

    # Process Pinecone results
    store_ids = set()
    for doc in state.initial_context or []:
        metadata = doc["metadata"]
        if metadata.get("mall_id") != state.mall_id:
            continue
        doc_type = metadata.get("type")
        if doc_type == "store":
            store = {
                "name": metadata.get("name_en"),
                "location": metadata.get("location_en"),
                "category": metadata.get("category_en"),
                "store_id": metadata.get("id"),
            }
            if store not in context["stores"]:
                context["stores"].append(store)
            store_ids.add(metadata.get("id"))
        elif doc_type == "offer":
            offer = {
                "id": metadata.get("id"),
                "description": metadata.get("description_en"),
                "store_id": metadata.get("store_id"),
                "store_name": metadata.get("store_name"),
                "location_en": metadata.get("location_en"),
            }
            context["offers"].append(offer)
            if metadata.get("store_id"):
                store_ids.add(metadata["store_id"])
        elif doc_type == "product":
            product = {
                "id": metadata.get("id"),
                "name": metadata.get("name_en"),
                "description": metadata.get("description_en"),
                "store_id": metadata.get("store_id"),
                "store_name": metadata.get("store_name"),
                "location_en": metadata.get("location_en"),
            }
            context["products"].append(product)
            if metadata.get("store_id"):
                store_ids.add(metadata["store_id"])
        elif doc_type == "event":
            context["events"].append({
                "name": metadata.get("name_en"),
                "date": metadata.get("start_time"),
                "location": metadata.get("location_en"),
            })
        elif doc_type == "service":
            context["services"].append({
                "name": metadata.get("name_en"),
                "description": metadata.get("description_en"),
            })
        elif doc_type == "amenity":
            context["amenities"].append({
                "name": metadata.get("name_en"),
                "location": metadata.get("location_en"),
            })

    # Fetch additional store details
    if store_ids:
        stores = await db_fetch_all_async(
            "SELECT store_id, name_en, location_en, category_en FROM stores WHERE store_id = ANY($1) AND mall_id = $2",
            (list(store_ids), state.mall_id)
        )
        store_map = {s["store_id"]: s for s in stores}
        for item in context["offers"] + context["products"]:
            if item.get("store_id") in store_map and not item.get("store_name"):
                store = store_map[item["store_id"]]
                item["store_name"] = store["name_en"]
                item["location_en"] = store["location_en"]

    state.context_data = context
    state.response = json.dumps(convert_to_json_safe(context))
    return state

async def generate_response(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you’re asking about. Please select a mall first! 😊"
        return state

    formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-6:]]) if state.conversation_history else "No prior conversation."
    response = await asyncio.to_thread(
        customer_chain.invoke,
        {
            "context": state.response,
            "query": state.query,
            "lang": state.language,
            "conversation_history": formatted_history,
            "mall_name": state.context_data.get("mall_name", "the mall"),
            "resolved_entity": state.context_data.get("resolved_entity", ""),
        },
    )
    state.response = response
    return state

# Workflow
customer_workflow = StateGraph(CustomerState)
customer_workflow.add_node("classify_intent", classify_intent)
customer_workflow.add_node("initial_retrieval", initial_retrieval)
customer_workflow.add_node("refine_context", refine_context)
customer_workflow.add_node("respond", generate_response)
customer_workflow.set_entry_point("classify_intent")
customer_workflow.add_conditional_edges(
    "classify_intent",
    lambda state: "respond" if state.intent.startswith("other_") and state.response else "initial_retrieval",
)
customer_workflow.add_edge("initial_retrieval", "refine_context")
customer_workflow.add_edge("refine_context", "respond")
customer_workflow.add_edge("respond", END)
customer_graph = customer_workflow.compile()