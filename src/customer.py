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
    # Fetch all brands with complete information including PMS unit codes for location
    brands = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.brand_name_ar, b.category_name, 
           b.description_en, b.store_phone_number, b.store_email, b.store_website,
           b.pms_unit_codes, bma.unique_property_id 
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id"""
    )
    
    # Fetch all products with complete information including price
    products = await db_fetch_all_async(
        """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
           p.is_featured, p.in_stock, b.brand_name_en
           FROM products p
           JOIN brands b ON p.brand_id = b.brand_id"""
    )
    
    # Fetch all engagements/offers with complete information
    engagements = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.brand_id, e.start_date, 
           e.end_date, e.terms_conditions_en, e.is_exclusive, e.unique_property_id
           FROM engagements e 
           WHERE e.type = 'offer'"""
    )
    
    # Clear existing graph to rebuild it completely
    knowledge_graph.clear()
    
    # Add brands/stores to knowledge graph with all available information
    for brand in brands:
        node_data = {
            "type": "store",
            "name": brand["brand_name_en"],
            "name_ar": brand.get("brand_name_ar", ""),
            "category": brand.get("category_name", ""),
            "description": brand.get("description_en", ""),
            "phone": brand.get("store_phone_number", ""),
            "email": brand.get("store_email", ""),
            "website": brand.get("store_website", ""),
            "mall_id": brand["unique_property_id"],
            "location": brand.get("pms_unit_codes", {})  # PMS codes for location
        }
        knowledge_graph.add_node(f"brand_{brand['brand_id']}", **node_data)
    
    # Add products to knowledge graph with complete information including price
    for product in products:
        node_data = {
            "type": "product",
            "name": product["name"],
            "description": product.get("description", ""),
            "price": float(product["price"]) if product.get("price") is not None else None,
            "category": product.get("category", ""),
            "brand_name": product.get("brand_name_en", ""),
            "in_stock": product.get("in_stock", True),
            "is_featured": product.get("is_featured", False)
        }
        knowledge_graph.add_node(f"product_{product['id']}", **node_data)
        knowledge_graph.add_edge(f"brand_{product['brand_id']}", f"product_{product['id']}")
    
    # Add engagements/offers to knowledge graph
    for engagement in engagements:
        node_data = {
            "type": "offer",
            "title": engagement.get("title_en", ""),
            "description": engagement.get("description_en", ""),
            "start_date": engagement.get("start_date", ""),
            "end_date": engagement.get("end_date", ""),
            "terms": engagement.get("terms_conditions_en", ""),
            "is_exclusive": bool(engagement.get("is_exclusive", 0)),
            "mall_id": engagement.get("unique_property_id")
        }
        knowledge_graph.add_node(f"engagement_{engagement['engagement_id']}", **node_data)
        if engagement["brand_id"]:
            knowledge_graph.add_edge(f"brand_{engagement['brand_id']}", f"engagement_{engagement['engagement_id']}")

# Intent Classification Prompt
intent_classification_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    You are CenomiAI, a mall assistant. Parse this query to determine the user's intent based on the query and conversation history.

    Valid entities: store, offer, product, event, service, amenity, loyalty
    Valid actions: info, navigate, recommend, list, balance, programs

    Query: "{query}"

    Previous conversation:
    {conversation_history}

    Instructions:
    - Use the conversation history to resolve vague terms like "it", "that", or "the store" to specific entities mentioned earlier.
    - If the query is a follow-up (e.g., "Where is it?"), link it to the most recent entity from history.
    - For broad queries (e.g., "What's good here?"), assume 'recommend' or 'list' based on context.
    - For loyalty-related queries, identify if the user is asking about their points balance or the programs they are enrolled in.
    - Extract specific details (e.g., store name) into collected_data.

    Return a JSON object with:
    - entity_type: What they're asking about (store, offer, product, etc., or loyalty)
    - action: What they want (info, navigate, recommend, list, balance, programs)
    - collected_data: Any details provided (e.g., "name": "Tiffany & Co.")

    Example outputs:
    ```json
    {{"entity_type": "product", "action": "info", "collected_data": {{"name": "wedding ring"}}}}
    {{"entity_type": "loyalty", "action": "balance", "collected_data": {{}}}}
    {{"entity_type": "loyalty", "action": "programs", "collected_data": {{}}}}
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
    conversation_id: str
    conversation_history: List[Dict[str, str]] = []
    response: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None
    initial_context: Optional[List[Dict[str, Any]]] = None
    suggestions: Optional[List[str]] = None
    mall_id: Optional[int] = None
    direct_response: Optional[str] = None

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
        state.response = "I'm having trouble understanding that 😅. Could you clarify what you're asking about?"
        return state

    state.intent = f"{intent_json['entity_type']}_{intent_json['action']}"
    state.context_data = state.context_data or {}
    if intent_json["collected_data"].get("name"):
        state.context_data["resolved_entity"] = intent_json["collected_data"]["name"]
    if state.intent.startswith("other_"):
        state.response = "I'm not sure what you mean 😅. Could you tell me more?"
    logger.info(f"Classified intent: {state.intent}, Resolved entity: {state.context_data.get('resolved_entity')}")
    return state

async def initial_retrieval(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
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
    all_brands = await db_fetch_all_async(
        "SELECT b.brand_name_en, b.category_name FROM brands b JOIN brand_mall_association bma ON b.brand_id = bma.brand_id WHERE bma.unique_property_id = $1", 
        (state.mall_id,)
    )
    store_names = [s["brand_name_en"].lower() for s in all_brands if s["brand_name_en"]]
    categories = set(s["category_name"].lower() for s in all_brands if s["category_name"])
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
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
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
    
    # Get mall name
    mall = await db_fetch_one_async("SELECT marketing_name AS name_en FROM malls WHERE unique_property_id = $1", (state.mall_id,))
    context["mall_name"] = mall["name_en"] if mall else "Unknown Mall"

    # For store queries, make sure to fetch ALL stores for a given mall
    if state.intent.startswith("store_"):
        store_name = state.context_data.get("resolved_entity", "").lower()
        
        # Fetch all stores for this mall directly from database for accuracy
        all_stores = await db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
               b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
               WHERE bma.unique_property_id = $1""", 
            (state.mall_id,)
        )
        
        for store in all_stores:
            store_data = {
                "name": store["brand_name_en"],
                "category": store.get("category_name", ""),
                "store_id": store["brand_id"],
                "description": store.get("description_en", ""),
                "phone": store.get("store_phone_number", ""),
                "email": store.get("store_email", ""),
                "website": store.get("store_website", ""),
                "location": store.get("pms_unit_codes", {})
            }
            
            # If this is the store being searched for, put it at the top
            if store_name and store_name in store["brand_name_en"].lower():
                context["stores"].insert(0, store_data)
            else:
                context["stores"].append(store_data)
    
    # Fetch product details including price
    if state.intent.startswith("product_"):
        product_name = state.context_data.get("resolved_entity", "").lower()
        
        # Search for products either by name or for a specific store
        if product_name:
            products = await db_fetch_all_async(
                """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
                   p.is_featured, p.in_stock, b.brand_name_en
                   FROM products p
                   JOIN brands b ON p.brand_id = b.brand_id
                   JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                   WHERE bma.unique_property_id = $1 AND 
                   (LOWER(p.name) LIKE $2 OR LOWER(b.brand_name_en) LIKE $2)""",
                (state.mall_id, f"%{product_name}%")
            )
        else:
            products = await db_fetch_all_async(
                """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
                   p.is_featured, p.in_stock, b.brand_name_en
                   FROM products p
                   JOIN brands b ON p.brand_id = b.brand_id
                   JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                   WHERE bma.unique_property_id = $1 
                   LIMIT 20""",
                (state.mall_id,)
            )
        
        for product in products:
            context["products"].append({
                "id": product["id"],
                "name": product["name"],
                "description": product.get("description", ""),
                "price": float(product["price"]) if product.get("price") is not None else None,
                "brand_id": product["brand_id"],
                "category": product.get("category", ""),
                "store_name": product.get("brand_name_en", ""),
                "in_stock": product.get("in_stock", True)
            })

    # Process Pinecone results
    brand_ids = set()
    for doc in state.initial_context or []:
        metadata = doc["metadata"]
        if metadata.get("mall_id") != state.mall_id:
            continue
        
        doc_type = metadata.get("type")
        if doc_type == "store" and not state.intent.startswith("store_"):
            # Only add store from vector search if not already doing a direct store query
            store = {
                "name": metadata.get("name_en"),
                "category": metadata.get("category_en"),
                "store_id": metadata.get("brand_id"),
                "description": metadata.get("description_en"),
            }
            if store not in context["stores"]:
                context["stores"].append(store)
            if metadata.get("brand_id"):
                brand_ids.add(metadata.get("brand_id"))
                
        elif doc_type == "engagement":
            # Determine if this is an offer or event based on type
            engagement_type = metadata.get("type", "").lower()
            if engagement_type == "offer":
                offer = {
                    "id": metadata.get("engagement_id"),
                    "description": metadata.get("description_en"),
                    "brand_id": metadata.get("brand_id"),
                    "store_name": "",  # Will be filled in later
                    "start_date": metadata.get("start_date"),
                    "end_date": metadata.get("end_date")
                }
                context["offers"].append(offer)
                if metadata.get("brand_id"):
                    brand_ids.add(metadata["brand_id"])
            elif engagement_type == "event":
                context["events"].append({
                    "name": metadata.get("name_en"),
                    "date": metadata.get("start_date"),
                    "description": metadata.get("description_en"),
                })
                
        elif doc_type == "product" and not state.intent.startswith("product_"):
            # Only add product from vector search if not already doing a direct product query
            product = {
                "id": metadata.get("id"),
                "name": metadata.get("name"),
                "brand_id": metadata.get("brand_id"),
                "category": metadata.get("category"),
                "store_name": metadata.get("brand_name_en"),
            }
            context["products"].append(product)
            if metadata.get("brand_id"):
                brand_ids.add(metadata["brand_id"])
                
        elif doc_type == "service":
            context["services"].append({
                "name": metadata.get("name"),
            })

    # Fetch additional brand details for offers and products
    if brand_ids:
        brands = await db_fetch_all_async(
            """SELECT brand_id, brand_name_en, category_name, description_en, 
               store_phone_number, store_email, store_website, pms_unit_codes 
               FROM brands WHERE brand_id = ANY($1)""",
            (list(brand_ids),)
        )
        brand_map = {b["brand_id"]: b for b in brands}
        
        for item in context["offers"] + context["products"]:
            if item.get("brand_id") in brand_map and not item.get("store_name"):
                brand = brand_map[item["brand_id"]]
                item["store_name"] = brand["brand_name_en"]
                if "category" not in item and brand["category_name"]:
                    item["category"] = brand["category_name"]
                if "location" not in item and brand.get("pms_unit_codes"):
                    item["location"] = brand["pms_unit_codes"]

    # If no stores found but resolved entity exists, try a direct DB lookup
    if not context["stores"] and state.context_data.get("resolved_entity"):
        store_name = state.context_data["resolved_entity"].lower()
        stores = await db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
               b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
               WHERE bma.unique_property_id = $1 AND LOWER(b.brand_name_en) LIKE $2""", 
            (state.mall_id, f"%{store_name}%")
        )
        
        for store in stores:
            context["stores"].append({
                "name": store["brand_name_en"],
                "category": store.get("category_name", ""),
                "store_id": store["brand_id"],
                "description": store.get("description_en", ""),
                "phone": store.get("store_phone_number", ""),
                "email": store.get("store_email", ""),
                "website": store.get("store_website", ""),
                "location": store.get("pms_unit_codes", {})
            })

    state.context_data = context
    state.response = json.dumps(convert_to_json_safe(context))
    return state

async def generate_response(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
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

async def fetch_loyalty_data(state: CustomerState) -> CustomerState:
    # Since loyalty tables don't exist in the schema, we'll return a message that it's not available
    state.response = "I'm sorry, but the loyalty program features are not available at this time."
    return state

# Workflow
customer_workflow = StateGraph(CustomerState)
customer_workflow.add_node("classify_intent", classify_intent)
customer_workflow.add_node("initial_retrieval", initial_retrieval)
customer_workflow.add_node("refine_context", refine_context)
customer_workflow.add_node("respond", generate_response)
customer_workflow.add_node("fetch_loyalty_data", fetch_loyalty_data)
customer_workflow.set_entry_point("classify_intent")

def route_after_classify(state: CustomerState):
    if state.intent in ["loyalty_balance", "loyalty_programs"]:
        return "fetch_loyalty_data"
    elif state.intent.startswith("other_") and state.response:
        return "respond"
    else:
        return "initial_retrieval"
    
customer_workflow.add_conditional_edges(
    "classify_intent",
    route_after_classify,
    {
        "fetch_loyalty_data": "fetch_loyalty_data",
        "respond": "respond",
        "initial_retrieval": "initial_retrieval",
    }
)
customer_workflow.add_edge("fetch_loyalty_data", END)
customer_workflow.add_edge("initial_retrieval", "refine_context")
customer_workflow.add_edge("refine_context", "respond")
customer_workflow.add_edge("respond", END)
customer_graph = customer_workflow.compile()