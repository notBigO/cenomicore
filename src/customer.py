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
    
    # Fetch ALL engagements (both offers and events) with complete information
    engagements = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, e.start_date, 
           e.end_date, e.terms_conditions_en, e.is_exclusive, e.unique_property_id
           FROM engagements e"""
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
    
    # Add all engagements to knowledge graph
    for engagement in engagements:
        engagement_type = engagement.get("type", "").lower()
        node_data = {
            "type": engagement_type,
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
# intent_classification_prompt = PromptTemplate(
#     input_variables=["query", "conversation_history"],
#     template="""
#     You are CenomiAI, a mall assistant. Parse this query to determine the user's intent based on the query and conversation history.

#     Valid entities: store, offer, product, event, service, amenity, loyalty
#     Valid actions: info, navigate, recommend, list, balance, programs

#     Query: "{query}"

#     Previous conversation:
#     {conversation_history}

#     Instructions:
#     - Use the conversation history to resolve vague terms like "it", "that", or "the store" to specific entities mentioned earlier.
#     - If the query is a follow-up (e.g., "Where is it?"), link it to the most recent entity from history.
#     - For broad queries (e.g., "What's good here?"), assume 'recommend' or 'list' based on context.
#     - For loyalty-related queries, identify if the user is asking about their points balance or the programs they are enrolled in.
#     - Extract specific details (e.g., store name) into collected_data.
#     - Identify if there's a specific type preference mentioned (e.g., "Italian" food, "sports" shoes, "luxury" brands) and add to type_preference.

#     Return a JSON object with:
#     - entity_type: What they're asking about (store, offer, product, etc., or loyalty)
#     - action: What they want (info, navigate, recommend, list, balance, programs)
#     - collected_data: Any details provided (e.g., "name": "Tiffany & Co.")
#     - type_preference: Any specific type/category mentioned (e.g., "Italian", "sports", "casual", "luxury")

#     Example outputs:
#     ```json
#     {{"entity_type": "product", "action": "info", "collected_data": {{"name": "wedding ring"}}, "type_preference": "luxury"}}
#     {{"entity_type": "store", "action": "recommend", "collected_data": {{}}, "type_preference": "dining"}}
#     {{"entity_type": "loyalty", "action": "balance", "collected_data": {{}}, "type_preference": null}}
#     """
# )
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
    - For broad queries (e.g., "What’s good here?"), assume 'recommend' or 'list' based on context.
    - For loyalty-related queries, identify if the user is asking about their points balance or the programs they are enrolled in.
    - Extract specific details (e.g., store name) into collected_data.

    Return a JSON object with:
    - entity_type: What they’re asking about (store, offer, product, etc., or loyalty)
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

# Updated Customer Response Prompt
# customer_prompt = PromptTemplate(
#     input_variables=["context", "query", "lang", "conversation_history", "mall_name", "resolved_entity", "type_preference", "needs_type_follow_up"],
#     template="""
#     You are CenomiAI, a professional and knowledgeable assistant for {mall_name} mall.
    
#     # Core Identity
#     - Respond in {lang} with a concise, professional tone
#     - Use emojis 😊 sparingly to keep interactions engaging
#     - Provide accurate, precise information about {mall_name} mall
    
#     # Context Awareness
#     - Maintain continuity based on: {conversation_history}
#     - Use resolved entity "{resolved_entity}" as the subject unless contradicted
#     - Connect to previously mentioned entities without explicitly stating "since we were talking about X"
#     - For follow-up questions, build on previous exchanges without restating context
    
#     # Type Preference Framework
#     - Type preference: "{type_preference}" - When provided, filter results to match this preference
#     - Needs follow-up for type: {needs_type_follow_up}
#     - For queries about food/dining: If no type preference, show general options and ask about cuisine preferences (Italian, Indian, Fast food, etc.)
#     - For queries about products: If no type preference, show general options and ask about specific types (Sports shoes, Formal shoes, etc.)
#     - For queries about stores: If no type preference, show general options and ask about specific categories they're interested in
    
#     # Follow-up Structure
#     - When no type preference AND needs_type_follow_up is true:
#       1. List the relevant stores/products/services first
#       2. End with "Do you have a preference for any specific type of [product/cuisine/service]?" to guide the conversation
#     - When type preference IS provided:
#       1. Show only the items matching that preference
#       2. End with a friendly note like "Enjoy your meal!" for food or "Hope you find the perfect pair!" for shoes
    
#     # Response Guidelines
    
#     ## Store Information
#     - Format location codes properly: 
#       * "FF" = "First Floor" (e.g., FF08 becomes "First Floor, Shop #08")
#       * "GF" = "Ground Floor" (e.g., GF12 becomes "Ground Floor, Shop #12") 
#       * "BSW" = "Basement West" (e.g., BSW001 becomes "Basement West, Shop #001")
#     - Show only 3-5 location codes maximum and mention "and additional locations" if there are more
#     - Include category and description of store offerings
#     - For neighboring stores, highlight those with adjacent shop numbers on the same floor
    
#     ## Product Queries
#     - Include price, availability, features, and formatted store location
#     - Structure as clear, numbered lists for multiple items
    
#     ## Offers & Events
#     - Include specific details (discount amounts, conditions, dates) for both offers and events
#     - If a specific store is mentioned, list its offers/events directly without saying "Since we were talking about X..."
    
#     ## Navigation Assistance
#     - Provide clear, step-by-step directions using proper floor names (not codes)
#     - Reference landmarks and nearby stores as navigation points
    
#     # Special Handling Instructions
#     - Keep responses concise and to the point
#     - Format location codes in human-readable form (e.g., "First Floor, Shop #12" instead of "FF12")
#     - Avoid overwhelming users with too many location codes at once
#     - When asked about neighboring stores, identify those with similar location codes
#     - For services and amenities, provide complete details including location and description
#     - Avoid self-reassuring phrases like "Since we were talking about Zara..." - instead, directly address the question
    
#     # Contextual Information Processing
    
#     Carefully analyze the provided context about {mall_name}:
#     {context}
    
#     Review the conversation history to maintain continuity without explicitly mentioning it:
#     {conversation_history}
    
#     Now respond to the current query with concise, professional information:
#     "{query}"
#     """
# )
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

    - Keep the response concise and to the point. Preferably within 1 line.
    
    ## Store Information
    - Provide specific details: location (floor, section), operating hours.
    - Include relevant category and description of what the store offers
    - If the user asks about a store not mentioned in context, acknowledge this and suggest similar stores in {mall_name} 
    - Do not give out any sort of contact information for stores. 
    
    
    ## Product Queries (including shopping lists)
    - For the product requested, match to specific stores that have the product in {mall_name}.
    - Include product details: price, offers, availability, features and store location
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
    
    
    ## Vague/Open-Ended Queries
    - For broad requests ("What's good here?", "I'm so bored", "I'm so hungry"), propose a structured plan with multiple options
    - Segment recommendations by categories (shopping, dining, entertainment)
    - Ground suggestions in user's previous interests if available from conversation history
    - Present a clear, actionable itinerary that covers different areas of the mall
    
    ## Personalization
    - Remember and reference previous interactions within the same session
    
    # Special Handling Instructions
    
    - Ask one follow up question at the end of your response if the type of product is not clear, or if the cuisine for dining is not clear or if the type of event is not clear for the same query type else end your response with a nice note.
    - If information is not available in context, clearly state this and provide the most relevant alternative from {mall_name}
    - Maintain consistent personality throughout the conversation, building rapport over multiple exchanges
    
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
    type_preference: Optional[str] = None  # Added field for type preference
    needs_type_follow_up: bool = False  # Added flag to indicate if we need to ask for type preference

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
    
    # Set resolved entity if present
    if intent_json["collected_data"].get("name"):
        state.context_data["resolved_entity"] = intent_json["collected_data"]["name"]
    
    # Set type preference if present
    state.type_preference = intent_json.get("type_preference")
    
    # Determine if we need a type follow-up based on the entity type and whether a preference was provided
    if state.intent.startswith(("product_", "store_", "service_")) and not state.type_preference:
        state.needs_type_follow_up = True
    else:
        state.needs_type_follow_up = False
    
    if state.intent.startswith("other_"):
        state.response = "I'm not sure what you mean 😅. Could you tell me more?"
    
    logger.info(f"Classified intent: {state.intent}, Resolved entity: {state.context_data.get('resolved_entity')}, Type preference: {state.type_preference}")
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
    filter_dict = {"mall_id": state.mall_id}
    try:
        results = await asyncio.to_thread(
            index.query, 
            vector=avg_vector, 
            top_k=25, 
            include_metadata=True, 
            filter=filter_dict
        )
        matches = results.get("matches", [])
        state.initial_context = [{"id": doc["id"], "score": float(doc["score"]), "metadata": doc["metadata"]} for doc in matches]
    except Exception as e:
        logger.error(f"Pinecone query error: {e}")
        state.initial_context = []
    
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    return state

async def refine_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Please select a mall first."
        return state

    context = {
        "stores": [],
        "offers": [],
        "events": [],
        "products": [],
        "services": [],
        "amenities": [],
        "mall_name": "",
        "neighboring_stores": [],  # Field for neighboring stores
        "category_types": {        # New field for organizing available category types
            "food": set(),         # Food/restaurant types (Italian, Fast Food, etc.)
            "product": set(),      # Product types (Sports, Casual, Electronics, etc.)
            "store": set()         # Store categories (Fashion, Electronics, etc.)
        }
    }
    
    # Get mall name
    mall = await db_fetch_one_async("SELECT marketing_name AS name_en FROM malls WHERE unique_property_id = $1", (state.mall_id,))
    context["mall_name"] = mall["name_en"] if mall else "Unknown Mall"

    # Fetch all engagements (offers and events) without any date filtering
    # to show past, current, and future engagements
    engagements = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
           e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, e.unique_property_id
           FROM engagements e 
           WHERE e.unique_property_id = $1""",
        (state.mall_id,)
    )
    
    for engagement in engagements:
        # Get associated brand information
        brand = None
        if engagement.get("brand_id"):
            brand = await db_fetch_one_async(
                "SELECT brand_name_en, category_name FROM brands WHERE brand_id = $1",
                (engagement["brand_id"],)
            )
        
        engagement_type = engagement.get("type", "").lower()
        
        if engagement_type == "offer":
            offer_data = {
                "id": engagement["engagement_id"],
                "title": engagement.get("title_en", ""),
                "description": engagement.get("description_en", ""),
                "brand_id": engagement.get("brand_id"),
                "store_name": brand["brand_name_en"] if brand else "Unknown Store",
                "category": brand.get("category_name", "") if brand else "",
                "start_date": engagement.get("start_date", ""),
                "end_date": engagement.get("end_date", ""),
                "terms": engagement.get("terms_conditions_en", ""),
                "is_exclusive": bool(engagement.get("is_exclusive", 0))
            }
            context["offers"].append(offer_data)
        
        elif engagement_type == "event":
            event_data = {
                "id": engagement["engagement_id"],
                "name": engagement.get("title_en", ""),
                "description": engagement.get("description_en", ""),
                "brand_id": engagement.get("brand_id"),
                "store_name": brand["brand_name_en"] if brand else None,
                "start_date": engagement.get("start_date", ""),
                "end_date": engagement.get("end_date", ""),
                "terms": engagement.get("terms_conditions_en", "")
            }
            context["events"].append(event_data)

    # For store queries, make sure to fetch ALL stores for a given mall
    store_name = state.context_data.get("resolved_entity", "").lower() if state.context_data else ""
    target_store_data = None
    
    # Always fetch all stores for complete data
    all_stores = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
           b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
           WHERE bma.unique_property_id = $1""", 
        (state.mall_id,)
    )
    
    food_related_categories = ['restaurant', 'cafe', 'food', 'dining', 'bakery', 'coffee']
    
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
        
        # Add to category lists
        category = store.get("category_name", "").lower()
        if category:
            # Extract food types from description if it's a food place
            is_food_place = any(food_term in category for food_term in food_related_categories)
            
            if is_food_place:
                # Add to food categories
                context["category_types"]["food"].add(category)
                
                # Try to extract specific cuisine types from description
                description = store.get("description_en", "").lower()
                cuisine_types = ["italian", "indian", "chinese", "japanese", "american", "mexican", 
                                "thai", "fast food", "mediterranean", "middle eastern", "french"]
                for cuisine in cuisine_types:
                    if cuisine in description:
                        context["category_types"]["food"].add(cuisine)
            else:
                # Add to store categories
                context["category_types"]["store"].add(category)
        
        # If this is the store being searched for, put it at the top and remember it for finding neighbors
        if store_name and store_name in store["brand_name_en"].lower():
            context["stores"].insert(0, store_data)
            target_store_data = store_data
        else:
            context["stores"].append(store_data)
    
    # Find neighboring stores if a specific store was searched for
    if target_store_data and target_store_data.get("location"):
        # Extract location codes for the target store
        location_codes = target_store_data["location"]
        neighboring_stores = []
        
        # Function to parse location code and find neighbors
        def get_location_prefix_and_number(code):
            import re
            if not isinstance(code, str):
                return None, None
                
            match = re.match(r'([A-Za-z]+)(\d+.*)', code)
            if match:
                prefix, number_part = match.groups()
                try:
                    number_match = re.match(r'(\d+)', number_part)
                    if number_match:
                        number = int(number_match.group(1))
                        return prefix, number
                except (ValueError, AttributeError):
                    pass
            return None, None
        
        # Find stores with adjacent location codes
        for location_code in location_codes:
            prefix, number = get_location_prefix_and_number(location_code)
            if prefix and number is not None:
                # Check for adjacent numbers (±1, ±2)
                adjacent_codes = [
                    f"{prefix}{number-2}", f"{prefix}{number-1}", 
                    f"{prefix}{number+1}", f"{prefix}{number+2}"
                ]
                
                for store in context["stores"]:
                    if store == target_store_data:
                        continue
                    
                    store_locations = store.get("location", [])
                    if any(code in adjacent_codes for code in store_locations):
                        if store not in neighboring_stores:
                            neighboring_stores.append(store)
        
        # Add neighboring stores to context
        context["neighboring_stores"] = neighboring_stores[:5]  # Limit to 5 neighbors
    
    # Fetch service information using the correct column names from DB schema
    services = await db_fetch_all_async(
        """SELECT s.id AS service_id, s.name, s.description, s.location, s.is_available
           FROM services s
           WHERE s.unique_property_id = $1""",
        (state.mall_id,)
    )
    
    for service in services:
        service_data = {
            "name": service.get("name", ""),
            "description": service.get("description", ""),
            "location": service.get("location", ""),
            "is_available": service.get("is_available", True)
        }
        context["services"].append(service_data)
    
    # Fetch product details including price
    if state.intent and state.intent.startswith("product_"):
        product_name = state.context_data.get("resolved_entity", "").lower() if state.context_data else ""
        
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
            product_category = product.get("category", "").lower()
            if product_category:
                context["category_types"]["product"].add(product_category)
                
            context["products"].append({
                "id": product["id"],
                "name": product["name"],
                "description": product.get("description", ""),
                "price": float(product["price"]) if product.get("price") is not None else None,
                "brand_id": product["brand_id"],
                "category": product_category,
                "store_name": product.get("brand_name_en", ""),
                "in_stock": product.get("in_stock", True)
            })

    # Process Pinecone results for other entities that might be relevant
    brand_ids = set()
    for doc in state.initial_context or []:
        metadata = doc["metadata"]
        if metadata.get("mall_id") != state.mall_id:
            continue
        
        doc_type = metadata.get("type")
        if doc_type == "store" and not (state.intent and state.intent.startswith("store_")):
            # Only add store from vector search if not already doing a direct store query
            store_category = metadata.get("category_en", "").lower()
            if store_category:
                context["category_types"]["store"].add(store_category)
                
            store = {
                "name": metadata.get("name_en"),
                "category": store_category,
                "store_id": metadata.get("brand_id"),
                "description": metadata.get("description_en"),
            }
            if not any(s.get("name") == store["name"] for s in context["stores"]):
                context["stores"].append(store)
            if metadata.get("brand_id"):
                brand_ids.add(metadata.get("brand_id"))
                
        elif doc_type == "product" and not (state.intent and state.intent.startswith("product_")):
            # Only add product from vector search if not already doing a direct product query
            product_category = metadata.get("category", "").lower()
            if product_category:
                context["category_types"]["product"].add(product_category)
                
            product = {
                "id": metadata.get("id"),
                "name": metadata.get("name"),
                "brand_id": metadata.get("brand_id"),
                "category": product_category,
                "store_name": metadata.get("brand_name_en"),
            }
            if not any(p.get("name") == product["name"] for p in context["products"]):
                context["products"].append(product)
            if metadata.get("brand_id"):
                brand_ids.add(metadata["brand_id"])
                
        elif doc_type == "service":
            service = {
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "location": metadata.get("location")
            }
            if not any(s.get("name") == service["name"] for s in context["services"]):
                context["services"].append(service)

    # Fetch additional brand details for products
    if brand_ids:
        brands = await db_fetch_all_async(
            """SELECT brand_id, brand_name_en, category_name, description_en, 
               store_phone_number, store_email, store_website, pms_unit_codes 
               FROM brands WHERE brand_id = ANY($1)""",
            (list(brand_ids),)
        )
        brand_map = {b["brand_id"]: b for b in brands}
        
        for item in context["products"]:
            if item.get("brand_id") in brand_map and not item.get("store_name"):
                brand = brand_map[item["brand_id"]]
                item["store_name"] = brand["brand_name_en"]
                if "category" not in item and brand["category_name"]:
                    item["category"] = brand["category_name"]
                    context["category_types"]["product"].add(brand["category_name"].lower())
                if "location" not in item and brand.get("pms_unit_codes"):
                    item["location"] = brand["pms_unit_codes"]

    # If no stores found but resolved entity exists, try a direct DB lookup
    if not context["stores"] and state.context_data and state.context_data.get("resolved_entity"):
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
            store_category = store.get("category_name", "").lower()
            if store_category:
                context["category_types"]["store"].add(store_category)
                
            context["stores"].append({
                "name": store["brand_name_en"],
                "category": store_category,
                "store_id": store["brand_id"],
                "description": store.get("description_en", ""),
                "phone": store.get("store_phone_number", ""),
                "email": store.get("store_email", ""),
                "website": store.get("store_website", ""),
                "location": store.get("pms_unit_codes", {})
            })
    
    # Filter results based on type_preference if provided
    if state.type_preference:
        type_pref = state.type_preference.lower()
        
        # Filter stores
        if any(type_pref in category for category in context["category_types"]["store"]):
            context["stores"] = [
                store for store in context["stores"] 
                if store.get("category", "").lower() and type_pref in store["category"].lower()
            ]
        
        # Filter food places
        elif any(type_pref in category for category in context["category_types"]["food"]):
            food_stores = []
            for store in context["stores"]:
                category = store.get("category", "").lower()
                description = store.get("description", "").lower()
                
                if (category and any(food_term in category for food_term in food_related_categories) and
                    (type_pref in category or type_pref in description)):
                    food_stores.append(store)
            
            if food_stores:
                context["stores"] = food_stores
        
        # Filter products
        if any(type_pref in category for category in context["category_types"]["product"]):
            context["products"] = [
                product for product in context["products"]
                if product.get("category", "").lower() and type_pref in product["category"].lower()
                or product.get("description", "").lower() and type_pref in product["description"].lower()
            ]
    
    # Convert sets to lists for JSON serialization
    context["category_types"]["food"] = sorted(list(context["category_types"]["food"]))
    context["category_types"]["product"] = sorted(list(context["category_types"]["product"]))
    context["category_types"]["store"] = sorted(list(context["category_types"]["store"]))

    state.context_data = context
    state.response = json.dumps(convert_to_json_safe(context))
    return state

async def generate_response(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state

    formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-6:]]) if state.conversation_history else "No prior conversation."
    
    # Pass type_preference and needs_type_follow_up
    mall_name = "the mall"
    resolved_entity = ""
    
    if state.context_data:
        mall_name = state.context_data.get("mall_name", "the mall")
        resolved_entity = state.context_data.get("resolved_entity", "")
    
    # Detect follow-up type preferences in the conversation history
    if state.conversation_history and not state.type_preference:
        last_message = next((msg for msg in reversed(state.conversation_history) if msg["role"] == "user"), None)
        if last_message and state.needs_type_follow_up:
            # Check if the user is responding to a type preference follow-up
            # This is a simple check - the NLP model should do the heavy lifting
            common_types = ["italian", "indian", "chinese", "fast food", "sports", "casual", 
                           "formal", "kids", "women", "men", "luxury", "budget", "electronics"]
            for type_name in common_types:
                if type_name.lower() in last_message["content"].lower():
                    state.type_preference = type_name
                    state.needs_type_follow_up = False
                    break
    
    try:
        response = await asyncio.to_thread(
            customer_chain.invoke,
            {
                "context": state.response,
                "query": state.query,
                "lang": state.language,
                "conversation_history": formatted_history,
                "mall_name": mall_name,
                "resolved_entity": resolved_entity,
                "type_preference": state.type_preference or "",
                "needs_type_follow_up": state.needs_type_follow_up
            }
        )
        state.response = response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        state.response = "I'm having trouble processing your request right now. Please try again."
    
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
    elif state.intent and state.intent.startswith("other_") and state.response:
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