from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel as PydanticBaseModel, Field, SecretStr
from pydantic.json import pydantic_encoder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from datetime import datetime
import json
import os
import asyncio
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils import db_fetch_all_async, db_fetch_one_async, convert_to_json_safe, DateTimeEncoder, REDIS_CLIENT, logger
import networkx as nx
import spacy
from langchain_openai import ChatOpenAI
import concurrent.futures
import functools


# Load spaCy NLP model for store name and category extraction
nlp = spacy.load("en_core_web_sm")

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index("cenomicore")
index = pc.Index("cenomiprod")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")


# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY environment variable is not set")
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Create SecretStr for OpenAI API key
openai_api_key = SecretStr(OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# Knowledge graph for relationships
knowledge_graph = nx.Graph()

# Add preloading function to warm up the system
async def preload_system():
    """Preload common resources to improve cold-start performance"""
    logger.info("Preloading system resources for improved performance...")
    
    try:
        # Preload embeddings model
        await asyncio.to_thread(embeddings.embed_query, "warmup query")
        
        # Preload spaCy NLP model
        await asyncio.to_thread(nlp, "warmup text for NLP model")
        
        # Preload LLM with a simple query
        await asyncio.to_thread(llm.invoke, "hello")
        
        # Preload common mall data
        malls = await db_fetch_all_async(
            "SELECT unique_property_id as mall_id, marketing_name as name FROM malls LIMIT 10"
        )
        
        for mall in malls:
            cache_key = f"mall_info:{mall['mall_id']}"
            if not optimized_cache.get(cache_key):
                # Fetch and cache basic mall info
                mall_info = await db_fetch_one_async(
                    """SELECT marketing_name, marketing_name_ar, city, country, mall_information
                    FROM malls WHERE unique_property_id = $1""",
                    (mall['mall_id'],)
                )
                if mall_info:
                    optimized_cache.set(cache_key, mall_info, ex=3600)  # 1 hour cache
        
        logger.info("System preloading completed successfully")
    except Exception as e:
        logger.error(f"Error during system preloading: {e}")

# Initialize the batch LLM processor on startup
batch_llm_processor = None

# Modified populate_knowledge_graph with better performance
async def populate_knowledge_graph():
    """Populate knowledge graph with optimized performance"""
    logger.info("Populating knowledge graph...")
    
    try:
        # Run these queries in parallel
        brands_query = db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.brand_name_ar, b.category_name, 
               b.description_en, b.store_phone_number, b.store_email, b.store_website,
               b.pms_unit_codes, bma.unique_property_id 
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id"""
        )
        
        products_query = db_fetch_all_async(
            """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
               p.is_featured, p.in_stock, b.brand_name_en
               FROM products p
               JOIN brands b ON p.brand_id = b.brand_id"""
        )
        
        engagements_query = db_fetch_all_async(
            """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, e.start_date, 
               e.end_date, e.terms_conditions_en, e.is_exclusive, e.unique_property_id
               FROM engagements e"""
        )
        
        # Wait for all queries to complete
        brands, products, engagements = await asyncio.gather(brands_query, products_query, engagements_query)
        
        # Clear existing graph to rebuild it completely
        knowledge_graph.clear()
        
        # Use thread pool for CPU-bound tasks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process brands in a thread
            executor.submit(add_brands_to_graph, brands)
            
            # Process products in a thread
            executor.submit(add_products_to_graph, products)
            
            # Process engagements in a thread
            executor.submit(add_engagements_to_graph, engagements)
        
        logger.info(f"Knowledge graph populated with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges")
        
        # Preload system in background
        asyncio.create_task(preload_system())
        
    except Exception as e:
        logger.error(f"Error populating knowledge graph: {e}")

def add_brands_to_graph(brands):
    """Add brands to the knowledge graph"""
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
            "location": brand.get("pms_unit_codes", {})
        }
        knowledge_graph.add_node(f"brand_{brand['brand_id']}", **node_data)

def add_products_to_graph(products):
    """Add products to the knowledge graph"""
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
    

def add_engagements_to_graph(engagements):
    """Add engagement data to the knowledge graph."""
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

# Add the new prompt for customer query classification
customer_query_classification_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    You are CenomiAI, a mall assistant. Parse this query to determine the high-level category of the user's question based on the query and conversation history.

    Query: "{query}"

    Previous conversation:
    {conversation_history}

    Instructions:
    - Use the conversation history to resolve vague terms like "it", "that", or "the store" to specific entities mentioned earlier.
    - If the query is a follow-up (e.g., "Where is it?"), link it to the most recent entity from history.
    - For queries about food, dining, or restaurants, classify as product_info_query with a focus on food.
    - For queries about planning a visit to the mall with family/kids, classify as family_planning_query.
    - For navigation questions ("how do I get to X"), classify based on the destination (store, service, etc.)

    Return ONLY ONE of these query types (no JSON, just the exact string):
    - product_info_query: When asking about products, stores, brands, restaurants or specific items
    - mall_info_query: When asking about the mall itself (location, timings, facilities, directions)
    - offer_or_event_info_query: When asking about offers, events, sales, promotions, or discounts
    - services_info_query: When asking about available services (parking, wheelchair access, restrooms, etc.)
    - family_planning_query: When asking about child-friendly activities, family itineraries, or visit planning
    - visit_planning_query: When asking for suggestions about what to do in the mall or creating an itinerary
    - fallback_query: When the query doesn't clearly fit any of the above categories

    Examples:
    "What time does the mall close?" → mall_info_query
    "Are there any discounts this week?" → offer_or_event_info_query
    "Where can I find Nike shoes?" → product_info_query
    "Is wheelchair service available?" → services_info_query
    "I want to get something nice for my wife" → product_info_query
    "What restaurants are there?" → product_info_query
    "Is there a sale at Zara?" → offer_or_event_info_query
    "Where is the nearest restroom?" → services_info_query
    "I'm coming with my kids tomorrow, what should we do?" → family_planning_query
    "What's good for a 2-hour visit?" → visit_planning_query
    "I want to eat today" → product_info_query
    "My kids are with me" → family_planning_query
    """
)
customer_query_classification_chain = customer_query_classification_prompt | llm | StrOutputParser()

# Existing intent classification prompt - No changes needed
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
    - Remember that store locations are stored in "pms_unit_codes", not "address"

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

# Updated Customer Response Prompt
customer_prompt = PromptTemplate(
    input_variables=["context", "query", "lang", "conversation_history", "mall_name", "resolved_entity", "topic_turn_count", "conversation_topic"],
    template="""
    You are CenomiAI — a mall assistant at {mall_name}, designed to help shoppers find information, plan visits, and discover stores, offers, and events.

    # CRITICAL INSTRUCTION
    - ONLY provide information that is explicitly present in the database information below
    - NEVER invent, hallucinate, or make up ANY information not included in the provided context
    - If you don't know something or it's not in the data, say "I don't have that information" or similar
    - Do NOT reference stores, products, offers, events, or amenities that don't explicitly appear in the context data
    - Consider the context data as your ONLY source of truth

    # Response Style
    - Be concise, friendly and conversational, as if you're a helpful friend who knows the mall well
    - Keep responses very short - typically 1-3 sentences maximum
    - When listing items (stores, products, offers), ALWAYS limit to a maximum of 3 results
    - Use bullet points for lists to improve readability
    - Responses should feel natural in both text and voice formats - imagine someone listening to your response

    # Image Handling (IMPORTANT)
    - When mentioning brands, stores, offers, events, services, or products that have image_url data, ALWAYS include the image
    - Use markdown image format: ![Description](image_url)
    - Place the image after mentioning the brand/store/offer/product - typically at the end of the bullet point
    - Only include images when there is a valid image_url in the data (not null, not empty string)
    - If multiple items have images, include at most one image per item

    # Location Format Rules
    - NEVER include raw location codes like "FF", "GF", "BSW" in your responses
    - Always convert these codes to human-readable descriptions:
      * "FF" → "First Floor"
      * "GF" → "Ground Floor" 
      * "BSW" → "Basement West"
      * "F1" → "First Floor"
      * "F2" → "Second Floor"
      * "F3" → "Third Floor"
    - Example: Instead of "FF08", say "First Floor, Shop 8"
    - Example: Instead of "GF12", say "Ground Floor, Shop 12"
    - If you see numbers after location codes, treat them as shop numbers
    - For store locations, use "pms_unit_codes" field, NOT "address"
    - For mall addresses, use "address_en" or "address_ar" based on the language
    - Mall address information is inside "mall_information.MallContact.Address1En" and "Address2En"

    # Conversation Context
    {conversation_history}
    
    # Conversation State
    Current topic: {conversation_topic}
    Turn count on this topic: {topic_turn_count}

    # Follow-up Suggestions
    - For store queries: Suggest directions, similar stores, or current offers
    - For product queries: Suggest filtering by price, brand, or viewing similar items
    - For mall info: Suggest other useful information (parking, operating hours)
    - For events/offers: Suggest filtering by category or time period
    - For family visits: Suggest kid-friendly options or services

    # Information to Include
    - Store details: Location (floor/section), category, and brief description
    - Product info: Price, availability, store location
    - Offers: Discount amount, conditions, validity period
    - Events: Location, timing, any special instructions
    - Services: Location, availability, requirements
    
    # Follow-up Question Control (STRICTLY FOLLOW THIS)
    - If topic_turn_count = 1: Ask ONE follow-up question to refine information
    - If topic_turn_count >= 2: DO NOT ask follow-up questions - respond with finality
    - NEVER ask more than one follow-up question in a response
    - After the first exchange on a topic, your goal is to conclude the topic naturally
    
    # Response Structure
    - For turn 1: Provide information and ask ONE relevant follow-up question
    - For turn 2+: Provide final information without any further questions
    - Keep all responses brief and to-the-point regardless of turn count
    - For turn 2+, end with a brief closing statement like "Enjoy your visit!" or "Hope that helps!"

    # Mall Database Information (YOUR ONLY SOURCE OF TRUTH)
    {context}

    User Query: {query}
    
    Respond in {lang} in a friendly, conversational tone, using ONLY information from the Mall Database Information above. Your response should be structured to work well for both text and voice:
    
    1. Direct answer with only the most important details (1-3 bullets if listing items)
    2. Include relevant images using markdown format when available (![Description](image_url))
    3. For turn 1 only: ONE simple follow-up question
    4. For turn 2+: Brief, friendly closing (no questions)
    """
)

customer_chain = customer_prompt | llm | StrOutputParser()

# Response format class
class ResponseFormat(PydanticBaseModel):
    response: str
    recommendations: Optional[List[Dict[str, str]]] = Field(default=None)
    is_recommendation_format: bool = Field(default=False)
    follow_up_question: Optional[str] = Field(default=None)
    
    def dict(self, *args, **kwargs):
        """Override dict method to ensure fields are properly serialized"""
        return {
            "response": self.response,
            "recommendations": self.recommendations,
            "is_recommendation_format": self.is_recommendation_format,
            "follow_up_question": self.follow_up_question
        }
    
    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

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
    query_type: Optional[str] = None  # Added field for high-level query classification
    conversation_topic: Optional[str] = None  # Track current conversation topic
    topic_turn_count: int = 0  # Track number of turns on current topic
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None  # Store formatted response as object or dict
    mall_data: Optional[Dict[str, Any]] = None  # Store mall data
    _combined_classification_result: Optional[Dict[str, Any]] = None  # Store combined classification result

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = False  # Allow access to _attributes

# Add the new node for high-level query classification
async def classify_query_type(state: CustomerState) -> CustomerState:
    combined_result = await optimized_classification(state)
    if combined_result:
        state.query_type = combined_result.get("query_type")
        logger.info(f"Classified query type: {state.query_type}")
    else:
        logger.warning("Could not classify query type")
        state.query_type = "general_query"  # Default fallback
    return state

# Update existing classify_intent function to use the high-level query type for context
async def classify_intent(state: CustomerState) -> CustomerState:
    # Reuse the same result from combined classification if already performed
    if hasattr(state, "_combined_classification_result") and state._combined_classification_result:
        combined_result = state._combined_classification_result
    else:
        combined_result = await optimized_classification(state)
        state._combined_classification_result = combined_result
    
    # Ensure we have a valid intent, defaulting to fallback if none
    state.intent = combined_result.get("intent") if combined_result else "other_general"
    state.context_data = state.context_data or {}
    
    # Set resolved entity if present
    if combined_result and combined_result.get("resolved_entity"):
        state.context_data["resolved_entity"] = combined_result["resolved_entity"]
    
    # Set type preference if present
    if combined_result and combined_result.get("type_preference"):
        state.type_preference = combined_result["type_preference"]
    
    # Determine if we need a type follow-up
    if combined_result:
        state.needs_type_follow_up = combined_result.get("needs_type_follow_up", False)
    
    # Only check startswith if state.intent is not None
    if state.intent and state.intent.startswith("other_"):
        state.response = "I'm not sure what you mean 😅. Could you tell me more?"
    
    logger.info(f"Classified intent: {state.intent}, Resolved entity: {state.context_data.get('resolved_entity')}, Type preference: {state.type_preference}")
    return state

# Create specialized context retrieval functions

async def retrieve_product_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state

    # Specialized query for products
    cache_key = f"product_context:{state.query}:{state.intent}:{state.user_id or 'anon'}:{state.mall_id}"
    cached_context = REDIS_CLIENT.get(cache_key)
    if cached_context:
        state.initial_context = json.loads(cached_context)
        return state

    # Extract product or store names from query using NLP
    query_items = [item.strip() for item in state.query.split("\n") if item.strip()] if "\n" in state.query else [state.query]
    product_name = state.context_data.get("resolved_entity", "").lower() if state.context_data else ""
    
    # If no specific product name, try to extract it from the query
    if not product_name:
        doc = nlp(" ".join(query_items).lower())
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"]:
                product_name = ent.text
            break

    # Build query vector focused on products
    product_query_vector = embeddings.embed_query(f"product {product_name if product_name else state.query}")
    
    # Pinecone query with filter specifically for products
    filter_dict = {"mall_id": state.mall_id, "type": "product"}
    try:
        results = await asyncio.to_thread(
            index.query, 
            vector=product_query_vector, 
            top_k=15,  # Reduced number - more focused 
            include_metadata=True, 
            filter=filter_dict
        )
        matches = results.get("matches", [])
        state.initial_context = [{"id": doc["id"], "score": float(doc["score"]), "metadata": doc["metadata"]} for doc in matches]
        
        # If we don't find products, try searching for stores that might have those products
        if len(matches) < 3 and product_name:
            store_filter = {"mall_id": state.mall_id, "type": "store"}
            store_results = await asyncio.to_thread(
                index.query,
                vector=embeddings.embed_query(f"store selling {product_name}"),
                top_k=5,
                include_metadata=True,
                filter=store_filter
            )
            store_matches = store_results.get("matches", [])
            for doc in store_matches:
                state.initial_context.append({"id": doc["id"], "score": float(doc["score"]) * 0.8, "metadata": doc["metadata"]})
    
    except Exception as e:
        logger.error(f"Pinecone query error for product context: {e}")
        state.initial_context = []
    
    # Before returning, process any location codes
    for item in state.initial_context:
        if "metadata" in item and item["metadata"]:
            if "location" in item["metadata"]:
                item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    return state

async def retrieve_mall_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state
    
    # Specialized query for mall information
    cache_key = f"mall_context:{state.query}:{state.mall_id}"
    cached_context = REDIS_CLIENT.get(cache_key)
    if cached_context:
        state.initial_context = json.loads(cached_context)
        return state
    
    # For mall info, we focus on amenities, directions, hours, and general mall data
    mall_query_vector = embeddings.embed_query(f"mall information {state.query}")
    
    # Get mall information directly from database with the correct fields
    mall_info = await db_fetch_one_async(
        """SELECT marketing_name, marketing_name_ar, city, country, mall_information, gps_coordinates
        FROM malls WHERE unique_property_id = $1""",
        (state.mall_id,)
    )
    
    state.initial_context = []
    
    if mall_info:
        # Extract address information from the mall_information JSON field
        address_en = None
        address_ar = None
        contact_phone = None
        contact_email = None
        opening_hours = []
        mall_description_en = None
        mall_description_ar = None
        map_url = None
        
        if mall_info.get("mall_information"):
            try:
                mall_data = mall_info["mall_information"]
                if isinstance(mall_data, str):
                    mall_data = json.loads(mall_data)
                
                # Extract address from MallContact
                if "MallContact" in mall_data:
                    contact_data = mall_data["MallContact"]
                    address_lines_en = []
                    if contact_data.get("Address1En"):
                        address_lines_en.append(contact_data["Address1En"])
                    if contact_data.get("Address2En"):
                        address_lines_en.append(contact_data["Address2En"])
                    address_en = ", ".join(address_lines_en)
                    
                    address_lines_ar = []
                    if contact_data.get("Address1Ar"):
                        address_lines_ar.append(contact_data["Address1Ar"])
                    if contact_data.get("Address2Ar"):
                        address_lines_ar.append(contact_data["Address2Ar"])
                    address_ar = ", ".join(address_lines_ar)
                    
                    contact_phone = contact_data.get("Phone")
                    contact_email = contact_data.get("Email")
                
                # Extract opening hours if available
                if "MallTiming" in mall_data and isinstance(mall_data["MallTiming"], list):
                    opening_hours = mall_data["MallTiming"]
                
                # Extract mall description
                mall_description_en = mall_data.get("MallDescriptionEn")
                mall_description_ar = mall_data.get("MallDescriptionAr")
                
                # Extract map URL
                map_url = mall_data.get("GoogleMapURL") or mall_data.get("MallMapEn")
                
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error processing mall_information JSON: {e}")
        
        # Create a synthetic context entry for the mall itself
        mall_metadata = {
            "type": "mall",
            "name_en": mall_info.get("marketing_name", ""),
            "name_ar": mall_info.get("marketing_name_ar", ""),
            "address_en": address_en,
            "address_ar": address_ar,
            "city": mall_info.get("city", ""),
            "country": mall_info.get("country", ""),
            "description_en": mall_description_en,
            "description_ar": mall_description_ar,
            "contact_phone": contact_phone,
            "contact_email": contact_email,
            "opening_hours": opening_hours,
            "map_url": map_url,
            "gps_coordinates": mall_info.get("gps_coordinates", ""),
            "mall_id": state.mall_id
        }
        state.initial_context.append({"id": f"mall_{state.mall_id}", "score": 1.0, "metadata": mall_metadata})
    
    # Also fetch amenities
    try:
        amenity_filter = {"mall_id": state.mall_id, "type": "amenity"}
        amenity_results = await asyncio.to_thread(
            index.query,
            vector=mall_query_vector,
            top_k=10,
            include_metadata=True,
            filter=amenity_filter
        )
        amenity_matches = amenity_results.get("matches", [])
        for doc in amenity_matches:
            state.initial_context.append({"id": doc["id"], "score": float(doc["score"]), "metadata": doc["metadata"]})
    except Exception as e:
        logger.error(f"Pinecone query error for mall context: {e}")
    
    # Before returning, process any location codes
    for item in state.initial_context:
        if "metadata" in item and item["metadata"]:
            if "location" in item["metadata"]:
                item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    return state

async def retrieve_offer_event_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state
    
    # Specialized query for offers and events
    cache_key = f"offer_event_context:{state.query}:{state.mall_id}"
    cached_context = REDIS_CLIENT.get(cache_key)
    if cached_context:
        try:
            if isinstance(cached_context, bytes):
                cached_context = cached_context.decode('utf-8')
            state.initial_context = json.loads(cached_context)
            return state
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error decoding cached offer/event context: {e}")
            # Continue with retrieving fresh data
    
    # Get current date for filtering current/future events
    current_date = datetime.now().isoformat()
    
    # Directly query database for latest offers and events
    engagements = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
           e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, b.brand_name_en
           FROM engagements e 
           LEFT JOIN brands b ON e.brand_id = b.brand_id
           WHERE e.unique_property_id = $1 AND 
           (e.end_date >= $2 OR e.end_date IS NULL)
           ORDER BY e.start_date ASC""",
        (state.mall_id, current_date)
    )
    
    # Format engagements as initial context
    state.initial_context = []
    for engagement in engagements:
        engagement_type = engagement.get("type", "").lower()
        metadata = {
            "type": engagement_type,
            "title": engagement.get("title_en", ""),
            "description": engagement.get("description_en", ""),
            "start_date": engagement.get("start_date", ""),
            "end_date": engagement.get("end_date", ""),
            "terms": engagement.get("terms_conditions_en", ""),
            "is_exclusive": bool(engagement.get("is_exclusive", 0)),
            "mall_id": state.mall_id,
            "brand_id": engagement.get("brand_id"),
            "brand_name": engagement.get("brand_name_en", "")
        }
        state.initial_context.append({
            "id": f"engagement_{engagement['engagement_id']}",
            "score": 1.0,  # Direct database lookup, high confidence
            "metadata": metadata
        })
    
    # If looking for a specific brand's offers, prioritize those
    if state.context_data and state.context_data.get("resolved_entity"):
        brand_name = state.context_data.get("resolved_entity", "").lower()
        if brand_name:  # Ensure it's not None or empty
            for item in state.initial_context:
                if item["metadata"].get("brand_name", "").lower() == brand_name:
                    item["score"] = 1.5  # Boost score for matching brand
    
    # Sort by score descending
    state.initial_context.sort(key=lambda x: x["score"], reverse=True)
    
    # Before returning, process any location codes
    for item in state.initial_context:
        if "metadata" in item and item["metadata"]:
            if "location" in item["metadata"]:
                item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    # Cache the results
    try:
        REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    except Exception as e:
        logger.error(f"Error caching offer/event context: {e}")
    
    return state

async def retrieve_services_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state
    
    # Specialized query for services
    cache_key = f"services_context:{state.query}:{state.mall_id}"
    cached_context = REDIS_CLIENT.get(cache_key)
    if cached_context:
        state.initial_context = json.loads(cached_context)
        return state
    
    # Directly query database for services
    services = await db_fetch_all_async(
        """SELECT s.id, s.name, s.description, s.location, s.is_available
           FROM services s
           WHERE s.unique_property_id = $1""",
        (state.mall_id,)
    )
    
    # Format services as initial context
    state.initial_context = []
    for service in services:
        metadata = {
            "type": "service",
            "name": service.get("name", ""),
            "description": service.get("description", ""),
            "location": service.get("location", ""),
            "is_available": service.get("is_available", True),
            "mall_id": state.mall_id
        }
        state.initial_context.append({
            "id": f"service_{service['id']}",
            "score": 1.0,  # Direct database lookup
            "metadata": metadata
        })
    
    # Also include amenities as they're often related to services
    try:
        service_query_vector = embeddings.embed_query(f"mall service {state.query}")
        amenity_filter = {"mall_id": state.mall_id, "type": "amenity"}
        amenity_results = await asyncio.to_thread(
            index.query,
            vector=service_query_vector,
            top_k=5,
            include_metadata=True,
            filter=amenity_filter
        )
        amenity_matches = amenity_results.get("matches", [])
        for doc in amenity_matches:
            state.initial_context.append({"id": doc["id"], "score": float(doc["score"]), "metadata": doc["metadata"]})
    except Exception as e:
        logger.error(f"Pinecone query error for services context: {e}")
    
    # Before returning, process any location codes
    for item in state.initial_context:
        if "metadata" in item and item["metadata"]:
            if "location" in item["metadata"]:
                item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    return state

async def retrieve_family_planning_context(state: CustomerState) -> CustomerState:
    """Specialized query function for family planning and kid-friendly activities"""
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state
    
    # Cache for family planning context
    cache_key = f"family_planning_context:{state.query}:{state.mall_id}"
    cached_context = REDIS_CLIENT.get(cache_key)
    if cached_context:
        state.initial_context = json.loads(cached_context)
        return state
    
    # Initialize context
    state.initial_context = []
    
    # First, get kid-friendly stores
    kid_friendly_categories = ["toys", "kids", "children", "baby", "family", "play", "games"]
    food_family_keywords = ["family meal", "kids menu", "children menu", "play area"]
    
    # Get mall information for operating hours and facilities
    mall_info = await db_fetch_one_async(
        """SELECT marketing_name, description, opening_hours, map_url, contact_info 
        FROM malls WHERE unique_property_id = $1""",
        (state.mall_id,)
    )
    
    if mall_info:
        # Create a synthetic context entry for the mall itself with family focus
        mall_metadata = {
            "type": "mall",
            "name": mall_info.get("marketing_name", ""),
            "description": mall_info.get("description", ""),
            "opening_hours": mall_info.get("opening_hours", ""),
            "map_url": mall_info.get("map_url", ""),
            "contact_info": mall_info.get("contact_info", ""),
            "mall_id": state.mall_id
        }
        state.initial_context.append({"id": f"mall_{state.mall_id}", "score": 1.0, "metadata": mall_metadata})
    
    # Fetch family-friendly stores
    all_stores = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
           b.pms_unit_codes
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
           WHERE bma.unique_property_id = $1""", 
        (state.mall_id,)
    )
    
    for store in all_stores:
        category = store.get("category_name", "").lower()
        description = store.get("description_en", "").lower() if store.get("description_en") else ""
        
        # Check if this is a kid-friendly store based on category or description
        is_kid_friendly = any(kid_term in category for kid_term in kid_friendly_categories) or \
                          any(kid_term in description for kid_term in kid_friendly_categories)
        
        if is_kid_friendly:
            metadata = {
                "type": "store",
                "name_en": store["brand_name_en"],
                "category_en": category,
                "description_en": description,
                "brand_id": store["brand_id"],
                "location": store.get("pms_unit_codes", {}),
                "mall_id": state.mall_id,
                "is_kid_friendly": True
            }
            state.initial_context.append({
                "id": f"brand_{store['brand_id']}",
                "score": 0.95,
                "metadata": metadata
            })
    
    # Fetch family-friendly restaurants
    restaurants = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
           b.pms_unit_codes
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
           WHERE bma.unique_property_id = $1 AND 
           (LOWER(b.category_name) LIKE '%restaurant%' OR 
            LOWER(b.category_name) LIKE '%food%' OR 
            LOWER(b.category_name) LIKE '%cafe%' OR
            LOWER(b.category_name) LIKE '%dining%')""", 
        (state.mall_id,)
    )
    
    for restaurant in restaurants:
        description = restaurant.get("description_en", "").lower() if restaurant.get("description_en") else ""
        
        # Check if this is a family-friendly restaurant
        is_family_friendly = any(term in description for term in food_family_keywords)
        
        if is_family_friendly:
            metadata = {
                "type": "store",
                "name_en": restaurant["brand_name_en"],
                "category_en": restaurant.get("category_name", ""),
                "description_en": description,
                "brand_id": restaurant["brand_id"],
                "location": restaurant.get("pms_unit_codes", {}),
                "mall_id": state.mall_id,
                "is_family_friendly": True
            }
            state.initial_context.append({
                "id": f"restaurant_{restaurant['brand_id']}",
                "score": 0.9,
                "metadata": metadata
            })
    
    # Fetch family-oriented events and offers
    current_date = datetime.now().isoformat()
    engagements = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
           e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, b.brand_name_en
           FROM engagements e 
           LEFT JOIN brands b ON e.brand_id = b.brand_id
           WHERE e.unique_property_id = $1 AND 
           (e.end_date >= $2 OR e.end_date IS NULL)""",
        (state.mall_id, current_date)
    )
    
    for engagement in engagements:
        title = engagement.get("title_en", "").lower()
        description = engagement.get("description_en", "").lower()
        
        # Check if this is a family-oriented event or offer
        is_family_oriented = any(kid_term in title or kid_term in description for kid_term in kid_friendly_categories)
        
        if is_family_oriented:
            engagement_type = engagement.get("type", "").lower()
            metadata = {
                "type": engagement_type,
                "title": engagement.get("title_en", ""),
                "description": engagement.get("description_en", ""),
                "start_date": engagement.get("start_date", ""),
                "end_date": engagement.get("end_date", ""),
                "terms": engagement.get("terms_conditions_en", ""),
                "is_exclusive": bool(engagement.get("is_exclusive", 0)),
                "mall_id": state.mall_id,
                "brand_id": engagement.get("brand_id"),
                "brand_name": engagement.get("brand_name_en", ""),
                "is_family_oriented": True
            }
            state.initial_context.append({
                "id": f"engagement_{engagement['engagement_id']}",
                "score": 0.95,
                "metadata": metadata
            })

    # Fetch services like play areas, nursing rooms, family restrooms
    # family_services = await db_fetch_all_async(
        # """SELECT s.id, s.name, s.description, s.location, s.is_available
        #    FROM services s
        #    WHERE s.unique_property_id = $1 AND 
        #    (LOWER(s.name) LIKE '%family%' OR 
        #     LOWER(s.name) LIKE '%kid%' OR 
        #     LOWER(s.name) LIKE '%child%' OR
        #     LOWER(s.name) LIKE '%play%' OR
        #     LOWER(s.name) LIKE '%baby%' OR
        #     LOWER(s.name) LIKE '%stroller%' OR
        #     LOWER(s.name) LIKE '%nursing%')""",
    #     (state.mall_id,)
    # )
    family_services = await db_fetch_all_async(
        """SELECT s.id, s.name, s.description, s.description_ar, s.location, s.is_available
           FROM services s
           WHERE s.unique_property_id = $1""",
        (state.mall_id,)
    )
    
    for service in family_services:
        metadata = {
            "type": "service",
            "name": service.get("name", ""),
            "description": service.get("description", ""),
            "description_ar": service.get("description_ar", ""),
            "location": service.get("location", ""),
            "is_available": service.get("is_available", True),
            "mall_id": state.mall_id,
            "is_family_service": True
        }
        state.initial_context.append({
            "id": f"family_service_{service['id']}",
            "score": 1.0, # High priority for family services
            "metadata": metadata
        })
    
    # Sort the results by score
    state.initial_context.sort(key=lambda x: x["score"], reverse=True)
    
    # Before returning, process any location codes
    for item in state.initial_context:
        if "metadata" in item and item["metadata"]:
            if "location" in item["metadata"]:
                item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    # Cache the results
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    return state

async def retrieve_visit_planning_context(state: CustomerState) -> CustomerState:
    """Specialized query function for visit planning and itinerary creation"""
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state
    
    # Cache for visit planning context
    cache_key = f"visit_planning_context:{state.query}:{state.mall_id}"
    cached_context = REDIS_CLIENT.get(cache_key)
    if cached_context:
        state.initial_context = json.loads(cached_context)
        return state
    
    # Initialize context
    state.initial_context = []
    
    # First, get mall information for operating hours and facilities
    mall_info = await db_fetch_one_async(
        """SELECT marketing_name, marketing_name_ar, city, country, mall_information, gps_coordinates 
        FROM malls WHERE unique_property_id = $1""",
        (state.mall_id,)
    )
    
    if mall_info:
        # Extract address information from the mall_information JSON field
        address_en = None
        address_ar = None
        contact_phone = None
        contact_email = None
        opening_hours = []
        mall_description_en = None
        mall_description_ar = None
        map_url = None
        
        if mall_info.get("mall_information"):
            try:
                mall_data = mall_info["mall_information"]
                if isinstance(mall_data, str):
                    mall_data = json.loads(mall_data)
                
                # Extract address from MallContact
                if "MallContact" in mall_data:
                    contact_data = mall_data["MallContact"]
                    address_lines_en = []
                    if contact_data.get("Address1En"):
                        address_lines_en.append(contact_data["Address1En"])
                    if contact_data.get("Address2En"):
                        address_lines_en.append(contact_data["Address2En"])
                    address_en = ", ".join(address_lines_en)
                    
                    address_lines_ar = []
                    if contact_data.get("Address1Ar"):
                        address_lines_ar.append(contact_data["Address1Ar"])
                    if contact_data.get("Address2Ar"):
                        address_lines_ar.append(contact_data["Address2Ar"])
                    address_ar = ", ".join(address_lines_ar)
                    
                    contact_phone = contact_data.get("Phone")
                    contact_email = contact_data.get("Email")
                
                # Extract opening hours if available
                if "MallTiming" in mall_data and isinstance(mall_data["MallTiming"], list):
                    opening_hours = mall_data["MallTiming"]
                
                # Extract mall description
                mall_description_en = mall_data.get("MallDescriptionEn")
                mall_description_ar = mall_data.get("MallDescriptionAr")
                
                # Extract map URL
                map_url = mall_data.get("GoogleMapURL") or mall_data.get("MallMapEn")
                
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error processing mall_information JSON: {e}")
        
        # Create a synthetic context entry for the mall itself
        mall_metadata = {
            "type": "mall",
            "name_en": mall_info.get("marketing_name", ""),
            "name_ar": mall_info.get("marketing_name_ar", ""),
            "address_en": address_en,
            "address_ar": address_ar,
            "city": mall_info.get("city", ""),
            "country": mall_info.get("country", ""),
            "description_en": mall_description_en,
            "description_ar": mall_description_ar,
            "contact_phone": contact_phone,
            "contact_email": contact_email,
            "opening_hours": opening_hours,
            "map_url": map_url,
            "gps_coordinates": mall_info.get("gps_coordinates", ""),
            "mall_id": state.mall_id
        }
        state.initial_context.append({"id": f"mall_{state.mall_id}", "score": 1.0, "metadata": mall_metadata})
    
    # Get current date for filtering current/future events
    current_date = datetime.now().isoformat()
    
    # Fetch current events
    events = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
           e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, b.brand_name_en
           FROM engagements e 
           LEFT JOIN brands b ON e.brand_id = b.brand_id
           WHERE e.unique_property_id = $1 AND 
           e.type = 'events' AND
           (e.end_date >= $2 OR e.end_date IS NULL)
           ORDER BY e.start_date ASC
           LIMIT 5""",
        (state.mall_id, current_date)
    )
    
    for event in events:
        metadata = {
            "type": "event",
            "title": event.get("title_en", ""),
            "description": event.get("description_en", ""),
            "start_date": event.get("start_date", ""),
            "end_date": event.get("end_date", ""),
            "terms": event.get("terms_conditions_en", ""),
            "mall_id": state.mall_id,
            "brand_id": event.get("brand_id"),
            "brand_name": event.get("brand_name_en", "")
        }
        state.initial_context.append({
            "id": f"event_{event['engagement_id']}",
            "score": 0.95,
            "metadata": metadata
        })
    
    # Fetch exclusive or highlighted offers
    offers = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
           e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, b.brand_name_en
           FROM engagements e 
           LEFT JOIN brands b ON e.brand_id = b.brand_id
           WHERE e.unique_property_id = $1 AND 
           e.type = 'offer' AND
           e.is_exclusive = 1 AND
           (e.end_date >= $2 OR e.end_date IS NULL)
           ORDER BY e.is_exclusive DESC, e.start_date ASC
           LIMIT 5""",
        (state.mall_id, current_date)
    )
    
    for offer in offers:
        metadata = {
            "type": "offer",
            "title": offer.get("title_en", ""),
            "description": offer.get("description_en", ""),
            "start_date": offer.get("start_date", ""),
            "end_date": offer.get("end_date", ""),
            "terms": offer.get("terms_conditions_en", ""),
            "is_exclusive": True,
            "mall_id": state.mall_id,
            "brand_id": offer.get("brand_id"),
            "brand_name": offer.get("brand_name_en", "")
        }
        state.initial_context.append({
            "id": f"offer_{offer['engagement_id']}",
            "score": 0.9,
            "metadata": metadata
        })
    
    # Fetch popular dining options
    restaurants = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
           b.pms_unit_codes
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
           WHERE bma.unique_property_id = $1 AND 
           (LOWER(b.category_name) LIKE '%restaurant%' OR 
            LOWER(b.category_name) LIKE '%food%' OR 
            LOWER(b.category_name) LIKE '%cafe%' OR
            LOWER(b.category_name) LIKE '%dining%')
           LIMIT 5""", 
        (state.mall_id,)
    )
    
    for restaurant in restaurants:
        metadata = {
            "type": "store",
            "name_en": restaurant["brand_name_en"],
            "category_en": restaurant.get("category_name", ""),
            "description_en": restaurant.get("description_en", ""),
            "brand_id": restaurant["brand_id"],
            "location": restaurant.get("pms_unit_codes", {}),
            "mall_id": state.mall_id
        }
        state.initial_context.append({
            "id": f"restaurant_{restaurant['brand_id']}",
            "score": 0.85,
            "metadata": metadata
        })
    
    # Fetch popular shopping stores
    stores = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
           b.pms_unit_codes
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
           WHERE bma.unique_property_id = $1 AND 
           (LOWER(b.category_name) LIKE '%fashion%' OR 
            LOWER(b.category_name) LIKE '%clothing%' OR 
            LOWER(b.category_name) LIKE '%apparel%' OR
            LOWER(b.category_name) LIKE '%accessories%')
           LIMIT 5""", 
        (state.mall_id,)
    )
    
    for store in stores:
        metadata = {
            "type": "store",
            "name_en": store["brand_name_en"],
            "category_en": store.get("category_name", ""),
            "description_en": store.get("description_en", ""),
            "brand_id": store["brand_id"],
            "location": store.get("pms_unit_codes", {}),
            "mall_id": state.mall_id
        }
        state.initial_context.append({
            "id": f"store_{store['brand_id']}",
            "score": 0.8,
            "metadata": metadata
        })
    
    # Fetch entertainment options
    entertainment = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
           b.pms_unit_codes
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
           WHERE bma.unique_property_id = $1 AND 
           (LOWER(b.category_name) LIKE '%entertainment%' OR 
            LOWER(b.category_name) LIKE '%cinema%' OR 
            LOWER(b.category_name) LIKE '%movie%' OR
            LOWER(b.category_name) LIKE '%game%' OR
            LOWER(b.category_name) LIKE '%play%' OR
            LOWER(b.category_name) LIKE '%arcade%')
           LIMIT 3""", 
        (state.mall_id,)
    )
    
    for venue in entertainment:
        metadata = {
            "type": "store",
            "name_en": venue["brand_name_en"],
            "category_en": venue.get("category_name", ""),
            "description_en": venue.get("description_en", ""),
            "brand_id": venue["brand_id"],
            "location": venue.get("pms_unit_codes", {}),
            "mall_id": state.mall_id,
            "is_entertainment": True
        }
        state.initial_context.append({
            "id": f"entertainment_{venue['brand_id']}",
            "score": 0.9,
            "metadata": metadata
        })
    
    # Before returning, process any location codes
    for item in state.initial_context:
        if "metadata" in item and item["metadata"]:
            if "location" in item["metadata"]:
                item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    # Cache the results
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    return state

async def retrieve_fallback_context(state: CustomerState) -> CustomerState:
    # Use original initial_retrieval as fallback
    await initial_retrieval(state)
    
    # Before returning, process any location codes
    for item in state.initial_context or []:
        if "metadata" in item and item["metadata"]:
            if "location" in item["metadata"]:
                item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    return state

# Add this function before the refine_context function to process location codes
def convert_location_codes(location_data):
    """Convert raw location codes to human-readable format"""
    if not location_data:
        return location_data
    
    if isinstance(location_data, list):
        # Handle list of location codes
        converted_locations = []
        for code in location_data:
            if isinstance(code, str):
                converted = convert_single_location_code(code)
                converted_locations.append(converted)
            else:
                converted_locations.append(code)
        return converted_locations
    elif isinstance(location_data, dict):
        # Handle dictionary of location data
        converted_locations = {}
        for key, code in location_data.items():
            if isinstance(code, str):
                converted = convert_single_location_code(code)
                converted_locations[key] = converted
            else:
                converted_locations[key] = code
        return converted_locations
    elif isinstance(location_data, str):
        # Handle single location code
        return convert_single_location_code(location_data)
    
    return location_data

def convert_single_location_code(code):
    """Convert a single location code to human-readable format"""
    import re
    
    if not isinstance(code, str):
        return code
    
    # Define common location code prefixes and their readable formats
    location_map = {
        "FF": "First Floor",
        "GF": "Ground Floor",
        "BSW": "Basement West",
        "BSE": "Basement East",
        "BS": "Basement",
        "F1": "First Floor",
        "F2": "Second Floor",
        "F3": "Third Floor",
        "F4": "Fourth Floor",
        "F5": "Fifth Floor",
        "L1": "Level 1",
        "L2": "Level 2",
        "L3": "Level 3",
        "G": "Ground Floor"
    }
    
    # Match a location prefix followed by digits
    match = re.match(r'([A-Za-z]+)(\d+.*)', code)
    if match:
        prefix, number_part = match.groups()
        if prefix in location_map:
            return f"{location_map[prefix]}, Shop {number_part}"
    
    # If no match found or prefix not in our map, return the original code
    return code

# Modify the beginning of the refine_context function to process location codes
async def refine_context(state: CustomerState) -> CustomerState:
    """
    Refine the context based on the intent and query type.
    This function ensures we're only using data from the database or vector store.
    """
    if not state.mall_id:
        state.response = "Please select a mall first."
        return state

    # Check if we have any initial context
    if not state.initial_context or len(state.initial_context) == 0:
        logger.warning("No initial context found for refining")
        if state.language == "ar":
            state.response = "عذراً، لا يمكنني العثور على معلومات ذات صلة. هل يمكنك إعادة صياغة سؤالك؟"
        else:
            state.response = "Sorry, I couldn't find any relevant information. Could you rephrase your question?"
        return state

    # Sort initial context by score for better retrieval (higher scores first)
    state.initial_context = sorted(state.initial_context, key=lambda x: x.get("score", 0), reverse=True)
    
    # Limit to top 10 results for performance 
    top_results = state.initial_context[:10]

    # Create a detailed context from the database results
    details = []
    for item in top_results:
        # Skip if no metadata (shouldn't happen with proper retrieval)
        if not item.get("metadata"):
            continue
            
        # Add item details based on its metadata
        # Only include fields that exist in the database
        metadata = item["metadata"]
        
        # Construct detail string with validation to ensure no missing fields
        detail = {}
        
        # Common fields
        detail["id"] = metadata.get("id", "")
        detail["type"] = metadata.get("type", "")
        detail["score"] = item.get("score", 0)
        
        # Handle different types of data
        if metadata.get("type") == "store":
            # Store-specific fields (only include what's in the database)
            detail["name"] = metadata.get("name", "")
            detail["category"] = metadata.get("category", "")
            detail["location"] = metadata.get("pms_unit_codes", "")
            detail["description"] = metadata.get("description", "")
            if "image_url" in metadata and metadata["image_url"]:
                detail["image_url"] = metadata["image_url"]
                
        elif metadata.get("type") == "product":
            # Product-specific fields
            detail["name"] = metadata.get("name", "")
            detail["price"] = metadata.get("price", "")
            detail["brand_name"] = metadata.get("brand_name", "")
            detail["description"] = metadata.get("description", "")
            detail["store_name"] = metadata.get("store_name", "")
            if "image_url" in metadata and metadata["image_url"]:
                detail["image_url"] = metadata["image_url"]
                
        elif metadata.get("type") == "engagement":
            # Engagement-specific fields (offers, events)
            detail["title"] = metadata.get("title_en", "")
            detail["brand_name"] = metadata.get("brand_name", "")
            detail["start_date"] = metadata.get("start_date", "")
            detail["end_date"] = metadata.get("end_date", "")
            detail["description"] = metadata.get("description_en", "")
            if "image_url" in metadata and metadata["image_url"]:
                detail["image_url"] = metadata["image_url"]
                
        elif metadata.get("type") == "amenity" or metadata.get("type") == "service":
            # Amenity/service-specific fields
            detail["name"] = metadata.get("name", "")
            detail["location"] = metadata.get("location", "")
            detail["description"] = metadata.get("description", "")
            if "image_url" in metadata and metadata["image_url"]:
                detail["image_url"] = metadata["image_url"]
                
        # Add the structured detail to our results
        details.append(detail)

    # Prepare the response with only database information
    response = ""
    
    # Add a prefix to indicate this is database content
    response += "DATABASE CONTENT:\n\n"
    
    # Format the details as a structured response
    if details:
        for i, detail in enumerate(details):
            response += f"Item {i+1}:\n"
            # Only include fields that have values (from the database)
            for key, value in detail.items():
                if value:  # Only include non-empty values
                    response += f"- {key}: {value}\n"
            response += "\n"
    else:
        response = "No relevant information found in the database."

    # Set the response with only data from the database
    state.response = response
    
    # Log the refined context for debugging
    logger.info(f"Refined context (first 200 chars): {state.response[:200]}...")
    
    return state

# Add back the generate_response function with conversation tracking
async def generate_response(state: CustomerState) -> CustomerState:
    # Format conversation history
    formatted_history = format_conversation_history(state.conversation_history, state.language)
    
    # Get mall name for the prompt
    mall_name = "Cenomi Mall"  # Default
    if state.mall_id and hasattr(state, "mall_data") and state.mall_data:
        mall_info = state.mall_data.get("mall_information", {})
        if mall_info and isinstance(mall_info, dict):
            mall_name_key = "NameEn" if state.language == "en" else "NameAr"
            mall_name = mall_info.get(mall_name_key, "Cenomi Mall")
    
    # Get resolved entity if available
    resolved_entity = ""
    if state.context_data and state.context_data.get("resolved_entity"):
        resolved_entity = state.context_data.get("resolved_entity")
    
    logger.info(f"Generating response with context: {state.response[:100] if state.response else ''}...")
    
    try:
        # Ensure we have valid context data
        if not state.response or state.response.strip() == "":
            if state.language == "ar":
                state.response = "لم أتمكن من العثور على معلومات حول ذلك. هل يمكنك توضيح استفسارك؟"
            else:
                state.response = "I couldn't find information about that. Could you clarify your question?"
            return state
        
        # Ensure we're only using database data
        context_data = state.response
        
        # Add a warning for developers about hallucination
        context_data = f"""
IMPORTANT FOR SYSTEM: This is database information only. NEVER invent or hallucinate additional details.
If information is not found, clearly state that you don't have that information.

{context_data}
"""
        
        # Use a more efficient approach with asyncio
        input_dict = {
            "context": context_data,
            "query": state.query,
            "lang": state.language,
            "conversation_history": formatted_history,
            "mall_name": mall_name,
            "resolved_entity": resolved_entity,
            "topic_turn_count": state.topic_turn_count,
            "conversation_topic": state.conversation_topic
        }
        
        # The customer_chain returns a string, not a dictionary
        response = await asyncio.to_thread(customer_chain.invoke, input_dict)
        
        # Check for empty response
        if not response or len(response.strip()) == 0:
            if state.language == "ar":
                state.response = "أواجه مشكلة في معالجة طلبك حاليًا. يرجى المحاولة مرة أخرى."
            else:
                state.response = "I'm having trouble processing your request right now. Please try again."
            return state
        
        # Format response for display - response is already a string
        state.response = response
        
        # Post-process to remove any hallucinated store or product references
        # Implement more strict filtering if needed
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        if state.language == "ar":
            state.response = "أواجه مشكلة في معالجة طلبك حاليًا. يرجى المحاولة مرة أخرى."
        else:
            state.response = "I'm having trouble processing your request right now. Please try again."
    
    return state

# Helper function to extract type preferences from conversation history
def extract_type_preference(conversation_history, language):
    """Extract type preference from conversation history"""
    # Check if the user is responding to a type preference follow-up
    last_message = next((msg for msg in reversed(conversation_history) if msg["role"] == "user"), None)
    if not last_message:
        return None
        
    # Common types in English and Arabic
    common_types = ["italian", "indian", "chinese", "fast food", "sports", "casual", 
                   "formal", "kids", "women", "men", "luxury", "budget", "electronics"]
    
    arabic_common_types = ["إيطالي", "هندي", "صيني", "وجبات سريعة", "رياضة", "غير رسمي", 
                          "رسمي", "أطفال", "نساء", "رجال", "فاخر", "اقتصادي", "إلكترونيات"]
    
    all_types = common_types + arabic_common_types
    
    for type_name in all_types:
        if type_name.lower() in last_message["content"].lower():
            return type_name
            
    return None

# Helper function to process LLM response
def process_response(response, intent, topic_turn_count):
    """Process LLM response to format with recommendations and extract follow-up questions"""
    # Check if this is a product listing, store listing, or offer listing
    # If so, format the response as recommendations
    needs_recommendation_format = False
    recommendations = []
    follow_up_question = None
    
    # Helper function to extract image URLs from markdown format
    def extract_images_from_markdown(text):
        import re
        # Match markdown image pattern: ![alt text](url)
        image_pattern = r"!\[(.*?)\]\((.*?)\)"
        images = re.findall(image_pattern, text)
        # Return a list of tuples (alt_text, url)
        return images
    
    # Extract all images from the response
    all_images = extract_images_from_markdown(response)
    
    # Create a clean response without image markdown
    clean_response = response
    if all_images:
        for alt_text, url in all_images:
            # Remove the markdown image from the clean response
            clean_response = clean_response.replace(f"![{alt_text}]({url})", "").strip()
    
    # Determine if we need to format as a recommendation list
    if intent and intent in ["product_list", "store_list", "offer_list", "product_recommend", "store_recommend", "offer_recommend"]:
        needs_recommendation_format = True
        
        # Extract recommendations and follow-up question from the response
        # Parse bullet points or numbered lists
        lines = clean_response.split('\n')
        content_lines = []
        question_line = None
        
        for line in lines:
            stripped = line.strip()
            # Check if line is a follow-up question
            if stripped and (stripped.endswith('?') or '?' in stripped):
                question_line = stripped
            # Check if line is a recommendation (bullet point or numbered item)
            elif stripped and (stripped.startswith('•') or stripped.startswith('-') or 
                             stripped.startswith('*') or 
                             (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ['.', ')'])):
                content_lines.append(stripped)
            elif stripped:
                content_lines.append(stripped)
        
        # Convert to recommendation format
        if content_lines:
            # Parse up to 3 recommendations
            for i, line in enumerate(content_lines[:3]):
                # Remove bullet point or number prefix
                if line.startswith(('•', '-', '*')):
                    clean_line = line[1:].strip()
                elif len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')']:
                    clean_line = line[2:].strip()
                else:
                    clean_line = line.strip()
                
                # Split into title and description if possible
                if ':' in clean_line:
                    title, desc = clean_line.split(':', 1)
                    rec = {"title": title.strip(), "description": desc.strip()}
                else:
                    rec = {"title": clean_line, "description": ""}
                
                # Add image URL if available for this recommendation
                # Match an image to this recommendation if possible
                if all_images and i < len(all_images):
                    rec["image_url"] = all_images[i][1]  # Use the URL from the image tuple
                
                recommendations.append(rec)
        
        follow_up_question = question_line
        
        # If this is turn 2+, remove follow-up question to respect turn logic
        if topic_turn_count >= 2 and follow_up_question:
            follow_up_question = None
    
    # Create formatted response
    response_format = {
        "response": clean_response,
        "recommendations": recommendations if needs_recommendation_format else None,
        "is_recommendation_format": needs_recommendation_format,
        "follow_up_question": follow_up_question
    }
    
    return response_format

# Add back the initial_retrieval function needed by retrieve_fallback_context
async def initial_retrieval(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        if state.language == "ar":
            state.response = "عذراً! أحتاج إلى معرفة المركز التجاري الذي تسأل عنه. يرجى اختيار مركز تجاري أولاً! 😊"
        else:
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
    entity_type = state.intent.split("_")[0] if state.intent else "general"
    query_prefix = intent_prefixes.get(state.intent, entity_type) if state.intent else "general"

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
    
    # Process any location codes before returning
    if state.initial_context:
        for item in state.initial_context:
            if "metadata" in item and item["metadata"]:
                # Check for locations in any of the expected fields
                if "pms_unit_codes" in item["metadata"]:
                    item["metadata"]["location"] = convert_location_codes(item["metadata"]["pms_unit_codes"])
                elif "location" in item["metadata"]:
                    item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
    return state

async def fetch_loyalty_data(state: CustomerState) -> CustomerState:
    # Since loyalty tables don't exist in the schema, we'll return a message that it's not available
    state.response = "I'm sorry, but the loyalty program features are not available at this time."
    return state

# Update the workflow
customer_workflow = StateGraph(CustomerState)
customer_workflow.add_node("classify_query_type", classify_query_type)
customer_workflow.add_node("classify_intent", classify_intent)
customer_workflow.add_node("retrieve_product_context", retrieve_product_context)
customer_workflow.add_node("retrieve_mall_context", retrieve_mall_context)
customer_workflow.add_node("retrieve_offer_event_context", retrieve_offer_event_context)
customer_workflow.add_node("retrieve_services_context", retrieve_services_context)
customer_workflow.add_node("retrieve_family_planning_context", retrieve_family_planning_context)
customer_workflow.add_node("retrieve_visit_planning_context", retrieve_visit_planning_context)
customer_workflow.add_node("retrieve_fallback_context", retrieve_fallback_context)
customer_workflow.add_node("refine_context", refine_context)
customer_workflow.add_node("respond", generate_response)
customer_workflow.add_node("fetch_loyalty_data", fetch_loyalty_data)

# Set entry point to the new node
customer_workflow.set_entry_point("classify_query_type")

# Route from query_type classification to intent classification
customer_workflow.add_edge("classify_query_type", "classify_intent")

# Route from intent classification to the appropriate context retrieval function
def route_after_intent_classify(state: CustomerState):
    if state.intent in ["loyalty_balance", "loyalty_programs"]:
        return "fetch_loyalty_data"
    elif state.intent and state.intent.startswith("other_") and state.response:
        return "respond"
    else:
        # Use the query_type to determine which retrieval function to use
        if state.query_type == "product_info_query":
            return "retrieve_product_context"
        elif state.query_type == "mall_info_query":
            return "retrieve_mall_context"
        elif state.query_type == "offer_or_event_info_query":
            return "retrieve_offer_event_context"
        elif state.query_type == "services_info_query":
            return "retrieve_services_context"
        elif state.query_type == "family_planning_query":
            return "retrieve_family_planning_context"
        elif state.query_type == "visit_planning_query":
            return "retrieve_visit_planning_context"
        else:
            return "retrieve_fallback_context"
    
customer_workflow.add_conditional_edges(
    "classify_intent",
    route_after_intent_classify,
    {
        "fetch_loyalty_data": "fetch_loyalty_data",
        "respond": "respond",
        "retrieve_product_context": "retrieve_product_context",
        "retrieve_mall_context": "retrieve_mall_context",
        "retrieve_offer_event_context": "retrieve_offer_event_context",
        "retrieve_services_context": "retrieve_services_context",
        "retrieve_family_planning_context": "retrieve_family_planning_context",
        "retrieve_visit_planning_context": "retrieve_visit_planning_context",
        "retrieve_fallback_context": "retrieve_fallback_context",
    }
)

# Connect all retrieval nodes to refine_context
customer_workflow.add_edge("retrieve_product_context", "refine_context")
customer_workflow.add_edge("retrieve_mall_context", "refine_context")
customer_workflow.add_edge("retrieve_offer_event_context", "refine_context")
customer_workflow.add_edge("retrieve_services_context", "refine_context")
customer_workflow.add_edge("retrieve_family_planning_context", "refine_context")
customer_workflow.add_edge("retrieve_visit_planning_context", "refine_context")
customer_workflow.add_edge("retrieve_fallback_context", "refine_context")

# Finish the workflow
customer_workflow.add_edge("fetch_loyalty_data", END)
customer_workflow.add_edge("refine_context", "respond")
customer_workflow.add_edge("respond", END)

# Compile the workflow
customer_graph = customer_workflow.compile()

# Setup thread pool for CPU-bound tasks
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Create optimized batched LLM processor
class BatchLLMProcessor:
    def __init__(self, llm):
        self.llm = llm
        self.queue = []
        self.results = {}
        self.max_batch_size = 3  # Set optimal batch size
        
    async def add_to_queue(self, prompt_id, prompt, parser=None):
        """Add a prompt to the batch queue and return a future for the result"""
        self.queue.append((prompt_id, prompt, parser))
        # Process immediately if batch size reached
        if len(self.queue) >= self.max_batch_size:
            await self.process_batch()
            
    async def process_batch(self):
        """Process all queued prompts in a single batch"""
        if not self.queue:
            return
            
        # Prepare batch inputs
        batch_inputs = []
        batch_ids = []
        batch_parsers = []
        
        for prompt_id, prompt, parser in self.queue:
            batch_inputs.append(prompt)
            batch_ids.append(prompt_id)
            batch_parsers.append(parser)
            
        # Process batch with LLM
        try:
            responses = await self.llm.abatch(batch_inputs)
            
            # Apply parsers and store results
            for i, response in enumerate(responses):
                if batch_parsers[i]:
                    # Apply parser if provided
                    parsed_response = await batch_parsers[i].ainvoke(response)
                    self.results[batch_ids[i]] = parsed_response
                else:
                    self.results[batch_ids[i]] = response
                    
        except Exception as e:
            logger.error(f"Error processing LLM batch: {e}")
            # Set error result for all requests in batch
            for prompt_id in batch_ids:
                self.results[prompt_id] = f"Error: {str(e)}"
                
        # Clear the queue
        self.queue = []
        
    async def get_result(self, prompt_id):
        """Get the result for a specific prompt"""
        if prompt_id in self.results:
            return self.results[prompt_id]
        # If not processed yet, process the batch
        await self.process_batch()
        return self.results.get(prompt_id)

# Initialize the batch processor
batch_llm = BatchLLMProcessor(llm)

# Cache for expensive operations
class OptimizedCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.memory_cache = {}
        self.ttl = 300  # 5 minutes default TTL
        
    def get(self, key):
        """Get item from cache with memory-first approach"""
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            return self.memory_cache[key]
            
        # Then try Redis
        cached = self.redis.get(key)
        if cached:
            try:
                if isinstance(cached, bytes):
                    cached = cached.decode('utf-8')
                result = json.loads(cached)
                # Update memory cache
                self.memory_cache[key] = result
                return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error decoding cached value for {key}: {e}")
                # Return the raw value if JSON parsing fails
                return cached
        return None
        
    def set(self, key, value, ex=None):
        """Set item in both memory and Redis cache"""
        ttl = ex or self.ttl
        # Set in memory cache
        self.memory_cache[key] = value
        try:
            # Set in Redis with expiry
            self.redis.set(key, json.dumps(value, cls=DateTimeEncoder), ex=ttl)
        except Exception as e:
            logger.error(f"Error setting cache for {key}: {e}")

# Initialize optimized cache
optimized_cache = OptimizedCache(REDIS_CLIENT)

# Create combined LLM chain for classification to reduce sequential calls
combined_classification_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    You are CenomiAI, a mall assistant. Analyze this query and conversation history to determine:
    
    1. The high-level query type (QUERY_TYPE)
    2. The specific intent with entity and action (INTENT)
    
    Query: "{query}"
    
    Previous conversation:
    {conversation_history}
    
    First, determine the QUERY_TYPE (one of these exact strings):
    - product_info_query: When asking about products, stores, brands, restaurants or specific items
    - mall_info_query: When asking about the mall itself (location, timings, facilities, directions)
    - offer_or_event_info_query: When asking about offers, events, sales, promotions, or discounts
    - services_info_query: When asking about available services (parking, wheelchair access, restrooms, etc.)
    - family_planning_query: When asking about child-friendly activities, family itineraries, or visit planning
    - visit_planning_query: When asking for suggestions about what to do in the mall or creating an itinerary
    - fallback_query: When the query doesn't clearly fit any of the above categories
    
    Then, determine the INTENT as a JSON object with:
    - entity_type: The entity being asked about (store, offer, product, event, service, amenity, loyalty)
    - action: What they want (info, navigate, recommend, list, balance, programs)
    - collected_data: Any specific details (e.g., store name, product type)
    
    Format your response exactly as follows:
    
    QUERY_TYPE: query_type_here
    INTENT: {"entity_type": "type", "action": "action", "collected_data": {"name": "entity_name"}}
    """
)

# Optimized combined classification chain
combined_classifier = combined_classification_prompt | llm | StrOutputParser()

# Modify the original classification functions to use the new optimized approach
async def optimized_classification(state: CustomerState) -> Dict[str, Any]:
    """Combined function to classify both query type and intent in one LLM call"""
    formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-6:]]) if state.conversation_history else "No prior conversation."
    
    cache_key = f"combined_classification:{state.query}:{hash(formatted_history)}"
    cached_result = optimized_cache.get(cache_key)
    
    if cached_result:
        return cached_result
    
    try:
        # Make a single LLM call instead of two sequential ones
        combined_result = await combined_classifier.ainvoke({
            "query": state.query, 
            "conversation_history": formatted_history
        })
        
        # Parse the result
        lines = combined_result.strip().split('\n')
        query_type = None
        intent_json = None
        
        for line in lines:
            if line.startswith("QUERY_TYPE:"):
                query_type = line.replace("QUERY_TYPE:", "").strip()
            elif line.startswith("INTENT:"):
                try:
                    intent_json_str = line.replace("INTENT:", "").strip()
                    intent_json = json.loads(intent_json_str)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse intent JSON: {line}")
                    intent_json = {"entity_type": "other", "action": "info", "collected_data": {}}
        
        # Validate query type
        valid_query_types = ["product_info_query", "mall_info_query", "offer_or_event_info_query", 
                            "services_info_query", "family_planning_query", "visit_planning_query", 
                            "fallback_query"]
        
        if not query_type or query_type not in valid_query_types:
            query_type = "fallback_query"
            
        # Process intent
        intent = "other_info"
        resolved_entity = None
        type_preference = None
        needs_type_follow_up = False
        
        if intent_json:
            try:
                intent = f"{intent_json['entity_type']}_{intent_json['action']}"
                
                if intent_json.get("collected_data", {}).get("name"):
                    resolved_entity = intent_json["collected_data"]["name"]
                
                type_preference = intent_json.get("type_preference")
                
                if intent.startswith(("product_", "store_", "service_")) and not type_preference:
                    needs_type_follow_up = True
            except (KeyError, TypeError):
                pass
        
        result = {
            "query_type": query_type,
            "intent": intent,
            "resolved_entity": resolved_entity,
            "type_preference": type_preference,
            "needs_type_follow_up": needs_type_follow_up
        }
        
        # Cache the result
        optimized_cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in optimized classification: {e}")
        return {
            "query_type": "fallback_query",
            "intent": "other_info",
            "resolved_entity": None,
            "type_preference": None,
            "needs_type_follow_up": False
        }

# Optimize Pinecone vector search with better parallelization and caching
async def optimized_vector_search(query: str, filter_dict: Dict[str, Any], top_k: int = 15):
    """Optimized vector search with improved caching and error handling"""
    # Create a cache key based on the query and filters
    cache_key = f"vector_search:{hash(query)}:{hash(json.dumps(filter_dict, sort_keys=True))}:{top_k}"
    cached_result = optimized_cache.get(cache_key)
    
    if cached_result:
        return cached_result
    
    try:
        # Embed the query in a thread to not block the event loop
        query_vector = await asyncio.to_thread(
            embeddings.embed_query,
            query
        )
        
        # Ensure filter_dict is properly formatted for Pinecone
        # Convert any type fields from string to int if needed
        if filter_dict and "type" in filter_dict and isinstance(filter_dict["type"], str):
            # Create a mapping of type names to integers if needed by your schema
            type_mapping = {
                "product": 1,
                "store": 2,
                "service": 3,
                "engagement": 4,
                "amenity": 5
            }
            
            # Convert string type to int if in mapping
            if filter_dict["type"] in type_mapping:
                filter_dict["type"] = type_mapping[filter_dict["type"]]
        
        # Perform the vector search with proper error handling
        try:
            results = await asyncio.to_thread(
                index.query,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Safety check for results
            if not results or not isinstance(results, dict):
                logger.warning(f"Unexpected results format from Pinecone: {type(results)}")
                return []
                
            # Safely extract matches
            matches = results.get("matches", [])
            if not isinstance(matches, list):
                logger.warning(f"Unexpected matches format in Pinecone results: {type(matches)}")
                return []
                
            # Process results safely
            search_results = []
            for doc in matches:
                if isinstance(doc, dict) and "id" in doc and "score" in doc and "metadata" in doc:
                    search_results.append({
                        "id": doc["id"],
                        "score": float(doc["score"]), 
                        "metadata": doc["metadata"]
                    })
                else:
                    logger.warning(f"Skipping invalid match in Pinecone results: {doc}")
                
        except Exception as pinecone_error:
            logger.error(f"Error in Pinecone query: {pinecone_error}")
            return []
        
        # Cache the results
        optimized_cache.set(cache_key, search_results, ex=300)  # 5 minute cache
        
        return search_results
    except Exception as e:
        logger.error(f"Error in optimized vector search: {e}")
        return []

# Parallelize database operations for context data
async def parallel_db_fetch(queries):
    """Run multiple database queries in parallel"""
    results = await asyncio.gather(*[db_fetch_all_async(*query) for query in queries])
    return results

# Modified classify_query_type and classify_intent to use the optimized classification
async def classify_query_type(state: CustomerState) -> CustomerState:
    combined_result = await optimized_classification(state)
    if combined_result:
        state.query_type = combined_result.get("query_type")
        logger.info(f"Classified query type: {state.query_type}")
    else:
        logger.warning("Could not classify query type")
        state.query_type = "general_query"  # Default fallback
    return state

async def classify_intent(state: CustomerState) -> CustomerState:
    # Reuse the same result from combined classification if already performed
    if hasattr(state, "_combined_classification_result") and state._combined_classification_result:
        combined_result = state._combined_classification_result
    else:
        combined_result = await optimized_classification(state)
        state._combined_classification_result = combined_result
    
    # Ensure we have a valid intent, defaulting to fallback if none
    state.intent = combined_result.get("intent") if combined_result else "other_general"
    state.context_data = state.context_data or {}
    
    # Set resolved entity if present
    if combined_result and combined_result.get("resolved_entity"):
        state.context_data["resolved_entity"] = combined_result["resolved_entity"]
    
    # Set type preference if present
    if combined_result and combined_result.get("type_preference"):
        state.type_preference = combined_result["type_preference"]
    
    # Determine if we need a type follow-up
    if combined_result:
        state.needs_type_follow_up = combined_result.get("needs_type_follow_up", False)
    
    # Only check startswith if state.intent is not None
    if state.intent and state.intent.startswith("other_"):
        state.response = "I'm not sure what you mean 😅. Could you tell me more?"
    
    logger.info(f"Classified intent: {state.intent}, Resolved entity: {state.context_data.get('resolved_entity')}, Type preference: {state.type_preference}")
    return state

# Optimize context retrieval functions

# Add parallel database fetching helper
async def parallel_context_fetch(mall_id: int, entity_name: str = None, current_date: str = None):
    """Fetch all the contextual data needed in parallel"""
    
    # Define the queries to run
    query_tasks = []
    
    # Get mall information
    query_tasks.append(db_fetch_one_async(
        """SELECT marketing_name, marketing_name_ar, city, country, mall_information, gps_coordinates
        FROM malls WHERE unique_property_id = $1""",
        (mall_id,)
    ))
    
    # Get stores data (with or without entity filter)
    if entity_name:
        query_tasks.append(db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
               b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes,
               b.brand_logo, b.banner_en, b.banner_ar
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
               WHERE bma.unique_property_id = $1 AND LOWER(b.brand_name_en) LIKE $2""", 
            (mall_id, f"%{entity_name.lower()}%")
        ))
    else:
        query_tasks.append(db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
               b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes,
               b.brand_logo, b.banner_en, b.banner_ar
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
               WHERE bma.unique_property_id = $1""", 
            (mall_id,)
        ))
    
    # Get products data
    query_tasks.append(db_fetch_all_async(
        """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
           p.is_featured, p.in_stock, b.brand_name_en, b.brand_logo
           FROM products p
           JOIN brands b ON p.brand_id = b.brand_id
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
           WHERE bma.unique_property_id = $1 
           LIMIT 20""",
        (mall_id,)
    ))
    
    # Get engagements (offers and events)
    if current_date:
        query_tasks.append(db_fetch_all_async(
            """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
               e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, e.unique_property_id,
               e.images_en, e.images_ar, b.brand_name_en
               FROM engagements e 
               LEFT JOIN brands b ON e.brand_id = b.brand_id
               WHERE e.unique_property_id = $1 AND 
               (e.end_date >= $2 OR e.end_date IS NULL)""",
            (mall_id, current_date)
        ))
    
    # Get services
    query_tasks.append(db_fetch_all_async(
        """SELECT s.id AS service_id, s.name, s.description, s.description_ar, s.location, s.is_available, s.icon_url
           FROM services s
           WHERE s.unique_property_id = $1""",
        (mall_id,)
    ))
    
    # Run all queries in parallel
    results = await asyncio.gather(*query_tasks, return_exceptions=True)
    
    # Process results
    mall_info = results[0] if not isinstance(results[0], Exception) else None
    stores = results[1] if not isinstance(results[1], Exception) else []
    products = results[2] if not isinstance(results[2], Exception) else []
    
    # Engagements and services depend on whether current_date was provided
    idx = 3
    engagements = []
    if current_date:
        engagements = results[idx] if not isinstance(results[idx], Exception) else []
        idx += 1
    
    services = results[idx] if not isinstance(results[idx], Exception) else []
    
    return {
        "mall_info": mall_info,
        "stores": stores,
        "products": products,
        "engagements": engagements,
        "services": services
    }

# Optimize the refine_context function which is a major bottleneck
async def refine_context(state: CustomerState) -> CustomerState:
    """
    Refine the context based on the intent and query type.
    This function ensures we're only using data from the database or vector store.
    """
    if not state.mall_id:
        state.response = "Please select a mall first."
        return state

    # Check if we have any initial context
    if not state.initial_context or len(state.initial_context) == 0:
        logger.warning("No initial context found for refining")
        if state.language == "ar":
            state.response = "عذراً، لا يمكنني العثور على معلومات ذات صلة. هل يمكنك إعادة صياغة سؤالك؟"
        else:
            state.response = "Sorry, I couldn't find any relevant information. Could you rephrase your question?"
        return state

    # Sort initial context by score for better retrieval (higher scores first)
    state.initial_context = sorted(state.initial_context, key=lambda x: x.get("score", 0), reverse=True)
    
    # Limit to top 10 results for performance 
    top_results = state.initial_context[:10]

    # Create a detailed context from the database results
    details = []
    for item in top_results:
        # Skip if no metadata (shouldn't happen with proper retrieval)
        if not item.get("metadata"):
            continue
            
        # Add item details based on its metadata
        # Only include fields that exist in the database
        metadata = item["metadata"]
        
        # Construct detail string with validation to ensure no missing fields
        detail = {}
        
        # Common fields
        detail["id"] = metadata.get("id", "")
        detail["type"] = metadata.get("type", "")
        detail["score"] = item.get("score", 0)
        
        # Handle different types of data
        if metadata.get("type") == "store":
            # Store-specific fields (only include what's in the database)
            detail["name"] = metadata.get("name", "")
            detail["category"] = metadata.get("category", "")
            detail["location"] = metadata.get("pms_unit_codes", "")
            detail["description"] = metadata.get("description", "")
            if "image_url" in metadata and metadata["image_url"]:
                detail["image_url"] = metadata["image_url"]
                
        elif metadata.get("type") == "product":
            # Product-specific fields
            detail["name"] = metadata.get("name", "")
            detail["price"] = metadata.get("price", "")
            detail["brand_name"] = metadata.get("brand_name", "")
            detail["description"] = metadata.get("description", "")
            detail["store_name"] = metadata.get("store_name", "")
            if "image_url" in metadata and metadata["image_url"]:
                detail["image_url"] = metadata["image_url"]
                
        elif metadata.get("type") == "engagement":
            # Engagement-specific fields (offers, events)
            detail["title"] = metadata.get("title_en", "")
            detail["brand_name"] = metadata.get("brand_name", "")
            detail["start_date"] = metadata.get("start_date", "")
            detail["end_date"] = metadata.get("end_date", "")
            detail["description"] = metadata.get("description_en", "")
            if "image_url" in metadata and metadata["image_url"]:
                detail["image_url"] = metadata["image_url"]
                
        elif metadata.get("type") == "amenity" or metadata.get("type") == "service":
            # Amenity/service-specific fields
            detail["name"] = metadata.get("name", "")
            detail["location"] = metadata.get("location", "")
            detail["description"] = metadata.get("description", "")
            if "image_url" in metadata and metadata["image_url"]:
                detail["image_url"] = metadata["image_url"]
                
        # Add the structured detail to our results
        details.append(detail)

    # Prepare the response with only database information
    response = ""
    
    # Add a prefix to indicate this is database content
    response += "DATABASE CONTENT:\n\n"
    
    # Format the details as a structured response
    if details:
        for i, detail in enumerate(details):
            response += f"Item {i+1}:\n"
            # Only include fields that have values (from the database)
            for key, value in detail.items():
                if value:  # Only include non-empty values
                    response += f"- {key}: {value}\n"
            response += "\n"
    else:
        response = "No relevant information found in the database."

    # Set the response with only data from the database
    state.response = response
    
    # Log the refined context for debugging
    logger.info(f"Refined context (first 200 chars): {state.response[:200]}...")
    
    return state

# Helper functions for parallel processing

def extract_mall_metadata(mall_data, mall_info):
    """Extract mall metadata from raw mall data"""
    metadata = {}
    
    # Extract address from MallContact
    if "MallContact" in mall_data:
        contact_data = mall_data["MallContact"]
        address_lines_en = []
        if contact_data.get("Address1En"):
            address_lines_en.append(contact_data["Address1En"])
        if contact_data.get("Address2En"):
            address_lines_en.append(contact_data["Address2En"])
        metadata["address_en"] = ", ".join(address_lines_en)
        
        address_lines_ar = []
        if contact_data.get("Address1Ar"):
            address_lines_ar.append(contact_data["Address1Ar"])
        if contact_data.get("Address2Ar"):
            address_lines_ar.append(contact_data["Address2Ar"])
        metadata["address_ar"] = ", ".join(address_lines_ar)
        
        metadata["contact_phone"] = contact_data.get("Phone")
        metadata["contact_email"] = contact_data.get("Email")
    
    # Extract opening hours if available
    if "MallTiming" in mall_data and isinstance(mall_data["MallTiming"], list):
        metadata["opening_hours"] = mall_data["MallTiming"]
    
    # Extract mall description
    metadata["description_en"] = mall_data.get("MallDescriptionEn")
    metadata["description_ar"] = mall_data.get("MallDescriptionAr")
    
    # Extract map URL
    metadata["map_url"] = mall_data.get("GoogleMapURL") or mall_data.get("MallMapEn")
    
    # Add other mall info
    metadata["name_en"] = mall_info.get("marketing_name", "")
    metadata["name_ar"] = mall_info.get("marketing_name_ar", "")
    metadata["city"] = mall_info.get("city", "")
    metadata["country"] = mall_info.get("country", "")
    metadata["gps_coordinates"] = mall_info.get("gps_coordinates", "")
    
    return metadata

def process_stores(stores, target_store_name, language, type_preference):
    """Process store data in parallel"""
    result = {
        "stores": [],
        "store_categories": set(),
        "food_categories": set(),
        "target_store": None
    }
    
    food_related_categories = ['restaurant', 'cafe', 'food', 'dining', 'bakery', 'coffee']
    
    for store in stores:
        # Extract image URLs
        image_url = None
        # First try brand logo as it's usually the most relevant
        if store.get("brand_logo"):
            image_url = store["brand_logo"]
        # Then try banner based on language
        elif language == "ar" and store.get("banner_ar"):
            image_url = store["banner_ar"]
        elif store.get("banner_en"):
            image_url = store["banner_en"]
            
        store_data = {
            "name": store["brand_name_en"],
            "category": store.get("category_name", ""),
            "store_id": store["brand_id"],
            "description": store.get("description_en", ""),
            "phone": store.get("store_phone_number", ""),
            "email": store.get("store_email", ""),
            "website": store.get("store_website", ""),
            "location": store.get("pms_unit_codes", {}),
            "image_url": image_url
        }
        
        # Add to category lists
        category = store.get("category_name", "").lower()
        if category:
            # Extract food types from description if it's a food place
            is_food_place = any(food_term in category for food_term in food_related_categories)
            
            if is_food_place:
                # Add to food categories
                result["food_categories"].add(category)
                
                # Try to extract specific cuisine types from description
                description = store.get("description_en", "").lower()
                cuisine_types = ["italian", "indian", "chinese", "japanese", "american", "mexican", 
                                "thai", "fast food", "mediterranean", "middle eastern", "french"]
                for cuisine in cuisine_types:
                    if cuisine in description:
                        result["food_categories"].add(cuisine)
            else:
                # Add to store categories
                result["store_categories"].add(category)
        
        # Check if this is the store being searched for
        if target_store_name and target_store_name in store["brand_name_en"].lower():
            result["stores"].insert(0, store_data)
            result["target_store"] = store_data
        else:
            result["stores"].append(store_data)
    
    # Apply type preference filtering if specified
    if type_preference:
        filtered_stores = []
        type_pref = type_preference.lower()
        
        # Filter stores by category
        for store in result["stores"]:
            category = store.get("category", "").lower()
            description = store.get("description", "").lower()
            
            # Check if it's a food place and matches the type preference
            is_food_place = category and any(food_term in category for food_term in food_related_categories)
            
            if ((is_food_place and (type_pref in category or type_pref in description)) or
                (not is_food_place and category and type_pref in category)):
                filtered_stores.append(store)
        
        # Only replace if we found matches
        if filtered_stores:
            result["stores"] = filtered_stores
    
    return result

def process_engagements(engagements, language):
    """Process engagements in parallel"""
    result = {
        "offers": [],
        "events": []
    }
    
    for engagement in engagements:
        # Extract image URLs
        image_url = None
        if language == "ar" and engagement.get("images_ar"):
            # Try to get Arabic image if language is Arabic
            if isinstance(engagement["images_ar"], str):
                image_url = engagement["images_ar"]
            elif isinstance(engagement["images_ar"], list) and len(engagement["images_ar"]) > 0:
                image_url = engagement["images_ar"][0]  # Take the first image if multiple exist
        elif engagement.get("images_en"):
            # Use English image as fallback
            if isinstance(engagement["images_en"], str):
                image_url = engagement["images_en"]
            elif isinstance(engagement["images_en"], list) and len(engagement["images_en"]) > 0:
                image_url = engagement["images_en"][0]  # Take the first image if multiple exist
        
        engagement_type = engagement.get("type", "").lower()
        
        if engagement_type == "offer":
            offer_data = {
                "id": engagement["engagement_id"],
                "title": engagement.get("title_en", ""),
                "description": engagement.get("description_en", ""),
                "brand_id": engagement.get("brand_id"),
                "store_name": engagement.get("brand_name_en", "Unknown Store"),
                "start_date": engagement.get("start_date", ""),
                "end_date": engagement.get("end_date", ""),
                "terms": engagement.get("terms_conditions_en", ""),
                "is_exclusive": bool(engagement.get("is_exclusive", 0)),
                "image_url": image_url
            }
            result["offers"].append(offer_data)
        
        elif engagement_type == "events":
            event_data = {
                "id": engagement["engagement_id"],
                "name": engagement.get("title_en", ""),
                "description": engagement.get("description_en", ""),
                "brand_id": engagement.get("brand_id"),
                "store_name": engagement.get("brand_name_en", ""),
                "start_date": engagement.get("start_date", ""),
                "end_date": engagement.get("end_date", ""),
                "terms": engagement.get("terms_conditions_en", ""),
                "image_url": image_url
            }
            result["events"].append(event_data)
    
    return result

def process_products(products, type_preference):
    """Process products in parallel"""
    result = {
        "products": [],
        "product_categories": set()
    }
    
    for product in products:
        product_category = product.get("category", "").lower()
        if product_category:
            result["product_categories"].add(product_category)
            
        product_data = {
            "id": product["id"],
            "name": product["name"],
            "description": product.get("description", ""),
            "price": float(product["price"]) if product.get("price") is not None else None,
            "brand_id": product["brand_id"],
            "category": product_category,
            "store_name": product.get("brand_name_en", ""),
            "in_stock": product.get("in_stock", True),
            "image_url": product.get("brand_logo")  # Use brand logo for product image
        }
        
        result["products"].append(product_data)
    
    # Apply type preference filtering if specified
    if type_preference:
        filtered_products = []
        type_pref = type_preference.lower()
        
        for product in result["products"]:
            category = product.get("category", "").lower()
            description = product.get("description", "").lower()
            
            if ((category and type_pref in category) or 
                (description and type_pref in description)):
                filtered_products.append(product)
        
        # Only replace if we found matches
        if filtered_products:
            result["products"] = filtered_products
    
    return result

def process_services(services):
    """Process services in parallel"""
    result = []
    
    for service in services:
        service_data = {
            "name": service.get("name", ""),
            "description": service.get("description", ""),
            "description_ar": service.get("description_ar", ""),
            "location": service.get("location", ""),
            "is_available": service.get("is_available", True),
            "image_url": service.get("icon_url")
        }
        result.append(service_data)
    
    return result

def find_neighboring_stores(target_store, all_stores, max_neighbors=5):
    """Find neighboring stores based on location"""
    neighboring_stores = []
    
    # Extract location codes for the target store
    location_codes = target_store["location"]
    
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
    for location_code in location_codes if isinstance(location_codes, list) else [location_codes]:
        prefix, number = get_location_prefix_and_number(location_code)
        if prefix and number is not None:
            # Check for adjacent numbers (±1, ±2)
            adjacent_codes = [
                f"{prefix}{number-2}", f"{prefix}{number-1}", 
                f"{prefix}{number+1}", f"{prefix}{number+2}"
            ]
            
            for store in all_stores:
                if store == target_store:
                    continue
                
                store_locations = store.get("location", [])
                if not store_locations:
                    continue
                    
                # Convert to list if not already
                if not isinstance(store_locations, list):
                    store_locations = [store_locations]
                
                if any(code in adjacent_codes for code in store_locations):
                    if store not in neighboring_stores:
                        neighboring_stores.append(store)
    
    return neighboring_stores[:max_neighbors]  # Limit to max_neighbors

def process_vector_results(vector_results, context, mall_id, intent):
    """Process vector search results to enhance context"""
    brand_ids = set()
    
    for doc in vector_results:
        metadata = doc["metadata"]
        if metadata.get("mall_id") != mall_id:
            continue
        
        doc_type = metadata.get("type")
        if doc_type == "store" and not (intent and intent.startswith("store_")):
            # Only add store from vector search if not already doing a direct store query
            store_category = metadata.get("category_en", "").lower()
            if store_category:
                context["category_types"]["store"].add(store_category)
                
            store = {
                "name": metadata.get("name_en"),
                "category": store_category,
                "store_id": metadata.get("brand_id"),
                "description": metadata.get("description_en"),
                "image_url": metadata.get("brand_logo") or metadata.get("banner_en")
            }
            if not any(s.get("name") == store["name"] for s in context["stores"]):
                context["stores"].append(store)
            if metadata.get("brand_id"):
                brand_ids.add(metadata.get("brand_id"))
                
        elif doc_type == "product" and not (intent and intent.startswith("product_")):
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
                "image_url": metadata.get("image_url")
            }
            if not any(p.get("name") == product["name"] for p in context["products"]):
                context["products"].append(product)
            if metadata.get("brand_id"):
                brand_ids.add(metadata["brand_id"])
                
        elif doc_type == "service":
            service = {
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "location": metadata.get("location"),
                "image_url": metadata.get("icon_url")
            }
            if not any(s.get("name") == service["name"] for s in context["services"]):
                context["services"].append(service)
    
    return context

def process_all_location_codes(context):
    """Process location codes in all context data"""
    # Process store locations
    for store in context["stores"]:
        if "location" in store:
            store["location"] = convert_location_codes(store["location"])
        # Also check for pms_unit_codes if location is not present
        elif "pms_unit_codes" in store:
            store["location"] = convert_location_codes(store["pms_unit_codes"])
    
    # Process product locations via store
    for product in context["products"]:
        if "location" in product:
            product["location"] = convert_location_codes(product["location"])
        # Also check for pms_unit_codes if location is not present
        elif "pms_unit_codes" in product:
            product["location"] = convert_location_codes(product["pms_unit_codes"])
    
    # Process service locations
    for service in context["services"]:
        if "location" in service:
            service["location"] = convert_location_codes(service["location"])
        # Also check for pms_unit_codes if location is not present
        elif "pms_unit_codes" in service:
            service["location"] = convert_location_codes(service["pms_unit_codes"])
    
    # Process neighboring stores locations
    for store in context["neighboring_stores"]:
        if "location" in store:
            store["location"] = convert_location_codes(store["location"])
        # Also check for pms_unit_codes if location is not present
        elif "pms_unit_codes" in store:
            store["location"] = convert_location_codes(store["pms_unit_codes"])
            
    return context

# Combine context retrieval based on query type to a single optimized function
async def retrieve_context(state: CustomerState) -> CustomerState:
    """Combined context retrieval function that uses query_type to determine approach"""
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state
    
    # Use cache based on query type, intent, and query
    cache_key = f"context:{state.query_type}:{state.intent}:{state.query}:{state.mall_id}"
    cached_context = optimized_cache.get(cache_key)
    if cached_context:
        state.initial_context = cached_context
        return state
    
    # Vector search parameters based on query type
    filter_dict = {"mall_id": state.mall_id}
    top_k = 15
    entity_name = state.context_data.get("resolved_entity", "").lower() if state.context_data else ""
    
    # Modify search query based on query type
    if state.query_type == "product_info_query":
        search_query = f"product {entity_name if entity_name else state.query}"
        # Using integer value for type according to the mapping
        filter_dict["type"] = 1  # 1 represents "product" in our mapping
        
    elif state.query_type == "mall_info_query":
        search_query = f"mall information {state.query}"
        # No type filter - we'll get mall, amenity, and general info
        
    elif state.query_type == "offer_or_event_info_query":
        search_query = f"offer event {entity_name if entity_name else state.query}"
        # We'll use DB queries directly for offers and events, but use vector search as backup
        
    elif state.query_type == "services_info_query":
        search_query = f"mall service {state.query}"
        # Using integer value for type according to the mapping
        filter_dict["type"] = 3  # 3 represents "service" in our mapping
        
    elif state.query_type == "family_planning_query":
        search_query = f"family children activities {state.query}"
        # No type filter - we want diverse results
        
    elif state.query_type == "visit_planning_query":
        search_query = f"mall visit itinerary {state.query}"
        # No type filter - we want diverse results
        
    else:  # fallback_query or unknown
        search_query = state.query
    
    # Perform vector search for semantic results
    context_results = await optimized_vector_search(search_query, filter_dict, top_k)
    
    # Enhance with direct database results based on query type
    if state.query_type == "offer_or_event_info_query":
        # Get current date for filtering current/future events
        current_date = datetime.now().isoformat()
        
        # Get offers and events directly from database
        engagements = await db_fetch_all_async(
            """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
               e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, b.brand_name_en
               FROM engagements e 
               LEFT JOIN brands b ON e.brand_id = b.brand_id
               WHERE e.unique_property_id = $1 AND 
               (e.end_date >= $2 OR e.end_date IS NULL)
               ORDER BY e.start_date ASC""",
            (state.mall_id, current_date)
        )
        
        # Add engagements to context with higher scores
        for engagement in engagements:
            engagement_type = engagement.get("type", "").lower()
            metadata = {
                "type": engagement_type,
                "title": engagement.get("title_en", ""),
                "description": engagement.get("description_en", ""),
                "start_date": engagement.get("start_date", ""),
                "end_date": engagement.get("end_date", ""),
                "terms": engagement.get("terms_conditions_en", ""),
                "is_exclusive": bool(engagement.get("is_exclusive", 0)),
                "mall_id": state.mall_id,
                "brand_id": engagement.get("brand_id"),
                "brand_name": engagement.get("brand_name_en", "")
            }
            
            # Higher score for direct database results
            context_results.append({
                "id": f"engagement_{engagement['engagement_id']}",
                "score": 1.0,  # High confidence for direct DB results
                "metadata": metadata
            })
        
        # Boost scores for matching entity name if provided
        if entity_name:
            for item in context_results:
                if (item["metadata"].get("title", "").lower().find(entity_name) > -1 or
                    item["metadata"].get("brand_name", "").lower().find(entity_name) > -1):
                    item["score"] *= 1.5  # Boost score by 50%
    
    # Process location codes
    for item in context_results:
        if "metadata" in item and item["metadata"]:
            if "location" in item["metadata"]:
                item["metadata"]["location"] = convert_location_codes(item["metadata"]["location"])
    
    # Sort by score and cache
    context_results.sort(key=lambda x: x["score"], reverse=True)
    state.initial_context = context_results
    
    # Cache the processed context
    optimized_cache.set(cache_key, context_results, ex=300)
    
    return state

# Update the workflow with optimized nodes
customer_workflow = StateGraph(CustomerState)

# Add nodes
customer_workflow.add_node("classify_query_type", classify_query_type)
customer_workflow.add_node("classify_intent", classify_intent)
customer_workflow.add_node("retrieve_context", retrieve_context)
customer_workflow.add_node("refine_context", refine_context)
customer_workflow.add_node("respond", generate_response)
customer_workflow.add_node("fetch_loyalty_data", fetch_loyalty_data)

# Set entry point
customer_workflow.set_entry_point("classify_query_type")

# Route from query_type classification to intent classification
customer_workflow.add_edge("classify_query_type", "classify_intent")

# Route from intent classification to appropriate next step
def route_after_intent_classify(state: CustomerState):
    if state.intent in ["loyalty_balance", "loyalty_programs"]:
        return "fetch_loyalty_data"
    elif state.intent and state.intent.startswith("other_") and state.response:
        return "respond"
    else:
        return "retrieve_context"
    
customer_workflow.add_conditional_edges(
    "classify_intent",
    route_after_intent_classify,
    {
        "fetch_loyalty_data": "fetch_loyalty_data",
        "respond": "respond",
        "retrieve_context": "retrieve_context",
    }
)

# Connect context retrieval to refine_context
customer_workflow.add_edge("retrieve_context", "refine_context")

# Finish the workflow
customer_workflow.add_edge("fetch_loyalty_data", END)
customer_workflow.add_edge("refine_context", "respond")
customer_workflow.add_edge("respond", END)

# Compile the workflow
customer_graph = customer_workflow.compile()

def format_conversation_history(conversation_history, language):
    """Format the conversation history for the prompt."""
    # Only include the latest 6 messages to keep context window smaller
    formatted_history = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in conversation_history[-6:]
    ]) if conversation_history else "No prior conversation."
    
    return formatted_history