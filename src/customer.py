from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel as PydanticBaseModel, Field
from pydantic.json import pydantic_encoder
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
from src.utils import db_fetch_all_async, db_fetch_one_async, convert_to_json_safe, DateTimeEncoder, REDIS_CLIENT, logger
import networkx as nx
import spacy
from langchain_openai import ChatOpenAI


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

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

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

    # Mall Database Information
    {context}

    User Query: {query}
    
    Respond in {lang} in a friendly, conversational tone. Your response should be structured to work well for both text and voice:
    
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

# Add the new node for high-level query classification
async def classify_query_type(state: CustomerState) -> CustomerState:
    formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-6:]]) if state.conversation_history else "No prior conversation."
    query_type = await customer_query_classification_chain.ainvoke({"query": state.query, "conversation_history": formatted_history})
    
    # Strip any whitespace and ensure we have a valid query type
    query_type = query_type.strip()
    valid_query_types = ["product_info_query", "mall_info_query", "offer_or_event_info_query", "services_info_query", "family_planning_query", "visit_planning_query", "fallback_query"]
    
    if query_type not in valid_query_types:
        logger.warning(f"Unexpected query_type result: '{query_type}', defaulting to fallback_query")
        query_type = "fallback_query"
    
    state.query_type = query_type
    logger.info(f"Classified query type: {state.query_type}")
    return state

# Update existing classify_intent function to use the high-level query type for context
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
        state.initial_context = json.loads(cached_context)
        return state
    
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
        brand_name = state.context_data.get("resolved_entity").lower()
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
    
    REDIS_CLIENT.set(cache_key, json.dumps(state.initial_context, cls=DateTimeEncoder), ex=300)
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

    # Fetch all engagements (offers and events) with date filtering
    # to show only current and future engagements
    current_date = datetime.now().isoformat()
    engagements = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
           e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, e.unique_property_id,
           e.images_en, e.images_ar
           FROM engagements e 
           WHERE e.unique_property_id = $1 AND 
           (e.end_date >= $2 OR e.end_date IS NULL)""",
        (state.mall_id, current_date)
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
        
        # Extract image URLs
        image_url = None
        if state.language == "ar" and engagement.get("images_ar"):
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
                "is_exclusive": bool(engagement.get("is_exclusive", 0)),
                "image_url": image_url
            }
            context["offers"].append(offer_data)
        
        elif engagement_type == "events":
            event_data = {
                "id": engagement["engagement_id"],
                "name": engagement.get("title_en", ""),
                "description": engagement.get("description_en", ""),
                "brand_id": engagement.get("brand_id"),
                "store_name": brand["brand_name_en"] if brand else None,
                "start_date": engagement.get("start_date", ""),
                "end_date": engagement.get("end_date", ""),
                "terms": engagement.get("terms_conditions_en", ""),
                "image_url": image_url
            }
            context["events"].append(event_data)

    # For store queries, make sure to fetch ALL stores for a given mall
    store_name = state.context_data.get("resolved_entity", "").lower() if state.context_data else ""
    target_store_data = None
    
    # Always fetch all stores for complete data
    all_stores = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
           b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes,
           b.brand_logo, b.banner_en, b.banner_ar
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
           WHERE bma.unique_property_id = $1""", 
        (state.mall_id,)
    )
    
    food_related_categories = ['restaurant', 'cafe', 'food', 'dining', 'bakery', 'coffee']
    
    for store in all_stores:
        # Extract image URLs
        image_url = None
        # First try brand logo as it's usually the most relevant
        if store.get("brand_logo"):
            image_url = store["brand_logo"]
        # Then try banner based on language
        elif state.language == "ar" and store.get("banner_ar"):
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
        """SELECT s.id AS service_id, s.name, s.description, s.description_ar, s.location, s.is_available, s.icon_url
           FROM services s
           WHERE s.unique_property_id = $1""",
        (state.mall_id,)
    )
    
    for service in services:
        service_data = {
            "name": service.get("name", ""),
            "description": service.get("description", ""),
            "description_ar": service.get("description_ar", ""),
            "location": service.get("location", ""),
            "is_available": service.get("is_available", True),
            "image_url": service.get("icon_url")
        }
        context["services"].append(service_data)
    
    # Fetch product details including price
    if state.intent and state.intent.startswith("product_"):
        product_name = state.context_data.get("resolved_entity", "").lower() if state.context_data else ""
        
        # Search for products either by name or for a specific store
        if product_name:
            products = await db_fetch_all_async(
                """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
                   p.is_featured, p.in_stock, b.brand_name_en, b.brand_logo
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
                   p.is_featured, p.in_stock, b.brand_name_en, b.brand_logo
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
                "in_stock": product.get("in_stock", True),
                "image_url": product.get("brand_logo")  # Use brand logo for product image
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
                "image_url": metadata.get("brand_logo") or metadata.get("banner_en")
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

    # Fetch additional brand details for products
    if brand_ids:
        brands = await db_fetch_all_async(
            """SELECT brand_id, brand_name_en, category_name, description_en, 
               store_phone_number, store_email, store_website, pms_unit_codes,
               brand_logo, banner_en, banner_ar 
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
                if "image_url" not in item or not item["image_url"]:
                    item["image_url"] = brand["brand_logo"] or brand["banner_en"] or brand["banner_ar"]

    # If no stores found but resolved entity exists, try a direct DB lookup
    if not context["stores"] and state.context_data and state.context_data.get("resolved_entity"):
        store_name = state.context_data["resolved_entity"].lower()
        stores = await db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
               b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes,
               b.brand_logo, b.banner_en, b.banner_ar
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
               WHERE bma.unique_property_id = $1 AND LOWER(b.brand_name_en) LIKE $2""", 
            (state.mall_id, f"%{store_name}%")
        )
        
        for store in stores:
            store_category = store.get("category_name", "").lower()
            if store_category:
                context["category_types"]["store"].add(store_category)
                
            # Extract image URL (prioritize brand logo, then banners)
            image_url = None
            if store.get("brand_logo"):
                image_url = store["brand_logo"]
            elif state.language == "ar" and store.get("banner_ar"):
                image_url = store["banner_ar"]
            elif store.get("banner_en"):
                image_url = store["banner_en"]
                
            context["stores"].append({
                "name": store["brand_name_en"],
                "category": store_category,
                "store_id": store["brand_id"],
                "description": store.get("description_en", ""),
                "phone": store.get("store_phone_number", ""),
                "email": store.get("store_email", ""),
                "website": store.get("store_website", ""),
                "location": store.get("pms_unit_codes", {}),
                "image_url": image_url
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

    # Check for any "address" requests and map them to pms_unit_codes
    if state.query.lower().find("address") > -1 or state.query.lower().find("location") > -1 or state.query.lower().find("where") > -1:
        for store in context["stores"]:
            # Make sure we use pms_unit_codes for location information
            if "pms_unit_codes" in store and not "location" in store:
                store["location"] = store["pms_unit_codes"]
    
    # At the end, preprocess all location codes to readable format
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

    state.context_data = context
    state.response = json.dumps(convert_to_json_safe(context))
    return state

# Add back the generate_response function with conversation tracking
async def generate_response(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        if state.language == "ar":
            state.response = "عذراً! أحتاج إلى معرفة المركز التجاري الذي تسأل عنه. يرجى اختيار مركز تجاري أولاً! 😊"
        else:
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
            
            # Add Arabic common types
            arabic_common_types = ["إيطالي", "هندي", "صيني", "وجبات سريعة", "رياضة", "غير رسمي", 
                                  "رسمي", "أطفال", "نساء", "رجال", "فاخر", "اقتصادي", "إلكترونيات"]
            
            all_types = common_types + arabic_common_types
            
            for type_name in all_types:
                if type_name.lower() in last_message["content"].lower():
                    state.type_preference = type_name
                    state.needs_type_follow_up = False
                    break
    
    # Track conversation topic for multi-turn handling
    current_topic = state.query_type or ""
    if state.intent:
        current_topic += "_" + state.intent
    
    # If we have a resolved entity, add it to the topic for better tracking
    if resolved_entity:
        current_topic += "_" + resolved_entity.lower().replace(" ", "_")
    
    # Check if this is continuing the same conversation topic
    if state.conversation_topic and current_topic and state.conversation_topic in current_topic:
        # Still on the same general topic
        state.topic_turn_count += 1
    else:
        # New topic
        state.conversation_topic = current_topic
        state.topic_turn_count = 1
    
    # Log conversation state for debugging
    logger.info(f"CONVERSATION STATE: Topic: {state.conversation_topic}, Turn count: {state.topic_turn_count}, Intent: {state.intent}, Query type: {state.query_type}")
    
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
                "topic_turn_count": state.topic_turn_count,
                "conversation_topic": state.conversation_topic
            }
        )
        
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
        if state.intent and state.intent in ["product_list", "store_list", "offer_list", "product_recommend", "store_recommend", "offer_recommend"]:
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
            if state.topic_turn_count >= 2 and follow_up_question:
                logger.info(f"REMOVING FOLLOW-UP QUESTION (turn {state.topic_turn_count}): {follow_up_question}")
                follow_up_question = None
        
        # Create formatted response
        response_format = ResponseFormat(
            response=clean_response,
            recommendations=recommendations if needs_recommendation_format else None,
            is_recommendation_format=needs_recommendation_format,
            follow_up_question=follow_up_question
        )
        
        # Store the response format as a serializable dictionary
        state.response_format = response_format.dict()
        state.response = response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        if state.language == "ar":
            state.response = "أواجه مشكلة في معالجة طلبك حاليًا. يرجى المحاولة مرة أخرى."
        else:
            state.response = "I'm having trouble processing your request right now. Please try again."
    
    return state

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