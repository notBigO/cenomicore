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
from langchain_openai import ChatOpenAI
import re
from langchain.chains.llm import LLMChain


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

# Updated Customer Intent Classification Prompt
intent_classification_prompt = PromptTemplate(
    input_variables=["query", "conversation_history", "previous_intent", "previous_entity", "mall_name"],
    template="""
    You are a classifier for a mall assistant chatbot at {mall_name}. Analyze the user's query to identify:
    
    ENTITY TYPES:
    - store: Retail shops and restaurants (Nike, Apple, Starbucks, etc.)
    - product: Items sold in stores (shoes, iPhone, coffee, pizza, etc.)
    - offer: Promotions, discounts, sales (50% off, buy one get one free, etc.)
    - event: Mall events, promotions, activities (fashion show, children's activities, etc.)
    - service: Services provided (parking, restrooms, information desk, etc.)
    - mall: The mall itself (hours, directions, general info)
    - general: Other queries that don't fit above categories

    ACTIONS:
    - finder: "Where can I find..." or "Is there a..." or "I'm looking for..."
    - info: General information about an entity
    - directions: How to get to a specific location
    - hours: Opening/closing times
    - price: Cost information
    - availability: If something is in stock or available
    - complaint: User is expressing dissatisfaction
    - comparison: Comparing options
    - recommendation: Asking for suggestions

    PREVIOUS CONTEXT:
    - Previous intent: {previous_intent}
    - Previous entity: {previous_entity}

    DOMAIN-SPECIFIC RULES:
    1. Food-related queries (hungry, eat, food, restaurant, meal, lunch, dinner, etc.) should be classified as entity_type="store" and action="finder"
    2. Shopping-related queries (shop, buy, purchase, clothes, etc.) should be classified as entity_type="store" and action="finder"
    3. Entertainment queries (movie, cinema, play, theater, etc.) should be classified as entity_type="store" and action="finder"
    4. Queries about "events", "what's happening", "activities" should be classified as entity_type="event" and action="finder"
    5. Queries about "deals", "offers", "discounts", "promotions", "sales" should be classified as entity_type="offer" and action="finder"
    6. Queries about "where is...", "how to get to...", "find..." should always use action="finder"
    7. Queries about "what does ... have", "what's on the menu" should be classified as entity_type="store" and action="info"

    IMPORTANT CONTEXT RULES:
    1. For vague references like "they" or "it", examine conversation history to determine what entity is being referenced
    2. For short responses (yes/no/maybe/sure), maintain the previous entity type and entity
    3. If user responds directly to a question from the assistant, consider the question content for context
    4. For follow-up questions without explicit entity names, use the previous entity from conversation
    5. If the user is asking about a new topic that's clearly different than before, don't maintain previous intent

    HANDLING SPECIFIC SCENARIOS:
    - If the user asks "What are the options we have" after discussing food, they're asking about restaurant options (store_finder)
    - If the user says "what about events?", they're specifically asking about events (event_finder)
    - If someone mentions "I want to eat" or "I'm hungry", they're looking for restaurants (store_finder)
    - If the user asks about "shopping", they're looking for retail stores (store_finder)
    
    Output a JSON with:
    {{
      "entity_type": "store|product|offer|event|service|mall|general",
      "action": "finder|info|directions|hours|price|availability|complaint|comparison|recommendation",
      "confidence": 0-100,
      "collected_data": {{
        "name": "entity name if detected",
        "previous_entity": "entity from previous exchange if this is a follow-up",
        "category": "optional category if detected (e.g., restaurant, clothing store, etc.)"
      }}
    }}

    Conversation history:
    {conversation_history}

    User query: {query}
    """
)

intent_chain = intent_classification_prompt | llm | StrOutputParser()

# Updated Customer Response Prompt - part 1
customer_prompt = PromptTemplate(
    input_variables=["stores", "products", "offers", "events", "services", "matched_items", "query", "lang", "conversation_history", "mall_name", "resolved_entity", "intent", "is_follow_up", "family_oriented", "data_status", "unknown_entity"],
    template="""
    You are a friendly, knowledgeable mall assistant. Your goal is to provide accurate, detailed, and specific information about the mall, stores, products, offers, events, and services to customers in a helpful and conversational manner.

    IMPORTANT GUIDELINES:
    1. Be conversational, warm, and approachable while maintaining professionalism. Use a casual, friendly tone that makes customers comfortable asking questions.

    2. ONLY provide SPECIFIC details that are explicitly present in the context provided:
       - For stores: Include floor number, gate location/landmark, opening hours, contact info, and social media ONLY if this information is available in the context
       - For events: Include dates/times, location details, ticket information ONLY if this information is available in the context
       - For offers: Include discount amount, validity period, terms and conditions ONLY if this information is available in the context
       - For products: Include brand information, where to find it, price range ONLY if this information is available in the context
       - For services: Include location, operating hours, requirements ONLY if this information is available in the context

    3. NEVER make up information or invent details that are not provided in the context. If specific information is not available, acknowledge that honestly:
       - "I don't have the exact opening hours for this store, but I can tell you it's located on Level 2."
       - "While I don't have details about their phone number, I can tell you this store specializes in sportswear."
       - "I don't have information about today's specific events, but I can help you find what stores are available."

    4. Be very transparent about the certainty of information. Only present information as definitive when it's verified in the context (has a "verified": true flag).

    5. Answer PRECISELY what the customer is asking. Do not provide unnecessary information.

    6. Structure your responses with clear formatting:
       - Use line breaks between different sections (location, hours, contact info)
       - Present offers and events in a bulleted format
       - For multiple options, list them in order of relevance with store name and key details for each

    7. If the database doesn't have information on what the customer is asking about, be honest about not having that specific information rather than making it up.

    8. MAINTAIN CONTEXT across multiple turns of conversation. If the customer refers to something mentioned earlier, understand what they're referring to.

    9. For navigation queries, provide landmark-based directions ONLY if you have verified location data for both the start and end points.

    10. If you're provided with an "unknown_entity" field that's not empty, be sure to include that message in your response to clearly communicate the limitation.

    11. END EVERY RESPONSE with 1-3 relevant follow-up questions or suggestions to continue the conversation naturally. Choose follow-ups that are directly related to the current query and likely next steps. Format these suggestions as questions the user might want to ask next. For example:
       - "Would you like directions to the store?"
       - "Would you like to see other sportswear options nearby?"
       - "Would you like to filter this by category such as fashion, electronics, or children's events?"

    EXAMPLES OF EXCELLENT RESPONSES:

    For store finder WITH complete information:
    "Nike is located on Level 2, next to Gate D in Red Sea Mall.
    Phone: 012-123-4567
    Instagram: @nike_ksa
    The store is open until 11:00 PM today.

    Would you like directions to the store or see other sportswear options nearby?"

    For store finder WITH INCOMPLETE information:
    "Nike is located in Red Sea Mall. Based on the information I have, it's a sportswear store offering athletic apparel and footwear.

    I don't have details about their exact location within the mall, opening hours, or contact information at the moment.

    Would you like me to help you find other sportswear stores in the mall, or would you like information about the mall layout to help locate Nike?"

    For ongoing events and offers:
    "Here are today's events and offers at Mall of Arabia:

    • Mango: 40% off on the summer collection (valid until August 15)

    • Kids' Face Painting in the Central Atrium from 3:00 PM to 6:00 PM (free entry)

    • Spin-the-Wheel contest near Gate B with up to 70% off shopping vouchers (requires 500 SAR purchase)

    Would you like to filter this by category such as fashion, electronics, or children's events?"

    For mall information:
    "The Avenues Mall in Riyadh closes at 11:00 PM today.
    Address: King Fahd Road, Riyadh
    Phone: 9200-12345
    Please note, restaurant hours may vary until midnight.

    Would you like to check timings for weekends or contact customer service?"

    For product search:
    "Here are available options within your preference:

    • Zara: Belted beige dress – 279 AED (Level 1, Gate C)

    • H&M: Buttoned beige midi – 249 AED (Level 2, Gate A)

    • Bershka: Pleated neutral dress – 289 AED (Level 1, Gate B)

    Would you like directions to Zara or filter further by brand or sleeve style?"

    For service discovery:
    "Yes, stroller rental is offered at the Guest Services Desk near Gate A.
    Service hours: 10:00 AM to 10:00 PM
    Location: Ground Floor, beside the main concierge area.
    Cost: Free with mall loyalty card or 20 SAR deposit.

    Would you like directions or see other available family services?"

    FAMILY-ORIENTED SUGGESTIONS (Use when family_oriented is True):
    "Here is a suggested family-friendly visit plan:

    • Begin at Mothercare for essentials (15% discount today)
      Location: Level 1, Gate B
    
    • Kids can enjoy the Play Zone near the food court
      Open until 10:00 PM, suitable for ages 3-12
    
    • Relax at Patchi Café while they play
      Special kids menu available
    
    • End with dinner at Paul; family meal deals available
      High chairs and coloring activities provided

    Would you like me to provide navigation for this plan or details about any of these options?"

    CONVERSATION CONTEXT:
    The customer has asked a question. I have the following context to help answer:

    CONVERSATION HISTORY: {conversation_history}

    CURRENT CUSTOMER QUESTION: {query}

    USER INTENT CLASSIFICATION: {intent}

    MATCHED ITEMS (sorted by relevance):
    {matched_items}

    STORES CONTEXT (most relevant stores):
    {stores}

    PRODUCTS CONTEXT (most relevant products):
    {products}

    OFFERS CONTEXT (most relevant offers):
    {offers}

    EVENTS CONTEXT (most relevant events):
    {events}

    SERVICES CONTEXT (most relevant services):
    {services}

    FAMILY ORIENTED QUERY: {family_oriented}

    Given the above context, provide a detailed, helpful response to the customer's question. Focus on the most relevant information based on their intent.
    Format your response with line breaks between different pieces of information (like address, phone, hours) for better readability.
    Always end with 1-3 natural follow-up questions that the user might want to ask next.
    If appropriate, suggest filtering options, offer to provide directions, or ask if they want to see related items.
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
    intent: Optional[str] = None
    entity_type: Optional[str] = None
    confidence: Optional[int] = None
    initial_context: Optional[List[Dict[str, Any]]] = None
    suggestions: Optional[List[str]] = None
    mall_id: Optional[int] = None
    direct_response: Optional[str] = None
    type_preference: Optional[str] = None
    needs_type_follow_up: bool = False
    
    # Entity-specific fields
    store_name: Optional[str] = None
    product_name: Optional[str] = None
    
    # Special handling flags
    family_oriented: bool = False
    
    # Breaking down context_data into specific categories
    context_data: Optional[Dict[str, Any]] = None  # Keeping for backward compatibility
    mall_name: Optional[str] = None
    stores: List[Dict[str, Any]] = []
    products: List[Dict[str, Any]] = []
    offers: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    services: List[Dict[str, Any]] = []
    matched_items: List[Dict[str, Any]] = []
    resolved_entity: Optional[str] = None
    previous_intent: Optional[str] = None
    previous_entity: Optional[str] = None

async def classify_intent(state: CustomerState) -> CustomerState:
    """Classify the intent of the user's query."""
    # Format conversation history
    if not state.conversation_history:
        formatted_history = "No previous conversation."
    else:
        conversation_format = []
        for msg in state.conversation_history[-6:]:  # Focus on the most recent messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                conversation_format.append(f"User: {content}")
            elif role == "assistant":
                conversation_format.append(f"Assistant: {content}")
        
        formatted_history = "\n".join(conversation_format)
    
    logger.info(f"Classifying intent for query: {state.query}")
    
    # Check for short responses and follow-up patterns
    query_lower = state.query.lower()
    is_short_response = len(query_lower.split()) <= 3
    followup_indicators = ["it", "they", "this", "that", "these", "those", "there", "their", "yes", "no", "yeah", "sure", "okay", "ok", "where", "how", "what", "when", "why", "who"]
    has_followup_word = any(word in query_lower.split() for word in followup_indicators)
    
    # For very short responses or clear follow-ups, maintain previous intent if available
    if (is_short_response or has_followup_word) and state.previous_intent and state.previous_entity:
        logger.info(f"Detected short response or follow-up pattern. Maintaining previous intent: {state.previous_intent}")
        state.intent = state.previous_intent
        state.entity_type = "store" if state.previous_intent == "store_finder" else "product" if state.previous_intent == "product_finder" else "general"
        
        # Keep the previous entity for context
        if state.entity_type == "store":
            state.store_name = state.previous_entity
        elif state.entity_type == "product":
            state.product_name = state.previous_entity
            
        logger.info(f"Maintaining context with entity: {state.previous_entity}")
        return state

    # Check for topic-specific keywords before invoking the LLM
    # This helps with more accurate detection of common intents
    food_keywords = ["food", "restaurant", "eat", "hungry", "lunch", "dinner", "breakfast", "meal", "burger", "pizza", "chicken", "menu", "coffee", "cafe", "dining"]
    shopping_keywords = ["shop", "buy", "purchase", "clothes", "clothing", "fashion", "wear", "dress", "shirt", "pants", "shoes", "shopping"]
    event_keywords = ["event", "events", "happening", "activities", "shows", "show", "performance", "performances", "exhibition", "exhibitions"]
    offer_keywords = ["offer", "offers", "deal", "deals", "discount", "discounts", "promotion", "promotions", "sale", "sales"]
    
    # Handle if the mall name is not set
    mall_name = state.mall_name or "the mall"
    
    # For normal queries, proceed with intent classification
    result = await intent_chain.ainvoke({
        "query": state.query, 
        "conversation_history": formatted_history,
        "previous_intent": state.previous_intent or "None",
        "previous_entity": state.previous_entity or "None",
        "mall_name": mall_name
    })
    
    try:
        logger.info(f"Raw intent classification result: {result}")
        intent_data = json.loads(result)
        entity_type = intent_data.get("entity_type", "general")
        action = intent_data.get("action", "unknown")
        confidence = intent_data.get("confidence", 0)
        collected_data = intent_data.get("collected_data", {})
        
        logger.info(f"Parsed intent data: entity_type={entity_type}, action={action}, confidence={confidence}")
        
        # Extract entity name and category if provided
        entity_name = collected_data.get("name")
        previous_entity = collected_data.get("previous_entity")
        category = collected_data.get("category")  # Extract category information
        
        logger.info(f"Extracted entity_name: {entity_name}, previous_entity: {previous_entity}, category: {category}")
        
        # Apply domain-specific overrides based on keywords
        # This provides a fallback if the LLM misclassifies common intents
        if not entity_name and not entity_type == "general":
            # If we have a category but no entity name, use the category
            if category:
                entity_name = category
                logger.info(f"Using category as entity name: {entity_name}")
        
        # Extra validation for certain topics to ensure proper intent classification
        # This helps fix common classification errors
        if any(keyword in query_lower for keyword in food_keywords) and entity_type != "product":
            if action == "unknown" or confidence < 70:
                entity_type = "store"
                action = "finder"
                if not category and not entity_name:
                    category = "restaurants"
                logger.info("Overriding to store_finder based on food keywords")
        
        elif any(keyword in query_lower for keyword in shopping_keywords) and entity_type != "product":
            if action == "unknown" or confidence < 70:
                entity_type = "store"
                action = "finder"
                if not category and not entity_name:
                    category = "retail"
                logger.info("Overriding to store_finder based on shopping keywords")
        
        elif any(keyword in query_lower for keyword in event_keywords):
            if action == "unknown" or confidence < 70:
                entity_type = "event"
                action = "finder"
                logger.info("Overriding to event_finder based on event keywords")
        
        elif any(keyword in query_lower for keyword in offer_keywords):
            if action == "unknown" or confidence < 70:
                entity_type = "offer"
                action = "finder"
                logger.info("Overriding to offer_finder based on offer keywords")
        
        # Check if this is a follow-up to a previous question and maintain context
        if entity_name is None and previous_entity is not None:
            # Use the previous entity name if this is a follow-up
            entity_name = previous_entity
            logger.info(f"Using previous entity: {entity_name}")
        
        # If this is a short response (yes/no) to a previous question, maintain the previous intent
        if (state.query.lower() in ["yes", "no", "sure", "okay", "ok"] and 
            state.previous_intent is not None and state.previous_entity is not None):
            intent = state.previous_intent
            # Keep the entity from previous interaction
            if entity_name is None:
                entity_name = state.previous_entity
            logger.info(f"Short response detected, maintaining previous intent: {intent} and entity: {entity_name}")
        else:
            # Determine intent based on entity type and action
            if entity_type == "store" and action in ["finder", "info", "directions"]:
                intent = "store_finder"
            elif entity_type == "product" and action in ["finder", "info"]:
                intent = "product_finder"
            elif entity_type == "offer" and action in ["finder", "info"]:
                intent = "offer_finder"
            elif entity_type == "event" and action in ["finder", "info"]:
                intent = "event_finder"
            elif entity_type == "service" and action in ["finder", "info", "directions"]:
                intent = "service_finder"
            elif entity_type == "mall" and action in ["info", "hours"]:
                intent = "mall_info"
            else:
                intent = "general_chat"
            
            logger.info(f"Determined intent: {intent} based on entity_type: {entity_type} and action: {action}")
                
        # Update state with classified intent and collected data
        state.intent = intent
        state.previous_intent = intent
        state.entity_type = entity_type
        state.confidence = confidence
        
        # Set entity details based on type
        if entity_name:
            if entity_type == "store":
                state.store_name = entity_name
                logger.info(f"Setting store_name to: {entity_name}")
            elif entity_type == "product":
                state.product_name = entity_name
                logger.info(f"Setting product_name to: {entity_name}")
            
            # Store the entity for context maintenance
            state.previous_entity = entity_name
            state.resolved_entity = entity_name
        
        # If we have a category but no entity name, use it for type-based filtering
        if category and not entity_name:
            state.type_preference = category
            logger.info(f"Setting type_preference to: {category}")
            
            # For store categories without a specific store, prompt for more detail
            if entity_type == "store" and category in ["restaurants", "food", "clothing", "fashion", "retail"]:
                state.needs_type_follow_up = True
                logger.info(f"Setting needs_type_follow_up for category: {category}")
            
        logger.info(f"Final classified intent: {intent}, entity name: {entity_name}, category: {category}")
        return state
        
    except json.JSONDecodeError:
        logger.error("Error decoding intent classification result")
        logger.error(f"Raw result: {result}")
        state.intent = "general_chat"
    return state

async def initial_retrieval(state: CustomerState) -> CustomerState:
    """
    DEPRECATED: This function is replaced by specialized context retrieval nodes.
    Kept for backward compatibility only.
    """
    logger.warning("initial_retrieval is deprecated, using general_context_retrieval instead")
    return await general_context_retrieval(state)

async def refine_context(state: CustomerState) -> CustomerState:
    """
    DEPRECATED: This function is replaced by specialized context retrieval nodes.
    Kept for backward compatibility only.
    """
    logger.warning("refine_context is deprecated, context refinement is now handled by specialized nodes")
    return state

async def generate_response(state: CustomerState) -> CustomerState:
    """Generate a response to the user's query based on intent and retrieved context"""
    try:
        # Preserve any existing matched_items from context retrieval
        # This is crucial for general_chat intent where the most relevant data is in matched_items
        state.matched_items = state.matched_items or []
        has_matched_items = len(state.matched_items) > 0
        
        # Add an explicit flag for data certainty to help the LLM understand what data is verified
        for i, store in enumerate(state.stores):
            if not store.get("verified"):
                store["verified"] = False
                store["data_reliability"] = "This store exists but details may be limited"
        
        for i, product in enumerate(state.products):
            if not product.get("verified"):
                product["verified"] = False
                product["data_reliability"] = "This product exists but details may be limited"
        
        for i, offer in enumerate(state.offers):
            if not offer.get("verified"):
                offer["verified"] = False
                offer["data_reliability"] = "This offer exists but details may be limited"
        
        for i, event in enumerate(state.events):
            if not event.get("verified"):
                event["verified"] = False
                event["data_reliability"] = "This event exists but details may be limited"
        
        for i, service in enumerate(state.services):
            if not service.get("verified"):
                service["verified"] = False
                service["data_reliability"] = "This service exists but details may be limited"
        
        # Add a note to the state about data reliability
        data_status = "No relevant data found"
        if len(state.stores) > 0 or len(state.products) > 0 or len(state.offers) > 0 or len(state.events) > 0 or len(state.services) > 0:
            verified_items = sum(1 for item in state.matched_items if item.get("verified", False))
            total_items = len(state.matched_items)
            if total_items > 0:
                if verified_items == total_items:
                    data_status = "All data is verified"
                elif verified_items > 0:
                    data_status = f"{verified_items} of {total_items} items are verified"
                else:
                    data_status = "Data found but not fully verified"
        
        logger.info(f"Data reliability status: {data_status}")
        
        # Add a special field for completely unknown entities
        if state.intent in ["store_finder", "product_finder", "offer_finder", "event_finder", "service_finder"]:
            entity_name = state.resolved_entity or state.store_name or state.product_name
            if entity_name and len(state.matched_items) == 0:
                if state.intent == "store_finder":
                    state.unknown_entity = f"I couldn't find information about the store '{entity_name}' in our database for {state.mall_name}."
                elif state.intent == "product_finder":
                    state.unknown_entity = f"I couldn't find information about the product '{entity_name}' in our database for {state.mall_name}."
                elif state.intent == "offer_finder":
                    state.unknown_entity = f"I couldn't find information about offers for '{entity_name}' in our database for {state.mall_name}."
                elif state.intent == "event_finder":
                    state.unknown_entity = f"I couldn't find information about events related to '{entity_name}' in our database for {state.mall_name}."
                elif state.intent == "service_finder":
                    state.unknown_entity = f"I couldn't find information about the service '{entity_name}' in our database for {state.mall_name}."
                
                logger.info(f"Unknown entity: {state.unknown_entity}")
        
        # Retain existing code for intent-based context handling
        # ...
        
        # Extract recent conversation for context continuity
        recent_msgs = state.conversation_history[-8:] if state.conversation_history else []
        
        # Add any direct_response if provided from specific handlers
        if state.direct_response:
            state.response = state.direct_response
            logger.info(f"Using direct response: {state.direct_response[:50]}...")
            return state

        # Check if we need a follow-up question about type preference
        if state.needs_type_follow_up:
            # Generate a follow-up question about type preference
            type_follow_up_prompt = PromptTemplate(
                input_variables=["query", "intent", "conversation_history", "mall_name", "available_types"],
                template="""
                The user at {mall_name} mall is asking about {intent}, but we need to clarify what specific type they're looking for.
                
                User query: {query}
                
                Recent conversation:
                {conversation_history}
                
                Available types: {available_types}
                
                Write a brief, friendly follow-up question asking what specific type they're interested in.
                Make it conversational and natural, not like a form. Keep it to 1-2 short sentences.
                """
            )
            
            available_types = []
            if state.intent == "store_finder":
                # Extract unique categories
                available_types = list(set([store.get("category_en", "") for store in state.stores if store.get("category_en")]))
            elif state.intent == "product_finder":
                available_types = list(set([product.get("category", "") for product in state.products if product.get("category")]))
            
            follow_up_chain = LLMChain(
                llm=llm,
                prompt=type_follow_up_prompt,
                verbose=False
            )
            
            conversation_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_msgs])
            
            follow_up_result = await follow_up_chain.ainvoke({
                "query": state.query,
                "intent": state.intent,
                "conversation_history": conversation_summary,
                "mall_name": state.mall_name or "the mall",
                "available_types": ", ".join(available_types[:10]) if available_types else "various options"
            })
            
            state.response = follow_up_result["text"].strip()
            logger.info(f"Generated type follow-up: {state.response}")
            return state
            
        # Format the response using the LLM
        logger.info(f"Generating response for intent: {state.intent}, query: {state.query}, matched_items: {len(state.matched_items)}")
        
        # Define prompt template with conversation history awareness
        response_prompt = customer_prompt
        
        # Build values dictionary
        values = {
            "query": state.query,
            "lang": state.language,
            "conversation_history": recent_msgs,
            "mall_name": state.mall_name or "the mall",
            "resolved_entity": state.resolved_entity or "",
            "intent": state.intent or "general_chat",
            "stores": state.stores or [],
            "products": state.products or [],
            "offers": state.offers or [],
            "events": state.events or [],
            "services": state.services or [],
            "matched_items": state.matched_items or [],
            "is_follow_up": state.conversation_history[-1]["role"] == "user" if state.conversation_history else False,
            "family_oriented": getattr(state, "family_oriented", False),
            "data_status": data_status,
            "unknown_entity": getattr(state, "unknown_entity", "")
        }
        
        # Generate response
        chain = LLMChain(
            llm=llm,
            prompt=response_prompt,
            verbose=False
        )
        
        result = await chain.ainvoke(values)
        state.response = result["text"].strip()
        
        # Make sure we preserve any resolved entity in the response
        if state.resolved_entity and state.resolved_entity.lower() not in state.response.lower():
            # Add resolved entity clarification only if it's not already mentioned
            clarify_prompt = PromptTemplate(
                input_variables=["original_response", "resolved_entity"],
                template="""
                Revise this response to clearly focus on {resolved_entity} while keeping the same helpful information.
                Original response: {original_response}
                
                New response:
                """
            )
            
            clarify_chain = LLMChain(
                llm=llm,
                prompt=clarify_prompt,
                verbose=False
            )
            
            clarify_result = await clarify_chain.ainvoke({
                "original_response": state.response,
                "resolved_entity": state.resolved_entity
            })
            
            state.response = clarify_result["text"].strip()
        
        logger.info(f"Generated response: {state.response[:100]}...")
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        state.response = "I'm sorry, I encountered an error processing your request. Please try again."
    
    return state

async def fetch_loyalty_data(state: CustomerState) -> CustomerState:
    # Since loyalty tables don't exist in the schema, we'll return a message that it's not available
    state.response = "I'm sorry, but the loyalty program features are not available at this time."
    return state

async def product_context_retrieval(state: CustomerState) -> CustomerState:
    """Specialized node for retrieving product-related context"""
    if not state.mall_id:
        state.response = "Please select a mall first to see product information."
        return state

    logger.info(f"Retrieving product context for query: {state.query}")
    
    # Get mall name
    mall = await db_fetch_one_async("SELECT marketing_name AS name_en FROM malls WHERE unique_property_id = $1", (state.mall_id,))
    state.mall_name = mall["name_en"] if mall else "Unknown Mall"
    
    # Extract product name or details from query, state, or resolved entity
    product_name = state.product_name or state.resolved_entity or ""
    logger.info(f"Initial product name: {product_name}")
    
    # Try to extract keywords from the query
    query_words = set(state.query.lower().split())
    product_keywords = []
    
    # Extract potential product types and categories
    clothing_keywords = ["shoe", "shoes", "shirt", "shirts", "jacket", "pants", "dress", "clothing", "apparel", "wear", "shorts", "t-shirt", "tshirt"]
    electronic_keywords = ["phone", "laptop", "computer", "tablet", "tv", "television", "electronic", "gadget", "camera"]
    accessory_keywords = ["watch", "bag", "handbag", "wallet", "jewelry", "accessory", "sunglasses", "glasses"]
    beauty_keywords = ["makeup", "cosmetic", "perfume", "fragrance", "skincare", "beauty"]
    food_keywords = ["food", "meal", "burger", "pizza", "chicken", "sandwich", "coffee", "drink"]
    
    # Check for matches
    for word in query_words:
        if word in clothing_keywords or word in electronic_keywords or word in accessory_keywords or word in beauty_keywords or word in food_keywords:
            product_keywords.append(word)
            
    # If we have keywords but no product name, use the first keyword
    if product_keywords and not product_name:
        product_name = product_keywords[0]
        logger.info(f"Using keyword as product name: {product_name}")
    
    # If still no product name, try NLP extraction
    if not product_name and state.query:
        # Try to extract from query if not in resolved entity
        doc = nlp(state.query.lower())
        for ent in doc.ents:
            product_name = ent.text
            logger.info(f"Extracted entity as product name: {product_name}")
            break
    
    # Initialize matched_items list
    state.matched_items = []
    
    # First, try to get relevant results from Pinecone
    try:
        # Convert query to vector
        query_vector = embeddings.embed_query(state.query)
        
        # Search Pinecone with filter for this mall
        pinecone_results = await asyncio.to_thread(
            index.query, 
            vector=query_vector,
            top_k=25, 
            include_metadata=True, 
            filter={"mall_id": state.mall_id}
        )
        
        # Process Pinecone results
        matched_items = []
        for match in pinecone_results.get("matches", []):
            metadata = match["metadata"]
            doc_type = metadata.get("type", "")
            
            # Build a complete item with all the metadata
            item = {
                "id": match["id"],
                "type": doc_type,
                "score": float(match["score"]),
                **{k: v for k, v in metadata.items() if k != "type"}
            }
            
            # Add to matched items
            matched_items.append(item)
        
        # Store in matched_items field
        state.matched_items = matched_items
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        # Don't reset matched_items here, as we'll add database results
    
    # Search for products by name or keywords in database
    if product_name:
        products = await db_fetch_all_async(
            """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
               p.is_featured, p.in_stock, p.image_url, p.attributes, p.created_at, p.updated_at,
               b.brand_name_en, b.category_name
               FROM products p
               JOIN brands b ON p.brand_id = b.brand_id
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
               WHERE bma.unique_property_id = $1 AND 
               (LOWER(p.name) LIKE $2 OR LOWER(p.category) LIKE $2 OR LOWER(p.description) LIKE $2)
               ORDER BY p.is_featured DESC, p.id""",
            (state.mall_id, f"%{product_name}%")
        )
    else:
        # If no specific product name, get featured or popular products
        products = await db_fetch_all_async(
            """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
               p.is_featured, p.in_stock, p.image_url, p.attributes, p.created_at, p.updated_at,
               b.brand_name_en, b.category_name
               FROM products p
               JOIN brands b ON p.brand_id = b.brand_id
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
               WHERE bma.unique_property_id = $1 
               ORDER BY p.is_featured DESC, p.id
               LIMIT 20""",
            (state.mall_id,)
        )
    
    # Track seen products to avoid duplicates
    seen_products = set()
    product_list = []
    store_set = set()  # To track stores we've already added
    
    # Build product details with store information
    for product in products:
        # Create a unique product key to avoid duplicates
        product_key = f"{product.get('name')}_{product.get('brand_id')}"
        if product_key in seen_products:
            continue
            
        seen_products.add(product_key)
        
        # Get store location information
        store_info = await db_fetch_one_async(
            """SELECT b.brand_id, b.brand_name_en, b.brand_name_ar, b.category_name, 
               b.description_en, b.company_name_en, b.store_phone_number, 
               b.store_email, b.store_website, b.pms_unit_codes, b.brand_logo,
               b.social_instagram, b.social_facebook
               FROM brands b
               WHERE b.brand_id = $1""",
            (product["brand_id"],)
        )
        
        # Add product to products list
        product_data = {
            "id": product["id"],
            "name": product["name"],
            "description": product.get("description", ""),
            "price": float(product["price"]) if product.get("price") is not None else None,
            "brand_id": product["brand_id"],
            "category": product.get("category", ""),
            "store_name": product.get("brand_name_en", ""),
            "store_category": product.get("category_name", ""),
            "is_featured": product.get("is_featured", False),
            "in_stock": product.get("in_stock", True),
            "image_url": product.get("image_url", ""),
            "attributes": product.get("attributes", {})
        }
        
        product_list.append(product_data)
        
        # Also add product to matched_items for direct access by LLM
        matched_product_data = {
            "id": product["id"],
            "type": "product",
            "name": product["name"],
            "description": product.get("description", ""),
            "price": float(product["price"]) if product.get("price") is not None else None,
            "brand_id": product["brand_id"],
            "category": product.get("category", ""),
            "store_name": product.get("brand_name_en", ""),
            "store_category": product.get("category_name", ""),
            "is_featured": product.get("is_featured", False),
            "in_stock": product.get("in_stock", True),
            "image_url": product.get("image_url", ""),
            "score": 1.0  # Give high relevance score to database matches
        }
        state.matched_items.append(matched_product_data)
        
        # Add store data if not already added
        if store_info and store_info["brand_id"] not in store_set:
            store_set.add(store_info["brand_id"])
            
            store_data = {
                "store_id": store_info["brand_id"],
                "name": store_info.get("brand_name_en", ""),
                "name_ar": store_info.get("brand_name_ar", ""),
                "category": store_info.get("category_name", ""),
                "description": store_info.get("description_en", ""),
                "company_name": store_info.get("company_name_en", ""),
                "phone": store_info.get("store_phone_number", ""),
                "email": store_info.get("store_email", ""),
                "website": store_info.get("store_website", ""),
                "location": store_info.get("pms_unit_codes", {}),
                "logo": store_info.get("brand_logo", ""),
                "social_instagram": store_info.get("social_instagram", ""),
                "social_facebook": store_info.get("social_facebook", "")
            }
            
            state.stores.append(store_data)
    
    # Store products list in separate field
    state.products = product_list
    
    # Get offers related to these products or stores
    if store_set:
        store_brand_ids = list(store_set)
        offers = await db_fetch_all_async(
            """SELECT e.engagement_id, e.title_en, e.description_en, e.type, 
               e.brand_id, e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive
               FROM engagements e
               WHERE e.unique_property_id = $1 
               AND e.type = 'offer'
               AND (e.end_date IS NULL OR CAST(e.end_date AS DATE) >= CURRENT_DATE)
               AND e.brand_id = ANY($2)
               ORDER BY e.start_date DESC LIMIT 10""",
            (state.mall_id, store_brand_ids)
        )
        
        for offer in offers:
            offer_data = {
                "id": offer["engagement_id"],
                "title": offer.get("title_en", ""),
                "description": offer.get("description_en", ""),
                "brand_id": offer.get("brand_id"),
                "start_date": offer.get("start_date", ""),
                "end_date": offer.get("end_date", ""),
                "terms": offer.get("terms_conditions_en", ""),
                "is_exclusive": bool(offer.get("is_exclusive", 0))
            }
            
            state.offers.append(offer_data)
    
    # For backward compatibility, also set context_data
    state.context_data = {
        "products": state.products,
        "stores": state.stores,
        "offers": state.offers,
        "events": state.events,
        "services": state.services,
        "mall_name": state.mall_name,
        "matched_items": state.matched_items
    }
    
    return state

async def store_context_retrieval(state: CustomerState) -> CustomerState:
    """Specialized node for retrieving store-related context"""
    if not state.mall_id:
        state.response = "Please select a mall first to see store information."
        return state

    logger.info(f"Retrieving store context for query: {state.query}")
    
    # Get mall name
    mall = await db_fetch_one_async("SELECT marketing_name AS name_en FROM malls WHERE unique_property_id = $1", (state.mall_id,))
    state.mall_name = mall["name_en"] if mall else "Unknown Mall"
    
    # Extract store name from state, resolved entity or query
    store_name = state.store_name or state.resolved_entity or ""
    logger.info(f"Initial store name: {store_name}")
    
    # Try to extract store type from the query if no specific store name was provided
    if not store_name:
        # Check for food-related terms
        food_terms = ["restaurant", "food", "eat", "café", "cafe", "coffee", "dining", "hungry", "meal", "lunch", "dinner", "breakfast"]
        retail_terms = ["shop", "store", "retail", "boutique", "outlet", "clothes", "fashion", "shoes", "electronics"]
        service_terms = ["service", "salon", "barber", "spa", "bank", "atm", "pharmacy"]
        
        query_words = state.query.lower().split()
        
        if any(term in query_words for term in food_terms):
            store_name = "restaurant"
            logger.info("Detected food-related query, using 'restaurant' as store type")
        elif any(term in query_words for term in retail_terms):
            store_name = "retail"
            logger.info("Detected retail-related query, using 'retail' as store type")
        elif any(term in query_words for term in service_terms):
            store_name = "service"
            logger.info("Detected service-related query, using 'service' as store type")
    
    # Try to get relevant results from Pinecone first
    try:
        # Convert query to vector
        query_vector = embeddings.embed_query(state.query)
        
        # Search Pinecone with filter for this mall and store type
        pinecone_results = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=25,
            include_metadata=True,
            filter={"mall_id": state.mall_id, "type": "store"}
        )
        
        # Process Pinecone results
        matched_items = []
        for match in pinecone_results.get("matches", []):
            metadata = match["metadata"]
            
            # Build a complete item with all the metadata
            item = {
                "id": match["id"],
                "type": "store",
                "score": float(match["score"]),
                **{k: v for k, v in metadata.items() if k != "type"}
            }
            
            # Add to matched items
            matched_items.append(item)
        
        # Store in state
        state.matched_items = matched_items
        logger.info(f"Found {len(matched_items)} matched items from vector search")
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        state.matched_items = []
    
    target_store_data = None
    
    # Fetch stores based on name or all stores if no name specified
    if store_name:
        logger.info(f"Searching for stores with name like '{store_name}'")
        stores = await db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.brand_name_ar, b.category_name, 
               b.description_en, b.description_ar, b.company_name_en, b.company_name_ar,
               b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes,
               b.brand_logo, b.social_instagram, b.social_facebook, b.is_published
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
               WHERE bma.unique_property_id = $1 AND 
               (LOWER(b.brand_name_en) LIKE $2 OR LOWER(b.category_name) LIKE $2)""", 
            (state.mall_id, f"%{store_name.lower()}%")
        )
    else:
        logger.info(f"No specific store name, fetching all stores")
        stores = await db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.brand_name_ar, b.category_name, 
               b.description_en, b.description_ar, b.company_name_en, b.company_name_ar,
               b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes,
               b.brand_logo, b.social_instagram, b.social_facebook, b.is_published
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
               WHERE bma.unique_property_id = $1
               LIMIT 50""", 
        (state.mall_id,)
    )
    
    # Process and add stores to the state
    store_list = []
    logger.info(f"Found {len(stores)} stores matching the criteria")
    
    for store in stores:
        store_data = {
            "store_id": store["brand_id"],
            "name": store["brand_name_en"],
            "name_ar": store.get("brand_name_ar", ""),
            "category": store.get("category_name", ""),
            "description": store.get("description_en", ""),
            "description_ar": store.get("description_ar", ""),
            "company_name": store.get("company_name_en", ""),
            "company_name_ar": store.get("company_name_ar", ""),
            "phone": store.get("store_phone_number", ""),
            "email": store.get("store_email", ""),
            "website": store.get("store_website", ""),
            "location": store.get("pms_unit_codes", {}),
            "logo": store.get("brand_logo", ""),
            "social_instagram": store.get("social_instagram", ""),
            "social_facebook": store.get("social_facebook", ""),
            "is_published": store.get("is_published", True)
        }
        
        # If this is the target store, put it at the top
        if store_name and store_name.lower() in store["brand_name_en"].lower():
            store_list.insert(0, store_data)
            target_store_data = store_data
        else:
            store_list.append(store_data)
    
    # Store in state
    state.stores = store_list
    
    # Get products for these stores
    store_ids = [store["store_id"] for store in store_list[:10]]  # Limit to top 10 stores
    
    if store_ids:
        products = await db_fetch_all_async(
            """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
               p.is_featured, p.in_stock, p.image_url, p.attributes, p.created_at, p.updated_at,
               b.brand_name_en
               FROM products p
               JOIN brands b ON p.brand_id = b.brand_id
               WHERE p.brand_id = ANY($1)
               ORDER BY p.is_featured DESC, p.id
               LIMIT 50""",
            (store_ids,)
        )
        
        # Track seen products to avoid duplicates
        seen_products = set()
        product_list = []
        
        for product in products:
            # Create a unique product key to avoid duplicates
            product_key = f"{product.get('name')}_{product.get('brand_id')}"
            if product_key in seen_products:
                continue
                
            seen_products.add(product_key)
            
            # Add product
            product_data = {
                "id": product["id"],
                "name": product["name"],
                "description": product.get("description", ""),
                "price": float(product["price"]) if product.get("price") is not None else None,
                "brand_id": product["brand_id"],
                "category": product.get("category", ""),
                "store_name": product.get("brand_name_en", ""),
                "is_featured": product.get("is_featured", False),
                "in_stock": product.get("in_stock", True),
                "image_url": product.get("image_url", ""),
                "attributes": product.get("attributes", {})
            }
            
            product_list.append(product_data)
        
        # Store in state
        state.products = product_list
    
    # Get offers and events for these stores
    if store_ids:
        engagements = await db_fetch_all_async(
            """SELECT e.engagement_id, e.title_en, e.title_ar, e.description_en, e.description_ar, 
               e.type, e.brand_id, e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive,
               e.publish_date, e.ext_url, e.images_en, e.tenant_profile_id
               FROM engagements e
               WHERE e.unique_property_id = $1 AND e.brand_id = ANY($2)
               AND (e.end_date IS NULL OR CAST(e.end_date AS DATE) >= CURRENT_DATE)
               ORDER BY e.start_date DESC LIMIT 50""",
            (state.mall_id, store_ids)
        )
        
        offer_list = []
        event_list = []
        
        for engagement in engagements:
            engagement_type = engagement.get("type", "").lower()
                
                # Get store name for this engagement
            store_name = next((s["name"] for s in store_list if s["store_id"] == engagement["brand_id"]), "")
            
            if engagement_type == "offer":
                offer_data = {
                    "id": engagement["engagement_id"],
                    "title": engagement.get("title_en", ""),
                        "title_ar": engagement.get("title_ar", ""),
                    "description": engagement.get("description_en", ""),
                        "description_ar": engagement.get("description_ar", ""),
                    "brand_id": engagement.get("brand_id"),
                        "store_name": store_name,
                    "start_date": engagement.get("start_date", ""),
                    "end_date": engagement.get("end_date", ""),
                    "terms": engagement.get("terms_conditions_en", ""),
                        "is_exclusive": bool(engagement.get("is_exclusive", 0)),
                        "url": engagement.get("ext_url", ""),
                        "images": engagement.get("images_en", "")
                }
                offer_list.append(offer_data)
            
            elif engagement_type == "event":
                event_data = {
                    "id": engagement["engagement_id"],
                    "name": engagement.get("title_en", ""),
                        "name_ar": engagement.get("title_ar", ""),
                    "description": engagement.get("description_en", ""),
                        "description_ar": engagement.get("description_ar", ""),
                    "brand_id": engagement.get("brand_id"),
                        "store_name": store_name,
                    "start_date": engagement.get("start_date", ""),
                    "end_date": engagement.get("end_date", ""),
                    "terms": engagement.get("terms_conditions_en", "")
                }
                event_list.append(event_data)
            
            # Store in state
            state.offers = offer_list
            state.events = event_list
    
    # Find neighboring stores if a specific store was identified
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
                
                for store in state.stores:
                    if store == target_store_data:
                        continue
                    
                    store_locations = store.get("location", [])
                    if any(code in adjacent_codes for code in store_locations):
                        if store not in neighboring_stores:
                            neighboring_stores.append(store)
        
    # For backward compatibility, also set context_data
    state.context_data = {
        "products": state.products,
        "stores": state.stores,
        "offers": state.offers,
        "events": state.events,
        "services": state.services,
        "mall_name": state.mall_name,
        "matched_items": state.matched_items
    }
    
    return state

async def engagement_context_retrieval(state: CustomerState) -> CustomerState:
    """Specialized node for retrieving offers and events context"""
    if not state.mall_id:
        state.response = "Please select a mall first to see offers and events."
        return state
    
    logger.info(f"Retrieving engagements context for query: {state.query}")
    engagement_context = {
        "offers": [],
        "events": [],
        "stores": [],
        "mall_name": ""
    }
    
    try:
        # Get mall name
        mall = await db_fetch_one_async("SELECT marketing_name AS name_en FROM malls WHERE unique_property_id = $1", (state.mall_id,))
        engagement_context["mall_name"] = mall["name_en"] if mall else "Unknown Mall"
        
        # Extract store name if mentioned (for store-specific offers/events)
        store_name = state.context_data.get("resolved_entity", "").lower() if state.context_data else ""
        
        # Determine if looking for specific offer/event type from query
        event_keywords = ["event", "events", "show", "shows", "exhibition", "exhibitions", "concert", "concerts", "performance", "performances", "happening", "activities"]
        offer_keywords = ["offer", "offers", "deal", "deals", "discount", "discounts", "promotion", "promotions", "sale", "sales", "coupon", "coupons"]
        
        is_offer_focused = any(keyword in state.query.lower() for keyword in offer_keywords)
        is_event_focused = any(keyword in state.query.lower() for keyword in event_keywords)
        
        # Base query for engagements
        engagement_query = """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
           e.start_date, e.end_date, e.terms_conditions_en, e.is_exclusive, e.unique_property_id
           FROM engagements e 
           WHERE e.unique_property_id = $1"""
        
        engagement_params = [state.mall_id]
        
        # Add store filter if a store is mentioned
        if store_name:
            store = await db_fetch_one_async(
                """SELECT brand_id FROM brands 
                   WHERE LOWER(brand_name_en) LIKE $1 AND brand_id IN 
                   (SELECT brand_id FROM brand_mall_association WHERE unique_property_id = $2)""",
                (f"%{store_name}%", state.mall_id)
            )
            if store:
                engagement_query += " AND e.brand_id = $2"
                engagement_params.append(store["brand_id"])
                
                # Add the store details to context
                store_details = await db_fetch_one_async(
                    """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
                       b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes
                       FROM brands b WHERE b.brand_id = $1""",
                    (store["brand_id"],)
                )
                if store_details:
                    engagement_context["stores"].append({
                        "name": store_details["brand_name_en"],
                        "category": store_details.get("category_name", ""),
                        "store_id": store_details["brand_id"],
                        "description": store_details.get("description_en", ""),
                        "location": store_details.get("pms_unit_codes", {})
                    })
        
        # Add type filter if focused on offers or events
        if is_offer_focused and not is_event_focused:
            engagement_query += f" AND e.type = 'offer'"
        elif is_event_focused and not is_offer_focused:
            engagement_query += f" AND e.type = 'events'"
        
        # Filter for active engagements (not expired)
        engagement_query += " AND (e.end_date IS NULL OR CAST(e.end_date AS DATE) >= CURRENT_DATE)"
        
        # Order by start date (newest first)
        engagement_query += " ORDER BY e.start_date DESC"
        
        # Fetch engagements
        engagements = await db_fetch_all_async(engagement_query, tuple(engagement_params))
        
        for engagement in engagements:
            # Initialize engagement_type with a safe default
            engagement_type = engagement.get("type", "unknown").lower() if engagement.get("type") else "unknown"
            
            # Get associated brand information if not already fetched
            brand = None
            if engagement.get("brand_id") and not (store_name and engagement_context["stores"]):
                try:
                    brand = await db_fetch_one_async(
                        "SELECT brand_name_en, category_name, pms_unit_codes FROM brands WHERE brand_id = $1",
                        (engagement["brand_id"],)
                    )
                except Exception as e:
                    logger.error(f"Error fetching brand for engagement {engagement.get('engagement_id')}: {e}")
            
            try:
                if engagement_type == "offer":
                    offer_data = {
                        "id": engagement["engagement_id"],
                        "title": engagement.get("title_en", ""),
                        "description": engagement.get("description_en", ""),
                        "brand_id": engagement.get("brand_id"),
                        "store_name": (brand["brand_name_en"] if brand and "brand_name_en" in brand else 
                                      engagement_context["stores"][0]["name"] if engagement_context["stores"] else 
                                      "Unknown Store"),
                        "category": (brand.get("category_name", "") if brand else 
                                    engagement_context["stores"][0].get("category", "") if engagement_context["stores"] else 
                                    ""),
                        "location": (brand.get("pms_unit_codes", {}) if brand else 
                                    engagement_context["stores"][0].get("location", {}) if engagement_context["stores"] else 
                                    {}),
                        "start_date": engagement.get("start_date", ""),
                        "end_date": engagement.get("end_date", ""),
                        "terms": engagement.get("terms_conditions_en", ""),
                        "is_exclusive": bool(engagement.get("is_exclusive", 0))
                    }
                    engagement_context["offers"].append(offer_data)
                
                elif engagement_type == "events":
                    event_data = {
                        "id": engagement["engagement_id"],
                        "name": engagement.get("title_en", ""),
                        "description": engagement.get("description_en", ""),
                        "brand_id": engagement.get("brand_id"),
                        "store_name": (brand["brand_name_en"] if brand and "brand_name_en" in brand else 
                                      engagement_context["stores"][0]["name"] if engagement_context["stores"] else 
                                      "Unknown Store"),
                        "location": (brand.get("pms_unit_codes", {}) if brand else 
                                    engagement_context["stores"][0].get("location", {}) if engagement_context["stores"] else 
                                    {}),
                        "start_date": engagement.get("start_date", ""),
                        "end_date": engagement.get("end_date", ""),
                        "terms": engagement.get("terms_conditions_en", "")
                    }
                    engagement_context["events"].append(event_data)
                else:
                    logger.warning(f"Unknown engagement type: {engagement_type} for engagement {engagement.get('engagement_id')}")
            except Exception as e:
                logger.error(f"Error processing engagement {engagement.get('engagement_id')}: {e}")
        
        # Set context_data for backward compatibility
        state.context_data = engagement_context
        
        # Update the state's direct properties
        state.offers = engagement_context["offers"]
        state.events = engagement_context["events"]
        state.stores = engagement_context["stores"]
        state.mall_name = engagement_context["mall_name"]
    
    except Exception as e:
        logger.error(f"Error in engagement_context_retrieval: {e}")
        # Ensure we have empty lists instead of None
        state.offers = state.offers or []
        state.events = state.events or []
        state.stores = state.stores or []
        state.mall_name = state.mall_name or "Unknown Mall"
        
        # Also ensure context_data is initialized
        if not state.context_data:
            state.context_data = {
                "offers": [],
                "events": [],
                "stores": [],
                "mall_name": state.mall_name or "Unknown Mall"
            }
    
    return state

async def service_context_retrieval(state: CustomerState) -> CustomerState:
    """Specialized node for retrieving service-related context"""
    if not state.mall_id:
        state.response = "Please select a mall first to see service information."
        return state
    
    logger.info(f"Retrieving service context for query: {state.query}")
    
    # Get mall name
    mall = await db_fetch_one_async("SELECT marketing_name AS name_en FROM malls WHERE unique_property_id = $1", (state.mall_id,))
    state.mall_name = mall["name_en"] if mall else "Unknown Mall"
    
    # Extract service name from resolved entity or query
    service_name = state.resolved_entity or ""
    
    # Check for generic service inquiries
    service_keywords = ["service", "services", "facility", "facilities", "amenity", "amenities"]
    is_generic_service_query = any(keyword in state.query.lower() for keyword in service_keywords) and not service_name
    
    # Check for specific services in the query
    if not service_name:
        specific_services = {
            "parking": ["parking", "car park", "where to park"],
            "restroom": ["restroom", "bathroom", "toilet", "washroom"],
            "wifi": ["wifi", "internet", "connection", "wi-fi"],
            "prayer room": ["prayer", "mosque", "prayer room"],
            "nursing room": ["nursing", "baby", "breastfeeding", "changing"],
            "information": ["information desk", "info", "help desk", "customer service"]
        }
        
        query_lower = state.query.lower()
        for service_type, keywords in specific_services.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.info(f"Detected specific service query for: {service_type}")
                service_name = service_type
                break
    
    if is_generic_service_query:
        logger.info("Generic service inquiry detected. Fetching all mall services.")
        # Clear any potentially confusing service_name
        service_name = ""
    
    # Try Pinecone search first
    try:
        # Convert query to vector
        query_vector = embeddings.embed_query(state.query)
        
        # Search Pinecone with filter for this mall and service type
        pinecone_results = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=25,
            include_metadata=True,
            filter={"mall_id": state.mall_id, "type": "service"}
        )
        
        # Process Pinecone results
        matched_items = []
        for match in pinecone_results.get("matches", []):
            metadata = match["metadata"]
            
            # Build a complete item with all the metadata
            item = {
                "id": match["id"],
                "type": "service",
                "score": float(match["score"]),
                **{k: v for k, v in metadata.items() if k != "type"}
            }
            
            # Add to matched items
            matched_items.append(item)
        
        # Store in state
        state.matched_items = matched_items
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        state.matched_items = []
    
    try:
        # Fetch services based on name or all services if no name specified
        if service_name:
            services = await db_fetch_all_async(
                """SELECT s.id AS service_id, s.name, s.name_ar, s.description, s.description_ar, 
                   s.icon_url, s.is_available, s.location, s.created_at, s.updated_at
                   FROM services s
                   WHERE s.unique_property_id = $1 AND LOWER(s.name) LIKE $2""",
                (state.mall_id, f"%{service_name.lower()}%")
            )
        else:
            services = await db_fetch_all_async(
                """SELECT s.id AS service_id, s.name, s.name_ar, s.description, s.description_ar, 
                   s.icon_url, s.is_available, s.location, s.created_at, s.updated_at
           FROM services s
           WHERE s.unique_property_id = $1""",
        (state.mall_id,)
    )
    
        # Process and add services to state
        service_list = []
    
        for service in services:
            service_data = {
                        "service_id": service.get("service_id"),
                "name": service.get("name", ""),
                        "name_ar": service.get("name_ar", ""),
                        "description": service.get("description", ""),
                        "description_ar": service.get("description_ar", ""),
                        "icon_url": service.get("icon_url", ""),
                        "is_available": service.get("is_available", True),
                        "location": service.get("location", ""),
                        "created_at": service.get("created_at", ""),
                        "updated_at": service.get("updated_at", "")
                    }
            service_list.append(service_data)
        
        # Store in state
        state.services = service_list
        
    except Exception as e:
        logger.error(f"Error fetching services: {e}")
        state.services = []
    
    # For backward compatibility, also set context_data
    state.context_data = {
        "products": state.products,
        "stores": state.stores,
        "offers": state.offers,
        "events": state.events,
        "services": state.services,
        "mall_name": state.mall_name,
        "matched_items": state.matched_items
    }
    
    return state

async def mall_context_retrieval(state: CustomerState) -> CustomerState:
    """Specialized node for retrieving mall-related context"""
    if not state.mall_id:
        state.response = "Please select a mall first to see mall information."
        return state
    
    logger.info(f"Retrieving mall context for query: {state.query}")
    mall_context = {
        "mall_info": {},
        "services": []
    }
    
    try:
        # Get detailed mall information
        mall = await db_fetch_one_async(
            """SELECT m.unique_property_id, m.marketing_name AS name_en, m.marketing_name_ar AS name_ar, 
               m.city, m.country, m.mall_information, m.image, m.gps_coordinates
               FROM malls m WHERE m.unique_property_id = $1""", 
            (state.mall_id,)
        )
        
        if mall:
            mall_context["mall_info"] = {
                "id": mall["unique_property_id"],
                "name": mall["name_en"],
                "name_ar": mall.get("name_ar", ""),
                "city": mall.get("city", ""),
                "country": mall.get("country", ""),
                "information": mall.get("mall_information", ""),
                "image": mall.get("image", ""),
                "location": mall.get("gps_coordinates", "")
            }
        
        # Fetch core services for the mall
        services = await db_fetch_all_async(
            """SELECT s.id AS service_id, s.name, s.description, s.location, s.is_available
               FROM services s
               WHERE s.unique_property_id = $1""",
                (state.mall_id,)
            )
        
        mall_context["services"] = [
            {
                "name": service.get("name", ""),
            "description": service.get("description", ""),
            "location": service.get("location", ""),
            "is_available": service.get("is_available", True)
        }
            for service in services
        ]
    except Exception as e:
        logger.error(f"Error fetching mall information: {e}")
        # Provide basic information if there's an error
        if not mall_context["mall_info"] and state.mall_id:
            try:
                basic_mall = await db_fetch_one_async(
                    "SELECT marketing_name AS name_en FROM malls WHERE unique_property_id = $1", 
                    (state.mall_id,)
                )
                if basic_mall:
                    mall_context["mall_info"] = {"name": basic_mall["name_en"]}
            except Exception:
                mall_context["mall_info"] = {"name": "the mall"}
    
    state.context_data = mall_context
    
    # Update direct state properties
    if "mall_info" in mall_context:
        state.mall_name = mall_context["mall_info"].get("name", "")
    
    if "services" in mall_context:
        state.services = mall_context["services"]
    
    return state

async def general_context_retrieval(state: CustomerState) -> CustomerState:
    """Fallback context retrieval when we're unsure of the intent"""
    # This will be similar to the original initial_retrieval function
    # but with some enhancements for better fallback handling
    
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state

    logger.info(f"Using general context retrieval as fallback for query: {state.query}")
    
    # Get mall name
    mall = await db_fetch_one_async("SELECT marketing_name AS name_en FROM malls WHERE unique_property_id = $1", (state.mall_id,))
    state.mall_name = mall["name_en"] if mall else "Unknown Mall"
    
    # Check for follow-up indicators and previous context
    followup_indicators = ["it", "they", "that", "those", "this", "there", "these", "their", "yes", "no", "yeah", "sure", "okay", "ok"]
    query_lower = state.query.lower()
    
    # Handle possible follow-up questions based on previous intent
    if (any(word in query_lower.split() for word in followup_indicators) or len(query_lower.split()) <= 5) and state.previous_intent and state.previous_entity:
        logger.info(f"Detected potential follow-up. Previous intent: {state.previous_intent}, entity: {state.previous_entity}")
        
        # For store finder follow-ups
        if state.previous_intent == "store_finder":
            # Look for the store in the database
            store_results = await db_fetch_all_async(
                """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
                   b.store_phone_number, b.store_email, b.store_website, b.pms_unit_codes
                   FROM brands b 
                   JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                   WHERE bma.unique_property_id = $1 
                   AND LOWER(b.brand_name_en) LIKE $2""",
                (state.mall_id, f"%{state.previous_entity.lower()}%")
            )
            
            if store_results:
                # Clear previous store data and add the specific matches
                state.stores = []
                for store in store_results:
                    state.stores.append({
                        "store_id": store["brand_id"],
                        "name": store["brand_name_en"],
                        "category": store.get("category_name", ""),
                        "description": store.get("description_en", ""),
                        "phone": store.get("store_phone_number", ""),
                        "email": store.get("store_email", ""),
                        "website": store.get("store_website", ""),
                        "location": store.get("pms_unit_codes", {})
                    })
                
                # Get products for this store
                if state.stores:
                    products = await db_fetch_all_async(
                        """SELECT p.id, p.name, p.description, p.price, p.category,
                           p.brand_id, p.is_featured, p.in_stock, p.image_url
                           FROM products p
                           WHERE p.brand_id = $1
                           ORDER BY p.is_featured DESC, p.id
                           LIMIT 10""",
                        (store_results[0]["brand_id"],)
                    )
                    
                    state.products = []
                    for product in products:
                        state.products.append({
                            "id": product["id"],
                            "name": product["name"],
                            "description": product.get("description", ""),
                            "price": float(product["price"]) if product.get("price") is not None else None,
                            "brand_id": product["brand_id"],
                            "category": product.get("category", ""),
                            "store_name": store_results[0]["brand_name_en"],
                            "is_featured": product.get("is_featured", False),
                            "in_stock": product.get("in_stock", True)
                        })
                    
                    logger.info(f"Follow-up: found {len(state.products)} products for store {state.previous_entity}")
        
        # For product finder follow-ups
        elif state.previous_intent == "product_finder":
            # Look for products in the database
            product_results = await db_fetch_all_async(
                """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
                   p.is_featured, p.in_stock, p.image_url, p.attributes,
                   b.brand_name_en
                   FROM products p
                   JOIN brands b ON p.brand_id = b.brand_id
                   JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                   WHERE bma.unique_property_id = $1 AND 
                   (LOWER(p.name) LIKE $2 OR LOWER(p.category) LIKE $2)
                   ORDER BY p.is_featured DESC, p.id""",
                (state.mall_id, f"%{state.previous_entity.lower()}%")
            )
            
            if product_results:
                state.products = []
                for product in product_results:
                    state.products.append({
                "id": product["id"],
                "name": product["name"],
                "description": product.get("description", ""),
                "price": float(product["price"]) if product.get("price") is not None else None,
                "brand_id": product["brand_id"],
                        "category": product.get("category", ""),
                "store_name": product.get("brand_name_en", ""),
                        "is_featured": product.get("is_featured", False),
                        "in_stock": product.get("in_stock", True),
                        "attributes": product.get("attributes", {})
                    })
                    
                    logger.info(f"Follow-up: found {len(state.products)} matches for product {state.previous_entity}")
    
    # Check for food/restaurant related queries
    food_keywords = ["food", "restaurant", "eat", "hungry", "lunch", "dinner", "breakfast", "meal", "burger", "pizza", "chicken", "menu", "coffee"]
    if any(keyword in query_lower for keyword in food_keywords) and not state.stores:
        logger.info("Detected food-related query, fetching restaurants")
        restaurants = await db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en,
               b.store_phone_number, b.pms_unit_codes
               FROM brands b
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
               WHERE bma.unique_property_id = $1 
               AND (LOWER(b.category_name) LIKE '%restaurant%' 
                    OR LOWER(b.category_name) LIKE '%food%'
                    OR LOWER(b.category_name) LIKE '%cafe%'
                    OR LOWER(b.category_name) LIKE '%coffee%'
                    OR LOWER(b.category_name) LIKE '%dining%')
               LIMIT 10""",
            (state.mall_id,)
        )
        
        if restaurants:
            state.stores = []
            for restaurant in restaurants:
                state.stores.append({
                    "store_id": restaurant["brand_id"],
                    "name": restaurant["brand_name_en"],
                    "category": restaurant.get("category_name", ""),
                    "description": restaurant.get("description_en", ""),
                    "phone": restaurant.get("store_phone_number", ""),
                    "location": restaurant.get("pms_unit_codes", {})
                })
                
            logger.info(f"Found {len(state.stores)} restaurants for food query")
            
    # Check for clothing/fashion/shoe related queries
    clothing_keywords = ["clothes", "clothing", "fashion", "shirt", "pants", "jeans", "dress", "shoe", "shoes", "wear", "outfit", "shopping", "apparel"]
    if any(keyword in query_lower for keyword in clothing_keywords) and not state.stores:
        logger.info("Detected clothing/fashion-related query, fetching relevant stores")
        fashion_stores = await db_fetch_all_async(
            """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en,
               b.store_phone_number, b.pms_unit_codes
               FROM brands b
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
               WHERE bma.unique_property_id = $1 
               AND (LOWER(b.category_name) LIKE '%fashion%' 
                    OR LOWER(b.category_name) LIKE '%clothing%'
                    OR LOWER(b.category_name) LIKE '%apparel%'
                    OR LOWER(b.category_name) LIKE '%shoes%'
                    OR LOWER(b.category_name) LIKE '%wear%'
                    OR LOWER(b.category_name) LIKE '%jeans%'
                    OR LOWER(b.category_name) LIKE '%dress%')
               LIMIT 10""",
            (state.mall_id,)
        )
        
        if fashion_stores:
            state.stores = []
            for store in fashion_stores:
                state.stores.append({
                    "store_id": store["brand_id"],
                    "name": store["brand_name_en"],
                    "category": store.get("category_name", ""),
                    "description": store.get("description_en", ""),
                    "phone": store.get("store_phone_number", ""),
                    "location": store.get("pms_unit_codes", {})
                })
                
            logger.info(f"Found {len(state.stores)} fashion/clothing stores")
    
    # If we still don't have any stores, do an embedding-based query
    if not state.stores and not state.products:
        query_items = [item.strip() for item in state.query.split("\n") if item.strip()] if "\n" in state.query else [state.query]
        
        # Build query vector
        query_vectors = [embeddings.embed_query(item) for item in query_items]
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
            
            # Process the matched items
            matched_items = []
            for match in results.get("matches", []):
                metadata = match["metadata"]
                doc_type = metadata.get("type", "")
                
                # Build a complete item with all the metadata
                item = {
                    "id": match["id"],
                    "type": doc_type,
                    "score": float(match["score"]),
                    **{k: v for k, v in metadata.items() if k != "type"}
                }
                
                matched_items.append(item)
            
            # Store matched items in state
            state.matched_items = matched_items
            
            # Process specific types into their respective state fields
            for item in matched_items:
                doc_type = item.get("type", "")
                
                if doc_type == "store":
                    # Only add if not already in stores
                    if not any(s.get("store_id") == item.get("brand_id") for s in state.stores):
                        store_data = {
                            "store_id": item.get("brand_id"),
                            "name": item.get("name_en", ""),
                            "name_ar": item.get("name_ar", ""),
                            "category": item.get("category_en", ""),
                            "description": item.get("description_en", ""),
                            "company_name": item.get("company_name_en", ""),
                            "location": item.get("pms_unit_codes", {}),
                            "logo": item.get("brand_logo", "")
                        }
                        state.stores.append(store_data)
                
                elif doc_type == "product":
                    # Only add if not already in products
                    if not any(p.get("id") == item.get("id") for p in state.products):
                        product_data = {
                            "id": item.get("id"),
                            "name": item.get("name", ""),
                            "description": item.get("description", ""),
                            "price": item.get("price"),
                            "brand_id": item.get("brand_id"),
                            "category": item.get("category", ""),
                            "store_name": item.get("brand_name_en", "")
                        }
                        state.products.append(product_data)
                
                elif doc_type == "engagement":
                    engagement_type = item.get("type")
                    if engagement_type == "offer":
                        # Only add if not already in offers
                        if not any(o.get("id") == item.get("engagement_id") for o in state.offers):
                            offer_data = {
                                "id": item.get("engagement_id"),
                                "title": item.get("name_en", ""),
                                "description": item.get("description_en", ""),
                                "brand_id": item.get("brand_id"),
                                "start_date": item.get("start_date", ""),
                                "end_date": item.get("end_date", ""),
                                "terms": item.get("terms_conditions_en", "")
                            }
                            state.offers.append(offer_data)
                    elif engagement_type == "event":
                        # Only add if not already in events
                        if not any(e.get("id") == item.get("engagement_id") for e in state.events):
                            event_data = {
                                "id": item.get("engagement_id"),
                                "name": item.get("name_en", ""),
                                "description": item.get("description_en", ""),
                                "brand_id": item.get("brand_id"),
                                "start_date": item.get("start_date", ""),
                                "end_date": item.get("end_date", "")
                            }
                            state.events.append(event_data)
                
                elif doc_type == "service":
                    # Only add if not already in services
                    if not any(s.get("service_id") == item.get("id") for s in state.services):
                        service_data = {
                            "service_id": item.get("id"),
                            "name": item.get("name", ""),
                            "description": item.get("description", ""),
                            "location": item.get("location", ""),
                            "is_available": item.get("is_available", True)
                        }
                        state.services.append(service_data)
        except Exception as e:
            logger.error(f"Pinecone query error in general_context_retrieval: {e}")
    
    # If we still don't have any data, fetch some default stores
    if not state.stores:
        try:
            # Get top stores 
            top_stores = await db_fetch_all_async(
                """SELECT b.brand_id, b.brand_name_en, b.category_name
                   FROM brands b
                   JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                   WHERE bma.unique_property_id = $1
                   ORDER BY b.anchor_brand DESC
                   LIMIT 5""",
                (state.mall_id,)
            )
            
            # Process and add stores to state
            store_list = []
            for store in top_stores:
                store_data = {
                    "store_id": store["brand_id"],
                    "name": store["brand_name_en"],
                    "category": store.get("category_name", "")
                }
                store_list.append(store_data)
            
            state.stores = store_list
        except Exception as e:
            logger.error(f"Error fetching top stores: {e}")
    
    # Get top offers/events if we don't have any
    if not state.offers and not state.events:
        try:
            # Get top offers/events
            top_engagements = await db_fetch_all_async(
                """SELECT e.engagement_id, e.title_en, e.type, e.brand_id
                   FROM engagements e
                   WHERE e.unique_property_id = $1
                   AND (e.end_date IS NULL OR CAST(e.end_date AS DATE) >= CURRENT_DATE)
                   ORDER BY e.start_date DESC
                   LIMIT 3""",
                (state.mall_id,)
            )
            
            offer_list = []
            event_list = []
            
            for engagement in top_engagements:
                engagement_type = engagement.get("type", "").lower()
                if engagement_type == "offer":
                    offer_data = {
                        "id": engagement["engagement_id"],
                        "title": engagement.get("title_en", ""),
                        "brand_id": engagement.get("brand_id")
                    }
                    offer_list.append(offer_data)
                elif engagement_type == "event":
                    event_data = {
                        "id": engagement["engagement_id"],
                        "name": engagement.get("title_en", ""),
                        "brand_id": engagement.get("brand_id")
                    }
                    event_list.append(event_data)
            
            state.offers = offer_list
            state.events = event_list
        except Exception as e:
            logger.error(f"Error fetching top engagements: {e}")
    
    # For backward compatibility, also set context_data
    state.context_data = {
        "products": state.products,
        "stores": state.stores,
        "offers": state.offers,
        "events": state.events,
        "services": state.services,
        "mall_name": state.mall_name,
        "matched_items": state.matched_items
    }
    
    # Log what we found for debugging
    logger.info(f"General context retrieval found: {len(state.stores)} stores, {len(state.products)} products, {len(state.offers)} offers, {len(state.events)} events, {len(state.services)} services")
    
    return state

async def general_chat_retrieval(state: CustomerState) -> CustomerState:
    """Comprehensive retrieval for general chat questions that don't fit specific intents"""
    if not state.mall_id:
        state.response = "Please select a mall first to get information."
        return state
    
    logger.info(f"Performing general context retrieval for query: {state.query}")
    
    # Use the query directly for vector search first
    embeddings_result = []
    try:
        # Convert query to vector
        query_vector = embeddings.embed_query(state.query)
        
        # Search Pinecone with filter for this mall
        pinecone_results = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=10,  # Fewer results for general chat
            include_metadata=True,
            filter={"mall_id": state.mall_id}
        )
        
        # Process matched items
        matched_items = []
        for match in pinecone_results.get("matches", []):
            metadata = match["metadata"]
            doc_type = metadata.get("type", "")
            
            if doc_type == "store":
                store_id = metadata.get("brand_id")
                if store_id:
                    try:
                        # Get verified store data directly from database
                        db_store = await db_fetch_one_async(
            """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, 
                               b.store_phone_number, b.store_email, b.store_website,
                               b.social_instagram, b.pms_unit_codes
               FROM brands b 
               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
                               WHERE b.brand_id = $1 AND bma.unique_property_id = $2""",
                            (store_id, state.mall_id)
                        )
                        if db_store:
                            # Add verified flag and score from vector search
                            item = {
                                "type": "store",
                                "store_id": db_store["brand_id"],
                                "name": db_store["brand_name_en"],
                                "category": db_store.get("category_name", ""),
                                "description": db_store.get("description_en", ""),
                                "score": float(match["score"]),
                                "verified": True
                            }
                            
                            # Add contact info only if available
                            if db_store.get("store_phone_number"):
                                item["phone"] = db_store["store_phone_number"]
                            if db_store.get("store_email"):
                                item["email"] = db_store["store_email"]
                            if db_store.get("store_website"):
                                item["website"] = db_store["store_website"]
                            if db_store.get("social_instagram"):
                                item["social_instagram"] = db_store["social_instagram"]
                            
                            # Process location codes if available
                            if db_store.get("pms_unit_codes"):
                                try:
                                    location_codes = db_store["pms_unit_codes"]
                                    if isinstance(location_codes, str):
                                        location_codes = json.loads(location_codes)
                                    
                                    if isinstance(location_codes, dict):
                                        if "level" in location_codes:
                                            item["floor"] = location_codes["level"]
                                        if "gate" in location_codes:
                                            item["gate"] = location_codes["gate"]
                                except Exception as e:
                                    logger.error(f"Error processing location codes: {e}")
                            
                            # Add to matched items and stores list
                            matched_items.append(item)
                            
                            # Also add to the stores context in state
                            if not any(s.get("store_id") == store_id for s in state.stores):
                                state.stores.append(item)
                            
                    except Exception as e:
                        logger.error(f"Error fetching store {store_id}: {e}")
                        # Add unverified item from vector search as fallback
                        matched_items.append({
                            "type": "store",
                            "store_id": metadata.get("brand_id"),
                            "name": metadata.get("name"),
                            "category": metadata.get("category", ""),
                            "score": float(match["score"]),
                            "verified": False
                        })
            
            elif doc_type == "product":
                product_id = metadata.get("id")
                if product_id:
                    try:
                        # Get verified product data directly from database
                        db_product = await db_fetch_one_async(
                            """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id,
                               p.is_featured, p.in_stock, b.brand_name_en
                               FROM products p
                               JOIN brands b ON p.brand_id = b.brand_id
                               JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                               WHERE p.id = $1 AND bma.unique_property_id = $2""",
                            (product_id, state.mall_id)
                        )
                        if db_product:
                            # Add verified flag and score from vector search
                            item = {
                                "type": "product",
                                "id": db_product["id"],
                                "name": db_product["name"],
                                "description": db_product.get("description", ""),
                                "brand_id": db_product.get("brand_id"),
                                "category": db_product.get("category", ""),
                                "store_name": db_product.get("brand_name_en", ""),
                                "score": float(match["score"]),
                                "verified": True
                            }
                            
                            # Add price only if it exists
                            if db_product.get("price") is not None:
                                item["price"] = float(db_product["price"])
                            
                            # Add availability information only if it exists
                            if db_product.get("in_stock") is not None:
                                item["in_stock"] = db_product["in_stock"]
                            
                            # Add to matched items and products list
                            matched_items.append(item)
                            
                            # Also add to the products context in state
                            if not any(p.get("id") == product_id for p in state.products):
                                state.products.append(item)
                            
                    except Exception as e:
                        logger.error(f"Error fetching product {product_id}: {e}")
                        # Add unverified item from vector search as fallback
                        matched_items.append({
                            "type": "product",
                            "id": metadata.get("id"),
                            "name": metadata.get("name"),
                            "category": metadata.get("category", ""),
                            "score": float(match["score"]),
                            "verified": False
                        })
            
            elif doc_type == "engagement":
                engagement_id = metadata.get("engagement_id")
                engagement_type = metadata.get("type", "")  # Default to empty string
                if engagement_id:
                    try:
                        # Get verified engagement data directly from database
                        db_engagement = await db_fetch_one_async(
                            """SELECT e.engagement_id, e.title_en, e.description_en, e.type,
                               e.start_date, e.end_date, e.terms_conditions_en, e.brand_id, b.brand_name_en
                               FROM engagements e
                               LEFT JOIN brands b ON e.brand_id = b.brand_id
                               WHERE e.engagement_id = $1 AND e.unique_property_id = $2""",
                            (engagement_id, state.mall_id)
                        )
                        if db_engagement:
                            # Get type from database or default to empty string
                            engagement_type = db_engagement.get("type", "").lower() if db_engagement.get("type") else "unknown"
                            
                            # Add verified flag and score from vector search
                            item = {
                                "type": engagement_type,
                                "id": db_engagement["engagement_id"],
                                "title": db_engagement["title_en"],
                                "description": db_engagement.get("description_en", ""),
                                "brand_id": db_engagement.get("brand_id"),
                                "store_name": db_engagement.get("brand_name_en", ""),
                                "score": float(match["score"]),
                                "verified": True
                            }
                            
                            # Add date information only if it exists
                            if db_engagement.get("start_date"):
                                item["start_date"] = db_engagement["start_date"]
                            if db_engagement.get("end_date"):
                                item["end_date"] = db_engagement["end_date"]
                                
                                # Calculate if still valid today
                                today = datetime.now().date()
                                end_date = db_engagement["end_date"].date() if isinstance(db_engagement["end_date"], datetime) else db_engagement["end_date"]
                                if end_date:
                                    item["is_valid_today"] = today <= end_date
                            
                            # Add terms for offers
                            if engagement_type == "offer" and db_engagement.get("terms_conditions_en"):
                                item["terms"] = db_engagement["terms_conditions_en"]
                            
                            # Add to matched items and appropriate context in state
                            matched_items.append(item)
                            
                            # Also add to the context in state - safely handle different engagement types
                            if engagement_type == "offer":
                                if not any(o.get("id") == engagement_id for o in state.offers):
                                    state.offers.append(item)
                            elif engagement_type == "event":
                                if not any(e.get("id") == engagement_id for e in state.events):
                                    state.events.append(item)
                            
                    except Exception as e:
                        logger.error(f"Error fetching engagement {engagement_id}: {e}")
                        # Set a default engagement_type if we couldn't get it from db
                        if not engagement_type:
                            engagement_type = "unknown"
                        # Add unverified item from vector search as fallback
                        matched_items.append({
                            "type": engagement_type or "engagement",
                            "id": metadata.get("engagement_id"),
                            "title": metadata.get("title") or metadata.get("name", ""),
                            "score": float(match["score"]),
                            "verified": False
                        })
            
            elif doc_type == "service":
                service_id = metadata.get("id")
                if service_id:
                    try:
                        # Get verified service data directly from database
                        db_service = await db_fetch_one_async(
                            """SELECT s.id, s.name, s.description, s.location, s.opening_hours, s.is_available
                               FROM services s
                               WHERE s.id = $1 AND s.unique_property_id = $2""",
                            (service_id, state.mall_id)
                        )
                        if db_service:
                            # Add verified flag and score from vector search
                            item = {
                                "type": "service",
                                "service_id": db_service["id"],
                                "name": db_service["name"],
                                "description": db_service.get("description", ""),
                                "location": db_service.get("location", ""),
                                "hours": db_service.get("opening_hours", ""),
                                "is_available": db_service.get("is_available", True),
                                "score": float(match["score"]),
                                "verified": True
                            }
                            
                            # Add to matched items and services list
                            matched_items.append(item)
                            
                            # Also add to the services context in state
                            if not any(s.get("service_id") == service_id for s in state.services):
                                state.services.append(item)
                            
                    except Exception as e:
                        logger.error(f"Error fetching service {service_id}: {e}")
                        # Add unverified item from vector search as fallback
                        matched_items.append({
                            "type": "service",
                            "service_id": metadata.get("id"),
                            "name": metadata.get("name"),
                            "score": float(match["score"]),
                            "verified": False
                        })
        
        # Sort matched items by score
        matched_items.sort(key=lambda x: x.get("score", 0), reverse=True)
        embeddings_result = matched_items
        
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
    
    # Get top stores if we don't have any
    if not state.stores:
        try:
            # Get top stores
            top_stores = await db_fetch_all_async(
                """SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en,
                   b.store_phone_number, b.store_email, b.store_website,
                   b.social_instagram, b.pms_unit_codes
                   FROM brands b
                   JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                   WHERE bma.unique_property_id = $1
                   ORDER BY b.anchor_brand DESC
                   LIMIT 5""",
                (state.mall_id,)
            )
            
            # Process and add stores to state
            store_list = []
            for store in top_stores:
                store_data = {
                    "type": "store",
                    "store_id": store["brand_id"],
                    "name": store["brand_name_en"],
                    "category": store.get("category_name", ""),
                    "description": store.get("description_en", ""),
                    "verified": True
                }
                
                # Add contact info only if available
                if store.get("store_phone_number"):
                    store_data["phone"] = store["store_phone_number"]
                if store.get("store_email"):
                    store_data["email"] = store["store_email"]
                if store.get("store_website"):
                    store_data["website"] = store["store_website"]
                if store.get("social_instagram"):
                    store_data["social_instagram"] = store["social_instagram"]
                
                # Process location codes if available
                if store.get("pms_unit_codes"):
                    try:
                        location_codes = store["pms_unit_codes"]
                        if isinstance(location_codes, str):
                            location_codes = json.loads(location_codes)
                        
                        if isinstance(location_codes, dict):
                            if "level" in location_codes:
                                store_data["floor"] = location_codes["level"]
                            if "gate" in location_codes:
                                store_data["gate"] = location_codes["gate"]
                    except Exception as e:
                        logger.error(f"Error processing location codes: {e}")
                
                store_list.append(store_data)
            
            state.stores = store_list
        except Exception as e:
            logger.error(f"Error fetching top stores: {e}")
    
    # Get top offers/events if we don't have any
    if not state.offers and not state.events:
        try:
            # Get top offers/events
            top_engagements = await db_fetch_all_async(
                """SELECT e.engagement_id, e.title_en, e.description_en, e.type, 
                   e.start_date, e.end_date, e.terms_conditions_en, e.brand_id, 
                   b.brand_name_en
                   FROM engagements e
                   LEFT JOIN brands b ON e.brand_id = b.brand_id
                   WHERE e.unique_property_id = $1
                   AND (e.end_date IS NULL OR CAST(e.end_date AS DATE) >= CURRENT_DATE)
                   ORDER BY e.start_date DESC
                   LIMIT 5""",
                (state.mall_id,)
            )
            
            offer_list = []
            event_list = []
            
            for engagement in top_engagements:
                engagement_type = engagement.get("type", "").lower()
                engagement_data = {
                    "id": engagement["engagement_id"],
                    "title": engagement["title_en"],
                    "description": engagement.get("description_en", ""),
                    "brand_id": engagement.get("brand_id"),
                    "store_name": engagement.get("brand_name_en", ""),
                    "verified": True
                }
                
                # Add date information only if it exists
                if engagement.get("start_date"):
                    engagement_data["start_date"] = engagement["start_date"]
                if engagement.get("end_date"):
                    engagement_data["end_date"] = engagement["end_date"]
                    
                    # Calculate if still valid today
                    today = datetime.now().date()
                    end_date = engagement["end_date"].date() if isinstance(engagement["end_date"], datetime) else engagement["end_date"]
                    if end_date:
                        engagement_data["is_valid_today"] = today <= end_date
                
                if engagement_type == "offer":
                    if engagement.get("terms_conditions_en"):
                        engagement_data["terms"] = engagement["terms_conditions_en"]
                    engagement_data["type"] = "offer"
                    offer_list.append(engagement_data)
                elif engagement_type == "event":
                    engagement_data["type"] = "event"
                    event_list.append(engagement_data)
            
            state.offers = offer_list
            state.events = event_list
        except Exception as e:
            logger.error(f"Error fetching top engagements: {e}")
    
    # Combine vector search results with database results
    final_matched_items = embeddings_result.copy()
    
    # Add additional items to matched_items if not already there
    for store in state.stores:
        store_id = store.get("store_id")
        if store_id and not any(item.get("type") == "store" and item.get("store_id") == store_id for item in final_matched_items):
            store_copy = store.copy()
            if "score" not in store_copy:
                store_copy["score"] = 0.5  # Lower score for fallback results
            final_matched_items.append(store_copy)
    
    for product in state.products:
        product_id = product.get("id")
        if product_id and not any(item.get("type") == "product" and item.get("id") == product_id for item in final_matched_items):
            product_copy = product.copy()
            if "score" not in product_copy:
                product_copy["score"] = 0.5  # Lower score for fallback results
            final_matched_items.append(product_copy)
    
    for offer in state.offers:
        offer_id = offer.get("id")
        if offer_id and not any(item.get("type") == "offer" and item.get("id") == offer_id for item in final_matched_items):
            offer_copy = offer.copy()
            if "score" not in offer_copy:
                offer_copy["score"] = 0.5  # Lower score for fallback results
            final_matched_items.append(offer_copy)
    
    for event in state.events:
        event_id = event.get("id")
        if event_id and not any(item.get("type") == "event" and item.get("id") == event_id for item in final_matched_items):
            event_copy = event.copy()
            if "score" not in event_copy:
                event_copy["score"] = 0.5  # Lower score for fallback results
            final_matched_items.append(event_copy)
    
    for service in state.services:
        service_id = service.get("service_id")
        if service_id and not any(item.get("type") == "service" and item.get("service_id") == service_id for item in final_matched_items):
            service_copy = service.copy()
            if "score" not in service_copy:
                service_copy["score"] = 0.5  # Lower score for fallback results
            final_matched_items.append(service_copy)
    
    # Sort final matched items by score
    final_matched_items.sort(key=lambda x: x.get("score", 0), reverse=True)
    state.matched_items = final_matched_items[:20]  # Limit to top 20 results
    
    logger.info(f"General context retrieval complete, found {len(state.matched_items)} matched items")
    return state

def route_after_intent_classification(state: CustomerState):
    """Routes to the appropriate context retrieval node based on classified intent"""
    intent_to_node_mapping = {
        "store_finder": "store_context_retrieval",
        "product_finder": "product_context_retrieval",
        "offer_finder": "engagement_context_retrieval", 
        "event_finder": "engagement_context_retrieval",
        "service_finder": "service_context_retrieval",
        "mall_info": "mall_context_retrieval",
        "general_chat": "general_chat_retrieval",
    }
    
    # Log the intent routing decision
    logger.info(f"Routing intent '{state.intent}' to appropriate node")
    
    # Enhanced routing logic based on query content and context
    query_lower = state.query.lower()
    
    # Look for intent-related keywords in the query
    store_keywords = ["store", "shop", "restaurant", "cafe", "buy", "purchase", "eat", "food"]
    product_keywords = ["product", "item", "thing", "stuff", "goods", "merchandise", "apparel"]
    event_keywords = ["event", "events", "happening", "activities", "shows", "performance", "exhibition", "concert", "workshop"]
    offer_keywords = ["offer", "offers", "deal", "deals", "discount", "promotion", "sale", "coupon", "bargain", "special"]
    
    # Try to detect multiple intents in a single query
    has_store_intent = any(keyword in query_lower for keyword in store_keywords)
    has_product_intent = any(keyword in query_lower for keyword in product_keywords)
    has_event_intent = any(keyword in query_lower for keyword in event_keywords)
    has_offer_intent = any(keyword in query_lower for keyword in offer_keywords)
    
    # Count how many different intents are present
    intent_count = sum([has_store_intent, has_product_intent, has_event_intent, has_offer_intent])
    has_multiple_intents = intent_count > 1
    
    if has_multiple_intents:
        logger.info(f"Detected mixed query with multiple intents: {intent_count}")
        # For mixed queries, use the general_context_retrieval which fetches diverse data types
        return "general_context_retrieval"
    
    # Handle compound queries about events AND offers
    if (has_event_intent and has_offer_intent) or "promotion" in query_lower:
        logger.info("Detected compound query about events and offers")
        state.intent = "event_finder"  # This will make engagement_context_retrieval fetch both types
        return "engagement_context_retrieval"
    
    # Handle food-related queries more directly by routing to store_context
    food_keywords = ["food", "restaurant", "eat", "hungry", "lunch", "dinner", "breakfast", "meal", "burger", "pizza", "chicken", "menu", "coffee"]
    if any(keyword in query_lower for keyword in food_keywords) and state.intent == "general_chat":
        logger.info("Detected food-related query, enhancing routing to store_context_retrieval")
        # Set the type_preference to guide the store retrieval
        state.type_preference = "restaurant"
        return "store_context_retrieval"
    
    # Direct shopping queries to store context
    shopping_keywords = ["shop", "buy", "purchase", "clothes", "clothing", "fashion", "wear", "dress", "shirt", "pants", "shoes", "shopping"]
    if any(keyword in query_lower for keyword in shopping_keywords) and state.intent == "general_chat":
        logger.info("Detected shopping-related query, enhancing routing to store_context_retrieval")
        return "store_context_retrieval"
    
    # Direct event queries
    if has_event_intent and state.intent == "general_chat":
        logger.info("Detected event-related query, enhancing routing to engagement_context_retrieval")
        # Set internal parameters to ensure the engagement context focuses on events
        state.intent = "event_finder"
        return "engagement_context_retrieval"
    
    # Direct offer/promotion queries
    if has_offer_intent and state.intent == "general_chat":
        logger.info("Detected offer-related query, enhancing routing to engagement_context_retrieval")
        # Set internal parameters to ensure the engagement context focuses on offers
        state.intent = "offer_finder"
        return "engagement_context_retrieval"
    
    # Direct service-related queries
    service_keywords = ["service", "services", "facility", "facilities", "amenity", "amenities", 
                        "restroom", "bathroom", "parking", "wifi", "information desk", "help desk", 
                        "prayer room", "nursing room", "lost and found", "customer service"]
    if any(keyword in query_lower for keyword in service_keywords) and state.intent == "general_chat":
        logger.info("Detected service-related query, enhancing routing to service_context_retrieval")
        # Set internal parameters to ensure the service context is used
        state.intent = "service_finder"
        return "service_context_retrieval"
    
    # Check if we have store_name or product_name set but intent doesn't match
    if state.store_name and state.intent != "store_finder":
        logger.info(f"Store name '{state.store_name}' is set but intent is {state.intent}, routing to store_context_retrieval")
        state.intent = "store_finder"
        return "store_context_retrieval"
    
    if state.product_name and state.intent != "product_finder":
        logger.info(f"Product name '{state.product_name}' is set but intent is {state.intent}, routing to product_context_retrieval")
        state.intent = "product_finder" 
        return "product_context_retrieval"
    
    # Detect follow-up questions that might need context from a previous exchange
    followup_indicators = ["it", "they", "that", "those", "this", "there", "these", "their", "yes", "no", "yeah", "sure", "okay", "ok"]
    short_query = len(query_lower.split()) <= 3  # Very short queries are often follow-ups
    
    is_possible_followup = (any(word in query_lower.split() for word in followup_indicators) or short_query) and state.previous_intent
    
    if is_possible_followup and state.intent == "general_chat":
        logger.info(f"Detected possible follow-up to previous intent: {state.previous_intent}")
        # Try to maintain the previous intent routing for continuity
        if state.previous_intent in intent_to_node_mapping:
            node = intent_to_node_mapping[state.previous_intent]
            state.intent = state.previous_intent
            logger.info(f"Routing follow-up to previous node: {node}")
            return node
    
    # If we have a high confidence specific intent, route to the specialized node
    if state.intent in intent_to_node_mapping and state.confidence and state.confidence > 60:
        node = intent_to_node_mapping[state.intent]
        logger.info(f"High confidence routing to specialized node: {node}")
        return node
    
    # If we have a specific intent with medium confidence, route to the appropriate node
    if state.intent in intent_to_node_mapping:
        node = intent_to_node_mapping[state.intent]
        logger.info(f"Routing to specialized node: {node}")
        return node
    
    # Fallback for truly general queries
    logger.info("No specific intent match, falling back to general_context_retrieval")
    return "general_context_retrieval"

# Workflow
customer_workflow = StateGraph(CustomerState)

# Add all nodes to the workflow
customer_workflow.add_node("classify_intent", classify_intent)
customer_workflow.add_node("product_context_retrieval", product_context_retrieval)
customer_workflow.add_node("store_context_retrieval", store_context_retrieval)
customer_workflow.add_node("engagement_context_retrieval", engagement_context_retrieval)
customer_workflow.add_node("service_context_retrieval", service_context_retrieval)
customer_workflow.add_node("mall_context_retrieval", mall_context_retrieval)
customer_workflow.add_node("general_context_retrieval", general_context_retrieval)
customer_workflow.add_node("fetch_loyalty_data", fetch_loyalty_data)
customer_workflow.add_node("respond", generate_response)
customer_workflow.add_node("general_chat_retrieval", general_chat_retrieval)

# Set entry point
customer_workflow.set_entry_point("classify_intent")

# Add conditional edges for routing after intent classification
customer_workflow.add_conditional_edges(
    "classify_intent",
    route_after_intent_classification,
    {
        "product_context_retrieval": "product_context_retrieval",
        "store_context_retrieval": "store_context_retrieval",
        "engagement_context_retrieval": "engagement_context_retrieval",
        "service_context_retrieval": "service_context_retrieval",
        "mall_context_retrieval": "mall_context_retrieval",
        "general_context_retrieval": "general_context_retrieval",
        "fetch_loyalty_data": "fetch_loyalty_data",
        "general_chat_retrieval": "general_chat_retrieval"
    }
)

# Connect all context retrieval nodes to respond
customer_workflow.add_edge("product_context_retrieval", "respond")
customer_workflow.add_edge("store_context_retrieval", "respond")
customer_workflow.add_edge("engagement_context_retrieval", "respond")
customer_workflow.add_edge("service_context_retrieval", "respond")
customer_workflow.add_edge("mall_context_retrieval", "respond")
customer_workflow.add_edge("general_context_retrieval", "respond")
customer_workflow.add_edge("fetch_loyalty_data", "respond")
customer_workflow.add_edge("general_chat_retrieval", "respond")
customer_workflow.add_edge("respond", END)

# Compile the graph
customer_graph = customer_workflow.compile()

# Add new helper function for hybrid data retrieval after the async def general_context_retrieval function

async def verify_and_enrich_data(state: CustomerState) -> CustomerState:
    """
    Verify and enrich data fetched from vector search with direct PostgreSQL queries.
    This ensures we're only providing information that actually exists in the database.
    """
    logger.info(f"Verifying and enriching data for query: {state.query}")
    
    # Verify store data
    if state.stores:
        verified_stores = []
        for store in state.stores:
            store_id = store.get("store_id")
            if store_id:
                try:
                    # Direct query to get accurate store information
                    store_details = await db_fetch_one_async(
                        """SELECT b.brand_id, b.brand_name_en, b.brand_name_ar, b.category_name, 
                           b.description_en, b.store_phone_number, b.store_email, b.store_website,
                           b.social_instagram, b.social_facebook, b.pms_unit_codes,
                           bma.unique_property_id
                           FROM brands b 
                           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                           WHERE b.brand_id = $1 AND bma.unique_property_id = $2""",
                        (store_id, state.mall_id)
                    )
                    
                    if store_details:
                        # Update with verified information
                        verified_store = {
                            "store_id": store_details["brand_id"],
                            "name": store_details["brand_name_en"],
                            "name_ar": store_details.get("brand_name_ar"),
                            "category": store_details.get("category_name", ""),
                            "description": store_details.get("description_en", ""),
                            "verified": True
                        }
                        
                        # Add contact information only if it exists
                        if store_details.get("store_phone_number"):
                            verified_store["phone"] = store_details["store_phone_number"]
                        if store_details.get("store_email"):
                            verified_store["email"] = store_details["store_email"]
                        if store_details.get("store_website"):
                            verified_store["website"] = store_details["store_website"]
                        if store_details.get("social_instagram"):
                            verified_store["social_instagram"] = store_details["social_instagram"]
                        if store_details.get("social_facebook"):
                            verified_store["social_facebook"] = store_details["social_facebook"]
                        
                        # Process location codes if available
                        if store_details.get("pms_unit_codes"):
                            try:
                                location_codes = store_details["pms_unit_codes"]
                                if isinstance(location_codes, str):
                                    location_codes = json.loads(location_codes)
                                
                                if isinstance(location_codes, dict):
                                    if "level" in location_codes:
                                        verified_store["floor"] = location_codes["level"]
                                    if "gate" in location_codes:
                                        verified_store["gate"] = location_codes["gate"]
                            except Exception as e:
                                logger.error(f"Error processing location codes: {e}")
                        
                        verified_stores.append(verified_store)
                        logger.info(f"Verified store: {verified_store['name']}")
                    else:
                        # Store exists in vector DB but not in PostgreSQL for this mall
                        logger.warning(f"Store ID {store_id} not found in PostgreSQL for mall {state.mall_id}")
                except Exception as e:
                    logger.error(f"Error verifying store {store_id}: {e}")
        
        state.stores = verified_stores
    
    # Verify product data
    if state.products:
        verified_products = []
        for product in state.products:
            product_id = product.get("id")
            if product_id:
                try:
                    # Direct query to get accurate product information
                    product_details = await db_fetch_one_async(
                        """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
                           p.is_featured, p.in_stock, p.image_url, b.brand_name_en
                           FROM products p
                           JOIN brands b ON p.brand_id = b.brand_id
                           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                           WHERE p.id = $1 AND bma.unique_property_id = $2""",
                        (product_id, state.mall_id)
                    )
                    
                    if product_details:
                        # Update with verified information
                        verified_product = {
                            "id": product_details["id"],
                            "name": product_details["name"],
                            "description": product_details.get("description", ""),
                            "brand_id": product_details.get("brand_id"),
                            "category": product_details.get("category", ""),
                            "store_name": product_details.get("brand_name_en", ""),
                            "verified": True
                        }
                        
                        # Add price only if it exists
                        if product_details.get("price") is not None:
                            verified_product["price"] = float(product_details["price"])
                        
                        # Add availability information only if it exists
                        if product_details.get("in_stock") is not None:
                            verified_product["in_stock"] = product_details["in_stock"]
                        
                        # Get store location for this product
                        if product_details.get("brand_id"):
                            store_location = await db_fetch_one_async(
                                """SELECT b.pms_unit_codes
                                   FROM brands b 
                                   JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                                   WHERE b.brand_id = $1 AND bma.unique_property_id = $2""",
                                (product_details["brand_id"], state.mall_id)
                            )
                            
                            if store_location and store_location.get("pms_unit_codes"):
                                try:
                                    location_codes = store_location["pms_unit_codes"]
                                    if isinstance(location_codes, str):
                                        location_codes = json.loads(location_codes)
                                    
                                    if isinstance(location_codes, dict):
                                        location_info = []
                                        if "level" in location_codes:
                                            location_info.append(f"Level {location_codes['level']}")
                                        if "gate" in location_codes:
                                            location_info.append(f"Gate {location_codes['gate']}")
                                        
                                        if location_info:
                                            verified_product["store_location"] = ", ".join(location_info)
                                except Exception as e:
                                    logger.error(f"Error processing product location: {e}")
                        
                        verified_products.append(verified_product)
                        logger.info(f"Verified product: {verified_product['name']}")
                    else:
                        # Product exists in vector DB but not in PostgreSQL for this mall
                        logger.warning(f"Product ID {product_id} not found in PostgreSQL for mall {state.mall_id}")
                except Exception as e:
                    logger.error(f"Error verifying product {product_id}: {e}")
        
        state.products = verified_products
    
    # Verify offer data
    if state.offers:
        verified_offers = []
        for offer in state.offers:
            offer_id = offer.get("id")
            if offer_id:
                try:
                    # Direct query to get accurate offer information
                    offer_details = await db_fetch_one_async(
                        """SELECT e.engagement_id, e.title_en, e.description_en, e.start_date, e.end_date,
                           e.terms_conditions_en, e.brand_id, b.brand_name_en
                           FROM engagements e
                           LEFT JOIN brands b ON e.brand_id = b.brand_id
                           WHERE e.engagement_id = $1 AND e.unique_property_id = $2 AND e.type = 'offer'""",
                        (offer_id, state.mall_id)
                    )
                    
                    if offer_details:
                        # Update with verified information
                        verified_offer = {
                            "id": offer_details["engagement_id"],
                            "title": offer_details["title_en"],
                            "description": offer_details.get("description_en", ""),
                            "terms": offer_details.get("terms_conditions_en", ""),
                            "brand_id": offer_details.get("brand_id"),
                            "store_name": offer_details.get("brand_name_en", ""),
                            "verified": True
                        }
                        
                        # Add date information only if it exists
                        if offer_details.get("start_date"):
                            verified_offer["start_date"] = offer_details["start_date"]
                        if offer_details.get("end_date"):
                            verified_offer["end_date"] = offer_details["end_date"]
                            
                            # Calculate if offer is valid today
                            today = datetime.now().date()
                            end_date = offer_details["end_date"].date() if isinstance(offer_details["end_date"], datetime) else offer_details["end_date"]
                            verified_offer["is_valid_today"] = today <= end_date
                        
                        verified_offers.append(verified_offer)
                        logger.info(f"Verified offer: {verified_offer['title']}")
                    else:
                        # Offer exists in vector DB but not in PostgreSQL for this mall
                        logger.warning(f"Offer ID {offer_id} not found in PostgreSQL for mall {state.mall_id}")
                except Exception as e:
                    logger.error(f"Error verifying offer {offer_id}: {e}")
        
        state.offers = verified_offers
    
    # Verify event data
    if state.events:
        verified_events = []
        for event in state.events:
            event_id = event.get("id")
            if event_id:
                try:
                    # Direct query to get accurate event information
                    event_details = await db_fetch_one_async(
                        """SELECT e.engagement_id, e.title_en, e.description_en, e.start_date, e.end_date,
                           e.brand_id, b.brand_name_en, e.location_en
                           FROM engagements e
                           LEFT JOIN brands b ON e.brand_id = b.brand_id
                           WHERE e.engagement_id = $1 AND e.unique_property_id = $2 AND e.type = 'events'""",
                        (event_id, state.mall_id)
                    )
                    
                    if event_details:
                        # Update with verified information
                        verified_event = {
                            "id": event_details["engagement_id"],
                            "title": event_details["title_en"],
                            "description": event_details.get("description_en", ""),
                            "brand_id": event_details.get("brand_id"),
                            "store_name": event_details.get("brand_name_en", ""),
                            "location": event_details.get("location_en", ""),
                            "verified": True
                        }
                        
                        # Add date information only if it exists
                        if event_details.get("start_date"):
                            verified_event["start_date"] = event_details["start_date"]
                        if event_details.get("end_date"):
                            verified_event["end_date"] = event_details["end_date"]
                            
                            # Calculate if event is happening today
                            today = datetime.now().date()
                            start_date = event_details["start_date"].date() if isinstance(event_details["start_date"], datetime) else event_details["start_date"]
                            end_date = event_details["end_date"].date() if isinstance(event_details["end_date"], datetime) else event_details["end_date"]
                            
                            if start_date and end_date:
                                verified_event["is_today"] = start_date <= today <= end_date
                        
                        verified_events.append(verified_event)
                        logger.info(f"Verified event: {verified_event['title']}")
                    else:
                        # Event exists in vector DB but not in PostgreSQL for this mall
                        logger.warning(f"Event ID {event_id} not found in PostgreSQL for mall {state.mall_id}")
                except Exception as e:
                    logger.error(f"Error verifying event {event_id}: {e}")
        
        state.events = verified_events
    
    # Verify service data
    if state.services:
        verified_services = []
        for service in state.services:
            service_id = service.get("service_id")
            if service_id:
                try:
                    # Direct query to get accurate service information
                    service_details = await db_fetch_one_async(
                        """SELECT s.id, s.name, s.description, s.location, s.opening_hours, s.is_available
                           FROM services s
                           WHERE s.id = $1 AND s.unique_property_id = $2""",
                        (service_id, state.mall_id)
                    )
                    
                    if service_details:
                        # Update with verified information
                        verified_service = {
                            "service_id": service_details["id"],
                            "name": service_details["name"],
                            "description": service_details.get("description", ""),
                            "location": service_details.get("location", ""),
                            "hours": service_details.get("opening_hours", ""),
                            "is_available": service_details.get("is_available", True),
                            "verified": True
                        }
                        
                        verified_services.append(verified_service)
                        logger.info(f"Verified service: {verified_service['name']}")
                    else:
                        # Service exists in vector DB but not in PostgreSQL for this mall
                        logger.warning(f"Service ID {service_id} not found in PostgreSQL for mall {state.mall_id}")
                except Exception as e:
                    logger.error(f"Error verifying service {service_id}: {e}")
        
        state.services = verified_services

    # Also update matched_items with verified information
    if state.matched_items:
        for i, item in enumerate(state.matched_items):
            item_type = item.get("type")
            
            if item_type == "store":
                store_id = item.get("store_id")
                if store_id:
                    # Find this store in our verified stores
                    for store in state.stores:
                        if store.get("store_id") == store_id:
                            state.matched_items[i] = {**item, **store}
                            break
            
            elif item_type == "product":
                product_id = item.get("id")
                if product_id:
                    # Find this product in our verified products
                    for product in state.products:
                        if product.get("id") == product_id:
                            state.matched_items[i] = {**item, **product}
                            break
            
            elif item_type == "offer":
                offer_id = item.get("id")
                if offer_id:
                    # Find this offer in our verified offers
                    for offer in state.offers:
                        if offer.get("id") == offer_id:
                            state.matched_items[i] = {**item, **offer}
                            break
            
            elif item_type == "event":
                event_id = item.get("id")
                if event_id:
                    # Find this event in our verified events
                    for event in state.events:
                        if event.get("id") == event_id:
                            state.matched_items[i] = {**item, **event}
                            break
            
            elif item_type == "service":
                service_id = item.get("service_id")
                if service_id:
                    # Find this service in our verified services
                    for service in state.services:
                        if service.get("service_id") == service_id:
                            state.matched_items[i] = {**item, **service}
                            break
    
    return state

# Now update the route function to call this verification step
def route_after_intent_classification(state: CustomerState):
    # Skip straight to response generation for direct responses
    if state.direct_response:
        return generate_response
        
    # Skip to response for loyalty (not implemented yet)
    if state.intent == "loyalty":
        return fetch_loyalty_data
    
    # Choose appropriate retrieval method based on intent
    if state.intent == "product_finder":
        return [product_context_retrieval, verify_and_enrich_data, generate_response]
    elif state.intent == "store_finder":
        return [store_context_retrieval, verify_and_enrich_data, generate_response]
    elif state.intent in ["offer_finder", "event_finder"]:
        return [engagement_context_retrieval, verify_and_enrich_data, generate_response]
    elif state.intent == "service_finder":
        return [service_context_retrieval, verify_and_enrich_data, generate_response]
    elif state.intent == "mall_info":
        return [mall_context_retrieval, verify_and_enrich_data, generate_response]
    elif state.intent == "general_chat":
        # For general chat, we use a broader retrieval approach
        return [general_chat_retrieval, verify_and_enrich_data, generate_response]
    else:
        # Default to general context retrieval
        return [general_context_retrieval, verify_and_enrich_data, generate_response]