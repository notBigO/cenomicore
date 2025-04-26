from typing import Optional, List, Dict, Any
from pydantic import BaseModel as PydanticBaseModel
from langchain_core.prompts import PromptTemplate
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

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("cenomicore")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# LLM setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Knowledge graph
knowledge_graph = nx.Graph()

async def populate_knowledge_graph():
    brands = await db_fetch_all_async(
        """SELECT b.brand_id, b.brand_name_en, b.brand_name_ar, b.category_name, 
           b.description_en, b.pms_unit_codes, bma.unique_property_id 
           FROM brands b 
           JOIN brand_mall_association bma ON b.brand_id = bma.brand_id"""
    )
    products = await db_fetch_all_async(
        """SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, 
           p.is_featured, p.in_stock, b.brand_name_en
           FROM products p
           JOIN brands b ON p.brand_id = b.brand_id"""
    )
    engagements = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.type, e.brand_id, 
           e.start_date, e.end_date, e.unique_property_id
           FROM engagements e"""
    )
    knowledge_graph.clear()
    for brand in brands:
        knowledge_graph.add_node(
            f"brand_{brand['brand_id']}",
            type="store",
            name=brand["brand_name_en"],
            name_ar=brand.get("brand_name_ar", ""),
            category=brand.get("category_name", ""),
            description=brand.get("description_en", ""),
            mall_id=brand["unique_property_id"],
            location=brand.get("pms_unit_codes", {})
        )
    for product in products:
        knowledge_graph.add_node(
            f"product_{product['id']}",
            type="product",
            name=product["name"],
            description=product.get("description", ""),
            price=float(product["price"]) if product["price"] else None,
            category=product.get("category", ""),
            brand_name=product.get("brand_name_en", ""),
            in_stock=product.get("in_stock", True),
            is_featured=product.get("is_featured", False)
        )
        knowledge_graph.add_edge(f"brand_{product['brand_id']}", f"product_{product['id']}")
    for engagement in engagements:
        engagement_type = engagement.get("type", "").lower()
        knowledge_graph.add_node(
            f"engagement_{engagement['engagement_id']}",
            type=engagement_type,
            title=engagement.get("title_en", ""),
            description=engagement.get("description_en", ""),
            start_date=engagement.get("start_date", ""),
            end_date=engagement.get("end_date", ""),
            mall_id=engagement.get("unique_property_id")
        )
        if engagement["brand_id"]:
            knowledge_graph.add_edge(f"brand_{engagement['brand_id']}", f"engagement_{engagement['engagement_id']}")

# Intent Classification Prompt
intent_classification_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    You are CenomiAI, a mall assistant. Parse this query to determine the user's intent, considering the conversation history.

    Valid entities: store, offer, product, event, service, amenity, loyalty
    Valid actions: info, navigate, recommend, list, balance, programs

    Query: "{query}"

    Previous conversation:
    {conversation_history}

    Instructions:
    - Use the conversation history to resolve vague terms (e.g., "it", "that") to prior entities.
    - For follow-up questions (e.g., "Where is it?"), link to the most recent entity mentioned.
    - For broad queries (e.g., "What's good here?"), assume 'recommend' or 'list'.
    - Extract specific details into collected_data:
      - If the query mentions a specific product (e.g., "shoes", "watch"), set "name" to that product.
      - If the query mentions a category (e.g., "electronics", "footwear"), set "category" to that.
    - If the query is about buying or looking for something, assume it's a product query.

    Return JSON:
    - entity_type: What they're asking about (store, offer, product, etc.)
    - action: What they want (info, navigate, recommend, list, etc.)
    - collected_data: Details provided (e.g., "name": "shoes", "category": "footwear")

    Examples:
    ```json
    {{"entity_type": "product", "action": "recommend", "collected_data": {{"name": "shoes"}}}}
    {{"entity_type": "store", "action": "list", "collected_data": {{"category": "electronics"}}}}
    {{"entity_type": "offer", "action": "info", "collected_data": {{"name": "Black Friday Sale"}}}}
    ```
    """
)
intent_chain = intent_classification_prompt | llm | StrOutputParser()

# State Definition
class CustomerState(PydanticBaseModel):
    query: str
    user_id: Optional[str] = None
    language: str
    conversation_id: str
    conversation_history: List[Dict[str, str]] = []
    response: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None
    entity_type: Optional[str] = None
    action: Optional[str] = None
    collected_data: Optional[Dict[str, Any]] = None
    mall_id: Optional[int] = None
    needs_type_follow_up: bool = False

# Intent Classification Node
async def classify_intent(state: CustomerState) -> CustomerState:
    formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-6:]]) if state.conversation_history else "No prior conversation."
    try:
        intent_result = await intent_chain.ainvoke({
            "query": state.query,
            "conversation_history": formatted_history
        })
        cleaned_result = intent_result.strip().replace("```json", "").replace("```", "").strip()
        intent_json = json.loads(cleaned_result)
        state.entity_type = intent_json["entity_type"]
        state.action = intent_json["action"]
        state.collected_data = intent_json["collected_data"]
        state.intent = f"{state.entity_type}_{state.action}"
        state.needs_type_follow_up = state.entity_type in ["store", "product", "service"] and not state.collected_data.get("type_preference")
        logger.info(f"Intent classification result: {intent_json}")
    except json.JSONDecodeError:
        state.intent = "other_info"
        state.response = "I'm not sure what you mean 😅. Could you clarify?"
    except Exception as e:
        logger.error(f"Error in classify_intent: {str(e)}")
        state.intent = "other_info"
        state.response = f"Sorry, something went wrong: {str(e)}"
    return state

# Context Retrieval Nodes
async def retrieve_store_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Please select a mall first!"
        return state
    
    # Vector search if we have a name
    stores = []
    if state.collected_data and "name" in state.collected_data:
        store_name = state.collected_data.get("name", "").lower()
        if store_name:
            # Vector search in Pinecone
            try:
                query_embedding = embeddings.embed_query(store_name)
                vector_results = index.query(
                    vector=query_embedding,
                    filter={"type": "store", "mall_id": str(state.mall_id)},
                    top_k=5,
                    include_metadata=True
                )
                
                # Extract brand_ids from vector results
                brand_ids = []
                for match in vector_results.matches:
                    if match.score > 0.7:  # Similarity threshold
                        metadata = match.metadata
                        if "brand_id" in metadata:
                            brand_ids.append(metadata["brand_id"])
                
                # If we found any good matches, query the database with these IDs
                if brand_ids:
                    brand_id_list = ",".join([str(id) for id in brand_ids])
                    vector_db_query = f"""
                        SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, b.pms_unit_codes
                        FROM brands b 
                        JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
                        WHERE bma.unique_property_id = $1 AND b.brand_id IN ({brand_id_list})
                    """
                    vector_stores = await db_fetch_all_async(vector_db_query, (state.mall_id,))
                    stores.extend([dict(store) for store in vector_stores])
                    logger.info(f"Vector search found {len(vector_stores)} stores for '{store_name}'")
            except Exception as e:
                logger.error(f"Error in vector search for stores: {str(e)}")
    
    # Traditional database search as fallback or additional results
    if not stores and state.collected_data:
        store_name = state.collected_data.get("name", "").lower() if state.collected_data else ""
        query = """
            SELECT b.brand_id, b.brand_name_en, b.category_name, b.description_en, b.pms_unit_codes
            FROM brands b 
            JOIN brand_mall_association bma ON b.brand_id = bma.brand_id 
            WHERE bma.unique_property_id = $1
        """
        params = [state.mall_id]
        if store_name:
            query += " AND LOWER(b.brand_name_en) LIKE $2"
            params.append(f"%{store_name}%")
        db_stores = await db_fetch_all_async(query, tuple(params))
        stores.extend([dict(store) for store in db_stores if not any(s["brand_id"] == store["brand_id"] for s in stores)])
    
    # Limit to top 5 stores
    state.context_data = {"stores": stores[:5]}
    return state

async def retrieve_product_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Please select a mall first!"
        return state
    
    # Vector search first
    products = []
    if state.collected_data:
        product_name = state.collected_data.get("name", "").lower() if state.collected_data else ""
        product_category = state.collected_data.get("category", "").lower() if state.collected_data else ""
        
        search_text = product_name
        if product_category and not product_name:
            search_text = product_category
        elif product_category and product_name:
            search_text = f"{product_name} {product_category}"
            
        if search_text:
            try:
                query_embedding = embeddings.embed_query(search_text)
                vector_results = index.query(
                    vector=query_embedding,
                    filter={"type": "product", "mall_id": str(state.mall_id)},
                    top_k=5,
                    include_metadata=True
                )
                
                # Extract product IDs from vector results
                product_ids = []
                for match in vector_results.matches:
                    if match.score > 0.65:  # Similarity threshold
                        metadata = match.metadata
                        if "id" in metadata:
                            product_ids.append(metadata["id"])
                
                # If we found good matches, query the database with these IDs
                if product_ids:
                    product_id_list = ",".join([str(id) for id in product_ids])
                    vector_db_query = f"""
                        SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, p.in_stock, 
                              b.brand_name_en, b.pms_unit_codes
                        FROM products p
                        JOIN brands b ON p.brand_id = b.brand_id
                        JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
                        WHERE bma.unique_property_id = $1 AND p.id IN ({product_id_list})
                    """
                    vector_products = await db_fetch_all_async(vector_db_query, (state.mall_id,))
                    products.extend([dict(product) for product in vector_products])
                    logger.info(f"Vector search found {len(vector_products)} products for '{search_text}'")
            except Exception as e:
                logger.error(f"Error in vector search for products: {str(e)}")
    
    # Traditional database search as fallback or for additional results
    if not products and state.collected_data:
        product_name = state.collected_data.get("name", "").lower() if state.collected_data else ""
        product_category = state.collected_data.get("category", "").lower() if state.collected_data else ""
        
        query = """
            SELECT p.id, p.name, p.description, p.price, p.category, p.brand_id, p.in_stock, 
                  b.brand_name_en, b.pms_unit_codes
            FROM products p
            JOIN brands b ON p.brand_id = b.brand_id
            JOIN brand_mall_association bma ON b.brand_id = bma.brand_id
            WHERE bma.unique_property_id = $1
        """
        params = [state.mall_id]
        if product_name and product_category:
            query += " AND (LOWER(p.name) LIKE $2 OR LOWER(p.category) = $3)"
            params.extend([f"%{product_name}%", product_category])
        elif product_name:
            query += " AND LOWER(p.name) LIKE $2"
            params.append(f"%{product_name}%")
        elif product_category:
            query += " AND LOWER(p.category) = $2"
            params.append(product_category)
        
        db_products = await db_fetch_all_async(query, tuple(params))
        products.extend([dict(product) for product in db_products if not any(p["id"] == product["id"] for p in products)])
    
    state.context_data = {"products": products[:5]}  # Limit to 5
    if not products:
        logger.info(f"No products found for query: {state.query}")
    else:
        logger.info(f"Retrieved {len(products)} products for query: {state.query}")
    return state

async def retrieve_offer_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Please select a mall first!"
        return state
    
    # Vector search for offers
    offers = []
    search_text = ""
    if state.collected_data:
        offer_name = state.collected_data.get("name", "").lower() if state.collected_data else ""
        if offer_name:
            search_text = offer_name
            try:
                query_embedding = embeddings.embed_query(search_text)
                vector_results = index.query(
                    vector=query_embedding,
                    filter={"type": "engagement", "mall_id": str(state.mall_id)},
                    top_k=5,
                    include_metadata=True
                )
                
                # Extract engagement IDs from vector results
                engagement_ids = []
                for match in vector_results.matches:
                    if match.score > 0.7:  # Similarity threshold
                        metadata = match.metadata
                        if "engagement_id" in metadata and metadata.get("type") == "offer":
                            engagement_ids.append(metadata["engagement_id"])
                
                # If we found good matches, query the database with these IDs
                if engagement_ids:
                    engagement_id_list = ",".join([str(id) for id in engagement_ids])
                    vector_db_query = f"""
                        SELECT e.engagement_id, e.title_en, e.description_en, e.start_date, e.end_date, 
                              b.brand_name_en
                        FROM engagements e
                        LEFT JOIN brands b ON e.brand_id = b.brand_id
                        WHERE e.unique_property_id = $1 AND LOWER(e.type) = 'offer' 
                        AND e.engagement_id IN ({engagement_id_list})
                    """
                    vector_offers = await db_fetch_all_async(vector_db_query, (state.mall_id,))
                    offers.extend([dict(offer) for offer in vector_offers])
                    logger.info(f"Vector search found {len(vector_offers)} offers for '{search_text}'")
            except Exception as e:
                logger.error(f"Error in vector search for offers: {str(e)}")
    
    # Traditional database search for offers as fallback or additional results
    if not offers or not search_text:
        db_offers = await db_fetch_all_async(
            """SELECT e.engagement_id, e.title_en, e.description_en, e.start_date, e.end_date, 
                     b.brand_name_en
              FROM engagements e
              LEFT JOIN brands b ON e.brand_id = b.brand_id
              WHERE e.unique_property_id = $1 AND LOWER(e.type) = 'offer'""",
            (state.mall_id,)
        )
        offers.extend([dict(offer) for offer in db_offers if not any(o["engagement_id"] == offer["engagement_id"] for o in offers)])
    
    state.context_data = {"offers": offers[:5]}  # Limit to 5
    return state

async def retrieve_event_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Please select a mall first!"
        return state
    
    # Vector search for events
    events = []
    if state.collected_data:
        event_name = state.collected_data.get("name", "").lower() if state.collected_data else ""
        if event_name:
            try:
                query_embedding = embeddings.embed_query(event_name)
                vector_results = index.query(
                    vector=query_embedding,
                    filter={"type": "engagement", "mall_id": str(state.mall_id)},
                    top_k=5,
                    include_metadata=True
                )
                
                # Extract engagement IDs from vector results
                engagement_ids = []
                for match in vector_results.matches:
                    if match.score > 0.7:  # Similarity threshold
                        metadata = match.metadata
                        if "engagement_id" in metadata and metadata.get("type") == "event":
                            engagement_ids.append(metadata["engagement_id"])
                
                # If we found good matches, query the database with these IDs
                if engagement_ids:
                    engagement_id_list = ",".join([str(id) for id in engagement_ids])
                    vector_db_query = f"""
                        SELECT e.engagement_id, e.title_en, e.description_en, e.start_date, e.end_date, 
                              b.brand_name_en
                        FROM engagements e
                        LEFT JOIN brands b ON e.brand_id = b.brand_id
                        WHERE e.unique_property_id = $1 AND LOWER(e.type) = 'event' 
                        AND e.engagement_id IN ({engagement_id_list})
                    """
                    vector_events = await db_fetch_all_async(vector_db_query, (state.mall_id,))
                    events.extend([dict(event) for event in vector_events])
                    logger.info(f"Vector search found {len(vector_events)} events for '{event_name}'")
            except Exception as e:
                logger.error(f"Error in vector search for events: {str(e)}")
    
    # Traditional database search for events as fallback or for additional results
    if not events and state.collected_data:
        event_name = state.collected_data.get("name", "").lower() if state.collected_data else ""
        query = """
            SELECT e.engagement_id, e.title_en, e.description_en, e.start_date, e.end_date, 
                  b.brand_name_en
            FROM engagements e
            LEFT JOIN brands b ON e.brand_id = b.brand_id
            WHERE e.unique_property_id = $1 AND LOWER(e.type) = 'event'
        """
        params = [state.mall_id]
        if event_name:
            query += " AND LOWER(e.title_en) LIKE $2"
            params.append(f"%{event_name}%")
        db_events = await db_fetch_all_async(query, tuple(params))
        events.extend([dict(event) for event in db_events if not any(e["engagement_id"] == event["engagement_id"] for e in events)])
    
    state.context_data = {"events": events[:5]}  # Limit to 5
    return state

# Response Generation Prompts
store_response_prompt = PromptTemplate(
    input_variables=["store_data", "query", "lang", "collected_data"],
    template="""
    You are CenomiAI, a friendly mall assistant. Answer the user's query about stores in {lang}, considering the collected data.

    Store Data: {store_data}
    Collected Data: {collected_data}
    Query: "{query}"

    Keep it short, casual, and helpful. One or two sentences max.
    If location codes exist (e.g., "FF08"), format as "First Floor, Shop #08".
    End with a friendly note or question if needed.
    """
)

product_response_prompt = PromptTemplate(
    input_variables=["product_data", "query", "lang", "collected_data"],
    template="""
    You are CenomiAI, a friendly mall assistant. Answer the user's query about products in {lang}, considering the collected data.

    Product Data: {product_data}
    Collected Data: {collected_data}
    Query: "{query}"

    Keep it short and simple. Mention price and store if available, in one or two sentences.
    End with a nice note like "Hope you like it!" or a quick question.
    """
)

offer_response_prompt = PromptTemplate(
    input_variables=["offer_data", "query", "lang", "collected_data"],
    template="""
    You are CenomiAI, a friendly mall assistant. Answer the user's query about offers in {lang}, considering the collected data.

    Offer Data: {offer_data}
    Collected Data: {collected_data}
    Query: "{query}"

    Keep it concise, mention key details (discount, dates) in one sentence.
    End with something like "Great deal, right?" or a brief follow-up.
    """
)

event_response_prompt = PromptTemplate(
    input_variables=["event_data", "query", "lang", "collected_data"],
    template="""
    You are CenomiAI, a friendly mall assistant. Answer the user's query about events in {lang}, considering the collected data.

    Event Data: {event_data}
    Collected Data: {collected_data}
    Query: "{query}"

    Keep it short, include date and place in one sentence.
    End with a fun note like "See you there?" or a quick question.
    """
)

general_response_prompt = PromptTemplate(
    input_variables=["query", "lang", "conversation_history"],
    template="""
    You are CenomiAI, a friendly mall assistant. Answer this vague query in {lang}, considering the conversation history.

    Query: "{query}"
    Conversation History: {conversation_history}

    Give a short, helpful reply in one sentence. Suggest something fun or ask for more details.
    """
)

# Response Generation Nodes
async def generate_store_response(state: CustomerState) -> CustomerState:
    store_data = json.dumps(state.context_data.get("stores", []))
    collected_data = json.dumps(state.collected_data)
    chain = store_response_prompt | llm | StrOutputParser()
    state.response = await chain.ainvoke(
        {"store_data": store_data, "query": state.query, "lang": state.language, "collected_data": collected_data}
    )
    return state

async def generate_product_response(state: CustomerState) -> CustomerState:
    products = state.context_data.get("products", [])
    if not products:
        state.response = "Sorry, I couldn't find any products matching your query. Maybe try something else?"
        return state
    product_data = json.dumps([dict(product) for product in products])
    collected_data = json.dumps(state.collected_data)
    chain = product_response_prompt | llm | StrOutputParser()
    state.response = await chain.ainvoke(
        {"product_data": product_data, "query": state.query, "lang": state.language, "collected_data": collected_data}
    )
    return state

async def generate_offer_response(state: CustomerState) -> CustomerState:
    offer_data = json.dumps(state.context_data.get("offers", []))
    collected_data = json.dumps(state.collected_data)
    chain = offer_response_prompt | llm | StrOutputParser()
    state.response = await chain.ainvoke(
        {"offer_data": offer_data, "query": state.query, "lang": state.language, "collected_data": collected_data}
    )
    return state

async def generate_event_response(state: CustomerState) -> CustomerState:
    event_data = json.dumps(state.context_data.get("events", []))
    collected_data = json.dumps(state.collected_data)
    chain = event_response_prompt | llm | StrOutputParser()
    state.response = await chain.ainvoke(
        {"event_data": event_data, "query": state.query, "lang": state.language, "collected_data": collected_data}
    )
    return state

async def generate_general_response(state: CustomerState) -> CustomerState:
    formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-6:]]) if state.conversation_history else "No prior conversation."
    chain = general_response_prompt | llm | StrOutputParser()
    state.response = await chain.ainvoke(
        {"query": state.query, "lang": state.language, "conversation_history": formatted_history}
    )
    return state

# Workflow Setup
customer_workflow = StateGraph(CustomerState)
customer_workflow.add_node("classify_intent", classify_intent)
customer_workflow.add_node("retrieve_store_context", retrieve_store_context)
customer_workflow.add_node("retrieve_product_context", retrieve_product_context)
customer_workflow.add_node("retrieve_offer_context", retrieve_offer_context)
customer_workflow.add_node("retrieve_event_context", retrieve_event_context)
customer_workflow.add_node("generate_store_response", generate_store_response)
customer_workflow.add_node("generate_product_response", generate_product_response)
customer_workflow.add_node("generate_offer_response", generate_offer_response)
customer_workflow.add_node("generate_event_response", generate_event_response)
customer_workflow.add_node("generate_general_response", generate_general_response)

customer_workflow.set_entry_point("classify_intent")

def route_after_classify(state: CustomerState):
    if state.intent.startswith("store_"):
        return "retrieve_store_context"
    elif state.intent.startswith("product_"):
        return "retrieve_product_context"
    elif state.intent.startswith("offer_"):
        return "retrieve_offer_context"
    elif state.intent.startswith("event_"):
        return "retrieve_event_context"
    else:
        return "generate_general_response"

customer_workflow.add_conditional_edges(
    "classify_intent",
    route_after_classify,
    {
        "retrieve_store_context": "retrieve_store_context",
        "retrieve_product_context": "retrieve_product_context",
        "retrieve_offer_context": "retrieve_offer_context",
        "retrieve_event_context": "retrieve_event_context",
        "generate_general_response": "generate_general_response",
    }
)

customer_workflow.add_edge("retrieve_store_context", "generate_store_response")
customer_workflow.add_edge("retrieve_product_context", "generate_product_response")
customer_workflow.add_edge("retrieve_offer_context", "generate_offer_response")
customer_workflow.add_edge("retrieve_event_context", "generate_event_response")
customer_workflow.add_edge("generate_store_response", END)
customer_workflow.add_edge("generate_product_response", END)
customer_workflow.add_edge("generate_offer_response", END)
customer_workflow.add_edge("generate_event_response", END)
customer_workflow.add_edge("generate_general_response", END)

customer_graph = customer_workflow.compile()

# Add a new semantic search function
async def semantic_search(query_text, search_type=None, mall_id=None, top_k=5, threshold=0.7):
    """
    Perform semantic search across the vector database
    
    Args:
        query_text (str): The search query
        search_type (str, optional): The type of entity to search for (store, product, event, offer)
        mall_id (int, optional): The mall ID to filter results
        top_k (int): Number of results to return
        threshold (float): Similarity threshold (0-1)
        
    Returns:
        list: List of matching items with their metadata
    """
    try:
        if not query_text:
            return []
            
        query_embedding = embeddings.embed_query(query_text)
        filter_dict = {}
        
        if search_type:
            # Handle different entity types
            if search_type == "offer" or search_type == "event":
                filter_dict["type"] = "engagement"
            else:
                filter_dict["type"] = search_type
                
        if mall_id:
            filter_dict["mall_id"] = str(mall_id)
            
        vector_results = index.query(
            vector=query_embedding,
            filter=filter_dict,
            top_k=top_k,
            include_metadata=True
        )
        
        results = []
        for match in vector_results.matches:
            if match.score > threshold:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
                
        return results
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []