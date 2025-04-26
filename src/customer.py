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
    store_name = state.collected_data.get("name", "").lower()
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
    stores = await db_fetch_all_async(query, tuple(params))
    state.context_data = {"stores": [dict(store) for store in stores[:5]]}  # Limit to 5
    return state

async def retrieve_product_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Please select a mall first!"
        return state
    product_name = state.collected_data.get("name", "").lower()
    product_category = state.collected_data.get("category", "").lower()
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
    products = await db_fetch_all_async(query, tuple(params))
    state.context_data = {"products": [dict(product) for product in products[:5]]}  # Limit to 5
    logger.info(f"Retrieved {len(products)} products for query: {state.query}")
    return state

async def retrieve_offer_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Please select a mall first!"
        return state
    offers = await db_fetch_all_async(
        """SELECT e.engagement_id, e.title_en, e.description_en, e.start_date, e.end_date, 
                  b.brand_name_en
           FROM engagements e
           LEFT JOIN brands b ON e.brand_id = b.brand_id
           WHERE e.unique_property_id = $1 AND LOWER(e.type) = 'offer'""",
        (state.mall_id,)
    )
    state.context_data = {"offers": [dict(offer) for offer in offers[:5]]}  # Limit to 5
    return state

async def retrieve_event_context(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Please select a mall first!"
        return state
    event_name = state.collected_data.get("name", "").lower()
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
    events = await db_fetch_all_async(query, tuple(params))
    state.context_data = {"events": [dict(event) for event in events[:5]]}  # Limit to 5
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