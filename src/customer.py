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

# Load spaCy NLP model for store name extraction
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

knowledge_graph = nx.Graph()

async def populate_knowledge_graph():
    # Assumes 'stores' table has columns: store_id, name_en, mall_id
    stores = await db_fetch_all_async("SELECT store_id, name_en, mall_id FROM stores")
    # Assumes 'products' table has columns: product_id, name_en, store_id
    products = await db_fetch_all_async("SELECT product_id, name_en, store_id FROM products")
    # Assumes 'offers' table has columns: offer_id, description_en, store_id
    offers = await db_fetch_all_async("SELECT offer_id, description_en, store_id FROM offers")

    for store in stores:
        knowledge_graph.add_node(
            f"store_{store['store_id']}",
            type="store",
            name=store["name_en"],
            mall_id=store["mall_id"],
        )
    for product in products:
        knowledge_graph.add_node(
            f"product_{product['product_id']}", type="product", name=product["name_en"]
        )
        knowledge_graph.add_edge(f"store_{product['store_id']}", f"product_{product['product_id']}")
    for offer in offers:
        knowledge_graph.add_node(
            f"offer_{offer['offer_id']}", type="offer", description=offer["description_en"]
        )
        knowledge_graph.add_edge(f"store_{offer['store_id']}", f"offer_{offer['offer_id']}")

# Intent Classification Prompt
intent_classification_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    Classify the user's intent based on the query and conversation history.
    Possible intents: store_info, product_info, offer_info, dining_info, service_info, amenity_info, event_info, navigation, general_inquiry, other.
    - If the query involves a list of items to buy, classify as 'product_info'.
    - If the query is about finding stores without specific products, classify as 'store_info'.
    - If the query is vague or broad (e.g., "plan my day"), classify as 'general_inquiry'.

    Query: "{query}"

    Conversation History:
    {conversation_history}

    Respond with only the intent category.
    """
)
intent_chain = intent_classification_prompt | llm | StrOutputParser()

customer_prompt = PromptTemplate(
    input_variables=["context", "query", "lang", "conversation_history", "mall_name"],
    template="""
    You are CenomiAI, a friendly, proactive, and highly knowledgeable assistant for {mall_name} mall. Respond in {lang} with a warm, conversational tone, using emojis 😊 to keep it engaging. Your purpose is to assist with inquiries specific to {mall_name}—stores, products, dining, services, amenities, events, offers, navigation, and more—staying strictly within this mall's context. Assume the customer knows little about {mall_name} and may ask diverse, vague, or list-based questions. Use the query, context, and conversation history to craft accurate, detailed, and tailored responses that anticipate their needs, weaving in relevant suggestions naturally. All information must relate to {mall_name} only. If info is missing, make smart assumptions based on {mall_name}'s context/history or provide general guidance about {mall_name} while keeping it supportive and fun.

    Response Guidelines:
    - Store Queries: Share specifics (name, exact location, offerings) for stores in {mall_name}.
    - Product Queries (including lists): Match each item to specific stores/products/offers in {mall_name}. Structure as a numbered list if applicable.
    - Dining: Recommend options in {mall_name} by cuisine/vibe/needs with locations.
    - Offers & Events: Highlight current promotions/events in {mall_name} tied to the query.
    - Navigation: Provide directions within {mall_name} from an assumed start (e.g., main entrance).
    - Vague/List Queries: Propose a clear, delightful plan specific to {mall_name}.
    - Personalization: Use `user_id` for loyalty perks within {mall_name}.
    - Suggestions: Integrate relevant offers/products from {mall_name} seamlessly.

    Current Query:
    "{query}"

    Context about {mall_name}:
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
    intent: Optional[str] = None
    initial_context: Optional[List[Dict[str, Any]]] = None
    suggestions: Optional[List[str]] = None
    mall_id: Optional[int] = None

async def classify_intent(state: CustomerState) -> CustomerState:
    formatted_history = (
        "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-4:]])
        if state.conversation_history
        else ""
    )
    intent = await intent_chain.ainvoke({"query": state.query, "conversation_history": formatted_history})
    state.intent = intent.strip()
    if state.intent == "other":
        state.response = "I’m not sure what you mean 😅. Could you tell me more? Are you asking about stores, dining, or something else?"
    logger.info(f"Classified intent: {state.intent}")
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

    query_items = (
        [item.strip() for item in state.query.split("\n") if item.strip()]
        if "\n" in state.query
        else [state.query]
    )
    intent_prefixes = {
        "store_info": "store",
        "product_info": "product",
        "offer_info": "offer",
        "dining_info": "dining",
        "service_info": "service",
        "amenity_info": "amenity",
        "event_info": "event",
        "navigation": "navigation",
        "general_inquiry": "mall",
        "other": "mall",
    }
    query_prefix = intent_prefixes.get(state.intent, "mall")

    # NLP-based store name extraction
    store_name = None
    all_stores = await db_fetch_all_async(
        "SELECT name_en FROM stores WHERE mall_id = $1", (state.mall_id,)
    )
    store_names = [store["name_en"].lower() for store in all_stores]
    for item in query_items:
        doc = nlp(item)
        for ent in doc.ents:
            if ent.text.lower() in store_names:
                store_name = ent.text
                break
        if not store_name:  # Fallback: check if any store name is substring
            for name in store_names:
                if name in item.lower():
                    store_name = name
                    break
        if store_name:
            break

    if store_name:
        query_vectors = [embeddings.embed_query(f"{query_prefix} {store_name} products")]
        store = await db_fetch_one_async(
            "SELECT store_id FROM stores WHERE name_en ILIKE $1 AND mall_id = $2",
            (store_name, state.mall_id),
        )
        if store:
            products = await db_fetch_all_async(
                "SELECT product_id, name_en, description_en, price, currency FROM products WHERE store_id = $1",
                (store["store_id"],),
            )
            state.context_data = state.context_data or {"products": []}
            state.context_data["products"] = [
                {
                    "id": p["product_id"],  # Ensure ID is included
                    "name": p["name_en"],
                    "description": p["description_en"],
                    "price": p["price"],
                    "currency": p["currency"],
                    "store_id": store["store_id"],
                }
                for p in products
            ]
    else:
        query_vectors = [embeddings.embed_query(f"{query_prefix} {item}") for item in query_items]

    avg_vector = [sum(v[i] for v in query_vectors) / len(query_vectors) for i in range(len(query_vectors[0]))]

    filter = {"mall_id": state.mall_id}  # Always filter by mall_id

    # Enhance with history
    if state.conversation_history and any(
        word in state.query.lower() for word in ["it", "they", "that", "this", "there", "those", "meant"]
    ):
        last_response = next(
            (msg["content"] for msg in reversed(state.conversation_history[-4:]) if msg["role"] == "assistant"),
            "",
        )
        if last_response:
            history_vector = embeddings.embed_query(last_response)
            avg_vector = [(a + h) / 2 for a, h in zip(avg_vector, history_vector)]

    # Fetch from Pinecone (assumes metadata includes mall_id, type, name_en, etc.)
    results = await asyncio.to_thread(
        index.query,
        vector=avg_vector,
        top_k=25,
        include_metadata=True,
        filter=filter,
    )
    docs = results["matches"] if "matches" in results else []

    state.initial_context = [
        {"id": doc["id"], "score": float(doc["score"]), "metadata": doc["metadata"]} for doc in docs
    ]
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
        "services": [],
        "amenities": [],
        "products": [],
        "loyalty_programs": [],
        "customer_loyalty": [],
    }
    store_ids = set()

    # Add mall name (assumes 'malls' table has mall_id, name_en)
    mall = await db_fetch_one_async(
        "SELECT name_en FROM malls WHERE mall_id = $1", (state.mall_id,)
    )
    context["mall_name"] = mall["name_en"] if mall else "Unknown Mall"

    # Process initial context, ensuring mall_id matches
    for doc in state.initial_context or []:
        metadata = doc["metadata"]
        if metadata.get("mall_id") != state.mall_id:
            continue  # Skip if not from the selected mall
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
        elif doc_type == "offer":
            offer = {
                "id": metadata.get("id"),  # Ensure ID is included
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
                "id": metadata.get("id"),  # Ensure ID is included
                "name": metadata.get("name_en"),
                "description": metadata.get("description_en"),
                "price": metadata.get("price"),
                "currency": metadata.get("currency"),
                "store_id": metadata.get("store_id"),
                "store_name": metadata.get("store_name"),
                "location_en": metadata.get("location_en"),
            }
            context["products"].append(product)
            if metadata.get("store_id"):
                store_ids.add(metadata["store_id"])
        elif doc_type == "event":
            context["events"].append(
                {
                    "name": metadata.get("name_en"),
                    "date": metadata.get("start_time"),
                    "location": metadata.get("location_en"),
                }
            )
        elif doc_type == "service":
            context["services"].append(
                {
                    "name": metadata.get("name_en"),
                    "description": metadata.get("description_en"),
                }
            )
        elif doc_type == "amenity":
            context["amenities"].append(
                {"name": metadata.get("name_en"), "location": metadata.get("location_en")}
            )

    # Fetch store details with mall_id filter
    if store_ids:
        stores = await db_fetch_all_async(
            "SELECT store_id, name_en, location_en, category_en FROM stores WHERE store_id = ANY($1) AND mall_id = $2",
            (list(store_ids), state.mall_id),
        )
        store_map = {s["store_id"]: s for s in stores}
        for item in context["products"] + context["offers"]:
            if item.get("store_id") in store_map and not item.get("store_name"):
                store = store_map[item["store_id"]]
                item["store_name"] = store["name_en"]
                item["location_en"] = store["location_en"]
                item["category"] = store["category_en"]

    # Fetch loyalty data (assumes customer_loyalty and loyalty_programs tables)
    if state.user_id and state.user_id.startswith("c_"):
        customer_id = state.user_id[2:]
        loyalty = await db_fetch_one_async(
            "SELECT cl.points_balance, lp.name_en, lp.description_en FROM customer_loyalty cl JOIN loyalty_programs lp ON cl.loyalty_id = lp.loyalty_id WHERE cl.customer_id = $1",
            (customer_id,),
        )
        context["customer_loyalty"] = loyalty or {}

    state.context_data = context
    state.response = json.dumps(convert_to_json_safe(context))
    return state

async def suggest_related(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state

    suggestions = []
    if state.context_data:
        for product in state.context_data.get("products", []):
            if "id" in product:
                product_id = f"product_{product['id']}"
                if knowledge_graph.has_node(product_id):
                    related_offers = [
                        n
                        for n in knowledge_graph.neighbors(product_id)
                        if n.startswith("offer_") and knowledge_graph.nodes[n].get("mall_id") == state.mall_id
                    ][:2]
                    for offer_id in related_offers:
                        offer_data = knowledge_graph.nodes[offer_id]
                        suggestions.append(
                            f"While checking out {product['name']} at {product['store_name']}, grab this deal: {offer_data['description']}!"
                        )

        for offer in state.context_data.get("offers", []):
            if "id" in offer:
                offer_id = f"offer_{offer['id']}"
                if knowledge_graph.has_node(offer_id):
                    related_products = [
                        n
                        for n in knowledge_graph.neighbors(offer_id)
                        if n.startswith("product_") and knowledge_graph.nodes[n].get("mall_id") == state.mall_id
                    ][:2]
                    for product_id in related_products:
                        product_data = knowledge_graph.nodes[product_id]
                        suggestions.append(
                            f"Pair this offer ({offer['description']}) with {product_data['name']} at {offer['store_name']}!"
                        )

    state.suggestions = suggestions[:3]
    return state

async def generate_response(state: CustomerState) -> CustomerState:
    if not state.mall_id:
        state.response = "Oops! I need to know which mall you're asking about. Please select a mall first! 😊"
        return state

    formatted_history = (
        "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state.conversation_history[-6:]])
        if state.conversation_history
        else "No prior conversation."
    )
    context_with_suggestions = state.response or "Limited context available—making the best of it!"
    if state.suggestions:
        context_with_suggestions += "\n\nRelated Suggestions:\n" + "\n".join(state.suggestions)

    response = await asyncio.to_thread(
        customer_chain.invoke,
        {
            "context": context_with_suggestions,
            "query": state.query,
            "lang": state.language,
            "conversation_history": formatted_history,
            "mall_name": state.context_data.get("mall_name", "the mall"),
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "user_id": state.user_id,
        },
    )
    state.response = response
    return state

# Workflow
customer_workflow = StateGraph(CustomerState)
customer_workflow.add_node("classify_intent", classify_intent)
customer_workflow.add_node("initial_retrieval", initial_retrieval)
customer_workflow.add_node("refine_context", refine_context)
customer_workflow.add_node("suggest_related", suggest_related)
customer_workflow.add_node("respond", generate_response)
customer_workflow.set_entry_point("classify_intent")
customer_workflow.add_conditional_edges(
    "classify_intent",
    lambda state: "respond" if state.intent == "other" and state.response else "initial_retrieval",
)
customer_workflow.add_edge("initial_retrieval", "refine_context")
customer_workflow.add_edge("refine_context", "suggest_related")
customer_workflow.add_edge("suggest_related", "respond")
customer_workflow.add_edge("respond", END)
customer_graph = customer_workflow.compile()