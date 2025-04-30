from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from datetime import datetime
import json
import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils import db_fetch_one_async, db_fetch_all_async, db_execute_async, REDIS_CLIENT, logger, get_conversation_history

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("cenomicore")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

class TenantState(BaseModel):
    query: str
    user_id: str
    language: str = "en"
    conversation_id: str
    conversation_history: List[Dict[str, str]] = []
    entity_type: Optional[str] = None
    action: Optional[str] = None
    collected_data: Dict[str, Any] = {}
    current_step: Optional[str] = None
    store_name: Optional[str] = None
    response: Optional[str] = None
    offer_list: Optional[List[Dict[str, Any]]] = None

intent_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    You are CenomiAI's tenant assistant. Parse this query to determine the tenant's intent based on the current query and conversation history.

    Valid entities: store, offer, product
    Valid actions: create, update, delete, list

    Query: "{query}"

    Previous conversation:
    {conversation_history}

    Instructions:
    - Use the conversation history to resolve vague terms like "it", "that", or "the product/offer".
    - If the query mentions "update" or "change" and follows a prior product/offer creation or mention, assume it refers to modifying an existing entity.
    - If the query is ambiguous but follows a completed action (e.g., product creation), start fresh unless explicitly continuing.
    - Extract any specific details (e.g., name, description, store name) into collected_data.

    Return a JSON object with:
    - entity_type: What they're working with (store, offer, product)
    - action: What they want to do (create, update, delete, list)
    - collected_data: Any details provided (e.g., "name": "Blue Shirt", "description": "20% off", "store": "Zara")

    Return only valid JSON in triple backticks.
    """
)

tenant_prompt = PromptTemplate(
    input_variables=["message", "conversation_history", "entity_type", "action"],
    template="""
    You are CenomiAI, a helpful assistant for mall tenants. Format the following system message into a natural, conversational response:
    
    System message: {message}
    
    Current context:
    - Entity: {entity_type}
    - Action: {action}
    
    Previous conversation:
    {conversation_history}
    
    Make the response friendly and professional. Use emoji occasionally to add warmth. Focus on helping the tenant manage their store information efficiently.
    """
)

async def analyze_intent(state: TenantState) -> TenantState:
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.conversation_history[-4:]])
    intent_chain = intent_prompt | llm | StrOutputParser()
    intent_result = await intent_chain.ainvoke({"query": state.query, "conversation_history": formatted_history})
    start = intent_result.find("```json") + 7
    end = intent_result.rfind("```")
    intent_data = json.loads(intent_result[start:end].strip())
    
    logger.info(f"Intent parsed: {intent_data}")
    
    if not state.current_step:
        state.collected_data = {}
    
    state.entity_type = intent_data.get("entity_type")
    state.action = intent_data.get("action")
    state.collected_data.update(intent_data.get("collected_data", {}))
    
    tenant_id = int(state.user_id[2:]) if state.user_id.startswith("t_") else int(state.user_id)  # Convert to int
    user_stores = await db_fetch_all_async(
        "SELECT name_en FROM stores WHERE tenant_id = $1",
        (tenant_id,)  # Pass as int
    )
    if "store" in state.collected_data:
        requested_store = state.collected_data["store"].lower()
        matching_store = next((s for s in user_stores if s["name_en"].lower() == requested_store), None)
        if matching_store:
            state.store_name = matching_store["name_en"]
        else:
            state.response = f"I couldn't find '{requested_store}'. Your stores: {', '.join([s['name_en'] for s in user_stores])}"
            state.current_step = "select_store"
            return state
    elif len(user_stores) == 1:
        state.store_name = user_stores[0]["name_en"]
        logger.info(f"Default store set to {state.store_name}")
    elif len(user_stores) > 1 and not state.store_name:
        state.current_step = "select_store"
    
    return state

async def prompt_for_missing_info(state: TenantState) -> TenantState:
    tenant_id = int(state.user_id[2:]) if state.user_id.startswith("t_") else int(state.user_id)  # Convert to int
    user_stores = await db_fetch_all_async(
        "SELECT name_en FROM stores WHERE tenant_id = $1",
        (tenant_id,)  # Pass as int
    )
    
    if state.current_step == "select_store" or (len(user_stores) > 1 and not state.store_name):
        if not user_stores:
            state.response = "You don't have any stores yet. Contact mall management to get started!"
            return state
        store_list = "\n".join([f"{i+1}) {s['name_en']}" for i, s in enumerate(user_stores)])
        state.response = f"You've got multiple stores! Which one?\n{store_list}\nType the number!"
        state.current_step = "select_store"
        return state
    
    if not state.store_name and len(user_stores) == 1:
        state.store_name = user_stores[0]["name_en"]
        logger.info(f"Auto-set store_name to {state.store_name}")
    
    if state.entity_type == "offer":
        if state.action == "create":
            if "description" not in state.collected_data:
                state.response = f"What's the offer for {state.store_name}? (e.g., '20% off summer clothes')"
                state.current_step = "description"
            elif "start_date" not in state.collected_data:
                state.response = "When should it start? (e.g., 'today' or '2025-04-01')"
                state.current_step = "start_date"
            elif "end_date" not in state.collected_data:
                state.response = "When should it end? (e.g., '2025-04-30')"
                state.current_step = "end_date"
            else:
                await execute_operation(state)
        elif state.action == "update":
            if "description" not in state.collected_data:
                offers = await db_fetch_all_async(
                    "SELECT description_en, start_date, end_date FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = $1 AND tenant_id = $2)",
                    (state.store_name, tenant_id)  # Pass tenant_id as int
                )
                if not offers:
                    state.response = f"No offers found for {state.store_name}. Want to add one?"
                    state.entity_type = None
                    state.action = None
                    return state
                state.offer_list = offers
                offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(offers)])
                state.response = f"Which offer to update in {state.store_name}?\n{offer_list}\nType the number!"
                state.current_step = "select_offer"
            elif "update_field" not in state.collected_data:
                state.response = f"What do you want to change for '{state.collected_data['description']}'?\n1) Description\n2) Start Date\n3) End Date\nType the number!"
                state.current_step = "update_field"
            elif state.collected_data["update_field"] == "1" and "new_description" not in state.collected_data:
                state.response = f"What's the new description for '{state.collected_data['description']}'?"
                state.current_step = "new_description"
            elif state.collected_data["update_field"] == "2" and "new_start_date" not in state.collected_data:
                state.response = f"What's the new start date for '{state.collected_data['description']}'? (e.g., '2025-04-01')"
                state.current_step = "new_start_date"
            elif state.collected_data["update_field"] == "3" and "new_end_date" not in state.collected_data:
                state.response = f"What's the new end date for '{state.collected_data['description']}'? (e.g., '2025-04-30')"
                state.current_step = "new_end_date"
            else:
                await execute_operation(state)
        elif state.action == "delete":
            if "description" not in state.collected_data:
                offers = await db_fetch_all_async(
                    "SELECT description_en, start_date, end_date FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = $1 AND tenant_id = $2)",
                    (state.store_name, tenant_id)  # Pass tenant_id as int
                )
                if not offers:
                    state.response = f"No offers found for {state.store_name}. Want to add one?"
                    state.entity_type = None
                    state.action = None
                    return state
                state.offer_list = offers
                offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(offers)])
                state.response = f"Which offer to remove from {state.store_name}?\n{offer_list}\nType the number!"
                state.current_step = "select_offer"
            else:
                await execute_operation(state)
        elif state.action == "list":
            offers = await db_fetch_all_async(
                "SELECT description_en, start_date, end_date FROM offers WHERE store_id = (SELECT store_id FROM stores WHERE name_en = $1 AND tenant_id = $2)",
                (state.store_name, tenant_id)  # Pass tenant_id as int
            )
            if not offers:
                state.response = f"No offers in {state.store_name} yet. Want to add one?"
            else:
                offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(offers)])
                state.response = f"Here are your offers for {state.store_name}:\n{offer_list}\nAnything else?"
    
    elif state.entity_type == "product":
        if state.action == "create":
            if "name" not in state.collected_data:
                state.response = f"What's the product name for {state.store_name}? (e.g., 'Blue Shirt')"
                state.current_step = "name"
            elif "description" not in state.collected_data:
                state.response = f"What's the description for '{state.collected_data['name']}'? (e.g., 'Cotton, size M')"
                state.current_step = "description"
            elif "price" not in state.collected_data:
                state.response = f"How much does '{state.collected_data['name']}' cost? (e.g., '50')"
                state.current_step = "price"
            elif "currency" not in state.collected_data:
                state.response = f"What's the currency for '{state.collected_data['name']}'? (e.g., 'SAR')"
                state.current_step = "currency"
            else:
                await execute_operation(state)
        elif state.action == "update":
            if "name" not in state.collected_data:
                products = await db_fetch_all_async(
                    "SELECT name_en, description_en, price, currency FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = $1 AND tenant_id = $2)",
                    (state.store_name, tenant_id)  # Pass tenant_id as int
                )
                if not products:
                    state.response = f"No products found for {state.store_name}. Want to add one?"
                    state.entity_type = None
                    state.action = None
                    return state
                state.offer_list = products
                product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(products)])
                state.response = f"Which product to update in {state.store_name}?\n{product_list}\nType the number!"
                state.current_step = "select_product"
            elif "update_field" not in state.collected_data:
                state.response = f"What do you want to change for '{state.collected_data['name']}'?\n1) Name\n2) Description\n3) Price\n4) Currency\nType the number!"
                state.current_step = "update_field"
            elif state.collected_data["update_field"] == "1" and "new_name" not in state.collected_data:
                state.response = f"What's the new name for '{state.collected_data['name']}'?"
                state.current_step = "new_name"
            elif state.collected_data["update_field"] == "2" and "new_description" not in state.collected_data:
                state.response = f"What's the new description for '{state.collected_data['name']}'?"
                state.current_step = "new_description"
            elif state.collected_data["update_field"] == "3" and "new_price" not in state.collected_data:
                state.response = f"What's the new price for '{state.collected_data['name']}'? (e.g., '60')"
                state.current_step = "new_price"
            elif state.collected_data["update_field"] == "4" and "new_currency" not in state.collected_data:
                state.response = f"What's the new currency for '{state.collected_data['name']}'? (e.g., 'USD')"
                state.current_step = "new_currency"
            else:
                await execute_operation(state)
        elif state.action == "delete":
            if "name" not in state.collected_data:
                products = await db_fetch_all_async(
                    "SELECT name_en, description_en, price, currency FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = $1 AND tenant_id = $2)",
                    (state.store_name, tenant_id)
                )
                if not products:
                    state.response = f"No products found for {state.store_name}. Want to add one?"
                    state.entity_type = None
                    state.action = None
                    return state
                state.offer_list = products
                product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(products)])
                state.response = f"Which product to remove from {state.store_name}?\n{product_list}\nType the number!"
                state.current_step = "select_product"
            else:
                await execute_operation(state)
        elif state.action == "list":
            products = await db_fetch_all_async(
                "SELECT name_en, description_en, price, currency FROM products WHERE store_id = (SELECT store_id FROM stores WHERE name_en = $1 AND tenant_id = $2)",
                (state.store_name, tenant_id)
            )
            if not products:
                state.response = f"No products in {state.store_name} yet. Want to add one?"
            else:
                product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(products)])
                state.response = f"Here are your products for {state.store_name}:\n{product_list}\nAnything else?"
    
    return state

async def process_input(state: TenantState) -> TenantState:
    if not state.current_step:
        return await analyze_intent(state)
    
    tenant_id = int(state.user_id[2:]) if state.user_id.startswith("t_") else int(state.user_id)  # Convert to int
    user_stores = await db_fetch_all_async(
        "SELECT name_en FROM stores WHERE tenant_id = $1",
        (tenant_id,)  # Pass as int
    )
    
    if state.current_step == "select_store":
        try:
            choice = int(state.query.strip()) - 1
            if 0 <= choice < len(user_stores):
                state.store_name = user_stores[choice]["name_en"]
                state.current_step = None
                logger.info(f"Store selected: {state.store_name}")
            else:
                store_list = "\n".join([f"{i+1}) {s['name_en']}" for i, s in enumerate(user_stores)])
                state.response = f"Invalid option! Pick a number:\n{store_list}"
                return state
        except ValueError:
            store_list = "\n".join([f"{i+1}) {s['name_en']}" for i, s in enumerate(user_stores)])
            state.response = f"Please type a number! Here are your stores:\n{store_list}"
            return state
    
    elif state.current_step == "select_offer":
        try:
            choice = int(state.query.strip()) - 1
            if 0 <= choice < len(state.offer_list):
                state.collected_data["description"] = state.offer_list[choice]["description_en"]
                state.current_step = None
                state.offer_list = None
            else:
                offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(state.offer_list)])
                state.response = f"Invalid option! Pick a number:\n{offer_list}"
                return state
        except ValueError:
            offer_list = "\n".join([f"{i+1}) {o['description_en']} (Valid: {o['start_date']} to {o['end_date']})" for i, o in enumerate(state.offer_list)])
            state.response = f"Please type a number! Options:\n{offer_list}"
            return state
    
    elif state.current_step == "select_product":
        try:
            choice = int(state.query.strip()) - 1
            if 0 <= choice < len(state.offer_list):
                state.collected_data["name"] = state.offer_list[choice]["name_en"]
                state.current_step = None
                state.offer_list = None
            else:
                product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(state.offer_list)])
                state.response = f"Invalid option! Pick a number:\n{product_list}"
                return state
        except ValueError:
            product_list = "\n".join([f"{i+1}) {p['name_en']} - {p['description_en']} ({p['price']} {p['currency']})" for i, p in enumerate(state.offer_list)])
            state.response = f"Please type a number! Options:\n{product_list}"
            return state
    
    elif state.current_step == "update_field":
        try:
            choice = int(state.query.strip())
            if state.entity_type == "offer" and choice in [1, 2, 3]:
                state.collected_data["update_field"] = str(choice)
                state.current_step = None
            elif state.entity_type == "product" and choice in [1, 2, 3, 4]:
                state.collected_data["update_field"] = str(choice)
                state.current_step = None
            else:
                if state.entity_type == "offer":
                    state.response = "Invalid option! Choose: 1) Description, 2) Start Date, 3) End Date"
                else:
                    state.response = "Invalid option! Choose: 1) Name, 2) Description, 3) Price, 4) Currency"
                return state
        except ValueError:
            if state.entity_type == "offer":
                state.response = "Please type a number! 1) Description, 2) Start Date, 3) End Date"
            else:
                state.response = "Please type a number! 1) Name, 2) Description, 3) Price, 4) Currency"
            return state
    
    elif state.current_step in ["description", "new_description", "start_date", "end_date", "new_start_date", "new_end_date", 
                                "name", "new_name", "price", "new_price", "currency", "new_currency"]:
        state.collected_data[state.current_step] = state.query.strip()
        state.current_step = None
    
    return await prompt_for_missing_info(state)

async def execute_operation(state: TenantState) -> None:
    tenant_id = int(state.user_id[2:]) if state.user_id.startswith("t_") else int(state.user_id)  # Convert to int
    store = await db_fetch_one_async(
        "SELECT store_id, mall_id, name_en, location_en FROM stores WHERE name_en = $1 AND tenant_id = $2",
        (state.store_name, tenant_id)
    )
    if not store:
        state.response = f"I couldn't find {state.store_name} in your stores."
        return
    
    store_id = store["store_id"]
    store_name = store["name_en"]
    location_en = store["location_en"]
    logger.info(f"Executing {state.action} on {state.entity_type} for store_id: {store_id}")
    
    if state.entity_type == "offer":
        if state.action == "create":
            description = state.collected_data["description"]
            # Convert start_date to datetime.date
            start_date_str = state.collected_data["start_date"]
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date() if start_date_str != "today" else datetime.now().date()
            # Convert end_date to datetime.date
            end_date_str = state.collected_data["end_date"]
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            await db_execute_async(
                "INSERT INTO offers (store_id, description_en, description_ar, start_date, end_date) VALUES ($1, $2, $3, $4, $5)",
                (store_id, description, description, start_date, end_date)  # Pass datetime.date objects
            )
            offer = await db_fetch_one_async(
                "SELECT offer_id FROM offers WHERE store_id = $1 AND description_en = $2",
                (store_id, description)
            )
            if offer:
                offer_id = offer["offer_id"]
                vector = embeddings.embed_query(description)
                index.upsert(vectors=[{
                    "id": f"offer_{offer_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "offer",
                        "id": offer_id,
                        "description_en": description,
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Added '{description}' to {state.store_name} from {start_date} to {end_date}. Anything else? 😊"
            else:
                state.response = f"Failed to add '{description}'. Try again or contact support."
                
        elif state.action == "update":
            old_desc = state.collected_data["description"]
            update_field = state.collected_data["update_field"]
            offer = await db_fetch_one_async(
                "SELECT offer_id FROM offers WHERE store_id = $1 AND description_en = $2",
                (store_id, old_desc)
            )
            if not offer:
                state.response = f"Couldn't find '{old_desc}' in {state.store_name}. Want to list offers?"
                return
            offer_id = offer["offer_id"]
            if update_field == "1":
                new_desc = state.collected_data["new_description"]
                await db_execute_async(
                    "UPDATE offers SET description_en = $1, description_ar = $2 WHERE offer_id = $3",
                    (new_desc, new_desc, offer_id)
                )
                vector = embeddings.embed_query(new_desc)
                index.upsert(vectors=[{
                    "id": f"offer_{offer_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "offer",
                        "id": offer_id,
                        "description_en": new_desc,
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Updated '{old_desc}' to '{new_desc}' in {state.store_name}. Anything else? 😊"
            elif update_field == "2":
                new_start_date = state.collected_data["new_start_date"]
                await db_execute_async(
                    "UPDATE offers SET start_date = $1 WHERE offer_id = $2",
                    (new_start_date, offer_id)
                )
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Updated '{old_desc}' start date to {new_start_date} in {state.store_name}. Anything else? 😊"
            elif update_field == "3":
                new_end_date = state.collected_data["new_end_date"]
                await db_execute_async(
                    "UPDATE offers SET end_date = $1 WHERE offer_id = $2",
                    (new_end_date, offer_id)
                )
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Updated '{old_desc}' end date to {new_end_date} in {state.store_name}. Anything else? 😊"
        elif state.action == "delete":
            description = state.collected_data["description"]
            offer = await db_fetch_one_async(
                "SELECT offer_id FROM offers WHERE store_id = $1 AND description_en = $2",
                (store_id, description)
            )
            if offer:
                offer_id = offer["offer_id"]
                await db_execute_async("DELETE FROM offers WHERE offer_id = $1", (offer_id,))
                index.delete(ids=[f"offer_{offer_id}_en"])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Removed '{description}' from {state.store_name}. Anything else? 😊"
            else:
                state.response = f"Couldn't find '{description}' in {state.store_name}. Want to list offers?"
    
    elif state.entity_type == "product":
        if state.action == "create":
            name = state.collected_data["name"]
            description = state.collected_data.get("description")
            price = float(state.collected_data["price"])
            currency = state.collected_data["currency"]
            await db_execute_async(
                "INSERT INTO products (store_id, name_en, name_ar, description_en, description_ar, price, currency) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                (store_id, name, name, description, description, price, currency)
            )
            product = await db_fetch_one_async(
                "SELECT product_id FROM products WHERE store_id = $1 AND name_en = $2",
                (store_id, name)
            )
            if product:
                product_id = product["product_id"]
                vector = embeddings.embed_query(f"{name} {description or ''}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "mall_id": store["mall_id"],
                        "name_en": name,
                        "description_en": description or "",
                        "price": price,
                        "currency": currency,
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Added '{name}' ({description or 'No description'}) to {state.store_name} for {price} {currency}. Anything else? 😊"
            else:
                state.response = f"Failed to add '{name}'. Try again or contact support."
        elif state.action == "update":
            old_name = state.collected_data["name"]
            update_field = state.collected_data["update_field"]
            product = await db_fetch_one_async(
                "SELECT product_id FROM products WHERE store_id = $1 AND name_en = $2",
                (store_id, old_name)
            )
            if not product:
                state.response = f"Couldn't find '{old_name}' in {state.store_name}. Want to list products?"
                return
            product_id = product["product_id"]
            if update_field == "1":
                new_name = state.collected_data["new_name"]
                await db_execute_async(
                    "UPDATE products SET name_en = $1, name_ar = $2 WHERE product_id = $3",
                    (new_name, new_name, product_id)
                )
                vector = embeddings.embed_query(f"{new_name} {state.collected_data.get('description', '')}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "mall_id": store["mall_id"],
                        "name_en": new_name,
                        "description_en": state.collected_data.get("description", ""),
                        "price": float(state.collected_data.get("price", 0)),
                        "currency": state.collected_data.get("currency", ""),
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Updated '{old_name}' to '{new_name}' in {state.store_name}. Anything else? 😊"
            elif update_field == "2":
                new_desc = state.collected_data["new_description"]
                await db_execute_async(
                    "UPDATE products SET description_en = $1, description_ar = $2 WHERE product_id = $3",
                    (new_desc, new_desc, product_id)
                )
                vector = embeddings.embed_query(f"{old_name} {new_desc}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "name_en": old_name,
                        "description_en": new_desc,
                        "price": float(state.collected_data.get("price", 0)),
                        "currency": state.collected_data.get("currency", ""),
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Updated '{old_name}' description to '{new_desc}' in {state.store_name}. Anything else? 😊"
            elif update_field == "3":
                new_price = float(state.collected_data["new_price"])
                await db_execute_async(
                    "UPDATE products SET price = $1 WHERE product_id = $2",
                    (new_price, product_id)
                )
                vector = embeddings.embed_query(f"{old_name} {state.collected_data.get('description', '')}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "name_en": old_name,
                        "description_en": state.collected_data.get("description", ""),
                        "price": new_price,
                        "currency": state.collected_data.get("currency", ""),
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Updated '{old_name}' price to {new_price} in {state.store_name}. Anything else? 😊"
            elif update_field == "4":
                new_currency = state.collected_data["new_currency"]
                await db_execute_async(
                    "UPDATE products SET currency = $1 WHERE product_id = $2",
                    (new_currency, product_id)
                )
                vector = embeddings.embed_query(f"{old_name} {state.collected_data.get('description', '')}")
                index.upsert(vectors=[{
                    "id": f"product_{product_id}_en",
                    "values": vector,
                    "metadata": {
                        "type": "product",
                        "id": product_id,
                        "name_en": old_name,
                        "description_en": state.collected_data.get("description", ""),
                        "price": float(state.collected_data.get("price", 0)),
                        "currency": new_currency,
                        "store_id": store_id,
                        "store_name": store_name,
                        "location_en": location_en,
                        "lang": "en"
                    }
                }])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Updated '{old_name}' currency to {new_currency} in {state.store_name}. Anything else? 😊"
        elif state.action == "delete":
            name = state.collected_data["name"]
            product = await db_fetch_one_async(
                "SELECT product_id FROM products WHERE store_id = $1 AND name_en = $2",
                (store_id, name)
            )
            if product:
                product_id = product["product_id"]
                await db_execute_async("DELETE FROM products WHERE product_id = $1", (product_id,))
                index.delete(ids=[f"product_{product_id}_en"])
                REDIS_CLIENT.delete(f"context:*:{store_id}")
                state.response = f"Removed '{name}' from {state.store_name}. Anything else? 😊"
            else:
                state.response = f"Couldn't find '{name}' in {state.store_name}. Want to list products?"
    
    state.entity_type = None
    state.action = None
    state.collected_data = {}
    state.current_step = None
    state.offer_list = None

async def tenant_recognize_intent(state: TenantState) -> TenantState:
    conversation_history = await get_conversation_history(state.conversation_id)
    state.conversation_history = [{"role": msg.role, "content": msg.content} for msg in conversation_history]
    
    if not state.current_step:
        state = await analyze_intent(state)
        state = await prompt_for_missing_info(state)
    else:
        state = await process_input(state)
    
    if state.response:
        tenant_chain = tenant_prompt | llm | StrOutputParser()
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.conversation_history[-4:]])
        state.response = await tenant_chain.ainvoke({
            "message": state.response,
            "conversation_history": formatted_history,
            "entity_type": state.entity_type or "unknown",
            "action": state.action or "unknown"
        })
    else:
        state.response = "I'm not sure what you want to do. You can add, update, or remove offers or products—just let me know!"
    return state

tenant_workflow = StateGraph(TenantState)
tenant_workflow.add_node("process", tenant_recognize_intent)
tenant_workflow.set_entry_point("process")
tenant_workflow.add_edge("process", END)
tenant_graph = tenant_workflow.compile()