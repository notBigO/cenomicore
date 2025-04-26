# Cenomi Chatbot Developer Guide

This guide is designed for developers who want to integrate with, extend, or modify the Cenomi Chatbot system.

## Table of Contents

1. [API Reference](#api-reference)
2. [Integration Guide](#integration-guide)
3. [Extending the System](#extending-the-system)
4. [Database Schema](#database-schema)
5. [Vector Search](#vector-search)
6. [Testing and Deployment](#testing-and-deployment)

## API Reference

### Authentication

#### POST /login

Authenticates users and returns a user ID.

**Request:**

```json
{
  "email": "string",
  "password": "string"
}
```

**Response:**

```json
{
  "user_id": "string" // Prefixed with "t_" for tenants, "c_" for customers
}
```

**Status Codes:**

- 200: Success
- 401: Invalid credentials

### Customer Endpoints

#### POST /chat

Processes customer queries about the mall, stores, and products.

**Request:**

```json
{
  "text": "string",
  "user_id": "string",
  "language": "string",
  "conversation_id": "string",
  "mall_id": "integer"
}
```

**Response:**

```json
{
  "message": "string",
  "conversation_id": "string"
}
```

**Status Codes:**

- 200: Success
- 500: Server error

### Tenant Endpoints

#### POST /tenant/update

Handles tenant requests to update store information, products, and offers.

**Request:**

```json
{
  "text": "string",
  "user_id": "string",
  "language": "string",
  "conversation_id": "string"
}
```

**Response:**

```json
{
  "message": "string",
  "conversation_id": "string"
}
```

**Status Codes:**

- 200: Success
- 403: Unauthorized
- 500: Server error

### Other Endpoints

#### GET /malls

Returns a list of available malls.

**Response:**

```json
[
  {
    "mall_id": "string",
    "name_en": "string"
  }
]
```

**Status Codes:**

- 200: Success

## Integration Guide

### Integrating with Web Applications

To integrate the Cenomi Chatbot with your web application, follow these steps:

1. **Set up API client**

```javascript
// Example JavaScript client
async function sendChatMessage(
  text,
  userId,
  conversationId,
  mallId,
  language = "en"
) {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      user_id: userId,
      conversation_id: conversationId,
      mall_id: mallId,
      language,
    }),
  });

  return response.json();
}
```

2. **Handle user authentication**

```javascript
async function loginUser(email, password) {
  const response = await fetch("http://localhost:8000/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email,
      password,
    }),
  });

  if (response.status === 401) {
    throw new Error("Invalid credentials");
  }

  return response.json();
}
```

3. **Manage conversation state**

Store the `conversation_id` returned from the API to maintain conversation context across multiple interactions.

### Integrating with Mobile Applications

For mobile applications, use the same API endpoints with appropriate HTTP clients for your platform:

```swift
// Swift example (iOS)
func sendChatMessage(text: String, userId: String?, conversationId: String?, mallId: Int?) {
    let url = URL(string: "http://localhost:8000/chat")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")

    let body: [String: Any?] = [
        "text": text,
        "user_id": userId,
        "conversation_id": conversationId,
        "mall_id": mallId,
        "language": "en"
    ]

    request.httpBody = try? JSONSerialization.data(withJSONObject: body.compactMapValues { $0 })

    // Continue with URLSession.shared.dataTask...
}
```

```kotlin
// Kotlin example (Android)
fun sendChatMessage(text: String, userId: String?, conversationId: String?, mallId: Int?) {
    val url = "http://localhost:8000/chat"
    val jsonObject = JSONObject().apply {
        put("text", text)
        userId?.let { put("user_id", it) }
        conversationId?.let { put("conversation_id", it) }
        mallId?.let { put("mall_id", it) }
        put("language", "en")
    }

    // Continue with your HTTP client (Retrofit, OkHttp, etc.)
}
```

## Extending the System

### Adding New Intents

To add new intent types to the customer graph:

1. Modify the `classify_intent` function in `customer.py`:

```python
async def classify_intent(state: CustomerState) -> CustomerState:
    # Add your new intent type to the prompt
    intent_prompt = PromptTemplate(
        input_variables=["query", "conversation_history"],
        template="""
        ... existing template ...
        - new_intent_type: Description of when to use this intent
        ... rest of template ...
        """
    )
    # Implementation
```

2. Update the routing logic if necessary:

```python
def route_after_classify(state: CustomerState):
    if state.intent == "new_intent_type":
        return "custom_node"
    # Existing routes
```

3. Add a new processing node for the intent:

```python
async def handle_new_intent(state: CustomerState) -> CustomerState:
    # Implementation
    return state

# Add to workflow
customer_workflow.add_node("custom_node", handle_new_intent)
customer_workflow.add_edge("custom_node", "generate_response")
```

### Adding New Entity Types for Tenants

To add new entity types to the tenant graph:

1. Modify the `analyze_intent` function in `tenant.py` to recognize the new entity type:

```python
intent_prompt = PromptTemplate(
    input_variables=["query", "conversation_history"],
    template="""
    ... existing template ...
    Valid entities: store, offer, product, new_entity_type
    ... rest of template ...
    """
)
```

2. Add handling for the new entity type in `prompt_for_missing_info`:

```python
async def prompt_for_missing_info(state: TenantState) -> TenantState:
    # Existing code

    elif state.entity_type == "new_entity_type":
        if state.action == "create":
            # Handle creation
        elif state.action == "update":
            # Handle updates
        # etc.

    return state
```

3. Implement database operations in `execute_operation`:

```python
async def execute_operation(state: TenantState) -> None:
    # Existing code

    elif state.entity_type == "new_entity_type":
        if state.action == "create":
            # Database insert
            # Vector index update
        # Other actions

    # Clean up state
```

### Creating Custom Nodes

To add custom processing nodes to either graph:

1. Define your node function:

```python
async def custom_process(state: CustomerState) -> CustomerState:
    """
    Custom processing function.

    This node:
    - Takes input from previous nodes
    - Performs custom processing
    - Updates state for next nodes
    """
    # Implementation
    return state
```

2. Add it to the appropriate workflow:

```python
workflow.add_node("custom_process", custom_process)
workflow.add_edge("some_previous_node", "custom_process")
workflow.add_edge("custom_process", "some_next_node")
```

## Database Schema

The database schema includes several key tables for managing conversations, users, and mall data:

### Conversations

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id VARCHAR(255),
    meta_data JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### Conversation Messages

```sql
CREATE TABLE conversation_messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### Stores

```sql
CREATE TABLE stores (
    store_id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(tenant_id),
    mall_id INTEGER REFERENCES malls(unique_property_id),
    name_en VARCHAR(255) NOT NULL,
    name_ar VARCHAR(255),
    location_en VARCHAR(255),
    location_ar VARCHAR(255)
);
```

### Products

```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    name_en VARCHAR(255) NOT NULL,
    name_ar VARCHAR(255),
    description_en TEXT,
    description_ar TEXT,
    price DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(10) NOT NULL
);
```

### Offers

```sql
CREATE TABLE offers (
    offer_id SERIAL PRIMARY KEY,
    store_id INTEGER REFERENCES stores(store_id),
    description_en TEXT NOT NULL,
    description_ar TEXT,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL
);
```

## Vector Search

The Cenomi Chatbot uses Pinecone for vector search functionality. Here's how to work with the vector search system:

### Creating Vector Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')

# Generate embeddings
vector = embeddings.embed_query("Your text here")
```

### Storing Vectors in Pinecone

```python
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("cenomicore")

# Store vector
index.upsert(vectors=[{
    "id": "unique_id_string",
    "values": vector,
    "metadata": {
        "type": "entity_type",
        "id": entity_id,
        "name_en": "Entity name",
        # Additional metadata
    }
}])
```

### Querying Vectors

```python
# Basic query
results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True
)

# Query with filters
results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True,
    filter={
        "type": "product",
        "mall_id": specific_mall_id
    }
)
```

## Testing and Deployment

### Running Tests

The Cenomi Chatbot includes a test suite that can be run with:

```bash
python -m pytest tests/
```

For specific test categories:

```bash
python -m pytest tests/test_customer.py  # Test customer graph
python -m pytest tests/test_tenant.py    # Test tenant graph
python -m pytest tests/test_api.py       # Test API endpoints
```

### Deployment

To deploy the Cenomi Chatbot in production:

#### Docker Deployment

1. Build the Docker image:

```bash
docker build -t cenomi-chatbot .
```

2. Run with Docker Compose:

```bash
docker-compose up -d
```

#### Server Deployment

1. Set up a Python environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment variables in a `.env` file.

3. Run with Gunicorn (for production):

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api:app
```

#### Database Migrations

To apply database migrations:

```bash
make db-migrate
```

To create a new migration:

```bash
make db-revision
```

### Monitoring

The system integrates with LangSmith for monitoring AI interactions. To view traces:

1. Log in to your LangSmith account
2. Navigate to the "cenomi-bot" project
3. View traces for detailed information about AI interactions

### Performance Tuning

Key performance parameters to consider:

1. **Database Connection Pool**: Adjust the pool size in `utils.py`:

```python
DB_POOL = await asyncpg.create_pool(min_size=5, max_size=20, **DB_CONFIG_ASYNC)
```

2. **Redis Cache TTL**: Modify the expiration time for cached items:

```python
REDIS_CLIENT.set(cache_key, json.dumps(data), ex=600)  # Increase from 300s to 600s
```

3. **Vector Search Parameters**: Adjust the number of results returned:

```python
results = index.query(vector=query_vector, top_k=10)  # Increase from 5 to 10
```
