# Cenomi Chatbot Technical Overview

## Graph-Based Conversational Architecture

The Cenomi Chatbot is built using a graph-based approach to conversational AI, using LangGraph (built on LangChain) to manage complex conversational flows with structured state management. The system consists of two primary graph implementations:

1. Customer-facing graph (`customer_graph`)
2. Tenant-facing graph (`tenant_graph`)

Each graph maintains its own state object and defines a specific sequence of processing nodes to handle different types of conversations. This approach allows for:

- Clear separation of processing steps
- Structured state management
- Deterministic conversation flows
- Explicit handling of different user types

## State Objects

### CustomerState

The `CustomerState` class defines the state for customer-facing conversations:

```python
class CustomerState(PydanticBaseModel):
    query: str                                   # Customer's current query
    user_id: Optional[str] = None                # Optional customer ID
    language: str                                # Language code (en, ar, etc.)
    conversation_id: str                         # Unique conversation identifier
    conversation_history: List[Dict[str, str]] = [] # Previous messages
    response: Optional[str] = None               # Generated response
    context_data: Optional[Dict[str, Any]] = None # Retrieved context information
    intent: Optional[str] = None                 # Classified intent
    initial_context: Optional[List[Dict[str, Any]]] = None # Raw vector search results
    suggestions: Optional[List[str]] = None      # Follow-up suggestions
    mall_id: Optional[int] = None                # Mall identifier
    direct_response: Optional[str] = None        # Immediate response (bypassing processing)
    type_preference: Optional[str] = None        # User's preference for product types
    needs_type_follow_up: bool = False           # Flag for type preference follow-up
```

### TenantState

The `TenantState` class defines the state for tenant-facing conversations:

```python
class TenantState(BaseModel):
    query: str                                  # Tenant's current query
    user_id: str                                # Tenant ID (prefixed with "t_")
    language: str = "en"                        # Language code
    conversation_id: str                        # Unique conversation identifier
    conversation_history: List[Dict[str, str]] = [] # Previous messages
    entity_type: Optional[str] = None           # Entity being managed (store, offer, product)
    action: Optional[str] = None                # Action being performed (create, update, delete, list)
    collected_data: Dict[str, Any] = {}         # Information collected during conversation
    current_step: Optional[str] = None          # Current step in the information gathering process
    store_name: Optional[str] = None            # Name of store being managed
    response: Optional[str] = None              # Generated response
    offer_list: Optional[List[Dict[str, Any]]] = None # List of offers for selection
```

## Customer Graph Workflow

The customer graph is defined with the following nodes:

```python
customer_workflow = StateGraph(CustomerState)
customer_workflow.add_node("classify_intent", classify_intent)
customer_workflow.add_node("initial_retrieval", initial_retrieval)
customer_workflow.add_node("refine_context", refine_context)
customer_workflow.add_node("generate_response", generate_response)
customer_workflow.add_node("fetch_loyalty_data", fetch_loyalty_data)

# Define the edges
customer_workflow.set_entry_point("classify_intent")
customer_workflow.add_conditional_edges(
    "classify_intent",
    route_after_classify,
    {
        "loyalty": "fetch_loyalty_data",
        "default": "initial_retrieval"
    }
)
customer_workflow.add_edge("fetch_loyalty_data", "generate_response")
customer_workflow.add_edge("initial_retrieval", "refine_context")
customer_workflow.add_edge("refine_context", "generate_response")
customer_workflow.add_edge("generate_response", END)
```

### Node Details

#### 1. classify_intent

This node analyzes the customer's query to determine the intent type:

```python
async def classify_intent(state: CustomerState) -> CustomerState:
    """
    Classifies the user's query into specific intents.

    Supported intents:
    - store_location: Questions about where a store is located
    - store_category: Questions about types of stores
    - product_info: Questions about products
    - mall_facilities: Questions about mall amenities
    - mall_info: General mall information
    - offer_info: Questions about current offers or promotions
    - loyalty_info: Questions about loyalty program
    - navigation: Help with finding directions
    """
    # Implementation details...

    return state
```

Key features:

- Uses LLM to categorize the query into predefined intent types
- Identifies entities mentioned in the query
- Handles multi-language classification
- Processes conversation history for context

#### 2. initial_retrieval

This node retrieves relevant information based on the intent:

```python
async def initial_retrieval(state: CustomerState) -> CustomerState:
    """
    Retrieves initial context based on the query and intent.

    This node:
    - Converts the query to vector embeddings
    - Searches the knowledge base for relevant information
    - Retrieves store, product, or facility information
    - Structures the initial context for further processing
    """
    # Implementation details...

    return state
```

Key features:

- Generates vector embeddings from the query
- Performs semantic search against Pinecone index
- Uses metadata filters based on intent type
- Handles mall-specific context retrieval

#### 3. refine_context

This node processes and filters the retrieved context:

```python
async def refine_context(state: CustomerState) -> CustomerState:
    """
    Refines the retrieved context to improve response relevance.

    This node:
    - Sorts and filters context by relevance
    - Restructures data for response generation
    - Handles special cases for different intents
    - Prepares context for LLM-based response generation
    """
    # Implementation details...

    return state
```

Key features:

- Prioritizes information based on relevance scores
- Formats location data for navigation queries
- Combines information from different sources
- Handles special cases for product queries

#### 4. fetch_loyalty_data

This node is conditionally called for loyalty-related queries:

```python
async def fetch_loyalty_data(state: CustomerState) -> CustomerState:
    """
    Retrieves customer loyalty information.

    This node:
    - Fetches loyalty program details
    - Gets customer-specific point balances and rewards
    - Formats loyalty data for response generation
    """
    # Implementation details...

    return state
```

#### 5. generate_response

This node creates the final response using the Gemini model:

```python
async def generate_response(state: CustomerState) -> CustomerState:
    """
    Generates the final response to the user's query.

    This node:
    - Uses LLM to generate natural language responses
    - Incorporates context data retrieved from previous nodes
    - Formats the response according to intent type
    - Handles edge cases and fallbacks
    """
    # Implementation details...

    return state
```

Key features:

- Uses a context-aware prompt template
- Includes conversation history for continuity
- Formats responses based on intent type
- Handles multilingual response generation

## Tenant Graph Workflow

The tenant graph is defined with a simpler structure:

```python
tenant_workflow = StateGraph(TenantState)
tenant_workflow.add_node("process", tenant_recognize_intent)
tenant_workflow.set_entry_point("process")
tenant_workflow.add_edge("process", END)
```

The `tenant_recognize_intent` function combines multiple processing steps:

```python
async def tenant_recognize_intent(state: TenantState) -> TenantState:
    """
    Main processing function for tenant interactions.

    This function:
    1. Retrieves conversation history
    2. Analyzes intent if no current step is active
    3. Processes input for ongoing conversations
    4. Generates formatted responses
    """
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
```

### Tenant Graph Processing Steps

Although implemented as a single node, the tenant graph processing consists of several logical steps:

#### 1. analyze_intent

```python
async def analyze_intent(state: TenantState) -> TenantState:
    """
    Analyzes the tenant's intent regarding store management.

    This function:
    - Determines what entity the tenant wants to work with (store, offer, product)
    - Identifies the action to perform (create, update, delete, list)
    - Extracts initial data from the query
    - Handles store selection for multi-store tenants
    """
    # Implementation details...

    return state
```

Key features:

- Uses LLM to extract structured intent data
- Handles store identification and selection
- Validates tenant permissions for the requested store
- Extracts entity details from natural language queries

#### 2. prompt_for_missing_info

```python
async def prompt_for_missing_info(state: TenantState) -> TenantState:
    """
    Identifies and requests missing information for the current operation.

    This function:
    - Determines what additional details are needed
    - Generates appropriate prompts based on entity type and action
    - Handles special cases for different operations
    - Executes operations when all required data is collected
    """
    # Implementation details...

    return state
```

Key features:

- Implements a multi-step data collection process
- Generates context-aware prompts
- Validates store ownership
- Handles different data requirements for different entities

#### 3. process_input

```python
async def process_input(state: TenantState) -> TenantState:
    """
    Processes tenant responses during multi-step operations.

    This function:
    - Handles responses to previous prompts
    - Validates selections (store numbers, offer choices)
    - Updates the state with collected information
    - Determines the next step in the process
    """
    # Implementation details...

    return state
```

Key features:

- Handles numeric selections from lists
- Validates input against available options
- Supports structured form-like data collection
- Maintains conversation state across multiple turns

#### 4. execute_operation

```python
async def execute_operation(state: TenantState) -> None:
    """
    Executes database operations based on collected information.

    This function:
    - Creates, updates, or deletes entries in the database
    - Updates vector search indices
    - Invalidates cache entries
    - Generates confirmation messages
    """
    # Implementation details...
```

Key features:

- Performs database CRUD operations
- Updates Pinecone vector indices
- Clears Redis cache for affected entities
- Generates user-friendly confirmation messages

## Vector Search Implementation

The system uses Pinecone for semantic search functionality:

```python
# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("cenomicore")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
```

The vector search is used for:

1. **Product Search**: Finding products across the mall based on descriptions
2. **Store Search**: Locating stores by name, category, or product type
3. **Offer Search**: Finding current promotions and discounts

Each entity is stored with detailed metadata:

```python
# Example product vector
{
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
}
```

## LLM Integration

The system uses Google's Gemini model for natural language understanding and generation:

```python
# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
```

Key prompt templates include:

1. **Customer Response Prompt**: Generates customer-facing responses with context
2. **Intent Classification Prompt**: Analyzes query intent and entities
3. **Tenant Intent Prompt**: Parses tenant management requests
4. **Tenant Response Prompt**: Formats tenant-facing responses

Example prompt (simplified):

```python
customer_prompt = PromptTemplate(
    input_variables=["context", "query", "lang", "conversation_history"],
    template="""
    You are CenomiAI, a friendly and highly knowledgeable mall assistant designed to enhance
    the shopping experience. Respond in {lang} with a warm, conversational tone, using emojis
    to keep it engaging...

    Current Query:
    "{query}"

    Context:
    {context}

    Conversation History:
    {conversation_history}
    """
)
```

## Caching Strategy

The system uses Redis for caching conversation history:

```python
# Redis setup
REDIS_CLIENT = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
```

Caching is implemented for:

1. **Conversation History**: To avoid repeated database queries
2. **Context Data**: To speed up repeated queries
3. **Vector Search Results**: To improve response time for similar queries

Cache invalidation occurs when:

- Store information is updated
- Products are added, modified, or removed
- Offers are created or updated

## Transaction Handling

All database operations within the tenant graph are executed atomically to ensure data consistency. The system uses PostgreSQL's transaction capabilities through the `asyncpg` library.

Examples of transaction handling:

```python
# Creating a new offer
await db_execute_async(
    "INSERT INTO offers (store_id, description_en, description_ar, start_date, end_date) VALUES ($1, $2, $3, $4, $5)",
    (store_id, description, description, start_date, end_date)
)

# Updating a product
await db_execute_async(
    "UPDATE products SET name_en = $1, name_ar = $2 WHERE product_id = $3",
    (new_name, new_name, product_id)
)
```

## Performance Considerations

Key performance optimizations include:

1. **Async Processing**: All database and LLM operations use async/await for non-blocking I/O
2. **Intelligent Caching**: Redis caching of conversation history and query results
3. **Selective Vector Updates**: Only modified entities are updated in the vector index
4. **Conversation Pruning**: Only the most recent messages are used for context

## Security Implementation

The system implements several security features:

1. **Tenant Authentication**: Verifies tenant identity before allowing store modifications
2. **Store Ownership Validation**: Ensures tenants can only modify their own stores
3. **Input Validation**: Sanitizes all user inputs before database operations
4. **Parameterized Queries**: Prevents SQL injection attacks
