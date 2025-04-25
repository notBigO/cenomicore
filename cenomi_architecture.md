# CenomiCore AI Chatbot Architecture

## Overview

CenomiCore is a comprehensive AI assistant built for mall environments, using a sophisticated architecture that combines LangChain, vector databases, graph-based workflows, and hybrid retrieval systems.

## Core Components

### 1. Conversational Agents

- **Customer Agent**: Serves mall visitors, providing information about stores, products, offers, and events
- **Tenant Agent**: Serves mall tenants for management operations (less details in the code snippets provided)

### 2. Backend Infrastructure

- **FastAPI Application**: RESTful API endpoints for chat and tenant management
- **PostgreSQL Database**: Stores structured data about malls, brands, products, engagements
- **Pinecone Vector Database**: Stores embeddings for semantic search
- **Redis**: Caches conversation history and responses for performance
- **LangSmith**: Handles tracing and monitoring of AI operations

## Language Models & NLP

- **Primary LLM**: OpenAI's gpt-4o-mini (previously used Gemini)
- **Embeddings**: HuggingFace embeddings with paraphrase-multilingual-MiniLM-L12-v2
- **NLP Processing**: spaCy for entity extraction from queries

## Graph-Based Workflow Architecture

The system uses LangGraph (from LangChain) to implement directed state graphs for conversation flow:

### Customer Workflow Graph

```
classify_intent → [conditional routing]
                  ├── initial_retrieval → refine_context → respond → END
                  ├── fetch_loyalty_data → END
                  └── respond → END (for "other" intents)
```

1. **classify_intent node**:

   - Determines user's intent (e.g., store_info, product_recommend)
   - Extracts entity information and type preferences

2. **initial_retrieval node**:

   - Uses embeddings to find relevant vector matches in Pinecone
   - Caches results to reduce latency

3. **refine_context node**:

   - Builds comprehensive context using both vector search and relational database
   - Creates a structured context object with stores, products, offers, events, etc.
   - Handles type preferences and categories

4. **respond node**:

   - Generates natural language responses from the context
   - Personalizes based on conversation history

5. **fetch_loyalty_data node**:
   - Specialized node for loyalty-related queries

## Knowledge Representation

### 1. Vector Embeddings

The system uses `generate_embeddings.py` to create and store vector embeddings for:

- Malls (locations)
- Brands/Stores
- Products
- Engagements (offers and events)
- Services

Each entity is embedded in both English and Arabic using a multilingual model.

### 2. Knowledge Graph

A NetworkX graph structure represents relationships between:

- Brands/stores and their products
- Brands and their engagements (offers/events)
- Spatial relationships between stores

## Hybrid Retrieval System

The system implements a hybrid retrieval approach:

1. **Vector Search (Semantic Retrieval)**:

   - Converts user queries to embeddings
   - Searches Pinecone for semantically similar content
   - Filters by mall ID and other metadata

2. **Direct Database Queries**:

   - Retrieves structured data like store information and products
   - Handles filtering and sorting based on intent

3. **Knowledge Graph Traversal**:
   - Finds relationships between entities (e.g., neighboring stores)
   - Connects products to brands and offers

## Database Schema Highlights

- **Conversations**: Stores chat history with metadata
- **Brands**: Contains store information with PMS unit codes for mall location
- **Products**: Links to brands with attributes and availability
- **Engagements**: Stores both offers and events with time constraints
- **Services**: Stores amenity information

## Key Workflows

### 1. Chat Processing

1. User sends message through `/chat` endpoint
2. System detects language or uses specified language
3. Retrieves conversation history from DB or Redis cache
4. Creates CustomerState with query, history, and mall context
5. Processes through LangGraph workflow
6. Stores response in conversation history
7. Returns response to user

### 2. Intent Classification

Uses a carefully designed prompt to classify user intent into entity types:

- store, offer, product, event, service, amenity, loyalty
  And action types:
- info, navigate, recommend, list, balance, programs

### 3. Context Building

1. Performs vector search via Pinecone based on query
2. Retrieves relevant entities from PostgreSQL
3. Builds context information organized by entity type
4. Filters and prioritizes results based on intent and preferences
5. Adds neighboring stores when appropriate

### 4. Response Generation

Uses a prompt-based system with specific rules for different entity types:

- Store information includes location, category, description
- Product queries include price, availability, store location
- Dining recommendations include cuisine and location
- Offers include details, conditions, dates

## Performance Optimization

- **Redis Caching**: Stores conversation history and initial search results
- **Batched Processing**: Handles embeddings in batches to avoid overloading Pinecone
- **Multilingual Support**: Built-in support for English and Arabic

## System Integration Points

- **Mall Property Management**: Integrates with PMS unit codes for store locations
- **Customer Management**: Links conversations to user IDs
- **Tenant Management**: Separate workflow for store management
