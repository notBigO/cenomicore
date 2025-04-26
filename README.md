# Cenomi Chatbot

## Table of Contents

1. [Introduction](#introduction)
2. [Project Architecture](#project-architecture)
3. [System Components](#system-components)
   - [API Service](#api-service)
   - [Customer Graph](#customer-graph)
   - [Tenant Graph](#tenant-graph)
   - [Terminal Interface](#terminal-interface)
4. [Workflow](#workflow)
   - [Customer Interaction Flow](#customer-interaction-flow)
   - [Tenant Interaction Flow](#tenant-interaction-flow)
5. [Data Flow](#data-flow)
6. [Database Schema](#database-schema)
7. [Setup and Installation](#setup-and-installation)
8. [Environment Variables](#environment-variables)
9. [Integration Points](#integration-points)
10. [Contribution Guidelines](#contribution-guidelines)

## Introduction

The Cenomi Chatbot is an intelligent virtual assistant designed to enhance the shopping experience for mall customers and provide management tools for store tenants. The system leverages AI to answer customer queries about store locations, products, services, and offers, while also enabling tenants to manage their store information, products, and promotional offers.

The application is built using FastAPI as the backend framework, with Google's Gemini AI powering the conversational intelligence. It utilizes Pinecone for vector search capabilities, PostgreSQL for data storage, and Redis for caching.

## Project Architecture

The Cenomi Chatbot follows a modular architecture with clear separation of concerns:

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│                │      │                │      │                │
│  Client Apps   │◄────►│  API Service   │◄────►│  AI Services   │
│                │      │                │      │                │
└────────────────┘      └───────┬────────┘      └────────────────┘
                                │                        ▲
                                ▼                        │
                        ┌────────────────┐      ┌────────────────┐
                        │                │      │                │
                        │   Database     │◄────►│ Vector Search  │
                        │                │      │                │
                        └────────────────┘      └────────────────┘
```

- **Client Apps**: Terminal interface for interacting with the chatbot
- **API Service**: FastAPI endpoints handling chat requests and tenant updates
- **AI Services**: Gemini model for natural language processing
- **Database**: PostgreSQL for structured data storage
- **Vector Search**: Pinecone for semantic search capabilities

## System Components

### API Service

The API service (`api.py`) is the core interface that handles incoming requests from users. It provides the following endpoints:

- `POST /login`: Authenticates users (both customers and tenants)
- `POST /chat`: Processes customer queries about the mall, stores, and products
- `POST /tenant/update`: Handles tenant requests to update store information, products, and offers
- `GET /malls`: Returns a list of available malls

The API service routes requests to the appropriate processing nodes based on the user type (customer or tenant) and maintains conversation history.

### Customer Graph

The customer graph (`customer.py`) defines the conversation flow for mall customers. It consists of the following nodes:

1. **Intent Classification**: Analyzes the customer's query to determine the intent
2. **Initial Retrieval**: Fetches relevant information based on the query
3. **Context Refinement**: Processes and filters the retrieved information
4. **Response Generation**: Creates natural language responses using the Gemini model

The customer graph handles various customer intents, including:

- Store location queries
- Product availability inquiries
- Mall facility questions
- Promotions and offers
- Navigation assistance

### Tenant Graph

The tenant graph (`tenant.py`) manages the conversation flow for mall tenants. It consists of the following nodes:

1. **Intent Analysis**: Determines the tenant's intent regarding store, product, or offer management
2. **Missing Info Prompting**: Collects necessary information through a guided conversation
3. **Input Processing**: Processes tenant responses to prompts
4. **Operation Execution**: Performs database operations based on tenant requests

The tenant graph supports the following operations:

- Creating, updating, and deleting offers
- Managing product information
- Listing current offers and products
- Selecting which store to manage (for tenants with multiple stores)

### Terminal Interface

The terminal interface (`chat_terminal.py`) provides a command-line interface for interacting with the chatbot. It supports:

- User authentication
- Mall selection
- Conversation with the AI assistant
- Command-based interactions (`exit`, `login`, `mall`)

## Workflow

### Customer Interaction Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│              │     │              │     │              │     │              │
│  Classify    │────►│  Initial     │────►│  Refine      │────►│  Generate    │
│  Intent      │     │  Retrieval   │     │  Context     │     │  Response    │
│              │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

The customer workflow follows these steps:

1. **Intent Classification**:

   - Analyzes the query to determine what the customer is asking about
   - Categories include: store queries, product queries, navigation, services, etc.
   - Output: Structured intent data with query type and entities

2. **Initial Retrieval**:

   - Converts query to vector embeddings
   - Searches vector database for semantically relevant information
   - Retrieves store, product, or facility information based on intent
   - Output: Raw context data from vector search

3. **Context Refinement**:

   - Processes and filters the retrieved context
   - Prioritizes information based on relevance
   - Handles special cases like location queries
   - Output: Structured, filtered context ready for response generation

4. **Response Generation**:
   - Uses Gemini AI to generate natural language responses
   - Incorporates retrieved context and conversation history
   - Formats response based on intent type
   - Output: Human-friendly response text

### Tenant Interaction Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│              │     │              │     │              │     │              │
│  Analyze     │────►│  Prompt for  │────►│  Process     │────►│  Execute     │
│  Intent      │     │  Missing Info│     │  Input       │     │  Operation   │
│              │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

The tenant workflow follows these steps:

1. **Intent Analysis**:

   - Parses tenant queries to identify entity type (store, offer, product)
   - Determines action (create, update, delete, list)
   - Extracts initial data from query
   - Output: Structured intent with entity, action, and collected data

2. **Missing Info Prompting**:

   - Identifies what additional information is needed
   - Generates appropriate prompts to collect data
   - Handles store selection for multi-store tenants
   - Output: Tenant-facing question or processing decision

3. **Input Processing**:

   - Processes tenant responses to prompts
   - Validates selections (e.g., store numbers, offer choices)
   - Updates state with collected information
   - Output: Updated state with collected data

4. **Operation Execution**:
   - Performs database operations (create, update, delete)
   - Updates vector search index for retrieval
   - Clears Redis cache for affected entities
   - Output: Confirmation message or next steps

## Data Flow

The system processes data through several stages:

1. **User Input**:

   - Received through API endpoints
   - Packaged into state objects (CustomerState or TenantState)

2. **Context Retrieval**:

   - Vector embeddings generated from queries
   - Semantic search performed against Pinecone index
   - Database queries for additional context

3. **Response Processing**:

   - Context data structured and formatted
   - AI model generates natural language responses
   - Responses packaged into API responses

4. **Persistence**:
   - Conversations stored in PostgreSQL
   - Updates to products and offers reflected in database
   - Vector embeddings updated in Pinecone
   - Redis used for caching conversation history

## Database Schema

The system uses a PostgreSQL database with the following key tables:

- **conversations**: Stores conversation metadata and state
- **conversation_messages**: Contains individual messages within conversations
- **customers**: Customer account information
- **tenants**: Tenant account information
- **stores**: Store information linked to tenants
- **products**: Product information linked to stores
- **offers**: Promotional offers linked to stores
- **malls**: Mall information including name and location

## Setup and Installation

### Prerequisites

- Python 3.8+
- PostgreSQL
- Redis
- Pinecone account
- Google Gemini API key

### Installation Steps

1. Clone the repository

   ```bash
   git clone <repository-url>
   cd cenomi-chatbot
   ```

2. Create and activate a virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. Set up the environment variables (see Environment Variables section)

5. Run database migrations

   ```bash
   make db-migrate
   ```

6. Start the API server

   ```bash
   uvicorn src.api:app --reload
   ```

7. In a separate terminal, run the chat terminal
   ```bash
   python src/chat_terminal.py
   ```

## Environment Variables

Create a `.env` file with the following variables:

```
# Database
DB_NAME=cenomi_db
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# AI Services
GEMINI_API_KEY=your_gemini_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=cenomi-bot

# Vector Database
PINECONE_API_KEY=your_pinecone_api_key
```

## Integration Points

The system integrates with several external services:

- **Gemini AI**: For natural language processing and response generation
- **Pinecone**: For vector storage and semantic search
- **LangSmith**: For tracing and monitoring AI interactions
- **Redis**: For caching conversation history
- **PostgreSQL**: For persistent data storage

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

When contributing, please follow these best practices:

- Write comprehensive docstrings
- Follow the existing code style
- Add tests for new features
- Update documentation as needed
