# Cenomi Chatbot Backend Documentation

## Project Overview

The Cenomi Chatbot backend is a FastAPI-based service that powers an intelligent virtual assistant for mall customers. It answers questions about stores, products, offers, and services etc. The backend integrates Pinecone for semantic vector search and OpenAI's GPT-4o-mini for natural language understanding and response generation.

---

## Tech Stack

- **Language:** Python 3.8+
- **Framework:** FastAPI
- **Vector Search:** Pinecone
- **LLM:** OpenAI GPT-4o-mini (or Gemini, configurable)
- **Database:** PostgreSQL
- **Cache:** Redis
- **Other:** spaCy (NLP), HuggingFace Embeddings, LangChain, LangGraph

---

## API Endpoints

| Method | Path             | Description                                      | Request Model         | Response Model/Format          |
| ------ | ---------------- | ------------------------------------------------ | --------------------- | ------------------------------ |
| POST   | `/login`         | Authenticate user (customer or tenant)           | `{ email, password }` | `{ user_id }`                  |
| POST   | `/chat`          | Customer chat endpoint (main assistant)          | `ChatRequest`         | `ChatResponse`                 |
| POST   | `/tenant/update` | Tenant update endpoint (manage store/offer/etc.) | `UpdateRequest`       | `{ message, conversation_id }` |
| POST   | `/tts`           | Text-to-speech audio generation                  | `TTSRequest`          | `{ audio_base64, media_type }` |
| GET    | `/malls`         | List all available malls                         | -                     | `[ { mall_id, name_en } ]`     |
| GET    | `/`              | Health check                                     | -                     | `{ message }`                  |

### Request/Response Models

- **ChatRequest**: `{ text, audio?, user_id?, language?, conversation_id?, mall_id?, include_tts? }`
- **ChatResponse**: `{ message, conversation_id, audio_base64?, recommendations?, is_recommendation_format?, follow_up_question? }`
- **UpdateRequest**: `{ text, user_id, language?, conversation_id? }`
- **TTSRequest**: `{ text, language, speed? }`

---

## Vector Search (Pinecone)

- **Embedding:** User queries are embedded using HuggingFace's `paraphrase-multilingual-MiniLM-L12-v2` model.
- **Indexing:** All stores, products, amenities, etc., are indexed in Pinecone with metadata (e.g., type, mall_id).
- **Query:** For each user query, the backend:
  1. Embeds the query.
  2. Searches Pinecone with relevant filters (e.g., `mall_id`, `type`).
  3. Retrieves the top-k most relevant results, which are then used as context for the LLM.
- **Fallback:** If not enough results are found, the system may broaden the search (e.g., from products to stores).

---

## Environment Setup

### Prerequisites

- Python 3.8+
- PostgreSQL
- Redis
- Pinecone account
- OpenAI API key (or Gemini API key)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cenomi-chatbot
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   Create a `.env` file in the root directory with the following:

   ```env
   # Database
   DB_NAME=cenomi_db
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432

   # AI Services
   OPENAI_API_KEY=your_openai_api_key
   # or
   GEMINI_API_KEY=your_gemini_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_PROJECT=cenomi-bot

   # Vector Database
   PINECONE_API_KEY=your_pinecone_api_key
   ```

5. **Run database migrations:**
   ```bash
   make db-migrate
   ```
6. **Start the API server:**
   ```bash
   uvicorn src.api:app --reload
   ```
7. **(Optional) Start the chat terminal:**
   ```bash
   python src/chat_terminal.py
   ```

---

## Additional Notes

- **Tracing & Monitoring:** Integrates with LangSmith for tracing LLM calls.
- **Caching:** Uses Redis and in-memory cache for fast repeated queries and TTS.
- **Extensibility:** The workflow is modular and can be extended for new query types or business logic.
- **Security:** Ensure your `.env` file is not committed to version control.
