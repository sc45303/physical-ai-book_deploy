---
id: 2
title: Plan Book RAG System Implementation
stage: plan
date: 2025-12-07
surface: agent
model: Qwen
feature: book-rag-system
branch: 002-book-rag-system
user: default_user
command: /sp.plan
labels: [feature, plan, rag, books, architecture]
links:
  spec: ../specs/002-book-rag-system/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/002-book-rag-system/plan.md
  - specs/002-book-rag-system/research.md
  - specs/002-book-rag-system/data-model.md
  - specs/002-book-rag-system/quickstart.md
  - specs/002-book-rag-system/contracts/api-contracts.md
tests:
  - specs/002-book-rag-system/checklists/requirements.md
---

## Prompt

$ARGUMENTS

## Response snapshot

# Implementation Plan: Book-based RAG System with FastAPI, PostgreSQL and Qdrant Vector Database

**Branch**: `002-book-rag-system` | **Date**: 2025-12-07 | **Spec**: [link](spec.md)
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a Retrieval Augmented Generation (RAG) system that allows users to upload books in various formats (PDF, DOCX, TXT), process them into vector embeddings, store them in a vector database, and enable semantic search and AI-powered Q&A capabilities. The system will use FastAPI for the backend, PostgreSQL for metadata storage, Qdrant for vector storage, and an LLM (OpenAI/Gemini) for answer generation.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic, Qdrant, OpenAI/Gemini SDK, Unstructured (for document parsing)
**Storage**: PostgreSQL (Neon Serverless) for metadata and chat history, Qdrant (Cloud) for vector embeddings
**Testing**: pytest for unit and integration tests
**Target Platform**: Linux server (deployed as web API)
**Project Type**: Web application (backend API with potential frontend integration)
**Performance Goals**: Process books up to 1000 pages within 5 minutes, return search results within 2 seconds, support 99% uptime
**Constraints**: <200ms p95 response time for queries, must handle 1000+ concurrent users, answers must be grounded in book content without hallucination
**Scale/Scope**: Support up to 10,000 books, 1M+ vector embeddings, 100k+ daily queries

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Compliance Check:
1. **Accuracy**: Implementation will use real, existing technologies (FastAPI, PostgreSQL, Qdrant) as specified in the feature requirements
2. **Clarity**: The implementation will follow clean, well-documented code practices with clear variable names and comprehensive comments
3. **Depth & Rigor**: The system will implement proper RAG architecture with vector embeddings, semantic search, and LLM integration
4. **Traceability**: All code artifacts will connect explicitly back to the functional requirements in the spec
5. **Spec-Driven Workflow**: This plan follows the /sp.plan specification-driven approach as required
6. **Zero Ambiguity**: All components will be precisely defined with no room for interpretation in implementation steps

### Gates:
- ✅ All technologies mentioned exist and are accessible
- ✅ Implementation approach is technically sound
- ✅ Performance goals align with feature requirements
- ✅ Architecture follows industry best practices for RAG systems

## Project Structure

### Documentation (this feature)

```text
specs/002-book-rag-system/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── app/
│   ├── main.py
│   ├── models/
│   │   ├── book.py
│   │   ├── user.py
│   │   ├── chat_session.py
│   │   └── vector_embedding.py
│   ├── services/
│   │   ├── document_parser.py
│   │   ├── embedding_service.py
│   │   ├── vector_store_service.py
│   │   ├── rag_service.py
│   │   └── llm_service.py
│   ├── routes/
│   │   ├── upload.py
│   │   ├── query.py
│   │   ├── chat.py
│   │   └── search.py
│   └── database/
│       ├── database.py
│       └── connections.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── requirements.txt
├── .env.example
└── README.md
```

**Structure Decision**: Web application structure with backend API containing models, services, and routes to handle book uploads, processing, embedding, storage, and retrieval functions. The structure supports the multi-tier architecture required for a RAG system with document processing, vector storage, and LLM interaction.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|

# Research Summary: Book-based RAG System

## Decision: Technology Stack Selection
**Rationale**: Selected FastAPI + PostgreSQL + Qdrant + OpenAI/Gemini based on requirements for a robust RAG system
**Alternatives considered**: 
- LangChain vs. custom implementation for RAG pipeline
- Pinecone vs. Qdrant for vector storage
- Different LLM providers (Anthropic, OpenAI, Gemini)

## Decision: Document Processing Pipeline
**Rationale**: Using Unstructured library to handle multiple document formats (PDF, DOCX, TXT) with proper text extraction and formatting preservation
**Alternatives considered**:
- PyPDF2 vs. pdfplumber for PDF processing
- Custom parsing vs. established libraries
- Synchronous vs. asynchronous processing for uploads

## Decision: Embedding Model Selection
**Rationale**: Using OpenAI's text-embedding-3-large or similar high-quality embedding model for best semantic search results
**Alternatives considered**:
- Open-source models (Sentence Transformers, BERT-based)
- Different OpenAI embedding models (text-embedding-3-small vs. large)
- Provider-specific models (Cohere, Google)

## Decision: Chunking Strategy
**Rationale**: Implementing semantic chunking with overlap to preserve context across chunks while maintaining search efficiency
**Alternatives considered**:
- Fixed-size chunking (by characters/words/tokens)
- Sentence-boundary-aware chunking
- Recursive chunking by document structure

## Decision: Vector Storage Architecture
**Rationale**: Using Qdrant for vector similarity search with metadata filtering capabilities
**Alternatives considered**:
- Pinecone (managed but more expensive)
- Weaviate (open-source alternative)
- Custom solution with PostgreSQL + pgvector

## Decision: API Design Pattern
**Rationale**: RESTful API with FastAPI for type safety, automatic documentation, and async capabilities
**Alternatives considered**:
- GraphQL for more flexible querying
- gRPC for internal services
- Serverless functions for scalability

# Data Model: Book-based RAG System

## Entity: User
- **Fields**:
  - id (UUID/Integer): Unique identifier
  - email (String): User's email address
  - created_at (DateTime): Account creation timestamp
  - updated_at (DateTime): Last update timestamp
- **Relationships**:
  - One-to-Many: User has many Books
  - One-to-Many: User has many ChatSessions

## Entity: Book
- **Fields**:
  - id (UUID/Integer): Unique identifier
  - user_id (UUID/Integer): Foreign key to User
  - title (String): Book title
  - author (String): Book author
  - format (String): File format (PDF, DOCX, TXT)
  - file_path (String): Path to stored file
  - page_count (Integer): Number of pages in the book
  - upload_date (DateTime): When the book was uploaded
  - processing_status (String): Status of processing (pending, in_progress, completed, failed)
  - created_at (DateTime): Record creation timestamp
  - updated_at (DateTime): Last update timestamp
- **Relationships**:
  - Many-to-One: Book belongs to User
  - One-to-Many: Book has many BookChunks

## Entity: BookChunk
- **Fields**:
  - id (UUID/Integer): Unique identifier
  - book_id (UUID/Integer): Foreign key to Book
  - chunk_index (Integer): Sequential index of the chunk in the book
  - content (Text): The text content of the chunk
  - token_count (Integer): Number of tokens in the chunk
  - vector_id (String): ID in the vector database (Qdrant)
  - created_at (DateTime): Record creation timestamp
- **Relationships**:
  - Many-to-One: BookChunk belongs to Book
  - One-to-One: BookChunk maps to VectorEmbedding (by vector_id)

## Entity: ChatSession
- **Fields**:
  - id (UUID/Integer): Unique identifier
  - user_id (UUID/Integer): Foreign key to User
  - book_id (UUID/Integer): Foreign key to Book
  - session_name (String): Optional name for the session
  - mode (String): Chat mode ('global' or 'selected_text')
  - created_at (DateTime): Session creation timestamp
  - updated_at (DateTime): Last interaction timestamp
- **Relationships**:
  - Many-to-One: ChatSession belongs to User
  - Many-to-One: ChatSession belongs to Book
  - One-to-Many: ChatSession has many ChatMessages

## Entity: ChatMessage
- **Fields**:
  - id (UUID/Integer): Unique identifier
  - chat_session_id (UUID/Integer): Foreign key to ChatSession
  - role (String): Message role ('user' or 'assistant')
  - content (Text): The message content
  - sources (JSON): List of source chunks used for answer generation
  - created_at (DateTime): Message timestamp
- **Relationships**:
  - Many-to-One: ChatMessage belongs to ChatSession

## Entity: VectorEmbedding
- **Fields**:
  - id (String): ID in the vector database (Qdrant)
  - book_chunk_id (UUID/Integer): Reference to the BookChunk
  - model_used (String): The embedding model used
  - created_at (DateTime): When the embedding was created
- **Relationships**:
  - One-to-One: VectorEmbedding maps to BookChunk (by book_chunk_id)

## Validation Rules
- **User**: Email must be valid, unique
- **Book**: Title and author required, file format must be supported, user_id required
- **BookChunk**: Content required, token_count must be positive, chunk_index must be unique per book
- **ChatSession**: User_id and book_id required, mode must be valid
- **ChatMessage**: Role must be 'user' or 'assistant', content required, chat_session_id required

# API Contract: Book-based RAG System

## Upload Book Endpoint
- **Path**: `/api/v1/books/upload`
- **Method**: `POST`
- **Purpose**: Upload a book file for processing
- **Request**:
  - Content-Type: `multipart/form-data`
  - Form fields:
    - `file`: Book file (PDF, DOCX, or TXT)
    - `title`: Book title (optional, extracted from file if not provided)
    - `author`: Book author (optional)
- **Response**:
  - Success: `201 Created` with Book object
  - Error: `400 Bad Request` if file format invalid
  - Error: `413 Payload Too Large` if file exceeds size limits

## Get Processing Status
- **Path**: `/api/v1/books/{book_id}/status`
- **Method**: `GET`
- **Purpose**: Check the processing status of an uploaded book
- **Response**:
  - Success: `200 OK` with processing status
  - Error: `404 Not Found` if book doesn't exist

## Search Book Content
- **Path**: `/api/v1/books/{book_id}/search`
- **Method**: `POST`
- **Purpose**: Semantic search within a specific book
- **Request**:
  - Content-Type: `application/json`
  - Body: `{ "query": "search query text", "top_k": 5 }`
- **Response**:
  - Success: `200 OK` with array of relevant text chunks
  - Error: `404 Not Found` if book doesn't exist

## Chat with Book (Global Mode)
- **Path**: `/api/v1/chat/global`
- **Method**: `POST`
- **Purpose**: Ask questions about the book with access to all content
- **Request**:
  - Content-Type: `application/json`
  - Body: `{ "book_id": "uuid", "message": "user question", "session_id": "optional uuid" }`
- **Response**:
  - Success: `200 OK` with answer and sources
  - Error: `404 Not Found` if book doesn't exist

## Chat with Selected Text
- **Path**: `/api/v1/chat/selected-text`
- **Method**: `POST`
- **Purpose**: Ask questions about specific selected text only
- **Request**:
  - Content-Type: `application/json`
  - Body: `{ "book_id": "uuid", "selected_text": "text that was selected", "message": "user question", "session_id": "optional uuid" }`
- **Response**:
  - Success: `200 OK` with answer based only on selected text
  - Error: `404 Not Found` if book doesn't exist

## Start Chat Session
- **Path**: `/api/v1/sessions`
- **Method**: `POST`
- **Purpose**: Create a new chat session
- **Request**:
  - Content-Type: `application/json`
  - Body: `{ "book_id": "uuid", "session_name": "optional name", "mode": "global|selected_text" }`
- **Response**:
  - Success: `201 Created` with session object
  - Error: `404 Not Found` if book doesn't exist

## List User Books
- **Path**: `/api/v1/books`
- **Method**: `GET`
- **Purpose**: Retrieve all books owned by the authenticated user
- **Response**:
  - Success: `200 OK` with array of book objects

# Quickstart Guide: Book-based RAG System

## Prerequisites
- Python 3.11+
- PostgreSQL (Neon Serverless account)
- Qdrant Cloud account
- OpenAI or Google Gemini API key

## Setup

### 1. Clone and Install Dependencies
```bash
git clone [repository-url]
cd [repository-name]
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file with the following variables:
```env
NEON_DB_URL=postgresql://username:password@ep-xxxxx.us-east-1.aws.neon.tech/dbname?sslmode=require
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key  # Optional if using Gemini
EMBED_MODEL=text-embedding-3-large
CHAT_MODEL=gpt-4.1-mini
```

### 3. Database Setup
```bash
# Run database migrations
python -m alembic upgrade head
```

## Usage

### 1. Start the API Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 2. Upload a Book
```bash
curl -X POST "http://localhost:8000/api/v1/books/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/book.pdf" \
  -F "title=My Book Title" \
  -F "author=Author Name"
```

### 3. Check Processing Status
```bash
curl -X GET "http://localhost:8000/api/v1/books/{book_id}/status"
```

### 4. Query Your Book
```bash
curl -X POST "http://localhost:8000/api/v1/chat/global" \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "your-book-uuid",
    "message": "What is this book about?",
    "session_id": "optional-session-uuid"
  }'
```

## API Reference
For detailed API documentation, visit `http://localhost:8000/docs` when the server is running.

## Troubleshooting
- If uploads fail, check file size limits in your configuration
- If queries return no results, verify that your book has completed processing
- For vector search issues, confirm Qdrant connection and API keys

## Outcome

- ✅ Impact: Successfully created comprehensive implementation plan with all required artifacts
- 🧪 Tests: All artifacts follow the required specification format
- 📁 Files: Created plan.md, research.md, data-model.md, quickstart.md, and api-contracts.md
- 🔁 Next prompts: Ready for /sp.tasks to generate implementation tasks
- 🧠 Reflection: The plan provides a complete technical architecture for the book RAG system

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS - All artifacts meet specification requirements
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A