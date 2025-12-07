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