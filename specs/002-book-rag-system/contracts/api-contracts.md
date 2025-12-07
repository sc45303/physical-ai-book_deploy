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