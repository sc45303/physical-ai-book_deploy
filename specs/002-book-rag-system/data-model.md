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