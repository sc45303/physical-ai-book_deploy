from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from typing import Optional
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.models.user import User
from app.models.book import Book
from app.services.file_service import FileService
from app.services.document_parser import DocumentParserService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.database.vector_store import VectorStoreService
from app.models.responses import UploadResponse
from app.utils.logging import get_logger
from app.config import settings
import asyncio

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/books", tags=["books"])

# Global instances of services
file_service = FileService()
parser_service = DocumentParserService()
chunking_service = ChunkingService()
embedding_service = EmbeddingService()
vector_store_service = VectorStoreService()


async def process_uploaded_book(book_id: uuid.UUID, file_path: str):
    """
    Background task to process an uploaded book: parse, chunk, embed, and store
    """
    try:
        logger.info(f"Starting to process book with ID: {book_id}")

        # Parse the document
        text, file_format = await parser_service.parse_document(file_path)
        logger.info(f"Parsed document with format {file_format}, extracted {len(text)} characters")

        # Get page count if it's a PDF
        page_count = await parser_service.get_page_count(file_path)
        if page_count > 0:
            logger.info(f"Document has {page_count} pages")

        # Chunk the text
        chunks_with_tokens = chunking_service.chunk_text(
            text,
            method="tokens",
            max_size=512,
            overlap=50
        )
        logger.info(f"Text chunked into {len(chunks_with_tokens)} chunks using token-based method")

        # Process each chunk
        total_chunks = len(chunks_with_tokens)
        for idx, (chunk_text, token_count) in enumerate(chunks_with_tokens):
            logger.debug(f"Processing chunk {idx+1}/{total_chunks} with {token_count} tokens")

            # Create embedding
            embedding = await embedding_service.create_embedding(chunk_text)

            # Store in vector database
            vector_id = str(uuid.uuid4())
            success = await vector_store_service.store_embedding(
                vector_id=vector_id,
                vector=embedding,
                book_id=book_id,
                chunk_content=chunk_text,
                metadata={"chunk_index": idx, "token_count": token_count}
            )

            if success:
                logger.debug(f"Successfully stored chunk {idx+1}/{total_chunks} with vector ID: {vector_id}")
            else:
                logger.warning(f"Failed to store chunk {idx+1}/{total_chunks} with vector ID: {vector_id}")

        logger.info(f"Successfully processed and stored all {total_chunks} chunks for book ID: {book_id}")

    except Exception as e:
        logger.error(f"Error processing book {book_id}: {str(e)}", exc_info=True)
        # In a real implementation, we would update the book's status to "failed"
        raise


@router.post("/upload", response_model=UploadResponse)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a book file and initiate processing
    """
    logger.info(f"Starting upload process for file: {file.filename}")
    
    # Validate file size
    file_content = await file.read()
    file_size = len(file_content)
    
    if not file_service.validate_file_size(file_size):
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds limit of 50MB"
        )
    
    # Reset file pointer after reading
    await file.seek(0)
    
    # Save the uploaded file
    saved_file_path = await file_service.save_upload_file(file)
    
    # Extract title and author from filename if not provided
    if not title:
        title = file.filename.rsplit('.', 1)[0]  # Remove extension
    
    if not author:
        author = "Unknown"
    
    # Generate a unique book ID
    book_id = uuid.uuid4()
    
    # Create a new Book record in the database
    book = Book(
        user_id=uuid.uuid4(),  # In a real implementation, this would come from authentication
        title=title,
        author=author,
        format=file.filename.split('.')[-1].upper(),
        file_path=str(saved_file_path),
        processing_status="in_progress"
    )
    
    # In a complete implementation, we would add the book to the database session
    # db.add(book)
    # await db.commit()
    # await db.refresh(book)
    
    # Process the book in the background
    background_tasks.add_task(process_uploaded_book, book_id, str(saved_file_path))
    
    response = UploadResponse(
        id=book_id,
        title=book.title,
        author=book.author,
        format=book.format,
        processing_status=book.processing_status,
        message="File uploaded successfully, processing initiated in background"
    )
    
    logger.info(f"Upload initiated for book ID: {book_id}")
    
    return response