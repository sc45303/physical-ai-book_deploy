from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.models.book import Book
from app.models.responses import ProcessingStatusResponse
from app.utils.logging import get_logger
from app.exceptions import BookNotFoundException

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/books", tags=["books"])


@router.get("/{book_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    book_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Check the processing status of an uploaded book
    """
    logger.info(f"Checking processing status for book ID: {book_id}")
    
    # In a real implementation, this would query the database for the book by ID
    # For now, we'll return a mock response based on the status we might expect
    
    # Mock implementation since we don't have a real database setup
    # In a real implementation, we would do:
    # book = await db.get(Book, book_id)
    # if not book:
    #     raise BookNotFoundException(str(book_id))
    # 
    # response = ProcessingStatusResponse(
    #     book_id=book.id,
    #     processing_status=book.processing_status,
    #     progress=0.8 if book.processing_status == "in_progress" else 1.0,
    #     message=f"Book processing is {book.processing_status}"
    # )
    
    # For the mock implementation, assume the status is completed
    response = ProcessingStatusResponse(
        book_id=book_id,
        processing_status="completed",  # For demo purposes
        progress=1.0,
        message="Book processing completed"
    )
    
    logger.info(f"Retrieved processing status for book ID: {book_id} - Status: {response.processing_status}")
    
    return response