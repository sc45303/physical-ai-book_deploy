from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.services.rag_service import RAGService
from app.services.search_service import SearchService
from app.models.responses import ChatResponse
from app.utils.logging import get_logger
from app.config import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Global instances of services
rag_service = RAGService()
search_service = SearchService()


@router.post("/global", response_model=ChatResponse)
async def chat_global(
    book_id: UUID,
    message: str,
    session_id: Optional[UUID] = None,
    top_k: int = 5,
    max_context_tokens: int = 3000,
    include_history: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Chat endpoint for global mode - answers questions using all content in the book
    """
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(f"Processing global chat request for book {book_id} with message: {message[:50]}...")

    try:
        # If session_id is provided, retrieve conversation history
        conversation_history = []
        if session_id and include_history:
            conversation_history = await _get_conversation_history(db, session_id)

        # Use RAG service to answer the question
        result = await rag_service.answer_question(
            book_id=book_id,
            question=message,
            top_k=top_k,
            max_context_tokens=max_context_tokens
        )

        response = ChatResponse(
            response=result["answer"],
            sources=result["sources"],
            model_used=result["model_used"]
        )

        logger.info(f"Successfully generated response for global chat with book {book_id}")

        # Save the user message and AI response to the session
        if session_id:
            await _save_message_to_session(db, session_id, "user", message, [])
            await _save_message_to_session(db, session_id, "assistant", result["answer"], result["sources"])
        else:
            # Create a new session if one wasn't provided
            new_session_id = await _create_new_session(db, book_id, "global")
            await _save_message_to_session(db, new_session_id, "user", message, [])
            await _save_message_to_session(db, new_session_id, "assistant", result["answer"], result["sources"])
            logger.info(f"Created new session {new_session_id} for chat")

        return response

    except Exception as e:
        logger.error(f"Error in global chat for book {book_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


async def _get_conversation_history(db: AsyncSession, session_id: UUID, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history for a session

    Args:
        db: Database session
        session_id: ID of the chat session
        limit: Maximum number of messages to retrieve

    Returns:
        List of conversation messages
    """
    # In a real implementation, this would query the database for chat messages
    # For this mock implementation, return an empty list
    return []


async def _save_message_to_session(
    db: AsyncSession,
    session_id: UUID,
    role: str,
    content: str,
    sources: List[Dict[str, Any]]
) -> None:
    """
    Save a message to a chat session

    Args:
        db: Database session
        session_id: ID of the chat session
        role: 'user' or 'assistant'
        content: Content of the message
        sources: Source information for assistant responses
    """
    # In a real implementation, this would create and save a ChatMessage object
    # For this mock implementation, just log the action
    logger.debug(f"Saving {role} message to session {session_id}")


async def _create_new_session(db: AsyncSession, book_id: UUID, mode: str) -> UUID:
    """
    Create a new chat session

    Args:
        db: Database session
        book_id: ID of the book for this session
        mode: 'global' or 'selected_text'

    Returns:
        ID of the new session
    """
    # In a real implementation, this would create and save a ChatSession object
    # For this mock implementation, return a new UUID
    import uuid
    new_session_id = uuid.uuid4()
    logger.debug(f"Created new session {new_session_id} for book {book_id} in {mode} mode")
    return new_session_id


@router.post("/selected-text", response_model=ChatResponse)
async def chat_selected_text(
    book_id: UUID,
    selected_text: str,
    message: str,
    session_id: Optional[UUID] = None,
    max_context_tokens: int = 3000,
    db: AsyncSession = Depends(get_db)
):
    """
    Chat endpoint for selected text mode - answers questions using only the selected text
    """
    if not selected_text.strip():
        raise HTTPException(status_code=400, detail="Selected text cannot be empty")
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    logger.info(f"Processing selected text chat request for book {book_id} with message: {message[:50]}...")
    
    try:
        # Use RAG service to answer question from selected text only
        result = await rag_service.answer_question_from_selected_text(
            selected_text=selected_text,
            question=message,
            max_context_tokens=max_context_tokens
        )
        
        response = ChatResponse(
            response=result["answer"],
            sources=result["sources"],
            model_used=result["model_used"]
        )
        
        logger.info(f"Successfully generated response for selected text chat with book {book_id}")
        
        # In a complete implementation, we would also save this message to the session
        # db.add(ChatMessage(...))
        # await db.commit()
        
        return response
        
    except Exception as e:
        logger.error(f"Error in selected text chat for book {book_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@router.post("/session")
async def create_session(
    book_id: UUID,
    session_name: Optional[str] = None,
    mode: str = "global",  # Default to global mode
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new chat session
    """
    from app.models.chat_session import ChatSession
    import uuid
    
    if mode not in ["global", "selected_text"]:
        raise HTTPException(status_code=400, detail="Mode must be 'global' or 'selected_text'")
    
    logger.info(f"Creating new chat session for book {book_id} with mode {mode}")
    
    try:
        # Create new session in database
        session = ChatSession(
            user_id=uuid.uuid4(),  # Would come from authentication in real implementation
            book_id=book_id,
            session_name=session_name,
            mode=mode
        )
        
        # In a complete implementation, we would add the session to the database
        # db.add(session)
        # await db.commit()
        # await db.refresh(session)
        
        logger.info(f"Created new chat session with ID: {session.id}")
        
        return {
            "session_id": session.id,
            "book_id": book_id,
            "session_name": session_name,
            "mode": mode,
            "message": "Chat session created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating chat session for book {book_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating chat session: {str(e)}")