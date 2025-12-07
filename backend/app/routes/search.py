from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.services.search_service import SearchService
from app.models.responses import SearchResponse, SearchResult
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/books", tags=["search"])

# Global instance of search service
search_service = SearchService()


@router.post("/{book_id}/search")
async def search_book_content(
    book_id: UUID,
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    search_type: str = Query("semantic", regex="^(semantic|keyword|combined)$", description="Type of search to perform"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search for specific terms, concepts, or topics within a book
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Searching for '{query}' in book {book_id} using {search_type} search")
    
    try:
        search_results = []
        
        if search_type == "semantic":
            search_results = await search_service.semantic_search(book_id, query, top_k)
        elif search_type == "keyword":
            search_results = await search_service.keyword_search(book_id, query, top_k)
        elif search_type == "combined":
            search_results = await search_service.combined_search(book_id, query, top_k)
        
        # Format results to match the expected response model
        formatted_results = []
        for result in search_results:
            formatted_results.append(SearchResult(
                id=result['id'],
                content=result['content'],
                score=result['score']
            ))
        
        response = SearchResponse(
            query=query,
            results=formatted_results
        )
        
        logger.info(f"Search returned {len(formatted_results)} results for query: {query[:50]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching book {book_id} for query '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")


@router.get("/{book_id}/search-stats")
async def get_search_statistics(
    book_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics about the searchable content of a book
    """
    logger.info(f"Getting search statistics for book {book_id}")
    
    try:
        stats = await search_service.get_search_statistics(book_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting search statistics for book {book_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting search statistics: {str(e)}")


@router.post("/search-across-books")
async def search_across_books(
    user_id: UUID,
    query: str = Query(..., min_length=1, max_length=500, description="Search query across all user's books"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return per book"),
    db: AsyncSession = Depends(get_db)
):
    """
    Search across all books accessible to a user
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Searching across books for user {user_id} with query: {query[:50]}...")
    
    try:
        # Perform search across all books for the user
        results = await search_service.search_across_books(user_id, query, top_k)
        
        logger.info(f"Cross-book search returned {len(results)} results")
        
        return {"query": query, "results": results, "user_id": str(user_id)}
        
    except Exception as e:
        logger.error(f"Error performing cross-book search for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing cross-book search: {str(e)}")