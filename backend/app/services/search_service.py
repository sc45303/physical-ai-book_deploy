from typing import List, Dict, Any, Optional
from app.services.embedding_service import EmbeddingService
from app.database.vector_store import VectorStoreService
from app.utils.logging import get_logger
from uuid import UUID
import hashlib
import asyncio
from datetime import datetime, timedelta
from functools import wraps

logger = get_logger(__name__)


class SimpleCache:
    """
    A simple in-memory cache for storing search results
    In production, you would use Redis or another more robust caching system
    """
    def __init__(self, default_ttl: int = 3600):  # 1 hour default TTL
        self.cache = {}
        self.default_ttl = default_ttl

    def _get_key(self, book_id: UUID, query: str, search_type: str) -> str:
        """Generate a cache key for the search query"""
        key_string = f"{book_id}:{search_type}:{query}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get a value from the cache if it exists and hasn't expired"""
        if key in self.cache:
            value, expiry_time = self.cache[key]
            if datetime.now() < expiry_time:
                return value
            else:
                # Remove expired entry
                del self.cache[key]
        return None

    def set(self, key: str, value: List[Dict[str, Any]], ttl: Optional[int] = None) -> None:
        """Set a value in the cache with an expiration time"""
        if ttl is None:
            ttl = self.default_ttl
        expiry_time = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry_time)

    def delete(self, key: str) -> bool:
        """Delete a value from the cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

# Global cache instance
search_cache = SimpleCache()


class SearchService:
    """
    Service class to handle various search functionalities
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService()
    
    async def semantic_search(
        self,
        book_id: UUID,
        query: str,
        top_k: int = 5,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search within a specific book

        Args:
            book_id: ID of the book to search in
            query: Search query
            top_k: Number of top results to return
            use_cache: Whether to use cached results if available

        Returns:
            List of search results with content and similarity scores
        """
        logger.info(f"Performing semantic search in book {book_id} for query: {query}")

        # Generate cache key
        cache_key = search_cache._get_key(book_id, query, "semantic")

        # Check cache first
        if use_cache:
            cached_results = search_cache.get(cache_key)
            if cached_results is not None:
                logger.info(f"Cache hit for semantic search: {query}")
                return cached_results

        # Create embedding for the query
        query_embedding = await self.embedding_service.create_embedding(query)

        # Search in vector store
        search_results = await self.vector_store_service.search_similar(
            query_vector=query_embedding,
            book_id=book_id,
            top_k=top_k
        )

        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                'id': result['id'],
                'content': result['content'],
                'score': result['score'],
                'book_id': result['book_id']
            })

        logger.info(f"Semantic search returned {len(formatted_results)} results")

        # Cache the results
        if use_cache:
            search_cache.set(cache_key, formatted_results)

        return formatted_results
    
    async def keyword_search(
        self,
        book_id: UUID,
        query: str,
        top_k: int = 5,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search within a specific book using PostgreSQL full-text search

        Args:
            book_id: ID of the book to search in
            query: Search query
            top_k: Number of top results to return
            use_cache: Whether to use cached results if available

        Returns:
            List of search results
        """
        logger.info(f"Performing keyword search in book {book_id} for query: {query}")

        # Generate cache key
        cache_key = search_cache._get_key(book_id, query, "keyword")

        # Check cache first
        if use_cache:
            cached_results = search_cache.get(cache_key)
            if cached_results is not None:
                logger.info(f"Cache hit for keyword search: {query}")
                return cached_results

        # This is a placeholder implementation.
        # In a real implementation, we would use SQLAlchemy with PostgreSQL
        # to perform full-text search on the book_chunks table.
        #
        # Example SQL for PostgreSQL full-text search:
        # SELECT *, ts_rank(search_vector, plainto_tsquery('english', :query)) as rank
        # FROM book_chunks
        # WHERE book_id = :book_id
        #   AND search_vector @@ plainto_tsquery('english', :query)
        # ORDER BY rank DESC
        # LIMIT :top_k

        # For this implementation, we'll return an empty list and log that
        # full-text search would be implemented with database integration
        logger.info(f"Full-text search would be implemented with PostgreSQL tsvector in a complete implementation")

        # In a real implementation, we would:
        # 1. Connect to database using the session passed from the route
        # 2. Perform a full-text search using PostgreSQL's text search functions
        # 3. Return the results with relevance scores
        # 4. Cache those results

        # For now, return empty results to indicate this functionality would be implemented
        empty_results = []

        # Cache the empty results
        if use_cache:
            search_cache.set(cache_key, empty_results)

        return empty_results
    
    async def combined_search(
        self,
        book_id: UUID,
        query: str,
        top_k: int = 5,
        rerank_method: str = "score_fusion",  # Options: "score_fusion", "reciprocal_rank"
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform both semantic and keyword search and combine results with advanced ranking

        Args:
            book_id: ID of the book to search in
            query: Search query
            top_k: Number of top results to return
            rerank_method: Method for combining scores ("score_fusion" or "reciprocal_rank")
            use_cache: Whether to use cached results if available

        Returns:
            Combined and ranked list of search results
        """
        logger.info(f"Performing combined search in book {book_id} for query: {query}")

        # Generate cache key for combined search
        cache_key = search_cache._get_key(book_id, query, f"combined_{rerank_method}")

        # Check cache first
        if use_cache:
            cached_results = search_cache.get(cache_key)
            if cached_results is not None:
                logger.info(f"Cache hit for combined search: {query}")
                return cached_results

        # Get semantic search results
        semantic_results = await self.semantic_search(book_id, query, top_k * 2, use_cache=False)  # Don't cache sub-searches separately

        # Get keyword search results
        keyword_results = await self.keyword_search(book_id, query, top_k * 2, use_cache=False)  # Don't cache sub-searches separately

        if rerank_method == "reciprocal_rank":
            # Use Reciprocal Rank Fusion to combine results
            combined_results = self._reciprocal_rank_fusion(semantic_results, keyword_results, top_k)
        else:  # Default to score fusion
            combined_results = self._score_fusion_ranking(semantic_results, keyword_results, top_k)

        logger.info(f"Combined search returned {len(combined_results)} results using {rerank_method} method")

        # Cache the combined results
        if use_cache:
            search_cache.set(cache_key, combined_results)

        return combined_results

    def _score_fusion_ranking(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using a weighted score fusion approach

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            top_k: Number of top results to return

        Returns:
            Combined and ranked results
        """
        # Create a map of results by ID with combined scores
        combined_scores = {}

        # Add semantic scores (normalized)
        for idx, result in enumerate(semantic_results):
            result_id = result['id']
            # Normalize score to 0-1 range and apply weight
            normalized_score = result['score']  # Assuming scores are already normalized
            combined_scores[result_id] = {
                'content': result['content'],
                'base_result': result,
                'semantic_score': normalized_score,
                'keyword_score': 0.0,
                'combined_score': normalized_score * 0.7  # 70% weight to semantic
            }

        # Add keyword scores (if available) and update combined scores
        for idx, result in enumerate(keyword_results):
            result_id = result['id']
            # Note: keyword search currently returns empty in our implementation
            # In a real implementation, we would have actual keyword scores
            if result_id in combined_scores:
                # Update existing entry with keyword score
                keyword_score = result.get('score', 0.0)
                combined_scores[result_id]['keyword_score'] = keyword_score
                combined_scores[result_id]['combined_score'] = (
                    combined_scores[result_id]['semantic_score'] * 0.7 +
                    keyword_score * 0.3
                )  # 70% semantic, 30% keyword
            else:
                # Add new result with keyword score only
                keyword_score = result.get('score', 0.0)
                combined_scores[result_id] = {
                    'content': result['content'],
                    'base_result': result,
                    'semantic_score': 0.0,
                    'keyword_score': keyword_score,
                    'combined_score': keyword_score * 0.3  # 30% weight to keyword
                }

        # Create final results list with combined scores
        final_results = []
        for result_id, data in combined_scores.items():
            final_result = data['base_result'].copy()
            final_result['score'] = data['combined_score']
            final_result['semantic_score'] = data['semantic_score']
            final_result['keyword_score'] = data['keyword_score']
            final_results.append(final_result)

        # Sort by combined score in descending order
        final_results.sort(key=lambda x: x['score'], reverse=True)

        # Return top_k results
        return final_results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion algorithm

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            top_k: Number of top results to return

        Returns:
            Combined and ranked results using RRf
        """
        # Create a dictionary to track scores for each document
        rrf_scores = {}

        # Calculate RRF scores for semantic results
        k = 60  # Smoothing parameter (standard value)
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result['id']
            rrf_score = 1 / (k + rank)
            rrf_scores[doc_id] = {
                'base_result': result,
                'rrf_score': rrf_score,
                'semantic_rank': rank,
                'keyword_rank': None
            }

        # Calculate RRF scores for keyword results and add to existing scores
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = result['id']
            rrf_score = 1 / (k + rank)

            if doc_id in rrf_scores:
                # Document exists in both lists, combine scores
                rrf_scores[doc_id]['rrf_score'] += rrf_score
                rrf_scores[doc_id]['keyword_rank'] = rank
            else:
                # New document from keyword search
                rrf_scores[doc_id] = {
                    'base_result': result,
                    'rrf_score': rrf_score,
                    'semantic_rank': None,
                    'keyword_rank': rank
                }

        # Create final results list with RRF scores
        final_results = []
        for doc_id, data in rrf_scores.items():
            final_result = data['base_result'].copy()
            final_result['score'] = data['rrf_score']
            final_result['rrf_score'] = data['rrf_score']
            final_result['semantic_rank'] = data['semantic_rank']
            final_result['keyword_rank'] = data['keyword_rank']
            final_results.append(final_result)

        # Sort by RRF score in descending order
        final_results.sort(key=lambda x: x['score'], reverse=True)

        # Return top_k results
        return final_results[:top_k]
    
    async def search_across_books(
        self,
        user_id: UUID,  # This would be used to filter user's books in a complete implementation
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search across all books accessible to a user
        
        Args:
            user_id: ID of the user
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results across all user's books
        """
        logger.info(f"Performing search across books for user {user_id} with query: {query}")
        
        # This would require integration with the database to get all books for a user
        # For this implementation, we'll return an empty list
        # In a complete implementation, this would:
        # 1. Query the database for all books belonging to the user
        # 2. Perform semantic search in each book
        # 3. Combine and rank all results
        
        logger.warning("Cross-book search not fully implemented in this version")
        return []
    
    async def get_search_statistics(
        self,
        book_id: UUID
    ) -> Dict[str, Any]:
        """
        Get statistics about the searchable content of a book
        
        Args:
            book_id: ID of the book
            
        Returns:
            Dictionary containing search-related statistics
        """
        logger.info(f"Getting search statistics for book {book_id}")
        
        try:
            # Get number of searchable chunks
            vector_count = await self.vector_store_service.get_vector_count(book_id)
            
            stats = {
                "book_id": str(book_id),
                "total_chunks": vector_count,
                "searchable": vector_count > 0
            }
            
            logger.info(f"Retrieved search statistics for book {book_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting search statistics for book {book_id}: {str(e)}")
            return {
                "book_id": str(book_id),
                "total_chunks": 0,
                "searchable": False,
                "error": str(e)
            }