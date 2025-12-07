import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from app.config import settings
from app.exceptions import VectorStoreException
from app.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """
    Service class to handle vector storage operations using Qdrant
    """
    
    def __init__(self):
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=True  # Use gRPC for better performance if available
            )
            self.collection_name = "book_chunks"
            self.vector_size = 1536  # Default size for OpenAI embeddings; will be updated based on actual model
            self._initialize_collection()
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise VectorStoreException(f"Failed to connect to vector store: {str(e)}")
    
    def _initialize_collection(self):
        """
        Initialize the collection if it doesn't exist
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                # Create collection with specified vector parameters
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise VectorStoreException(f"Failed to initialize vector store collection: {str(e)}")
    
    async def store_embedding(
        self, 
        vector_id: str, 
        vector: List[float], 
        book_id: UUID, 
        chunk_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a vector embedding in Qdrant
        
        Args:
            vector_id: Unique identifier for the vector
            vector: The embedding vector
            book_id: ID of the book this chunk belongs to
            chunk_content: The original text content of the chunk
            metadata: Additional metadata to store with the vector
            
        Returns:
            True if successfully stored, False otherwise
        """
        try:
            # Prepare payload with metadata
            payload = {
                "book_id": str(book_id),
                "content": chunk_content,
                "created_at": "now"  # This will be updated by the DB
            }
            if metadata:
                payload.update(metadata)
            
            # Update vector size if needed
            if len(vector) != self.vector_size:
                self.vector_size = len(vector)
            
            # Upsert the point in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=vector_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            
            logger.info(f"Successfully stored embedding with ID: {vector_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding {vector_id}: {str(e)}")
            raise VectorStoreException(f"Failed to store embedding: {str(e)}")
    
    async def search_similar(
        self, 
        query_vector: List[float], 
        book_id: UUID, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the specified book
        
        Args:
            query_vector: The query embedding vector
            book_id: ID of the book to search within
            top_k: Number of top results to return
            
        Returns:
            List of similar vectors with their metadata
        """
        try:
            # Search with filtering to only get results from the specified book
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="book_id",
                            match=models.MatchValue(value=str(book_id))
                        )
                    ]
                ),
                limit=top_k
            )
            
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'content': result.payload.get('content', ''),
                    'score': result.score,
                    'book_id': result.payload.get('book_id', ''),
                    'metadata': {k: v for k, v in result.payload.items() if k not in ['content', 'book_id']}
                })
            
            logger.info(f"Found {len(results)} similar vectors for book {book_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {str(e)}")
            raise VectorStoreException(f"Failed to search vectors: {str(e)}")
    
    async def delete_by_book_id(self, book_id: UUID) -> bool:
        """
        Delete all vectors associated with a specific book
        
        Args:
            book_id: ID of the book to delete vectors for
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            # Delete points that match the book_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="book_id",
                                match=models.MatchValue(value=str(book_id))
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Successfully deleted all vectors for book {book_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors for book {book_id}: {str(e)}")
            raise VectorStoreException(f"Failed to delete vectors: {str(e)}")
    
    async def get_vector_count(self, book_id: UUID) -> int:
        """
        Get the count of vectors for a specific book
        
        Args:
            book_id: ID of the book
            
        Returns:
            Number of vectors stored for this book
        """
        try:
            count = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="book_id",
                            match=models.MatchValue(value=str(book_id))
                        )
                    ]
                )
            )
            
            return count.count
        except Exception as e:
            logger.error(f"Failed to get vector count for book {book_id}: {str(e)}")
            raise VectorStoreException(f"Failed to get vector count: {str(e)}")