import asyncio
import openai
from typing import List, Dict, Any
from app.config import settings
from app.utils.logging import get_logger
from app.exceptions import LLMException
import numpy as np

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service class to handle text embeddings using OpenAI or similar APIs
    """
    
    def __init__(self):
        # Initialize OpenAI client
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.provider = "openai"
        elif settings.gemini_api_key:
            # Initialize Gemini if available
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self.provider = "gemini"
                self.gemini_model = genai.GenerativeModel('models/embedding-001')
            except ImportError:
                raise LLMException("Google Generative AI library not installed for Gemini support")
        else:
            raise LLMException("No valid API key provided for embedding service")
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            if self.provider == "openai":
                response = await openai.embeddings.create(
                    input=text,
                    model=settings.embed_model
                )
                return response.data[0].embedding
            elif self.provider == "gemini":
                # Gemini doesn't have a direct embedding API in the standard SDK
                # This needs to be implemented with the appropriate Gemini embedding method
                # For now, we'll raise an exception until we implement Gemini embeddings properly
                raise LLMException("Gemini embeddings not yet implemented")
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise LLMException(f"Failed to create embedding: {str(e)}")
    
    async def create_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Create embeddings for a batch of texts
        
        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            List of lists of floats representing the embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                if self.provider == "openai":
                    response = await openai.embeddings.create(
                        input=batch,
                        model=settings.embed_model
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                else:
                    # Process each text individually for other providers
                    for text in batch:
                        embedding = await self.create_embedding(text)
                        all_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error creating embeddings batch: {str(e)}")
                raise LLMException(f"Failed to create embeddings: {str(e)}")
        
        return all_embeddings
    
    async def get_embedding_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings
        
        Returns:
            int: Number of dimensions in the embedding vectors
        """
        # Get a sample embedding to determine dimensions
        sample_embedding = await self.create_embedding("Sample text for dimension checking")
        return len(sample_embedding)
    
    async def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        # Convert to numpy arrays for efficient computation
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        similarity = dot_product / (norm_v1 * norm_v2)
        return float(similarity)
    
    async def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings to a query embedding
        
        Args:
            query_embedding: The embedding to search for
            candidate_embeddings: List of candidate embeddings to search in
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing 'index', 'similarity', and 'embedding' keys
        """
        similarities = []
        
        for idx, candidate_embedding in enumerate(candidate_embeddings):
            similarity = await self.cosine_similarity(query_embedding, candidate_embedding)
            similarities.append({
                'index': idx,
                'similarity': similarity,
                'embedding': candidate_embedding
            })
        
        # Sort by similarity in descending order and return top_k
        sorted_similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)
        return sorted_similarities[:top_k]