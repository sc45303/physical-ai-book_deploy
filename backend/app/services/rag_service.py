import asyncio
from typing import List, Dict, Any, Optional
from app.services.embedding_service import EmbeddingService
from app.database.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.utils.logging import get_logger
from app.exceptions import LLMException
from uuid import UUID

logger = get_logger(__name__)


class RAGService:
    """
    Service class to handle Retrieval Augmented Generation (RAG) functionality
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService()
        self.llm_service = LLMService()
    
    async def answer_question(
        self,
        book_id: UUID,
        question: str,
        top_k: int = 5,
        max_context_tokens: int = 3000,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question about book content using RAG

        Args:
            book_id: ID of the book to search in
            question: The question to answer
            top_k: Number of top results to retrieve
            max_context_tokens: Maximum number of tokens in context
            include_citations: Whether to include citations in the response

        Returns:
            Dict containing the answer and sources
        """
        logger.info(f"Answering question for book {book_id}: {question}")

        # Create embedding for the question
        question_embedding = await self.embedding_service.create_embedding(question)

        # Search for relevant chunks in the vector store
        search_results = await self.vector_store_service.search_similar(
            query_vector=question_embedding,
            book_id=book_id,
            top_k=top_k
        )

        logger.info(f"Found {len(search_results)} relevant chunks")

        # Process search results to create context with careful token management
        context_chunks = []
        token_count = 0

        for result in search_results:
            chunk_text = result.get('content', '')
            chunk_tokens = result.get('metadata', {}).get('token_count', 0)

            # Add to context if it doesn't exceed the limit
            if token_count + chunk_tokens <= max_context_tokens:
                context_chunks.append({
                    'id': result['id'],
                    'content': chunk_text,
                    'score': result['score'],
                    'source': result.get('source', 'unknown'),
                    'chunk_index': result.get('metadata', {}).get('chunk_index', -1)
                })
                token_count += chunk_tokens
            else:
                # If adding this chunk would exceed the limit, stop
                break

        if not context_chunks:
            logger.warning(f"No relevant content found for question: {question}")
            return {
                "answer": "I couldn't find relevant information in the book to answer your question.",
                "sources": [],
                "model_used": await self.llm_service.get_model_name()
            }

        # Construct context for the LLM with proper formatting for citations
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_parts.append(f"[Source {i+1}]: {chunk['content']}")

        context = "\n\n".join(context_parts)

        # Create prompt with citation instructions if requested
        if include_citations:
            prompt = f"""
            Based on the following context from a book, please answer the question.
            If the answer is not in the context, please say so explicitly.
            When referencing information from the context, cite the source using the format [Source X] where X is the source number.

            Context:
            {context}

            Question: {question}

            Answer:
            """
        else:
            prompt = f"""
            Based on the following context from a book, please answer the question.
            If the answer is not in the context, please say so explicitly.

            Context:
            {context}

            Question: {question}

            Answer:
            """

        try:
            answer = await self.llm_service.generate_response(prompt, grounding_check=True)
        except LLMException as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

        logger.info(f"Successfully generated answer for question: {question[:50]}...")

        # Prepare source citations
        sources = []
        for i, chunk in enumerate(context_chunks):
            sources.append({
                'id': chunk['id'],
                'content': chunk['content'],
                'score': chunk['score'],
                'source': chunk['source'],
                'chunk_index': chunk['chunk_index'],
                'citation_ref': f"[Source {i+1}]"
            })

        return {
            "answer": answer,
            "sources": sources,
            "model_used": await self.llm_service.get_model_name()
        }
    
    async def answer_question_from_selected_text(
        self,
        selected_text: str,
        question: str,
        max_context_tokens: int = 3000
    ) -> Dict[str, Any]:
        """
        Answer a question about selected text only

        Args:
            selected_text: The text that was selected by the user
            question: The question to answer
            max_context_tokens: Maximum number of tokens in context

        Returns:
            Dict containing the answer and sources
        """
        logger.info(f"Answering question from selected text: {question[:50]}...")

        # Preprocess and validate the selected text
        processed_text = await self._preprocess_selected_text(selected_text, max_context_tokens)

        if not processed_text:
            logger.warning("Selected text is empty or invalid after preprocessing")
            return {
                "answer": "The selected text is empty or invalid. Please select valid text to ask questions about.",
                "sources": [],
                "model_used": await self.llm_service.get_model_name()
            }

        # Construct context for the LLM using only the selected text
        prompt = f"""
        Based on the following selected text, please answer the question.
        Only use information from the provided text to answer the question.
        If the answer is not in the selected text, explicitly state that the answer cannot be found in the provided text.

        Selected text:
        {processed_text}

        Question: {question}

        Answer:
        """

        try:
            answer = await self.llm_service.generate_response(prompt, grounding_check=True)
        except LLMException as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

        logger.info(f"Successfully generated answer for question from selected text: {question[:50]}...")

        return {
            "answer": answer,
            "sources": [{"id": "selected_text", "content": processed_text[:200] + "..." if len(processed_text) > 200 else processed_text, "score": 1.0, "source": "user_selected"}],
            "model_used": await self.llm_service.get_model_name()
        }

    async def _preprocess_selected_text(self, selected_text: str, max_context_tokens: int) -> str:
        """
        Preprocess the selected text by validating and cleaning it

        Args:
            selected_text: The text that was selected by the user
            max_context_tokens: Maximum number of tokens allowed

        Returns:
            Preprocessed and validated selected text
        """
        if not selected_text or not selected_text.strip():
            return ""

        # Remove extra whitespace
        processed_text = ' '.join(selected_text.split())

        # Check length and truncate if necessary
        estimated_tokens = len(processed_text) // 4  # Rough estimation: 1 token ≈ 4 chars
        if estimated_tokens > max_context_tokens:
            logger.warning(f"Selected text is too long ({estimated_tokens} tokens), truncating to {max_context_tokens}")
            # Truncate to the allowed length (this is a simple truncation, in practice you might want to be more careful about sentence boundaries)
            allowed_chars = max_context_tokens * 4
            processed_text = processed_text[:allowed_chars]

        return processed_text
    
    async def get_relevant_chunks(
        self,
        book_id: UUID,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get relevant chunks from a book without generating an answer
        
        Args:
            book_id: ID of the book to search in
            query: The search query
            top_k: Number of top results to retrieve
            
        Returns:
            List of relevant chunks
        """
        logger.info(f"Retrieving relevant chunks for book {book_id} with query: {query}")
        
        # Create embedding for the query
        query_embedding = await self.embedding_service.create_embedding(query)
        
        # Search for relevant chunks in the vector store
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
                'source': result.get('source', 'unknown')
            })
        
        logger.info(f"Retrieved {len(formatted_results)} relevant chunks")
        
        return formatted_results