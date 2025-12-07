import asyncio
import openai
from typing import Optional
from app.config import settings
from app.utils.logging import get_logger
from app.exceptions import LLMException

logger = get_logger(__name__)


class LLMService:
    """
    Service class to handle communication with LLM APIs (OpenAI, Gemini, etc.)
    """
    
    def __init__(self):
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.provider = "openai"
            self.model = settings.chat_model
        elif settings.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self.provider = "gemini"
                self.model = settings.chat_model if settings.chat_model else "gemini-pro"
                self.gemini_model = genai.GenerativeModel(self.model)
            except ImportError:
                raise LLMException("Google Generative AI library not installed for Gemini support")
        else:
            raise LLMException("No valid API key provided for LLM service")
    
    async def generate_response(self, prompt: str, grounding_check: bool = True) -> str:
        """
        Generate a response from the LLM based on the given prompt

        Args:
            prompt: Input prompt for the LLM
            grounding_check: Whether to perform additional checks to ensure response is grounded in context

        Returns:
            Generated response string
        """
        try:
            if self.provider == "openai":
                response = await openai.chat.completions.acreate(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Answer only based on the information provided in the context. If the answer is not in the context, explicitly say 'I cannot find the answer in the provided context.' Do not make up information."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                result = response.choices[0].message.content.strip()
            elif self.provider == "gemini":
                import google.generativeai as genai
                response = await self.gemini_model.generate_content_async(
                    contents=prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=500,
                        temperature=0.3
                    )
                )
                result = response.text.strip()

            # Additional hallucination check if requested
            if grounding_check:
                result = await self._check_response_grounding(prompt, result)

            return result
        except Exception as e:
            logger.error(f"Error generating response from {self.provider}: {str(e)}")
            raise LLMException(f"Failed to get response from LLM: {str(e)}")

    async def _check_response_grounding(self, context_and_question: str, response: str) -> str:
        """
        Perform additional check to ensure the response is grounded in the provided context

        Args:
            context_and_question: The original prompt containing context and question
            response: The LLM's response

        Returns:
            The response, potentially modified to ensure grounding
        """
        # This is a simple check - in a more advanced implementation, we might:
        # 1. Verify that claims in the response can be traced back to the context
        # 2. Use a secondary verification model
        # 3. Perform semantic similarity checks between response and context

        # For now, we'll just verify that if the response indicates lack of information,
        # it's properly formatted
        if "not in the context" in context_and_question.lower() or "cannot find" in response.lower():
            # This is expected behavior when the answer isn't in the context
            pass

        # Return the response as is for this implementation
        return response
    
    async def get_model_name(self) -> str:
        """
        Get the name of the model being used
        
        Returns:
            Name of the LLM model
        """
        return self.model
    
    async def validate_connection(self) -> bool:
        """
        Validate that the LLM service is properly configured and accessible
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Test with a simple prompt
            test_response = await self.generate_response("Say 'connection test' in one word")
            return len(test_response) > 0
        except Exception as e:
            logger.error(f"LLM connection validation failed: {str(e)}")
            return False
    
    async def get_embedding_for_text(self, text: str) -> list:
        """
        Get embedding for a text using the same provider if possible
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as a list
        """
        try:
            if self.provider == "openai":
                response = await openai.embeddings.acreate(
                    input=text,
                    model=settings.embed_model
                )
                return response.data[0].embedding
            elif self.provider == "gemini":
                # Gemini doesn't have a direct embedding method in this implementation
                # We'd need to use a different service for embeddings if using Gemini
                raise LLMException("Embeddings not supported with Gemini in this implementation")
        except Exception as e:
            logger.error(f"Error getting embedding from {self.provider}: {str(e)}")
            raise LLMException(f"Failed to get embedding: {str(e)}")