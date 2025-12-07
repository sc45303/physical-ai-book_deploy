import re
from typing import List, Tuple
from app.utils.logging import get_logger
import tiktoken

logger = get_logger(__name__)


class ChunkingService:
    """
    Service class to handle text chunking for RAG system
    """
    
    def __init__(self):
        # Use OpenAI's encoding for token counting
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string
        
        Args:
            text: Input text
            
        Returns:
            int: Number of tokens
        """
        return len(self.encoder.encode(text))
    
    def chunk_by_tokens(
        self, 
        text: str, 
        max_tokens: int = 512,
        overlap_tokens: int = 50
    ) -> List[Tuple[str, int]]:
        """
        Chunk text based on token count with overlap
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            
        Returns:
            List of tuples containing (chunk_text, token_count)
        """
        if not text:
            return []
        
        # Split text into sentences to avoid breaking them
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks_with_tokens = []
        current_chunk = ""
        current_token_count = 0
        
        for sentence in sentences:
            sentence_token_count = self._count_tokens(sentence)
            
            # If adding this sentence would exceed the limit
            if current_token_count + sentence_token_count > max_tokens:
                # Add current chunk if it's not empty
                if current_chunk.strip():
                    chunks_with_tokens.append((current_chunk.strip(), current_token_count))
                
                # Start a new chunk with this sentence
                current_chunk = sentence
                current_token_count = sentence_token_count
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_token_count += sentence_token_count
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks_with_tokens.append((current_chunk.strip(), current_token_count))
        
        # Add overlap between chunks if needed
        if overlap_tokens > 0 and len(chunks_with_tokens) > 1:
            chunks_with_overlap = []
            
            for i, (chunk_text, token_count) in enumerate(chunks_with_tokens):
                if i == 0:
                    # First chunk: no previous chunk to overlap with
                    chunks_with_overlap.append((chunk_text, token_count))
                else:
                    # Get tokens from the previous chunk for overlap
                    prev_chunk_text, prev_token_count = chunks_with_tokens[i-1]
                    
                    # Get the overlap text from the end of the previous chunk
                    prev_tokens = self.encoder.encode(prev_chunk_text)
                    if len(prev_tokens) > overlap_tokens:
                        overlap_tokens_list = prev_tokens[-overlap_tokens:]
                    else:
                        overlap_tokens_list = prev_tokens
                    
                    overlap_text = self.encoder.decode(overlap_tokens_list)
                    
                    # Combine overlap text with current chunk
                    combined_chunk = overlap_text + " " + chunk_text
                    combined_token_count = self._count_tokens(combined_chunk)
                    
                    chunks_with_overlap.append((combined_chunk, combined_token_count))
            
            return chunks_with_overlap
        
        return chunks_with_tokens
    
    def chunk_by_characters(
        self, 
        text: str, 
        max_chars: int = 1000,
        overlap_chars: int = 100
    ) -> List[Tuple[str, int]]:
        """
        Chunk text based on character count with overlap
        
        Args:
            text: Input text to chunk
            max_chars: Maximum characters per chunk
            overlap_chars: Number of overlapping characters between chunks
            
        Returns:
            List of tuples containing (chunk_text, character_count)
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # If we're at the end, just take the remaining text
            if end >= len(text):
                chunk = text[start:]
                chunks.append((chunk, len(chunk)))
                break
            
            # Try to break at a sentence boundary if possible
            chunk = text[start:end]
            last_sentence_end = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
            
            if last_sentence_end != -1 and last_sentence_end > len(chunk) // 2:
                # Break at sentence boundary
                actual_end = start + last_sentence_end + 1
                chunk = text[start:actual_end]
                end = actual_end
            else:
                # Break at the max character limit
                chunk = text[start:end]
            
            # Add overlap if not the last chunk
            if end < len(text) and overlap_chars > 0:
                overlap_start = max(0, end - overlap_chars)
                overlap_text = text[overlap_start:end]
                next_chunk = overlap_text + text[end:end + max_chars]
                chunk = text[start:end]
            
            chunks.append((chunk, len(chunk)))
            
            # Move start to the next position after current chunk (minus overlap)
            start = end - overlap_chars if overlap_chars > 0 and end < len(text) else end
        
        return chunks
    
    def chunk_by_paragraph(self, text: str) -> List[Tuple[str, int]]:
        """
        Chunk text by paragraphs
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of tuples containing (chunk_text, character_count)
        """
        if not text:
            return []
        
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            if paragraph.strip():  # Only add non-empty paragraphs
                chunks.append((paragraph.strip(), len(paragraph.strip())))
        
        return chunks
    
    def chunk_text(
        self,
        text: str,
        method: str = "tokens",
        max_size: int = 512,
        overlap: int = 50
    ) -> List[Tuple[str, int]]:
        """
        Chunk text using the specified method
        
        Args:
            text: Input text to chunk
            method: Chunking method ("tokens", "characters", "paragraphs")
            max_size: Maximum size per chunk (tokens/characters)
            overlap: Overlap size (tokens/characters)
            
        Returns:
            List of tuples containing (chunk_text, size)
        """
        logger.info(f"Chunking text using method '{method}' with max_size {max_size} and overlap {overlap}")
        
        if method == "tokens":
            return self.chunk_by_tokens(text, max_size, overlap)
        elif method == "characters":
            return self.chunk_by_characters(text, max_size, overlap)
        elif method == "paragraphs":
            return self.chunk_by_paragraph(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")