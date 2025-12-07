from fastapi import HTTPException, status
from typing import Optional


class BookRAGException(HTTPException):
    """Base exception for the Book RAG system"""
    def __init__(self, detail: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(status_code=status_code, detail=detail)


class BookProcessingException(BookRAGException):
    """Exception raised when there's an error processing a book"""
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=status.HTTP_400_BAD_REQUEST)


class BookNotFoundException(BookRAGException):
    """Exception raised when a book is not found"""
    def __init__(self, book_id: str):
        super().__init__(
            detail=f"Book with ID {book_id} not found",
            status_code=status.HTTP_404_NOT_FOUND
        )


class UserNotFoundException(BookRAGException):
    """Exception raised when a user is not found"""
    def __init__(self, user_id: str):
        super().__init__(
            detail=f"User with ID {user_id} not found",
            status_code=status.HTTP_404_NOT_FOUND
        )


class ChatSessionNotFoundException(BookRAGException):
    """Exception raised when a chat session is not found"""
    def __init__(self, session_id: str):
        super().__init__(
            detail=f"Chat session with ID {session_id} not found",
            status_code=status.HTTP_404_NOT_FOUND
        )


class InvalidFileTypeException(BookRAGException):
    """Exception raised when an unsupported file type is uploaded"""
    def __init__(self, file_type: str):
        super().__init__(
            detail=f"File type {file_type} is not supported. Supported types: PDF, DOCX, TXT",
            status_code=status.HTTP_400_BAD_REQUEST
        )


class VectorStoreException(BookRAGException):
    """Exception raised when there's an issue with the vector store"""
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Vector store error: {detail}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class LLMException(BookRAGException):
    """Exception raised when there's an issue with the LLM service"""
    def __init__(self, detail: str):
        super().__init__(
            detail=f"LLM service error: {detail}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class ConfigurationException(BookRAGException):
    """Exception raised when there's a configuration issue"""
    def __init__(self, detail: str):
        super().__init__(
            detail=f"Configuration error: {detail}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )