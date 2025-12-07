from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class BookResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    author: str
    format: str
    file_path: str
    page_count: Optional[int] = None
    upload_date: datetime
    processing_status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BookChunkResponse(BaseModel):
    id: uuid.UUID
    book_id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    vector_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    book_id: uuid.UUID
    session_name: Optional[str]
    mode: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    chat_session_id: uuid.UUID
    role: str
    content: str
    sources: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    id: uuid.UUID
    title: str
    author: str
    format: str
    processing_status: str
    message: str


class ProcessingStatusResponse(BaseModel):
    book_id: uuid.UUID
    processing_status: str
    progress: Optional[float] = None
    message: str


class SearchResult(BaseModel):
    id: uuid.UUID
    content: str
    score: float
    source: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    model_used: str