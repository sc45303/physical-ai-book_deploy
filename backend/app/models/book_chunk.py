from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import relationship
from app.models.base import Base
from typing import TYPE_CHECKING
import uuid
from sqlalchemy.types import Uuid

if TYPE_CHECKING:
    from app.models.book import Book
    from app.models.vector_embedding import VectorEmbedding


class BookChunk(Base):
    __tablename__ = "book_chunks"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(Uuid(as_uuid=True), ForeignKey("books.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Sequential index of the chunk in the book
    content = Column(Text, nullable=False)  # The text content of the chunk
    token_count = Column(Integer, nullable=True)  # Number of tokens in the chunk
    vector_id = Column(String, nullable=True)  # ID in the vector database (Qdrant)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    book = relationship("Book", back_populates="chunks")