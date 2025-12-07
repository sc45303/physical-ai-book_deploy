from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.models.base import Base
from typing import TYPE_CHECKING
import uuid
from sqlalchemy.types import Uuid

if TYPE_CHECKING:
    from app.models.book_chunk import BookChunk


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id = Column(String, primary_key=True)  # ID in the vector database (Qdrant)
    book_chunk_id = Column(Uuid(as_uuid=True), nullable=False)  # Reference to the BookChunk
    model_used = Column(String, nullable=False)  # The embedding model used
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Note: Actual vector data is stored in Qdrant, not in PostgreSQL
    # This model is for tracking metadata about vector embeddings