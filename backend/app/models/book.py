from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import relationship
from app.models.base import Base
from typing import TYPE_CHECKING
import uuid
from sqlalchemy.types import Uuid

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.book_chunk import BookChunk


class Book(Base):
    __tablename__ = "books"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    author = Column(String, nullable=False)
    format = Column(String, nullable=False)  # PDF, DOCX, TXT
    file_path = Column(String, nullable=False)
    page_count = Column(Integer, nullable=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    processing_status = Column(String, nullable=False, default="pending")  # pending, in_progress, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="books")
    chunks = relationship("BookChunk", back_populates="book", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="book", cascade="all, delete-orphan")