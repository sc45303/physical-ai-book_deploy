from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import relationship
from app.models.base import Base
from typing import TYPE_CHECKING
import uuid
from sqlalchemy.types import Uuid

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.book import Book
    from app.models.chat_message import ChatMessage


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id"), nullable=False)
    book_id = Column(Uuid(as_uuid=True), ForeignKey("books.id"), nullable=False)
    session_name = Column(String, nullable=True)
    mode = Column(String, nullable=False)  # 'global' or 'selected_text'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    book = relationship("Book", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")