from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.orm import relationship
from app.models.base import Base
from typing import TYPE_CHECKING
import uuid
from sqlalchemy.types import Uuid

if TYPE_CHECKING:
    from app.models.book import Book
    from app.models.chat_session import ChatSession


class User(Base):
    __tablename__ = "users"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship to books
    books = relationship("Book", back_populates="user", cascade="all, delete-orphan")

    # Relationship to chat sessions
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")