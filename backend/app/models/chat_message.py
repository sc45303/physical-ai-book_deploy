from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.models.base import Base
from typing import TYPE_CHECKING
import uuid
from sqlalchemy.types import Uuid

if TYPE_CHECKING:
    from app.models.chat_session import ChatSession


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_session_id = Column(Uuid(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(JSONB, nullable=True)  # List of source chunks used for answer generation
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")