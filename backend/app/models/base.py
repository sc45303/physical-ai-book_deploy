from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.ext.declarative import as_declarative
from sqlalchemy.sql import func
import uuid
from sqlalchemy.types import Uuid

@as_declarative()
class Base:
    pass

# Common base with id, created_at, updated_at
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.sql import func
import uuid
from sqlalchemy.types import Uuid


class Base:
    __abstract__ = True

    @declared_attr
    def id(cls):
        return Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())