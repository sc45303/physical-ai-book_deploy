from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import AsyncGenerator
from dotenv import load_dotenv
import os

load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("NEON_DB_URL", "postgresql+asyncpg://user:password@localhost/dbname")

# Create async engine
engine = create_async_engine(DATABASE_URL)

# Create async session
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Base class for models
Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        yield session