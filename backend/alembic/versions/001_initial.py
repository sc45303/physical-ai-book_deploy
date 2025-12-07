"""Initial migration for Book RAG System

Revision ID: 001_initial
Revises: 
Create Date: 2025-12-07 11:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.UniqueConstraint('email')
    )
    
    # Create books table
    op.create_table('books',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('author', sa.String(), nullable=False),
        sa.Column('format', sa.String(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=False),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('upload_date', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('processing_status', sa.String(), nullable=False, default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'))
    )
    
    # Create book_chunks table
    op.create_table('book_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('book_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('books.id'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('vector_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'))
    )
    
    # Create chat_sessions table
    op.create_table('chat_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('book_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('books.id'), nullable=False),
        sa.Column('session_name', sa.String(), nullable=True),
        sa.Column('mode', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'))
    )
    
    # Create chat_messages table
    op.create_table('chat_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('chat_session_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('chat_sessions.id'), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('sources', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'))
    )
    
    # Create vector_embeddings table
    op.create_table('vector_embeddings',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('book_chunk_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_used', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'))
    )
    
    # Create indexes
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_books_user_id', 'books', ['user_id'])
    op.create_index('ix_book_chunks_book_id', 'book_chunks', ['book_id'])
    op.create_index('ix_chat_sessions_user_id', 'chat_sessions', ['user_id'])
    op.create_index('ix_chat_sessions_book_id', 'chat_sessions', ['book_id'])
    op.create_index('ix_chat_messages_session_id', 'chat_messages', ['chat_session_id'])


def downgrade():
    # Drop tables in reverse order to handle foreign key constraints
    op.drop_table('vector_embeddings')
    op.drop_table('chat_messages')
    op.drop_table('chat_sessions')
    op.drop_table('book_chunks')
    op.drop_table('books')
    op.drop_table('users')