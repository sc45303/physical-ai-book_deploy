from fastapi import FastAPI
from app.routes import upload, status, search, chat
from app.utils.logging import setup_logging
import uvicorn
import os

# Set up logging
setup_logging()

# Create FastAPI app instance
app = FastAPI(
    title="Book RAG System API",
    description="An API for uploading books, processing them into vector embeddings, and answering questions",
    version="1.0.0"
)

# Include routers
app.include_router(upload.router)
app.include_router(status.router)
app.include_router(search.router)
app.include_router(chat.router)

# Basic health check endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Book RAG System API"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Book RAG System is running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)