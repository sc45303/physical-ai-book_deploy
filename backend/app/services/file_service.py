import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
from app.utils.logging import get_logger
from app.config import settings
from app.exceptions import InvalidFileTypeException

logger = get_logger(__name__)


class FileService:
    """
    Service class to handle file operations including uploads
    """
    
    def __init__(self):
        # Create upload directory if it doesn't exist
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    async def save_upload_file(self, upload_file: UploadFile, destination: Optional[Path] = None) -> Path:
        """
        Save an uploaded file to the specified destination
        
        Args:
            upload_file: The uploaded file
            destination: Optional destination path. If not provided, one will be generated
            
        Returns:
            Path: Path to the saved file
        """
        if destination is None:
            # Generate a unique filename
            file_extension = Path(upload_file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            destination = self.upload_dir / unique_filename
        
        # Validate file type
        self._validate_file_type(upload_file.filename)
        
        # Save the file
        with open(destination, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
        
        logger.info(f"File saved successfully: {destination}")
        return destination
    
    def _validate_file_type(self, filename: str) -> None:
        """
        Validate the file type based on its extension

        Args:
            filename: Name of the file to validate

        Raises:
            InvalidFileTypeException: If the file type is not supported
        """
        file_extension = Path(filename).suffix.lower()

        if file_extension not in ['.pdf', '.docx', '.txt']:
            raise InvalidFileTypeException(file_extension)

    def validate_file_content_type(self, content_type: str) -> bool:
        """
        Validate the file content type (MIME type)

        Args:
            content_type: MIME type of the uploaded file

        Returns:
            bool: True if content type is valid, False otherwise
        """
        allowed_content_types = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/plain',
            'text/markdown'
        ]

        return content_type in allowed_content_types
    
    def validate_file_size(self, file_size: int, max_size_mb: int = 50) -> bool:
        """
        Validate that the file size is within the allowed limit
        
        Args:
            file_size: Size of the file in bytes
            max_size_mb: Maximum allowed size in megabytes
            
        Returns:
            bool: True if file size is within limits, False otherwise
        """
        max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        return file_size <= max_size_bytes
    
    def get_file_size(self, file_path: Path) -> int:
        """
        Get the size of a file in bytes
        
        Args:
            file_path: Path to the file
            
        Returns:
            int: Size of the file in bytes
        """
        return file_path.stat().st_size
    
    def get_file_info(self, file_path: Path) -> dict:
        """
        Get information about a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            dict: File information including size, extension, and modification time
        """
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "extension": file_path.suffix,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "name": file_path.name
        }
    
    def delete_file(self, file_path: Path) -> bool:
        """
        Delete a file
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            bool: True if file was deleted successfully, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted successfully: {file_path}")
                return True
            else:
                logger.warning(f"File does not exist: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate a unique filename by prepending a UUID
        
        Args:
            original_filename: Original filename
            
        Returns:
            str: Unique filename
        """
        file_path = Path(original_filename)
        unique_name = f"{uuid.uuid4()}{file_path.suffix}"
        return unique_name