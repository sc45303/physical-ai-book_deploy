import asyncio
from typing import List, Tuple
from pathlib import Path
import logging
from app.utils.logging import get_logger
from app.exceptions import InvalidFileTypeException

logger = get_logger(__name__)


class DocumentParserService:
    """
    Service class to handle document parsing for different file formats
    """
    
    @staticmethod
    def _validate_file_type(file_path: str) -> str:
        """
        Validate the file type and return the format
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File format ('PDF', 'DOCX', 'TXT')
            
        Raises:
            InvalidFileTypeException: If the file type is not supported
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return 'PDF'
        elif file_extension == '.docx':
            return 'DOCX'
        elif file_extension == '.txt':
            return 'TXT'
        else:
            raise InvalidFileTypeException(file_extension)
    
    @staticmethod
    async def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            import pypdf
            
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = pypdf.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            logger.warning("pypdf not installed. Using alternative PDF parsing method")
            # Alternative approach if pypdf is not available
            raise Exception("PDF parsing requires pypdf library. Please install it using 'pip install pypdf'")
    
    @staticmethod
    async def extract_text_from_docx(file_path: str) -> str:
        """
        Extract text from a DOCX file
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            str: Extracted text
        """
        try:
            from docx import Document
            
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text)
        except ImportError:
            logger.warning("python-docx not installed. Using alternative DOCX parsing method")
            # Alternative approach if python-docx is not available
            raise Exception("DOCX parsing requires python-docx library. Please install it using 'pip install python-docx'")
    
    @staticmethod
    async def extract_text_from_txt(file_path: str) -> str:
        """
        Extract text from a TXT file
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            str: Extracted text
        """
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            return txt_file.read()
    
    async def parse_document(self, file_path: str) -> Tuple[str, str]:
        """
        Parse a document and extract its text content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple[str, str]: A tuple containing (extracted_text, file_format)
        """
        logger.info(f"Starting to parse document: {file_path}")
        
        # Validate file type
        file_format = self._validate_file_type(file_path)
        logger.info(f"Detected file format: {file_format}")
        
        # Extract text based on file format
        if file_format == 'PDF':
            text = await self.extract_text_from_pdf(file_path)
        elif file_format == 'DOCX':
            text = await self.extract_text_from_docx(file_path)
        elif file_format == 'TXT':
            text = await self.extract_text_from_txt(file_path)
        else:
            raise InvalidFileTypeException(file_format)
        
        logger.info(f"Successfully parsed document: {file_path}. Extracted {len(text)} characters.")
        
        return text, file_format
    
    async def get_page_count(self, file_path: str) -> int:
        """
        Get the page count of a document (currently only supports PDF)
        
        Args:
            file_path: Path to the document file
            
        Returns:
            int: Number of pages in the document
        """
        file_format = self._validate_file_type(file_path)
        
        if file_format == 'PDF':
            try:
                import pypdf
                
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = pypdf.PdfReader(pdf_file)
                    return len(pdf_reader.pages)
            except ImportError:
                logger.warning("pypdf not installed, cannot get PDF page count")
                return 0
        else:
            # For non-PDF files, we can't directly determine pages
            # Could implement alternative logic if needed
            return 0