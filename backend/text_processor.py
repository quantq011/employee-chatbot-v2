"""
Text Processor Module
Handles text cleaning, normalization, and chunking for document processing
"""
import re
from typing import List, Dict, Optional
import logging
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
class TextProcessor:
    """Process and chunk text for embedding"""
   
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        remove_headers_footers: bool = True
    ):
        """
        Initialize text processor
       
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between chunks to preserve context
            remove_headers_footers: Whether to remove common headers/footers
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.remove_headers_footers = remove_headers_footers
   
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
       
        Args:
            text: Raw text to clean
           
        Returns:
            Cleaned text
        """
        if not text:
            return ""
       
        logger.debug(f"  Cleaning text (length: {len(text)})")
       
        # Remove null bytes and control characters
        text = text.replace('\x00', '')
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
       
        # Normalize whitespace - simplified to avoid catastrophic backtracking
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double (simplified)
       
        # Remove page numbers (common patterns)
        if self.remove_headers_footers:
            text = re.sub(r'\n\s*Page \d+\s*\n', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
       
        # Remove URLs (optional - comment out if you want to keep them)
        # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
       
        # Normalize unicode quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', '-').replace('–', '-')
       
        # Strip leading/trailing whitespace
        text = text.strip()
       
        logger.debug(f"  ✓ Cleaned (result length: {len(text)})")
       
        return text
   
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into overlapping chunks
       
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
           
        Returns:
            List of chunks with metadata
        """
        if not text:
            return []
       
        chunks = []
        text_length = len(text)
       
        # If text is smaller than chunk size, return as single chunk
        if text_length <= self.chunk_size:
            chunk_data = {
                'text': text,
                'chunk_index': 0,
                'total_chunks': 1,
                'start_char': 0,
                'end_char': text_length
            }
            if metadata:
                chunk_data.update(metadata)
            return [chunk_data]
       
        # Split into chunks with overlap
        start = 0
        chunk_index = 0
       
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
           
            # Try to break at sentence or paragraph boundary
            if end < text_length:
                # Look for paragraph break
                break_point = text.rfind('\n\n', start, end)
                if break_point == -1:
                    # Look for sentence break
                    break_point = text.rfind('. ', start, end)
                if break_point == -1:
                    # Look for any break
                    break_point = text.rfind(' ', start, end)
               
                if break_point > start:
                    end = break_point + 1
           
            chunk_text = text[start:end].strip()
           
            if chunk_text:
                chunk_data = {
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'start_char': start,
                    'end_char': end
                }
                if metadata:
                    chunk_data.update(metadata)
               
                chunks.append(chunk_data)
                chunk_index += 1
           
            # Move start position with overlap
            start = end - self.chunk_overlap if end < text_length else text_length
       
        # Update total_chunks in all chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
       
        logger.info(f"✓ Split text into {len(chunks)} chunks")
        return chunks
   
    def chunk_by_sections(
        self,
        text: str,
        section_pattern: str = r'\n#+\s+(.+?)\n',
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk text by sections (e.g., markdown headers)
       
        Args:
            text: Text to chunk
            section_pattern: Regex pattern to identify sections
            metadata: Optional metadata
           
        Returns:
            List of section chunks with metadata
        """
        chunks = []
       
        # Find all section headers
        sections = list(re.finditer(section_pattern, text))
       
        if not sections:
            # No sections found, fall back to regular chunking
            return self.chunk_text(text, metadata)
       
        for i, section_match in enumerate(sections):
            section_title = section_match.group(1)
            section_start = section_match.start()
           
            # Find section end (start of next section or end of text)
            if i < len(sections) - 1:
                section_end = sections[i + 1].start()
            else:
                section_end = len(text)
           
            section_text = text[section_start:section_end].strip()
           
            if section_text:
                chunk_data = {
                    'text': section_text,
                    'section_title': section_title,
                    'chunk_index': i,
                    'total_chunks': len(sections),
                    'chunk_type': 'section'
                }
                if metadata:
                    chunk_data.update(metadata)
               
                chunks.append(chunk_data)
       
        logger.info(f"✓ Split text into {len(chunks)} sections")
        return chunks
   
    def process_document(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        use_sections: bool = False
    ) -> List[Dict]:
        """
        Complete document processing: clean and chunk
       
        Args:
            text: Raw document text
            metadata: Document metadata
            use_sections: Whether to chunk by sections
           
        Returns:
            List of processed chunks
        """
        logger.info(f"  Processing document (length: {len(text)} chars, use_sections: {use_sections})")
       
        # Clean text
        logger.info(f"  Cleaning text...")
        cleaned_text = self.clean_text(text)
        logger.info(f"  ✓ Cleaned (length: {len(cleaned_text)} chars)")
       
        if not cleaned_text:
            logger.warning("No text after cleaning")
            return []
       
        # Chunk text
        logger.info(f"  Chunking text...")
        if use_sections:
            chunks = self.chunk_by_sections(cleaned_text, metadata=metadata)
        else:
            chunks = self.chunk_text(cleaned_text, metadata=metadata)
       
        logger.info(f"✓ Processed document into {len(chunks)} chunks")
        return chunks
 
 
# Example usage
if __name__ == "__main__":
    processor = TextProcessor(chunk_size=500, chunk_overlap=100)
   
    sample_text = """
    # Introduction
    This is a sample document with multiple sections.
   
    # Section 1
    Here is some content for section 1.
    It has multiple paragraphs.
   
    # Section 2
    And here is section 2 with more content.
    """
   
    # Test cleaning
    cleaned = processor.clean_text(sample_text)
    print(f"Cleaned text length: {len(cleaned)}")
   
    # Test chunking
    chunks = processor.process_document(sample_text, use_sections=True)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk.get('section_title', 'N/A')}")
 
 