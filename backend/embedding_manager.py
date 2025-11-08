"""
Embedding Manager Module
Centralized embedding generation and management
"""
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pathlib import Path
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")
 
 
class EmbeddingManager:
    """Manage embedding generation with caching and batch processing"""
   
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 16
    ):
        """
        Initialize embedding manager
       
        Args:
            model: Embedding model name
            api_key: OpenAI API key
            base_url: API endpoint URL
            batch_size: Number of texts to embed at once
        """
        self.batch_size = batch_size
       
        # Get configuration from environment or parameters
        self.api_key = api_key or os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_EMBEDDING_API_BASE") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
       
        # Ensure endpoint ends with /v1
        if not self.base_url.endswith('/v1'):
            self.base_url = self.base_url.rstrip('/') + '/v1'
       
        # Create HTTP client with SSL verification disabled for custom endpoints
        # http_client = httpx.Client(
        #     timeout=httpx.Timeout(30.0, connect=5.0),  # Reduced timeout to detect issues faster
        #     limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        #     verify=False
        # )
       
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key,
            base_url=self.base_url,
            #http_client=http_client
        )
       
        logger.info(f"✓ Embedding Manager initialized")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Endpoint: {self.base_url}")
        logger.info(f"  Batch size: {self.batch_size}")
   
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
       
        Args:
            text: Text to embed
           
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
       
        try:
            embedding = self.embeddings.embed_query(text.strip())
            return embedding
        except Exception as e:
            logger.error(f"✗ Error embedding text: {e}")
            raise
   
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
       
        Args:
            texts: List of texts to embed
           
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
       
        # Clean and validate texts
        cleaned_texts = []
        for text in texts:
            if text and text.strip():
                cleaned_texts.append(text.strip())
       
        if not cleaned_texts:
            raise ValueError("No valid texts to embed")
       
        try:
            embeddings = self.embeddings.embed_documents(cleaned_texts)
            logger.info(f"✓ Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"✗ Error embedding batch: {e}")
            raise
   
    def embed_documents(
        self,
        documents: List[Dict],
        text_key: str = 'text'
    ) -> List[Dict]:
        """
        Generate embeddings for multiple documents with metadata
       
        Args:
            documents: List of document dictionaries with text and metadata
            text_key: Key in document dict containing text to embed
           
        Returns:
            Documents with added 'embedding' field
        """
        if not documents:
            return []
       
        logger.info(f"📝 Embedding {len(documents)} documents...")
        logger.info(f"   Batch size: {self.batch_size}")
       
        # Process in batches
        results = []
        total_batches = (len(documents) - 1) // self.batch_size + 1
       
        for i in range(0, len(documents), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = documents[i:i + self.batch_size]
            batch_texts = [doc.get(text_key, '') for doc in batch]
           
            logger.info(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
           
            try:
                logger.info(f"   Calling embedding API...")
                batch_embeddings = self.embed_batch(batch_texts)
                logger.info(f"   ✓ Received {len(batch_embeddings)} embeddings")
               
                # Add embeddings to documents
                for doc, embedding in zip(batch, batch_embeddings):
                    doc['embedding'] = embedding
                    results.append(doc)
               
                logger.info(f"   ✓ Batch {batch_num}/{total_batches} complete")
               
            except Exception as e:
                logger.error(f"   ✗ Failed to embed batch {batch_num}: {e}")
                # Add documents without embeddings
                for doc in batch:
                    doc['embedding'] = None
                    doc['embedding_error'] = str(e)
                    results.append(doc)
       
        successful = sum(1 for doc in results if doc.get('embedding') is not None)
        logger.info(f"✓ Successfully embedded {successful}/{len(documents)} documents")
       
        return results
   
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model
       
        Returns:
            Embedding dimension
        """
        test_embedding = self.embed_text("test")
        return len(test_embedding)
 
 
# Example usage
if __name__ == "__main__":
    manager = EmbeddingManager()
   
    # Test single embedding
    embedding = manager.embed_text("Hello, world!")
    print(f"Embedding dimension: {len(embedding)}")
   
    # Test batch embedding
    texts = ["First text", "Second text", "Third text"]
    embeddings = manager.embed_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")
   
    # Test document embedding
    documents = [
        {'text': 'Document 1', 'metadata': {'source': 'file1.pdf'}},
        {'text': 'Document 2', 'metadata': {'source': 'file2.pdf'}},
    ]
    embedded_docs = manager.embed_documents(documents)
    print(f"Embedded {len(embedded_docs)} documents")
 
 