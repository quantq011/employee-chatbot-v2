"""
Vector Store Manager Module
Handles ChromaDB operations: add, update, delete, search
"""
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from pathlib import Path
import logging
import uuid
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
class VectorStoreManager:
    """Manage ChromaDB vector store operations"""
   
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        """
        Initialize vector store manager
       
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
       
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
       
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
       
        logger.info(f"✓ Vector Store initialized at {persist_directory}")
   
    def get_or_create_collection(
        self,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        """
        Get existing collection or create new one
       
        Args:
            collection_name: Name of collection (uses default if None)
            embedding_function: Custom embedding function for ChromaDB
           
        Returns:
            ChromaDB collection
        """
        name = collection_name or self.collection_name
       
        try:
            collection = self.client.get_collection(
                name=name,
                embedding_function=embedding_function
            )
            logger.info(f"✓ Loaded existing collection: {name}")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                embedding_function=embedding_function
            )
            logger.info(f"✓ Created new collection: {name}")
       
        return collection
   
    def add_documents(
        self,
        documents: List[Dict],
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Add documents to vector store
       
        Args:
            documents: List of documents with 'text', 'embedding', and 'metadata'
            collection_name: Target collection name
            embedding_function: Embedding function if needed
           
        Returns:
            Result summary
        """
        collection = self.get_or_create_collection(collection_name, embedding_function)
       
        # Prepare data for insertion
        ids = []
        embeddings = []
        metadatas = []
        texts = []
       
        for doc in documents:
            # Generate ID if not provided
            doc_id = doc.get('id') or str(uuid.uuid4())
            ids.append(doc_id)
           
            # Get embedding
            embedding = doc.get('embedding')
            if embedding is None:
                logger.warning(f"Document {doc_id} has no embedding, skipping")
                continue
            embeddings.append(embedding)
           
            # Get text
            text = doc.get('text', '')
            texts.append(text)
           
            # Prepare metadata (ChromaDB requires string/int/float values only)
            metadata = doc.get('metadata', {})
            clean_metadata = self._clean_metadata(metadata)
            metadatas.append(clean_metadata)
       
        if not ids:
            return {
                'success': False,
                'message': 'No valid documents to add',
                'added': 0
            }
       
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
           
            logger.info(f"✓ Added {len(ids)} documents to {collection.name}")
           
            return {
                'success': True,
                'message': f'Added {len(ids)} documents',
                'added': len(ids),
                'collection': collection.name
            }
           
        except Exception as e:
            logger.error(f"✗ Error adding documents: {e}")
            return {
                'success': False,
                'message': str(e),
                'added': 0
            }
   
    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Search vector store
       
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where_filter: Metadata filter (e.g., {'source': 'file1.pdf'})
            collection_name: Collection to search
            embedding_function: Embedding function
           
        Returns:
            Search results
        """
        collection = self.get_or_create_collection(collection_name, embedding_function)
       
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )
           
            logger.info(f"✓ Found {len(results['ids'][0])} results")
           
            return {
                'success': True,
                'ids': results['ids'][0] if results['ids'] else [],
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else []
            }
           
        except Exception as e:
            logger.error(f"✗ Error searching: {e}")
            return {
                'success': False,
                'error': str(e),
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': []
            }
   
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Update existing documents
       
        Args:
            ids: Document IDs to update
            documents: New document texts
            embeddings: New embeddings
            metadatas: New metadata
            collection_name: Collection name
            embedding_function: Embedding function
           
        Returns:
            Update result
        """
        collection = self.get_or_create_collection(collection_name, embedding_function)
       
        try:
            # Clean metadatas if provided
            clean_metadatas = None
            if metadatas:
                clean_metadatas = [self._clean_metadata(m) for m in metadatas]
           
            collection.update(
                ids=ids,
                embeddings=embeddings,
                metadatas=clean_metadatas,
                documents=documents
            )
           
            logger.info(f"✓ Updated {len(ids)} documents in {collection.name}")
           
            return {
                'success': True,
                'message': f'Updated {len(ids)} documents',
                'updated': len(ids)
            }
           
        except Exception as e:
            logger.error(f"✗ Error updating documents: {e}")
            return {
                'success': False,
                'message': str(e),
                'updated': 0
            }
   
    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        where_filter: Optional[Dict] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Delete documents from vector store
       
        Args:
            ids: Document IDs to delete
            where_filter: Metadata filter for deletion
            collection_name: Collection name
            embedding_function: Embedding function
           
        Returns:
            Deletion result
        """
        collection = self.get_or_create_collection(collection_name, embedding_function)
       
        try:
            collection.delete(
                ids=ids,
                where=where_filter
            )
           
            count = len(ids) if ids else "matching"
            logger.info(f"✓ Deleted {count} documents from {collection.name}")
           
            return {
                'success': True,
                'message': f'Deleted documents',
                'deleted': count
            }
           
        except Exception as e:
            logger.error(f"✗ Error deleting documents: {e}")
            return {
                'success': False,
                'message': str(e),
                'deleted': 0
            }
   
    def get_collection_stats(
        self,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about a collection
       
        Args:
            collection_name: Collection name
            embedding_function: Embedding function
           
        Returns:
            Collection statistics
        """
        collection = self.get_or_create_collection(collection_name, embedding_function)
       
        try:
            count = collection.count()
           
            return {
                'success': True,
                'collection_name': collection.name,
                'document_count': count
            }
           
        except Exception as e:
            logger.error(f"✗ Error getting stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }
   
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        Clean metadata to ensure ChromaDB compatibility
       
        Args:
            metadata: Raw metadata
           
        Returns:
            Cleaned metadata with only string/int/float values
        """
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings
                clean[key] = ', '.join(str(v) for v in value)
            else:
                # Convert other types to string
                clean[key] = str(value)
       
        # ChromaDB requires at least one metadata field
        if not clean:
            clean['source'] = 'unknown'
       
        return clean
 
 
# Example usage
if __name__ == "__main__":
    manager = VectorStoreManager()
   
    # Test adding documents
    documents = [
        {
            'id': 'doc1',
            'text': 'Sample document',
            'embedding': [0.1] * 1536,  # Fake embedding
            'metadata': {'source': 'test.pdf', 'page': 1}
        }
    ]
   
    result = manager.add_documents(documents)
    print(f"Add result: {result}")
   
    # Test search
    search_result = manager.search([0.1] * 1536, n_results=5)
    print(f"Search result: {search_result['success']}")
   
    # Test stats
    stats = manager.get_collection_stats()
    print(f"Collection stats: {stats}")
 
 