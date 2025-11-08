"""
Document Ingestion Pipeline
Orchestrates the complete workflow: load -> process -> embed -> store
"""
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
 
from document_loader import DocumentLoader
from text_processor import TextProcessor
from embedding_manager import EmbeddingManager
from vector_store_manager import VectorStoreManager
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
class DocumentIngestionPipeline:
    """Complete document ingestion pipeline"""
   
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 16,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents"
    ):
        """
        Initialize ingestion pipeline
       
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for embedding
            persist_directory: ChromaDB persist directory
            collection_name: Collection name
        """
        self.loader = DocumentLoader()
        self.processor = TextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_manager = EmbeddingManager(batch_size=batch_size)
        self.vector_store = VectorStoreManager(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
       
        logger.info("✓ Document Ingestion Pipeline initialized")
   
    def ingest_file(
        self,
        file_path: str,
        use_sections: bool = False,
        additional_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single file
       
        Args:
            file_path: Path to file
            use_sections: Whether to chunk by sections
            additional_metadata: Additional metadata to attach
           
        Returns:
            Ingestion result
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Ingesting file: {file_path}")
        logger.info(f"{'='*60}")
       
        # Step 1: Load document
        load_result = self.loader.load_document(file_path)
        if not load_result['success']:
            return {
                'success': False,
                'error': load_result['error'],
                'file_path': file_path,
                'chunks_processed': 0
            }
       
        # Step 2: Process and chunk text
        logger.info(f"Step 2: Processing and chunking text...")
        metadata = load_result['metadata']
        if additional_metadata:
            metadata.update(additional_metadata)
       
        chunks = self.processor.process_document(
            text=load_result['content'],
            metadata=metadata,
            use_sections=use_sections
        )
       
        if not chunks:
            return {
                'success': False,
                'error': 'No chunks generated from document',
                'file_path': file_path,
                'chunks_processed': 0
            }
       
        logger.info(f"✓ Created {len(chunks)} chunks")
       
        # Step 3: Generate embeddings
        logger.info(f"Step 3: Generating embeddings for {len(chunks)} chunks...")
        embedded_chunks = self.embedding_manager.embed_documents(chunks, text_key='text')
        logger.info(f"✓ Embedding complete")
       
        # Filter out failed embeddings
        valid_chunks = [c for c in embedded_chunks if c.get('embedding') is not None]
       
        if not valid_chunks:
            return {
                'success': False,
                'error': 'Failed to generate embeddings',
                'file_path': file_path,
                'chunks_processed': 0
            }
       
        logger.info(f"✓ {len(valid_chunks)} valid embeddings")
       
        # Step 4: Store in vector database
        logger.info(f"Step 4: Storing in vector database...")
        store_result = self.vector_store.add_documents(valid_chunks)
       
        logger.info(f"✓ Completed ingestion: {file_path}")
        logger.info(f"  Chunks: {len(valid_chunks)}")
        logger.info(f"  Added to vector store: {store_result['added']}")
       
        return {
            'success': True,
            'file_path': file_path,
            'chunks_processed': len(valid_chunks),
            'chunks_stored': store_result['added'],
            'collection': store_result.get('collection')
        }
   
    def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        use_sections: bool = False,
        additional_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a directory
       
        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            use_sections: Whether to chunk by sections
            additional_metadata: Additional metadata
           
        Returns:
            Ingestion summary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Ingesting directory: {directory_path}")
        logger.info(f"{'='*60}")
       
        # Load all documents
        load_results = self.loader.load_directory(directory_path, recursive=recursive)
       
        if not load_results:
            return {
                'success': False,
                'error': 'No documents found in directory',
                'directory_path': directory_path,
                'files_processed': 0
            }
       
        # Process each document
        results = []
        total_chunks = 0
        total_stored = 0
       
        for load_result in load_results:
            if not load_result['success']:
                logger.warning(f"Skipping {load_result.get('metadata', {}).get('filename', 'unknown')}: {load_result['error']}")
                continue
           
            # Process chunks
            metadata = load_result['metadata']
            if additional_metadata:
                metadata.update(additional_metadata)
           
            chunks = self.processor.process_document(
                text=load_result['content'],
                metadata=metadata,
                use_sections=use_sections
            )
           
            if not chunks:
                logger.warning(f"No chunks generated for {metadata.get('filename')}")
                continue
           
            # Generate embeddings
            embedded_chunks = self.embedding_manager.embed_documents(chunks, text_key='text')
            valid_chunks = [c for c in embedded_chunks if c.get('embedding') is not None]
           
            if not valid_chunks:
                logger.warning(f"No valid embeddings for {metadata.get('filename')}")
                continue
           
            # Store in vector database
            store_result = self.vector_store.add_documents(valid_chunks)
           
            total_chunks += len(valid_chunks)
            total_stored += store_result['added']
           
            results.append({
                'filename': metadata.get('filename'),
                'chunks': len(valid_chunks),
                'stored': store_result['added']
            })
       
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Directory ingestion complete")
        logger.info(f"  Files processed: {len(results)}")
        logger.info(f"  Total chunks: {total_chunks}")
        logger.info(f"  Total stored: {total_stored}")
        logger.info(f"{'='*60}")
       
        return {
            'success': True,
            'directory_path': directory_path,
            'files_processed': len(results),
            'total_chunks': total_chunks,
            'total_stored': total_stored,
            'file_details': results
        }
   
    def ingest_batch(
        self,
        file_paths: List[str],
        use_sections: bool = False,
        additional_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Ingest a batch of specific files
       
        Args:
            file_paths: List of file paths
            use_sections: Whether to chunk by sections
            additional_metadata: Additional metadata
           
        Returns:
            Batch ingestion summary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Ingesting batch of {len(file_paths)} files")
        logger.info(f"{'='*60}")
       
        results = []
        successful = 0
        total_chunks = 0
       
        for file_path in file_paths:
            result = self.ingest_file(
                file_path=file_path,
                use_sections=use_sections,
                additional_metadata=additional_metadata
            )
           
            results.append(result)
           
            if result['success']:
                successful += 1
                total_chunks += result['chunks_processed']
       
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Batch ingestion complete")
        logger.info(f"  Successful: {successful}/{len(file_paths)}")
        logger.info(f"  Total chunks: {total_chunks}")
        logger.info(f"{'='*60}")
       
        return {
            'success': True,
            'files_attempted': len(file_paths),
            'files_successful': successful,
            'total_chunks': total_chunks,
            'results': results
        }
 
 
# Example usage
if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
   
    # Ingest a single file
    result = pipeline.ingest_file("example.pdf")
    print(f"Single file result: {result}")
   
    # Ingest a directory
    result = pipeline.ingest_directory("./documents", recursive=True)
    print(f"Directory result: {result}")
 
 