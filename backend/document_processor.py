import os
import logging
from typing import List, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Updated imports for langchain
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except ImportError:
    # Fallback for older langchain versions
    from langchain.document_loaders import PyPDFLoader, TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chroma_db_path: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document processor with ChromaDB and embedding model.
        
        Args:
            chroma_db_path: Path to ChromaDB storage
            embedding_model: Name of the sentence transformer model
        """
        self.chroma_db_path = chroma_db_path
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection_name = "document_embeddings"
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def load_documents(self, documents_path: str) -> List[Document]:
        """
        Load documents from a directory containing PDF and TXT files.
        
        Args:
            documents_path: Path to directory containing documents
            
        Returns:
            List of loaded documents
        """
        documents = []
        docs_dir = Path(documents_path)
        
        if not docs_dir.exists():
            raise ValueError(f"Documents directory {documents_path} does not exist")
        
        for file_path in docs_dir.iterdir():
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded PDF: {file_path.name} ({len(docs)} pages)")
                
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded TXT: {file_path.name}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into chunks with overlap.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def store_embeddings(self, chunks: List[Document]):
        """
        Store document chunks and their embeddings in ChromaDB.
        
        Args:
            chunks: List of document chunks
        """
        # Prepare data for ChromaDB
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            metadata = {
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "chunk_index": i
            }
            
            ids.append(chunk_id)
            metadatas.append(metadata)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.generate_embeddings(texts)
        
        # Store in ChromaDB
        logger.info("Storing embeddings in ChromaDB...")
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB")
    
    def process_documents(self, documents_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Complete document processing pipeline: load, chunk, embed, and store.
        
        Args:
            documents_path: Path to documents directory
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        logger.info("Starting document processing pipeline...")
        
        # Load documents
        documents = self.load_documents(documents_path)
        
        # Chunk documents
        chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)
        
        # Store embeddings
        self.store_embeddings(chunks)
        
        logger.info("Document processing pipeline completed!")
    
    def search_similar_chunks(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar document chunks based on a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Process documents
    processor.process_documents("../data/sample_docs")
    
    # Test search
    results = processor.search_similar_chunks("What is machine learning?", n_results=3)
    print("Search results:", results)
