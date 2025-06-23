#!/usr/bin/env python3
"""
Simplified FastAPI backend for RAG Chatbot
This version has minimal dependencies and should work reliably
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not available. Please install: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False
    sys.exit(1)

# Try to import our modules
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    CORE_MODULES_AVAILABLE = True
except ImportError:
    print("Core modules not available. Please install: pip install chromadb sentence-transformers")
    CORE_MODULES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Simple RAG chatbot API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    n_results: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str

# Global variables
doc_processor = None
embedding_model = None

class SimpleDocumentProcessor:
    """Simplified document processor."""
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.initialized = False
        
        if CORE_MODULES_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB and embedding model."""
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path="../chroma_db")
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("document_embeddings")
                logger.info("Loaded existing document collection")
            except:
                self.collection = self.chroma_client.create_collection("document_embeddings")
                logger.info("Created new document collection")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {e}")
            self.initialized = False
    
    def process_documents(self, docs_path: str = "../data/sample_docs"):
        """Process documents and store embeddings."""
        if not self.initialized:
            return False
        
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Load documents
            documents = []
            docs_dir = Path(docs_path)
            
            if not docs_dir.exists():
                logger.error(f"Documents directory not found: {docs_path}")
                return False
            
            for pdf_file in docs_dir.glob("*.pdf"):
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {pdf_file.name}")
                except Exception as e:
                    logger.error(f"Failed to load {pdf_file.name}: {e}")
            
            if not documents:
                logger.error("No documents loaded")
                return False
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Generate embeddings and store
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            # Prepare data for ChromaDB
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": str(chunk.metadata.get("source", "unknown")),
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": i
                }
                metadatas.append(metadata)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Processed and stored {len(chunks)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5):
        """Search for relevant document chunks."""
        if not self.initialized:
            return None
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return None

# Initialize document processor
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    global doc_processor
    
    logger.info("Starting RAG Chatbot API...")
    
    if CORE_MODULES_AVAILABLE:
        doc_processor = SimpleDocumentProcessor()
        if doc_processor.initialized:
            logger.info("✅ Document processor initialized successfully")
        else:
            logger.warning("⚠️ Document processor failed to initialize")
    else:
        logger.warning("⚠️ Core modules not available")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "core_modules": CORE_MODULES_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/status")
async def get_status():
    """Get system status."""
    total_chunks = 0
    
    if doc_processor and doc_processor.initialized:
        try:
            total_chunks = doc_processor.collection.count()
        except:
            total_chunks = 0
    
    return {
        "document_processor_ready": doc_processor is not None and doc_processor.initialized,
        "llm_ready": False,  # We'll use simple responses
        "total_chunks": total_chunks,
        "model_info": {
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "simple_responses"
        }
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    if not doc_processor or not doc_processor.initialized:
        return {
            "answer": "Document processor not initialized. Please process documents first.",
            "sources": [],
            "confidence": 0.0,
            "query": request.message
        }
    
    try:
        # Search for relevant documents
        results = doc_processor.search_documents(request.message, request.n_results)
        
        if not results or not results.get("documents") or not results["documents"][0]:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "sources": [],
                "confidence": 0.0,
                "query": request.message
            }
        
        # Extract information
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        # Create a simple response based on the most relevant document
        most_relevant_doc = documents[0]
        
        # Simple response generation
        answer = f"Based on the documents, here's relevant information about '{request.message}':\n\n"
        answer += most_relevant_doc[:500] + "..." if len(most_relevant_doc) > 500 else most_relevant_doc
        
        # Prepare sources
        sources = []
        for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
            sources.append({
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", 0),
                "relevance_score": 1 - distance
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": max(1 - min(distances), 0.0) if distances else 0.0,
            "query": request.message
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "query": request.message
        }

@app.post("/process-documents")
async def process_documents():
    """Process documents endpoint."""
    if not doc_processor:
        raise HTTPException(status_code=503, detail="Document processor not available")
    
    try:
        success = doc_processor.process_documents()
        if success:
            return {"message": "Documents processed successfully"}
        else:
            return {"message": "Failed to process documents", "error": True}
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(query: str, n_results: int = 5):
    """Search documents endpoint."""
    if not doc_processor or not doc_processor.initialized:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    results = doc_processor.search_documents(query, n_results)
    if results:
        return results
    else:
        raise HTTPException(status_code=500, detail="Search failed")

if __name__ == "__main__":
    # Run the server
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )