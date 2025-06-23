import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Try to import our modules, with fallbacks
try:
    from document_processor import DocumentProcessor
    from llm_interface import LLMInterface, RAGSystem
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the backend directory")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot API using local LLM and ChromaDB",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    n_results: Optional[int] = 5
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query: str

class SystemStatus(BaseModel):
    document_processor_ready: bool
    llm_ready: bool
    total_chunks: int
    model_info: Dict[str, Any]

# Global instances
doc_processor = None
llm_interface = None
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global doc_processor, llm_interface, rag_system
    
    logger.info("Initializing RAG system...")
    
    # Initialize document processor
    try:
        doc_processor = DocumentProcessor(chroma_db_path="../chroma_db")
        logger.info("Document processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {e}")
        doc_processor = None
    
    # Initialize LLM interface
    try:
        # Look for model files in the models directory
        models_dir = Path("../models")
        model_path = None
        
        if models_dir.exists():
            for model_file in models_dir.glob("*.gguf"):
                model_path = str(model_file)
                break
        
        llm_interface = LLMInterface(model_path=model_path)
        logger.info(f"LLM interface initialized (Model available: {llm_interface.is_available()})")
        
        if not llm_interface.is_available():
            logger.warning("No local LLM model found. Using fallback responses.")
            
    except Exception as e:
        logger.error(f"Failed to initialize LLM interface: {e}")
        llm_interface = LLMInterface()  # Initialize without model
    
    # Initialize RAG system
    if doc_processor and llm_interface:
        rag_system = RAGSystem(doc_processor, llm_interface)
        logger.info("RAG system initialized successfully")
    else:
        logger.error("Failed to initialize RAG system")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/chat": "POST - Send a message to the chatbot",
            "/status": "GET - Check system status",
            "/process-documents": "POST - Process documents for RAG",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/status")
async def get_status():
    """Get system status."""
    # Get document count
    total_chunks = 0
    try:
        if doc_processor and doc_processor.collection:
            total_chunks = doc_processor.collection.count()
    except:
        total_chunks = 0
    
    # Get model info
    model_info = {
        "model_available": llm_interface.is_available() if llm_interface else False,
        "model_path": llm_interface.model_path if llm_interface else None,
        "embedding_model": doc_processor.embedding_model_name if doc_processor else "unknown"
    }
    
    return {
        "document_processor_ready": doc_processor is not None,
        "llm_ready": llm_interface.is_available() if llm_interface else False,
        "total_chunks": total_chunks,
        "model_info": model_info
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint for RAG queries."""
    if not rag_system:
        return {
            "answer": "RAG system not initialized. Please check the system status.",
            "sources": [],
            "confidence": 0.0,
            "query": request.message
        }
    
    try:
        # Process the query through RAG system
        result = rag_system.query(
            user_query=request.message,
            n_results=request.n_results
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "query": request.message
        }

@app.post("/process-documents")
async def process_documents(background_tasks: BackgroundTasks, documents_path: str = "../data/sample_docs"):
    """Process documents for RAG (runs in background)."""
    if not doc_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    def process_docs():
        try:
            logger.info(f"Starting document processing for: {documents_path}")
            doc_processor.process_documents(documents_path)
            logger.info("Document processing completed")
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
    
    background_tasks.add_task(process_docs)
    
    return {
        "message": "Document processing started in background",
        "documents_path": documents_path
    }

@app.get("/search")
async def search_documents(query: str, n_results: int = 5):
    """Search for similar document chunks (for testing)."""
    if not doc_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        results = doc_processor.search_similar_chunks(query, n_results)
        return results
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/reset-db")
async def reset_database():
    """Reset the ChromaDB database (for development)."""
    if not doc_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    
    try:
        # Reset the collection
        doc_processor.chroma_client.reset()
        
        # Recreate the collection
        doc_processor.collection = doc_processor.chroma_client.create_collection(
            name=doc_processor.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        return {"message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
