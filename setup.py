#!/usr/bin/env python3
"""
Setup script for RAG Chatbot
This script initializes the system and processes documents.
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.document_processor import DocumentProcessor
from backend.llm_interface import LLMInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main setup function."""
    logger.info("🚀 Starting RAG Chatbot Setup...")
    
    # Check if documents exist
    docs_path = Path("./data/sample_docs")
    if not docs_path.exists() or not list(docs_path.glob("*.pdf")) and not list(docs_path.glob("*.txt")):
        logger.warning("⚠️  No documents found in data/sample_docs/")
        logger.info("Please add PDF or TXT files to the data/sample_docs/ directory")
        return
    
    # Initialize document processor
    logger.info("📄 Initializing document processor...")
    try:
        processor = DocumentProcessor()
        logger.info("✅ Document processor initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize document processor: {e}")
        return
    
    # Process documents
    logger.info("🔄 Processing documents...")
    try:
        processor.process_documents(str(docs_path))
        logger.info("✅ Documents processed successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to process documents: {e}")
        return
    
    # Check for LLM model
    models_path = Path("./models")
    model_files = list(models_path.glob("*.gguf")) if models_path.exists() else []
    
    if model_files:
        logger.info(f"🧠 Found LLM model: {model_files[0].name}")
        try:
            llm = LLMInterface(model_path=str(model_files[0]))
            if llm.is_available():
                logger.info("✅ LLM model loaded successfully!")
            else:
                logger.warning("⚠️  LLM model failed to load")
        except Exception as e:
            logger.error(f"❌ Error loading LLM: {e}")
    else:
        logger.warning("⚠️  No LLM model found in models/ directory")
        logger.info("📥 To use local LLM:")
        logger.info("   1. Download a GGUF model from Hugging Face")
        logger.info("   2. Place it in the models/ directory")
        logger.info("   3. Restart the application")
    
    # Test search
    logger.info("🔍 Testing search functionality...")
    try:
        results = processor.search_similar_chunks("test query", n_results=3)
        if results.get("documents") and results["documents"][0]:
            logger.info(f"✅ Search working! Found {len(results['documents'][0])} results")
        else:
            logger.warning("⚠️  Search returned no results")
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
    
    logger.info("🎉 Setup completed!")
    logger.info("")
    logger.info("📝 Next steps:")
    logger.info("   1. Start the backend: cd backend && python main.py")
    logger.info("   2. Start the frontend: streamlit run frontend/streamlit_app.py")
    logger.info("   3. Open http://localhost:8501 in your browser")
    logger.info("")
    logger.info("🐳 Or use Docker:")
    logger.info("   docker-compose up --build")

if __name__ == "__main__":
    main()