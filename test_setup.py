#!/usr/bin/env python3
"""
Simple test script to verify RAG chatbot functionality
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import chromadb
        print("✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers imported successfully")
    except ImportError as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        return False
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    return True

def test_document_processor():
    """Test document processor functionality."""
    print("\n📄 Testing document processor...")
    
    try:
        from backend.document_processor import DocumentProcessor
        
        # Initialize processor
        processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        # Check if documents exist
        docs_path = Path("./data/sample_docs")
        pdf_files = list(docs_path.glob("*.pdf"))
        print(f"📁 Found {len(pdf_files)} PDF files")
        
        if pdf_files:
            print("✅ Documents ready for processing")
            
            # Test processing one document
            print("🔄 Processing documents...")
            processor.process_documents(str(docs_path))
            print("✅ Documents processed successfully!")
            
            # Test search
            print("🔍 Testing search...")
            results = processor.search_similar_chunks("test query", n_results=3)
            if results.get("documents") and results["documents"][0]:
                print(f"✅ Search working! Found {len(results['documents'][0])} results")
            else:
                print("⚠️ Search returned no results")
        else:
            print("❌ No PDF files found in data/sample_docs/")
            
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_api_components():
    """Test API components."""
    print("\n🔗 Testing API components...")
    
    try:
        from backend.main import app
        print("✅ FastAPI app created successfully")
        
        from backend.llm_interface import LLMInterface
        llm = LLMInterface()  # Without model
        print("✅ LLM interface initialized (fallback mode)")
        
    except Exception as e:
        print(f"❌ API components test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting RAG Chatbot Tests...\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing packages.")
        return
    
    # Test document processor
    if not test_document_processor():
        print("\n❌ Document processor tests failed.")
        return
    
    # Test API components
    if not test_api_components():
        print("\n❌ API component tests failed.")
        return
    
    print("\n🎉 All tests passed!")
    print("\n📝 Next steps:")
    print("1. Start the backend: python backend/main.py")
    print("2. Start the frontend: streamlit run frontend/streamlit_app.py")
    print("3. Open http://localhost:8501 in your browser")
    print("\n💡 Note: The system will work without a local LLM model,")
    print("   but responses will be basic. For better results:")
    print("   - Download a GGUF model to the models/ directory")
    print("   - Install llama-cpp-python with proper build tools")

if __name__ == "__main__":
    main()