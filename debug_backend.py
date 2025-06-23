#!/usr/bin/env python3
"""
Debug Backend - Test components individually
"""

import sys
import traceback
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_imports():
    """Test all imports."""
    print("🔍 Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported")
    except Exception as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported")
    except Exception as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB imported")
    except Exception as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers imported")
    except Exception as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        return False
    
    try:
        import torch
        import transformers
        print("✅ PyTorch and Transformers imported")
    except Exception as e:
        print(f"❌ PyTorch/Transformers import failed: {e}")
        return False
    
    return True

def test_backend_modules():
    """Test backend module imports."""
    print("\n🔍 Testing backend modules...")
    
    try:
        from backend.document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported")
    except Exception as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from backend.lightweight_llm import LightweightLLM
        print("✅ LightweightLLM imported")
    except Exception as e:
        print(f"❌ LightweightLLM import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from backend.llm_interface import LLMInterface
        print("✅ LLMInterface imported")
    except Exception as e:
        print(f"❌ LLMInterface import failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_lightweight_llm():
    """Test lightweight LLM loading."""
    print("\n🧠 Testing lightweight LLM...")
    
    try:
        from backend.lightweight_llm import LightweightLLM
        
        print("🔄 Loading DistilGPT-2...")
        llm = LightweightLLM("distilgpt2")
        
        if llm.is_available():
            print("✅ LightweightLLM loaded successfully!")
            
            # Test generation
            print("🔄 Testing text generation...")
            response = llm.generate_response("The benefits of AI include", max_length=50)
            print(f"✅ Generated: {response[:100]}...")
            
            return True
        else:
            print("❌ LightweightLLM failed to load")
            return False
            
    except Exception as e:
        print(f"❌ LightweightLLM test failed: {e}")
        traceback.print_exc()
        return False

def test_fastapi_app():
    """Test FastAPI app creation."""
    print("\n🔗 Testing FastAPI app...")
    
    try:
        from backend.main import app
        print("✅ FastAPI app created successfully")
        
        # Test a simple endpoint
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        print("🔄 Testing health endpoint...")
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ FastAPI app test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔧 Backend Debug Test")
    print("=" * 30)
    
    success = True
    
    # Test basic imports
    if not test_imports():
        success = False
    
    # Test backend modules
    if not test_backend_modules():
        success = False
    
    # Test lightweight LLM
    if not test_lightweight_llm():
        success = False
    
    # Test FastAPI app
    if not test_fastapi_app():
        success = False
    
    print("\n" + "=" * 30)
    if success:
        print("🎉 All tests passed! Backend should work.")
        print("\n💡 Try running the backend manually:")
        print("   python backend/main.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\n🔧 Common fixes:")
        print("   pip install --upgrade torch transformers")
        print("   pip install --upgrade fastapi uvicorn pydantic==1.10.12")

if __name__ == "__main__":
    main()