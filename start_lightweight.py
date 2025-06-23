#!/usr/bin/env python3
"""
Quick start script for RAG Chatbot with Lightweight LLM
"""

import subprocess
import sys
import os
from pathlib import Path

def install_packages():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    packages = [
        "torch",
        "transformers", 
        "chromadb",
        "sentence-transformers",
        "fastapi==0.68.0",
        "uvicorn==0.15.0", 
        "pydantic==1.10.12",
        "streamlit",
        "langchain",
        "langchain-community",
        "pypdf2",
        "python-multipart",
        "requests",
        "python-dotenv"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}")

def start_backend():
    """Start the FastAPI backend."""
    print("\n🚀 Starting backend...")
    backend_path = Path("backend")
    
    if not backend_path.exists():
        print("❌ Backend directory not found!")
        return None
    
    try:
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=backend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend."""
    print("🎨 Starting frontend...")
    
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return process
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main startup function."""
    print("🤖 RAG Chatbot with Lightweight LLM")
    print("=" * 50)
    
    # Check if documents exist
    docs_path = Path("data/sample_docs")
    if not docs_path.exists() or not any(docs_path.glob("*.pdf")):
        print("⚠️ No PDF documents found in data/sample_docs/")
        print("Please add your PDF files to the data/sample_docs/ directory")
        return
    
    pdf_count = len(list(docs_path.glob("*.pdf")))
    print(f"📄 Found {pdf_count} PDF documents")
    
    # Install packages
    install_packages()
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return
    
    print("⏳ Waiting for backend to start...")
    import time
    time.sleep(8)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        backend_process.terminate()
        return
    
    print("\n🎉 RAG Chatbot is starting!")
    print("📋 Services:")
    print("   🔗 API: http://localhost:8000")
    print("   🎨 Web UI: http://localhost:8501")
    print("\n📝 Next steps:")
    print("   1. Open http://localhost:8501 in your browser")
    print("   2. Click 'Process Documents' in the sidebar")
    print("   3. Wait for processing to complete")
    print("   4. Start asking questions!")
    print("\n💡 Features:")
    print("   ✅ Lightweight LLM (DistilGPT-2)")
    print("   ✅ Fast startup and responses")
    print("   ✅ Works on CPU (no GPU needed)")
    print("   ✅ Document search and RAG")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped!")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped!")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        
    finally:
        # Clean up processes
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()