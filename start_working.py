#!/usr/bin/env python3
"""
Working RAG Chatbot Startup Script
Uses the simplified backend that actually works
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🚀 Starting Working RAG Chatbot")
    print("=" * 40)
    
    # Check documents
    docs_path = Path("data/sample_docs")
    if docs_path.exists():
        pdf_count = len(list(docs_path.glob("*.pdf")))
        print(f"📄 Found {pdf_count} PDF documents")
    else:
        print("⚠️ No documents directory found")
    
    # Start simplified backend
    print("\n🔧 Starting simplified backend...")
    try:
        backend_process = subprocess.Popen(
            [sys.executable, "backend/simple_main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        print("✅ Backend process started")
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return
    
    # Wait for backend to be ready
    print("⏳ Waiting for backend to be ready...")
    for i in range(30):  # Wait up to 30 seconds
        if check_backend_health():
            print("✅ Backend is ready!")
            break
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}/30)")
    else:
        print("❌ Backend failed to start properly")
        backend_process.terminate()
        return
    
    # Start frontend
    print("\n🎨 Starting frontend...")
    try:
        frontend_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py", 
             "--server.headless", "true", "--server.port", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        print("✅ Frontend process started")
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        backend_process.terminate()
        return
    
    print("\n🎉 RAG Chatbot is now running!")
    print("📋 Access Points:")
    print("   🌐 Web Interface: http://localhost:8501")
    print("   🔗 API Backend: http://localhost:8000")
    print("   📊 API Status: http://localhost:8000/status")
    
    print("\n📝 Next Steps:")
    print("   1. Open http://localhost:8501 in your browser")
    print("   2. You should see 'API Online' in the sidebar")
    print("   3. Click 'Process Documents' to index your PDFs")
    print("   4. Start asking questions about your documents!")
    
    print("\n💡 What's Working:")
    print("   ✅ Document processing and indexing")
    print("   ✅ Vector search and retrieval")
    print("   ✅ Simple but effective responses")
    print("   ✅ Source attribution")
    print("   ✅ Chat history")
    
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Monitor processes
        while True:
            time.sleep(2)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("\n❌ Backend process ended!")
                break
            if frontend_process.poll() is not None:
                print("\n❌ Frontend process ended!")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    
    finally:
        # Clean up
        print("🧹 Stopping services...")
        try:
            backend_process.terminate()
            frontend_process.terminate()
            time.sleep(2)
            backend_process.kill()
            frontend_process.kill()
        except:
            pass
        print("✅ All services stopped")

if __name__ == "__main__":
    main()