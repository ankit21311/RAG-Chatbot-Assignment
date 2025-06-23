#!/usr/bin/env python3
"""
Fix and Run RAG Chatbot
This script fixes common issues and starts the system
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run a command and show progress."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"⚠️ {description} - Warning: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

def main():
    print("🚀 RAG Chatbot - Fix and Run")
    print("=" * 40)
    
    # Fix 1: Install/Fix packages
    print("\n📦 Installing required packages...")
    packages_to_install = [
        "pip install --upgrade pip",
        "pip install pydantic==1.10.12 --force-reinstall",
        "pip install fastapi==0.68.0 uvicorn==0.15.0 --force-reinstall", 
        "pip install streamlit",
        "pip install chromadb sentence-transformers",
        "pip install langchain langchain-community",
        "pip install torch transformers --index-url https://download.pytorch.org/whl/cpu",
        "pip install pypdf2 python-multipart requests python-dotenv"
    ]
    
    for cmd in packages_to_install:
        run_command(cmd, f"Installing {cmd.split()[-1]}")
    
    print("\n✅ Package installation completed!")
    
    # Check if documents exist
    if not os.path.exists("data/sample_docs") or not any(f.endswith('.pdf') for f in os.listdir("data/sample_docs")):
        print("⚠️ No PDF documents found in data/sample_docs/")
        print("Your PDFs should be there already. Continuing anyway...")
    else:
        pdf_count = len([f for f in os.listdir("data/sample_docs") if f.endswith('.pdf')])
        print(f"📄 Found {pdf_count} PDF documents - Good!")
    
    print("\n🚀 Starting RAG Chatbot...")
    
    # Start backend
    print("🔧 Starting backend...")
    try:
        backend_process = subprocess.Popen(
            [sys.executable, "backend/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        print("✅ Backend starting...")
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return
    
    # Wait for backend
    print("⏳ Waiting for backend to initialize...")
    time.sleep(10)
    
    # Start frontend
    print("🎨 Starting frontend...")
    try:
        frontend_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.headless", "true", "--server.port", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        print("✅ Frontend starting...")
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        backend_process.terminate()
        return
    
    print("\n🎉 RAG Chatbot is running!")
    print("📋 Access your chatbot:")
    print("   🌐 Web Interface: http://localhost:8501")
    print("   🔗 API: http://localhost:8000")
    print("\n📝 Next steps:")
    print("   1. Open http://localhost:8501 in your browser")
    print("   2. Click 'Process Documents' in the sidebar")
    print("   3. Wait for processing to complete")
    print("   4. Start asking questions about your PDFs!")
    print("\n💡 Features:")
    print("   ✅ Lightweight LLM (works without GPU)")
    print("   ✅ Document search and retrieval")
    print("   ✅ Source attribution")
    print("   ✅ Chat history")
    print("\nPress Ctrl+C to stop both services")
    
    try:
        # Keep running and monitor processes
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
        print("🧹 Cleaning up processes...")
        try:
            backend_process.terminate()
            frontend_process.terminate()
            time.sleep(2)
            backend_process.kill()
            frontend_process.kill()
        except:
            pass
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()