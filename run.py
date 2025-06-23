#!/usr/bin/env python3
"""
Run script for RAG Chatbot
Starts both backend and frontend services
"""

import os
import sys
import signal
import subprocess
import time
import logging
from pathlib import Path
import threading
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ServiceManager:
    def __init__(self):
        self.processes = []
        self.shutdown = False
    
    def start_backend(self):
        """Start the FastAPI backend."""
        logger.info("🚀 Starting FastAPI backend...")
        backend_path = Path(__file__).parent / "backend"
        
        try:
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=backend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes.append(("Backend", process))
            
            # Monitor backend output
            def monitor_backend():
                for line in iter(process.stdout.readline, ''):
                    if not self.shutdown:
                        print(f"[Backend] {line.strip()}")
                    else:
                        break
            
            threading.Thread(target=monitor_backend, daemon=True).start()
            return process
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return None
    
    def start_frontend(self):
        """Start the Streamlit frontend."""
        logger.info("🎨 Starting Streamlit frontend...")
        frontend_path = Path(__file__).parent / "frontend"
        
        try:
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                    "--server.address", "0.0.0.0",
                    "--server.port", "8501",
                    "--server.headless", "true"
                ],
                cwd=frontend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes.append(("Frontend", process))
            
            # Monitor frontend output
            def monitor_frontend():
                for line in iter(process.stdout.readline, ''):
                    if not self.shutdown:
                        print(f"[Frontend] {line.strip()}")
                    else:
                        break
            
            threading.Thread(target=monitor_frontend, daemon=True).start()
            return process
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return None
    
    def wait_for_backend(self, timeout=30):
        """Wait for backend to be ready."""
        logger.info("⏳ Waiting for backend to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Backend is ready!")
                    return True
            except:
                pass
            
            time.sleep(2)
        
        logger.error("❌ Backend failed to start within timeout")
        return False
    
    def shutdown_services(self):
        """Shutdown all services gracefully."""
        logger.info("🛑 Shutting down services...")
        self.shutdown = True
        
        for name, process in self.processes:
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    process.kill()
                    process.wait()
                
                logger.info(f"✅ {name} stopped")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        logger.info("🏁 All services stopped")
    
    def run(self):
        """Run the complete RAG chatbot system."""
        logger.info("🤖 Starting RAG Chatbot System...")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_services()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start backend
            backend_process = self.start_backend()
            if not backend_process:
                return
            
            # Wait for backend to be ready
            if not self.wait_for_backend():
                self.shutdown_services()
                return
            
            # Start frontend
            frontend_process = self.start_frontend()
            if not frontend_process:
                self.shutdown_services()
                return
            
            logger.info("🎉 RAG Chatbot is running!")
            logger.info("📋 Services:")
            logger.info("   🔗 API: http://localhost:8000")
            logger.info("   🎨 Web UI: http://localhost:8501")
            logger.info("")
            logger.info("Press Ctrl+C to stop all services")
            
            # Keep the main process alive
            while not self.shutdown:
                time.sleep(1)
                
                # Check if processes are still running
                for name, process in self.processes:
                    if process.poll() is not None:
                        logger.error(f"❌ {name} process died!")
                        self.shutdown_services()
                        return
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.shutdown_services()

def main():
    """Main entry point."""
    # Check if setup is needed
    docs_path = Path("./data/sample_docs")
    if not docs_path.exists() or not any(docs_path.glob("*.pdf")) and not any(docs_path.glob("*.txt")):
        logger.warning("⚠️  No documents found!")
        logger.info("Please add PDF or TXT files to data/sample_docs/ directory")
        logger.info("Then run: python setup.py")
        return
    
    # Check if ChromaDB exists
    chroma_path = Path("./chroma_db")
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        logger.warning("⚠️  ChromaDB not initialized!")
        logger.info("Run setup first: python setup.py")
        return
    
    # Start the system
    manager = ServiceManager()
    manager.run()

if __name__ == "__main__":
    main()