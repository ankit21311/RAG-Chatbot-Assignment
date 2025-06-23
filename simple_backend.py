#!/usr/bin/env python3
"""
Super Simple Backend for RAG Chatbot
Uses built-in HTTP server - no external dependencies needed
"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time
from pathlib import Path

# Simple document processor
class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.processed = False
        
    def load_documents(self):
        """Load documents from PDFs."""
        docs_path = Path("data/sample_docs")
        
        if not docs_path.exists():
            print("No documents directory found")
            return False
            
        try:
            # Try to use PyPDF2 if available
            import PyPDF2
            
            for pdf_file in docs_path.glob("*.pdf"):
                try:
                    with open(pdf_file, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        
                        self.documents.append({
                            "source": pdf_file.name,
                            "content": text,
                            "chunks": self._chunk_text(text)
                        })
                        print(f"Loaded {pdf_file.name}")
                except Exception as e:
                    print(f"Error loading {pdf_file.name}: {e}")
            
            self.processed = True
            print(f"Loaded {len(self.documents)} documents")
            return True
            
        except ImportError:
            print("PyPDF2 not available. Install with: pip install PyPDF2")
            return False
    
    def _chunk_text(self, text, chunk_size=1000):
        """Split text into chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def search(self, query):
        """Simple text search."""
        if not self.processed:
            return None
        
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            for i, chunk in enumerate(doc["chunks"]):
                if query_lower in chunk.lower():
                    results.append({
                        "source": doc["source"],
                        "chunk": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                        "relevance": chunk.lower().count(query_lower)
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:5]  # Return top 5 results

# Global RAG instance
rag = SimpleRAG()

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        
        if path == '/health':
            self._send_json({"status": "healthy"})
        elif path == '/status':
            self._send_json({
                "document_processor_ready": rag.processed,
                "llm_ready": False,
                "total_chunks": sum(len(doc["chunks"]) for doc in rag.documents),
                "model_info": {"type": "simple_search"}
            })
        elif path == '/':
            self._send_json({
                "message": "Simple RAG Chatbot API",
                "version": "1.0.0",
                "status": "running"
            })
        else:
            self._send_error(404, "Not Found")
    
    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path
        
        if path == '/chat':
            self._handle_chat()
        elif path == '/process-documents':
            self._handle_process_documents()
        else:
            self._send_error(404, "Not Found")
    
    def _handle_chat(self):
        """Handle chat requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('message', '')
            
            if not rag.processed:
                response = {
                    "answer": "Documents not processed yet. Please process documents first.",
                    "sources": [],
                    "confidence": 0.0,
                    "query": query
                }
            else:
                results = rag.search(query)
                
                if results:
                    # Create response from search results
                    answer = f"Based on the documents, here's what I found about '{query}':\n\n"
                    answer += results[0]["chunk"]
                    
                    sources = [{
                        "source": r["source"],
                        "relevance_score": r["relevance"] / 10.0
                    } for r in results]
                    
                    response = {
                        "answer": answer,
                        "sources": sources,
                        "confidence": min(results[0]["relevance"] / 10.0, 1.0),
                        "query": query
                    }
                else:
                    response = {
                        "answer": "I couldn't find relevant information about your query in the documents.",
                        "sources": [],
                        "confidence": 0.0,
                        "query": query
                    }
            
            self._send_json(response)
            
        except Exception as e:
            self._send_json({
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "query": ""
            })
    
    def _handle_process_documents(self):
        """Handle document processing."""
        try:
            success = rag.load_documents()
            if success:
                self._send_json({"message": "Documents processed successfully"})
            else:
                self._send_json({"message": "Failed to process documents", "error": True})
        except Exception as e:
            self._send_json({"message": f"Error: {str(e)}", "error": True})
    
    def _send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_error(self, code, message):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

def start_server():
    """Start the HTTP server."""
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, SimpleHandler)
    
    print("🚀 Simple RAG Backend Server")
    print("=" * 30)
    print("✅ Server running on http://localhost:8000")
    print("📋 Endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /status - System status")
    print("   POST /chat - Chat with documents")
    print("   POST /process-documents - Process PDFs")
    print("\n💡 No external dependencies required!")
    print("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    start_server()