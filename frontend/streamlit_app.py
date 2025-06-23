import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_available" not in st.session_state:
        st.session_state.api_available = False

def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_status():
    """Get system status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def send_chat_message(message: str, n_results: int = 5) -> Dict[str, Any]:
    """Send a chat message to the API."""
    try:
        payload = {
            "message": message,
            "n_results": n_results
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Error: {response.status_code} - {response.text}",
                "sources": [],
                "confidence": 0.0,
                "query": message
            }
    except Exception as e:
        return {
            "answer": f"Error connecting to API: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "query": message
        }

def process_documents():
    """Trigger document processing."""
    try:
        response = requests.post(f"{API_BASE_URL}/process-documents", timeout=10)
        return response.status_code == 200, response.text
    except Exception as e:
        return False, str(e)

def display_sources(sources: List[Dict[str, Any]]):
    """Display source information."""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    for i, source in enumerate(sources):
        with st.expander(f"Source {i+1}: {source.get('source', 'Unknown')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Page:** {source.get('page', 'N/A')}")
            with col2:
                st.write(f"**Relevance:** {source.get('relevance_score', 0):.2f}")

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🤖 RAG Chatbot")
    st.markdown("*Retrieval-Augmented Generation Chatbot with Local LLM*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check API health
        api_health = check_api_health()
        st.session_state.api_available = api_health
        
        if api_health:
            st.success("✅ API Online")
            
            # Get system status
            with st.spinner("Checking system status..."):
                status = get_system_status()
            
            if status:
                st.subheader("📊 System Info")
                
                # Document processor status
                if status.get("document_processor_ready"):
                    st.success("✅ Document Processor Ready")
                    st.write(f"**Total Chunks:** {status.get('total_chunks', 0)}")
                else:
                    st.error("❌ Document Processor Not Ready")
                
                # LLM status
                if status.get("llm_ready"):
                    st.success("✅ LLM Ready")
                else:
                    st.warning("⚠️ LLM Not Available (Using Fallback)")
                
                # Model info
                model_info = status.get("model_info", {})
                st.subheader("🧠 Model Info")
                st.write(f"**Embedding Model:** {model_info.get('embedding_model', 'Unknown')}")
                st.write(f"**LLM Available:** {'Yes' if model_info.get('model_available') else 'No'}")
                
                if model_info.get("model_path"):
                    st.write(f"**Model Path:** {model_info.get('model_path')}")
                
        else:
            st.error("❌ API Offline")
            st.info("Make sure the FastAPI server is running on port 8000")
        
        st.divider()
        
        # Document Management
        st.subheader("📁 Document Management")
        
        if st.button("🔄 Process Documents", disabled=not api_health):
            with st.spinner("Processing documents..."):
                success, message = process_documents()
                if success:
                    st.success("Document processing started!")
                    st.info("This may take a few minutes. Check system status above.")
                else:
                    st.error(f"Failed to start processing: {message}")
        
        st.info("Place your PDF/TXT files in the `data/sample_docs/` directory")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Chat Settings")
        n_results = st.slider("Number of relevant chunks", 1, 10, 5)
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if not api_health:
        st.error("🚨 Cannot connect to the API server. Please start the backend server first.")
        st.code("cd backend && python main.py", language="bash")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_chat_message(prompt, n_results)
            
            # Display answer
            st.markdown(response["answer"])
            
            # Display confidence if available
            if response.get("confidence", 0) > 0:
                st.caption(f"Confidence: {response['confidence']:.2f}")
            
            # Display sources
            if response.get("sources"):
                display_sources(response["sources"])
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "confidence": response.get("confidence", 0)
            })
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📝 Instructions:
    1. **Start the backend**: Run `python backend/main.py` 
    2. **Process documents**: Click "Process Documents" to index your PDFs
    3. **Ask questions**: Type your questions in the chat input
    4. **View sources**: Expand the source sections to see relevant document chunks
    
    ### 🔧 For Local LLM:
    - Download a GGUF model (e.g., Llama-2-7B-Chat) to the `models/` directory
    - Restart the backend to load the model
    """)

if __name__ == "__main__":
    main()