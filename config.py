"""
Configuration file for RAG Chatbot
Centralized configuration for all components
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Main configuration class."""
    
    # Directory paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data" / "sample_docs"
    MODELS_DIR = BASE_DIR / "models"
    CHROMA_DB_PATH = BASE_DIR / "chroma_db"
    
    # Document processing settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 5))
    
    # Embedding model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # LLM settings
    MODEL_PATH = os.getenv("MODEL_PATH", None)
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 512))
    LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", 4096))
    LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", 512))
    LLM_THREADS = int(os.getenv("LLM_THREADS", 4))
    LLM_GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", 0))
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # Streamlit settings
    STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
    
    # ChromaDB settings
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "document_embeddings")
    CHROMA_DISTANCE_METRIC = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_model_path(cls):
        """Get the first available model path."""
        if cls.MODEL_PATH and Path(cls.MODEL_PATH).exists():
            return cls.MODEL_PATH
        
        # Look for GGUF files in models directory
        if cls.MODELS_DIR.exists():
            for model_file in cls.MODELS_DIR.glob("*.gguf"):
                return str(model_file)
        
        return None
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.CHROMA_DB_PATH
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_llm_params(cls):
        """Get LLM initialization parameters."""
        return {
            "n_ctx": cls.LLM_CONTEXT_WINDOW,
            "n_batch": cls.LLM_BATCH_SIZE,
            "n_threads": cls.LLM_THREADS,
            "n_gpu_layers": cls.LLM_GPU_LAYERS,
            "verbose": False
        }
    
    @classmethod
    def get_document_processing_params(cls):
        """Get document processing parameters."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("🔧 RAG Chatbot Configuration:")
        print(f"   📁 Data Directory: {cls.DATA_DIR}")
        print(f"   🧠 Models Directory: {cls.MODELS_DIR}")
        print(f"   🗄️  ChromaDB Path: {cls.CHROMA_DB_PATH}")
        print(f"   📄 Chunk Size: {cls.CHUNK_SIZE}")
        print(f"   🔗 Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"   🤖 Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"   🌡️  LLM Temperature: {cls.LLM_TEMPERATURE}")
        print(f"   📊 Max Tokens: {cls.LLM_MAX_TOKENS}")
        print(f"   🖥️  API Host: {cls.API_HOST}:{cls.API_PORT}")
        print(f"   🎨 Streamlit Host: {cls.STREAMLIT_HOST}:{cls.STREAMLIT_PORT}")
        
        model_path = cls.get_model_path()
        if model_path:
            print(f"   🧠 LLM Model: {Path(model_path).name}")
        else:
            print("   ⚠️  No LLM model found")

# Create default .env file template
ENV_TEMPLATE = """
# RAG Chatbot Configuration
# Copy this to .env and customize as needed

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS=5

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM Settings
# MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=512
LLM_CONTEXT_WINDOW=4096
LLM_BATCH_SIZE=512
LLM_THREADS=4
LLM_GPU_LAYERS=0

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Streamlit Settings
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501

# ChromaDB Settings
CHROMA_COLLECTION_NAME=document_embeddings
CHROMA_DISTANCE_METRIC=cosine

# Logging
LOG_LEVEL=INFO
""".strip()

def create_env_template():
    """Create a .env template file if it doesn't exist."""
    env_file = Path(".env.template")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(ENV_TEMPLATE)
        print(f"📝 Created {env_file}")
        print("   Copy this to .env and customize as needed")

if __name__ == "__main__":
    # Create directories and config template
    Config.ensure_directories()
    create_env_template()
    Config.print_config()