import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Try to import lightweight LLM first
try:
    from lightweight_llm import LightweightLLM
    LIGHTWEIGHT_LLM_AVAILABLE = True
except ImportError:
    LIGHTWEIGHT_LLM_AVAILABLE = False

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python not available. Install it for local LLM support.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, model_path: Optional[str] = None, use_lightweight: bool = True, **kwargs):
        """
        Initialize the LLM interface with multiple options.
        
        Args:
            model_path: Path to GGUF model file (for llama-cpp-python)
            use_lightweight: Whether to use lightweight transformers model
            **kwargs: Additional arguments for Llama initialization
        """
        self.model_path = model_path
        self.use_lightweight = use_lightweight
        self.llm = None
        self.lightweight_llm = None
        
        # Default model parameters for llama-cpp-python
        self.default_params = {
            "n_ctx": 4096,
            "n_batch": 512,
            "n_threads": 4,
            "n_gpu_layers": 0,
            "verbose": False
        }
        self.default_params.update(kwargs)
        
        # Try to initialize the best available option
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the best available LLM option."""
        
        # Option 1: Try lightweight transformers model (recommended)
        if self.use_lightweight and LIGHTWEIGHT_LLM_AVAILABLE:
            try:
                logger.info("Initializing lightweight transformers model...")
                self.lightweight_llm = LightweightLLM("distilgpt2")  # Start with fastest
                
                if self.lightweight_llm.is_available():
                    logger.info("✅ Lightweight model loaded successfully!")
                    return
                else:
                    logger.warning("Lightweight model failed to load")
                    self.lightweight_llm = None
                    
            except Exception as e:
                logger.error(f"Failed to load lightweight model: {e}")
                self.lightweight_llm = None
        
        # Option 2: Try llama-cpp-python if model file exists
        if self.model_path and LLAMA_CPP_AVAILABLE:
            try:
                self._load_model()
                if self.llm:
                    return
            except Exception as e:
                logger.error(f"Failed to load llama-cpp model: {e}")
        
        # If no model loaded, provide informative message
        if not self.is_available():
            logger.warning("No LLM model loaded. Using fallback responses.")
            self._log_setup_instructions()
    
    def _load_model(self):
        """Load the LLM model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading LLM model from: {self.model_path}")
            self.llm = Llama(model_path=self.model_path, **self.default_params)
            logger.info("LLM model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.llm = None
    
    def download_model(self, model_name: str = "llama-2-7b-chat.Q4_K_M.gguf") -> str:
        """
        Download a model from Hugging Face.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Path to the downloaded model
        """
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / model_name
        
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return str(model_path)
        
        # For demo purposes, we'll provide instructions instead of auto-downloading
        logger.info(f"""
        To use a local LLM, please download a GGUF model file manually:
        
        1. Visit: https://huggingface.co/models?library=gguf
        2. Search for models like 'llama-2-7B-Chat' or 'mistral-7B'
        3. Download a .gguf file (Q4_K_M quantization is recommended)
        4. Place it in the 'models' directory as: {model_path}
        5. Restart the application
        
        Popular models:
        - TheBloke/Llama-2-7B-Chat-GGUF
        - TheBloke/Mistral-7B-Instruct-v0.2-GGUF
        - microsoft/DialoGPT-medium (for smaller systems)
        """)
        
        return str(model_path)
    
    def _log_setup_instructions(self):
        """Log setup instructions for users."""
        logger.info("💡 To enable better LLM responses:")
        logger.info("   Option 1 (Lightweight): pip install torch transformers")
        logger.info("   Option 2 (Advanced): Download GGUF model + pip install llama-cpp-python")
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response using the best available LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated response
        """
        
        # Use lightweight model if available
        if self.lightweight_llm and self.lightweight_llm.is_available():
            try:
                return self.lightweight_llm.generate_response(
                    prompt=prompt,
                    max_length=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            except Exception as e:
                logger.error(f"Lightweight model error: {e}")
        
        # Fall back to llama-cpp-python if available
        if self.llm:
            try:
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop or ["Human:", "User:", "\n\n"],
                    echo=False
                )
                return response["choices"][0]["text"].strip()
            except Exception as e:
                logger.error(f"Llama-cpp model error: {e}")
        
        # Final fallback
        return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """
        Fallback response when LLM is not available.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Fallback response
        """
        return """I can see your query, but I need a language model to provide detailed responses.

To enable better responses:

🚀 **Lightweight Option (Recommended):**
```bash
pip install torch transformers
```
Then restart the backend.

🔧 **Advanced Option:**
1. Download a GGUF model (e.g., Llama-2-7B-Chat)
2. Place it in the 'models' directory
3. Install: pip install llama-cpp-python
4. Restart the backend

The lightweight option works great for most use cases and is much faster to set up!"""
    
    def create_rag_prompt(self, query: str, context_chunks: List[str], max_context_length: int = 2000) -> str:
        """
        Create a RAG prompt combining query and context.
        
        Args:
            query: User query
            context_chunks: List of relevant document chunks
            max_context_length: Maximum length of context to include
            
        Returns:
            Formatted RAG prompt
        """
        # Combine context chunks
        context = "\n\n".join(context_chunks)
        
        # Adjust context length based on model type
        if self.lightweight_llm:
            max_context_length = 800  # Shorter for lightweight models
        
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Simple, effective prompt
        rag_prompt = f"""Based on the following information, answer the question accurately.

Information:
{context}

Question: {query}

Answer:"""
        
        return rag_prompt
    
    def answer_query(
        self, 
        query: str, 
        context_chunks: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3
    ) -> str:
        """
        Generate an answer for a query using RAG approach.
        
        Args:
            query: User query
            context_chunks: Relevant document chunks
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower for more focused answers)
            
        Returns:
            Generated answer
        """
        
        # Use lightweight model's RAG method if available
        if self.lightweight_llm and self.lightweight_llm.is_available():
            try:
                return self.lightweight_llm.answer_query(query, context_chunks, max_tokens)
            except Exception as e:
                logger.error(f"Lightweight RAG error: {e}")
        
        # Fall back to standard RAG prompt
        prompt = self.create_rag_prompt(query, context_chunks)
        response = self.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["Question:", "Information:", "\n\nQuestion:", "\n\nInformation:"]
        )
        
        return response
    
    def is_available(self) -> bool:
        """Check if any LLM is available."""
        return (
            (self.lightweight_llm and self.lightweight_llm.is_available()) or
            (self.llm is not None)
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.lightweight_llm and self.lightweight_llm.is_available():
            return {
                "type": "lightweight_transformers",
                "model_name": self.lightweight_llm.model_name,
                "device": self.lightweight_llm.device,
                "available": True
            }
        elif self.llm:
            return {
                "type": "llama_cpp",
                "model_path": self.model_path,
                "available": True
            }
        else:
            return {
                "type": "fallback",
                "available": False
            }

class RAGSystem:
    def __init__(self, document_processor, llm_interface):
        """
        Initialize the complete RAG system.
        
        Args:
            document_processor: DocumentProcessor instance
            llm_interface: LLMInterface instance
        """
        self.document_processor = document_processor
        self.llm_interface = llm_interface
    
    def query(self, user_query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            user_query: User's question
            n_results: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Retrieve relevant chunks
            search_results = self.document_processor.search_similar_chunks(
                user_query, n_results=n_results
            )
            
            # Extract text chunks
            context_chunks = search_results.get("documents", [[]])[0]
            metadatas = search_results.get("metadatas", [[]])[0]
            distances = search_results.get("distances", [[]])[0]
            
            if not context_chunks:
                return {
                    "answer": "I couldn't find relevant information in the documents to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "query": user_query
                }
            
            # Generate answer using LLM
            answer = self.llm_interface.answer_query(user_query, context_chunks)
            
            # Prepare response
            sources = []
            for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
                sources.append({
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", 0),
                    "chunk_index": metadata.get("chunk_index", i),
                    "relevance_score": 1 - distance  # Convert distance to similarity
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": max(1 - min(distances), 0.0) if distances else 0.0,
                "query": user_query
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "query": user_query
            }

if __name__ == "__main__":
    # Example usage
    llm = LLMInterface(use_lightweight=True)
    
    # Test basic generation
    response = llm.generate_response("Hello, how are you?")
    print("Response:", response)
    
    # Test RAG prompt creation
    prompt = llm.create_rag_prompt(
        "What is machine learning?",
        ["Machine learning is a subset of AI...", "ML algorithms learn from data..."]
    )
    print("RAG Prompt:", prompt)
