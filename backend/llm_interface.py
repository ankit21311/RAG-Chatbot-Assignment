import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python not available. Install it for local LLM support.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the LLM interface.
        
        Args:
            model_path: Path to the GGUF model file
            **kwargs: Additional arguments for Llama initialization
        """
        self.model_path = model_path
        self.llm = None
        
        # Default model parameters
        self.default_params = {
            "n_ctx": 4096,  # Context window
            "n_batch": 512,  # Batch size
            "n_threads": 4,  # Number of threads
            "n_gpu_layers": 0,  # GPU layers (0 for CPU-only)
            "verbose": False
        }
        
        # Update with provided kwargs
        self.default_params.update(kwargs)
        
        if model_path and LLAMA_CPP_AVAILABLE:
            self._load_model()
        elif not LLAMA_CPP_AVAILABLE:
            logger.warning("Using fallback text generation. Install llama-cpp-python for better results.")
    
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
        2. Search for models like 'llama-2-7b-chat' or 'mistral-7b'
        3. Download a .gguf file (Q4_K_M quantization is recommended)
        4. Place it in the 'models' directory as: {model_path}
        5. Restart the application
        
        Popular models:
        - TheBloke/Llama-2-7B-Chat-GGUF
        - TheBloke/Mistral-7B-Instruct-v0.2-GGUF
        - microsoft/DialoGPT-medium (for smaller systems)
        """)
        
        return str(model_path)
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated response
        """
        if self.llm is None:
            return self._fallback_response(prompt)
        
        try:
            # Generate response
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["Human:", "User:", "\n\n"],
                echo=False
            )
            
            generated_text = response["choices"][0]["text"].strip()
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """
        Fallback response when LLM is not available.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Fallback response
        """
        return """I apologize, but the local LLM is not currently available. 
        
To enable local LLM functionality:
1. Download a GGUF model file (e.g., Llama-2-7B-Chat)
2. Place it in the 'models' directory
3. Update the model path in the configuration
4. Restart the application

For now, I can see your query but cannot provide a detailed response based on the documents."""
    
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
        
        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Create RAG prompt
        rag_prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. If the answer is not in the context, say so.

Context:
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
        # Create RAG prompt
        prompt = self.create_rag_prompt(query, context_chunks)
        
        # Generate response
        response = self.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["Question:", "Context:", "\n\nQuestion:", "\n\nContext:"]
        )
        
        return response
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.llm is not None

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
                    "confidence": 0.0
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
    llm = LLMInterface()
    
    # Test basic generation
    response = llm.generate_response("Hello, how are you?")
    print("Response:", response)
    
    # Test RAG prompt creation
    prompt = llm.create_rag_prompt(
        "What is machine learning?",
        ["Machine learning is a subset of AI...", "ML algorithms learn from data..."]
    )
    print("RAG Prompt:", prompt)