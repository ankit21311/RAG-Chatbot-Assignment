import os
import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightLLM:
    """Lightweight LLM using small transformer models."""
    
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize with a lightweight model.
        
        Available models (from smallest to larger):
        - "distilgpt2" (82M parameters) - Fastest, most lightweight
        - "gpt2" (124M parameters) - Small and fast
        - "microsoft/DialoGPT-small" (117M parameters) - Good for conversation
        - "microsoft/DialoGPT-medium" (345M parameters) - Better quality
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the lightweight model."""
        try:
            logger.info(f"Loading lightweight model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Create pipeline for easier text generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"✅ Lightweight model loaded successfully!")
            logger.info(f"Model parameters: ~{self._get_model_size()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def _get_model_size(self):
        """Get approximate model size."""
        if self.model is None:
            return "Unknown"
        
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count > 1e9:
            return f"{param_count/1e9:.1f}B parameters"
        elif param_count > 1e6:
            return f"{param_count/1e6:.0f}M parameters"
        else:
            return f"{param_count/1e3:.0f}K parameters"
    
    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.pipeline is not None
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 200,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """Generate a response using the lightweight model."""
        if not self.is_available():
            return "Lightweight model not available. Please check the logs for errors."
        
        try:
            # Clean and prepare the prompt
            clean_prompt = prompt.strip()
            
            # Generate response
            with torch.no_grad():
                outputs = self.pipeline(
                    clean_prompt,
                    max_length=min(max_length, 512),  # Keep responses reasonable
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    return_full_text=False  # Only return generated text
                )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Clean up the response
            response = self._clean_response(generated_text, clean_prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I couldn't generate a response: {str(e)}"
    
    def _clean_response(self, generated_text: str, original_prompt: str) -> str:
        """Clean and format the generated response."""
        # Remove the original prompt if it's repeated
        if generated_text.startswith(original_prompt):
            generated_text = generated_text[len(original_prompt):].strip()
        
        # Split by common separators and take the first complete sentence/paragraph
        sentences = generated_text.split('.')
        if len(sentences) > 1:
            # Take first 2-3 sentences for a complete response
            clean_response = '. '.join(sentences[:3]).strip()
            if clean_response and not clean_response.endswith('.'):
                clean_response += '.'
        else:
            clean_response = generated_text.strip()
        
        # Remove any incomplete sentences at the end
        if len(clean_response) > 200:
            clean_response = clean_response[:200].rsplit('.', 1)[0] + '.'
        
        return clean_response if clean_response else "I need more context to provide a helpful response."
    
    def create_rag_prompt(self, query: str, context_chunks: List[str], max_context_length: int = 800) -> str:
        """Create a RAG prompt optimized for lightweight models."""
        # Combine and truncate context
        context = "\n\n".join(context_chunks)
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Create a simple, clear prompt
        rag_prompt = f"""Based on the following information, please answer the question.

Information:
{context}

Question: {query}
Answer:"""
        
        return rag_prompt
    
    def answer_query(self, query: str, context_chunks: List[str], max_length: int = 150) -> str:
        """Generate an answer using RAG approach with the lightweight model."""
        # Create prompt
        prompt = self.create_rag_prompt(query, context_chunks)
        
        # Generate response with conservative settings for better quality
        response = self.generate_response(
            prompt=prompt,
            max_length=max_length,
            temperature=0.3,  # Lower temperature for more focused answers
            top_p=0.8,
            top_k=40
        )
        
        return response

# Model recommendations by system capability
MODEL_RECOMMENDATIONS = {
    "fastest": "distilgpt2",           # 82M params - Very fast, basic responses
    "balanced": "gpt2",                # 124M params - Good balance of speed/quality
    "conversational": "microsoft/DialoGPT-small",  # 117M params - Better for chat
    "quality": "microsoft/DialoGPT-medium"         # 345M params - Best quality, slower
}

def get_recommended_model(preference: str = "fastest") -> str:
    """Get recommended model based on preference."""
    return MODEL_RECOMMENDATIONS.get(preference, "distilgpt2")

if __name__ == "__main__":
    # Test the lightweight LLM
    print("🧪 Testing Lightweight LLM...")
    
    # Test with the fastest model
    llm = LightweightLLM("distilgpt2")
    
    if llm.is_available():
        print("✅ Model loaded successfully!")
        
        # Test basic generation
        response = llm.generate_response("The benefits of artificial intelligence include")
        print(f"Basic generation: {response}")
        
        # Test RAG-style generation
        context = ["Artificial intelligence helps automate tasks and improve efficiency."]
        rag_response = llm.answer_query("What are the benefits of AI?", context)
        print(f"RAG response: {rag_response}")
        
    else:
        print("❌ Model failed to load")
    
    print("\n💡 Model Recommendations:")
    for pref, model in MODEL_RECOMMENDATIONS.items():
        print(f"   {pref}: {model}")