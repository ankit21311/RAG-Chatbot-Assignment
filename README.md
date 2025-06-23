# 🤖 RAG Chatbot with Local LLM

A **Retrieval-Augmented Generation (RAG)** chatbot that answers user queries based on custom documents using a local LLM
and ChromaDB vector store.

## 🎯 Features

- ✅ **Document Processing**: Load and index PDF/TXT files
- ✅ **Vector Search**: ChromaDB for efficient similarity search
- ✅ **Local LLM**: Run Llama 2, Mistral, or other GGUF models locally
- ✅ **FastAPI Backend**: RESTful API with comprehensive endpoints
- ✅ **Streamlit Frontend**: Beautiful web interface with chat history
- ✅ **No API Keys**: Completely local and offline inference
- ✅ **Source Attribution**: See which documents informed each answer
- ✅ **Configurable**: Adjust chunk size, retrieval count, and model parameters

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│  Document Store │
│                 │    │                 │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                        ┌─────────────────┐
                        │   Local LLM     │
                        │ (Llama/Mistral) │
                        └─────────────────┘
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd RAG-Chatbot-Assignment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Documents

Place your PDF or TXT files in the `data/sample_docs/` directory:

```
data/sample_docs/
├── document1.pdf
├── document2.pdf
├── research_paper.txt
└── manual.pdf
```

### 4. (Optional) Download a Local LLM

For better responses, download a GGUF model:

1. Visit [Hugging Face GGUF models](https://huggingface.co/models?library=gguf)
2. Download a model like `llama-2-7b-chat.Q4_K_M.gguf`
3. Place it in the `models/` directory

**Recommended models:**

- **Llama-2-7B-Chat** (TheBloke/Llama-2-7B-Chat-GGUF)
- **Mistral-7B-Instruct** (TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- **Zephyr-7B-Beta** (TheBloke/zephyr-7B-beta-GGUF)

### 5. Start the Backend

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### 6. Start the Frontend

In a new terminal:

```bash
streamlit run frontend/streamlit_app.py
```

The web interface will open at `http://localhost:8501`

### 7. Process Documents and Chat

1. Click "**Process Documents**" in the sidebar
2. Wait for indexing to complete
3. Start asking questions about your documents!

## 📁 Project Structure

```
RAG-Chatbot-Assignment/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── document_processor.py # Document loading & embedding
│   └── llm_interface.py     # LLM integration
├── frontend/
│   └── streamlit_app.py     # Streamlit web interface
├── data/
│   └── sample_docs/         # Your PDF/TXT documents
├── models/                  # Local LLM models (GGUF files)
├── chroma_db/              # ChromaDB vector database
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/chat` | POST | Send chat message |
| `/process-documents` | POST | Process documents |
| `/search` | GET | Search document chunks |

### Example API Usage

```bash
# Check status
curl http://localhost:8000/status

# Process documents
curl -X POST http://localhost:8000/process-documents

# Send chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file for custom settings:

```env
# Model settings
MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Document processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS=5

# Server settings
API_HOST=0.0.0.0
API_PORT=8000
```

### Model Parameters

Adjust in `backend/llm_interface.py`:

```python
default_params = {
    "n_ctx": 4096,        # Context window
    "n_batch": 512,       # Batch size
    "n_threads": 4,       # CPU threads
    "n_gpu_layers": 0,    # GPU layers (0 for CPU)
    "temperature": 0.7,   # Sampling temperature
}
```

## 📊 Performance Tips

### For Better Speed:

- Use quantized models (Q4_K_M or Q5_K_M)
- Increase `n_threads` for your CPU
- Enable GPU with `n_gpu_layers > 0` if available

### For Better Quality:

- Use larger models (13B or 30B if you have RAM)
- Lower temperature (0.1-0.3) for factual responses
- Increase chunk overlap for better context

### For Memory Efficiency:

- Use smaller quantized models (Q2_K or Q3_K_M)
- Reduce context window (`n_ctx`)
- Process documents in smaller batches

## 🧪 Testing

### Test Document Processing

```bash
cd backend
python document_processor.py
```

### Test LLM Interface

```bash
cd backend
python llm_interface.py
```

### Test API

```bash
# Start the server
python backend/main.py

# In another terminal
curl http://localhost:8000/health
```

## 🐛 Troubleshooting

### Common Issues

1. **"ChromaDB not found"**
   ```bash
   pip install chromadb
   ```

2. **"llama-cpp-python installation failed"**
   ```bash
   # For CPU only
   pip install llama-cpp-python
   
   # For GPU (CUDA)
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```

3. **"Model loading failed"**
    - Ensure model file is in `models/` directory
    - Check file permissions
    - Verify GGUF format

4. **"Out of memory"**
    - Use smaller quantized model
    - Reduce `n_ctx` parameter
    - Close other applications

5. **"Documents not processing"**
    - Check file formats (PDF/TXT only)
    - Verify file permissions
    - Check server logs for errors

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Notes

- This system runs locally - no data leaves your machine
- No API keys or external services required
- Documents are stored locally in ChromaDB
- LLM inference happens on your hardware

## 📈 Scaling

### For Production:

- Use PostgreSQL with pgvector instead of ChromaDB
- Implement user authentication
- Add rate limiting
- Use Redis for caching
- Deploy with Docker/Kubernetes

### For Multiple Users:

- Add user sessions
- Implement document permissions
- Use async processing for uploads
- Add monitoring and logging

## 🛠️ Advanced Usage

### Custom Embeddings

```python
from sentence_transformers import SentenceTransformer

# Use different embedding model
processor = DocumentProcessor(
    embedding_model="all-mpnet-base-v2"  # Higher quality
)
```

### Custom Chunking

```python
# Adjust chunk parameters
processor.process_documents(
    documents_path="./data/sample_docs",
    chunk_size=512,      # Smaller chunks
    chunk_overlap=100    # Less overlap
)
```

### Batch Processing

```python
# Process multiple document directories
directories = ["./data/docs1", "./data/docs2", "./data/docs3"]
for dir_path in directories:
    processor.process_documents(dir_path)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for sentence transformers
- **ChromaDB** for vector database
- **Langchain** for document processing
- **Streamlit** for the web interface
- **FastAPI** for the backend API
- **llama.cpp** for local LLM inference

## 📚 Additional Resources

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Happy Chatting! 🚀**