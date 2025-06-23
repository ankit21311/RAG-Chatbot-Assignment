# 🔧 Manual Startup Guide

## If the automated scripts don't work, follow these steps:

### **Step 1: Start Frontend Only (Working)**

```bash
python -m streamlit run frontend/streamlit_app.py --server.port 8501
```

Then open: `http://localhost:8501`

### **Step 2: The System Will Still Work!**

Even without the backend, you can:

- ✅ See the beautiful interface
- ✅ Use the document processing features
- ✅ Basic chat functionality
- ⚠️ Responses will be simple (no advanced LLM)

### **Step 3: If You Want the Full Backend (Optional)**

```bash
# In a new terminal:
cd backend
python main.py
```

## **What You Get:**

### **Frontend Only (Port 8501):**

- Beautiful Streamlit interface
- Document upload and processing
- Basic chat functionality
- System status display

### **Frontend + Backend (Ports 8501 + 8000):**

- Everything above PLUS:
- Advanced LLM responses
- Better document analysis
- Full RAG pipeline
- API endpoints

## **Troubleshooting:**

### **"This site can't be reached" on port 8000:**

- This is normal - the backend might not be running
- Use port 8501 instead: `http://localhost:8501`

### **"streamlit command not found":**

```bash
pip install streamlit
python -m streamlit run frontend/streamlit_app.py
```

### **Import errors:**

```bash
pip install chromadb sentence-transformers streamlit
```

## **Quick Success Route:**

1. **Run**: `python -m streamlit run frontend/streamlit_app.py`
2. **Open**: `http://localhost:8501`
3. **Use**: The interface will work for document processing!

**Your RAG chatbot interface is ready to use!** 🚀