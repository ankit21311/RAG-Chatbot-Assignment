@echo off
echo 🚀 Starting RAG Chatbot...
echo.

echo 📦 Installing required packages...
pip install chromadb sentence-transformers fastapi uvicorn streamlit langchain langchain-community pypdf2 python-multipart requests python-dotenv

echo.
echo 🤖 Starting FastAPI backend...
start "RAG Backend" cmd /k "cd backend && python main.py"

echo.
echo ⏳ Waiting for backend to start...
timeout /t 10 /nobreak > nul

echo.
echo 🎨 Starting Streamlit frontend...
start "RAG Frontend" cmd /k "streamlit run frontend/streamlit_app.py"

echo.
echo 🎉 RAG Chatbot is starting!
echo 📋 Services:
echo    🔗 API: http://localhost:8000
echo    🎨 Web UI: http://localhost:8501
echo.
echo ℹ️  Two new windows will open:
echo    - Backend (FastAPI server)
echo    - Frontend (Streamlit app)
echo.
echo 📝 Next steps:
echo    1. Wait for both services to start
echo    2. Open http://localhost:8501 in your browser
echo    3. Click "Process Documents" in the sidebar
echo    4. Start asking questions!
echo.
pause