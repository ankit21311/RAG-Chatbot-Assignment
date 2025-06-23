# RAG Chatbot Startup Script for PowerShell
Write-Host "🚀 Starting RAG Chatbot..." -ForegroundColor Green
Write-Host ""

Write-Host "📦 Installing required packages..." -ForegroundColor Yellow
pip install chromadb sentence-transformers fastapi uvicorn streamlit langchain langchain-community pypdf2 python-multipart requests python-dotenv

Write-Host ""
Write-Host "🤖 Starting FastAPI backend..." -ForegroundColor Blue
Start-Process PowerShell -ArgumentList "-NoExit", "-Command", "cd backend; python main.py"

Write-Host ""
Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "🎨 Starting Streamlit frontend..." -ForegroundColor Magenta
Start-Process PowerShell -ArgumentList "-NoExit", "-Command", "streamlit run frontend/streamlit_app.py"

Write-Host ""
Write-Host "🎉 RAG Chatbot is starting!" -ForegroundColor Green
Write-Host "📋 Services:" -ForegroundColor White
Write-Host "   🔗 API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "   🎨 Web UI: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "ℹ️  Two new PowerShell windows will open:" -ForegroundColor White
Write-Host "   - Backend (FastAPI server)" -ForegroundColor White
Write-Host "   - Frontend (Streamlit app)" -ForegroundColor White
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Wait for both services to start" -ForegroundColor White
Write-Host "   2. Open http://localhost:8501 in your browser" -ForegroundColor White
Write-Host "   3. Click 'Process Documents' in the sidebar" -ForegroundColor White
Write-Host "   4. Start asking questions!" -ForegroundColor White
Write-Host ""

# Wait for user input
Read-Host "Press Enter to continue..."