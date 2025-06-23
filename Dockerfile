FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY models/ ./models/
COPY README.md .

# Create necessary directories
RUN mkdir -p chroma_db

# Expose ports
EXPOSE 8000 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start FastAPI server in background\n\
cd /app/backend && python main.py &\n\
\n\
# Wait a bit for the API to start\n\
sleep 5\n\
\n\
# Start Streamlit frontend\n\
streamlit run /app/frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set the default command
CMD ["/app/start.sh"]