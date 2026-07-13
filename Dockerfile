FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example .env

WORKDIR /app/backend

EXPOSE 8000 8501

# Start both backend and frontend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run ../frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
