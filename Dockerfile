# ----------------------------------------------------
# Stage 1: Build Frontend (React + TypeScript + Vite)
# ----------------------------------------------------
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ----------------------------------------------------
# Stage 2: Python Backend & Production Runtime
# ----------------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python backend dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code, data, and trained model artifacts
COPY backend ./backend

# Copy built static frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure non-root permissions for Hugging Face Spaces compatibility
RUN chmod -R 777 /app

# Default port: 7860 (Hugging Face default) or Render $PORT override
ENV PORT=7860
EXPOSE 7860

# Launch unified FastAPI server serving both API and frontend
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
