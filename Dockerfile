FROM python:3.12-slim

WORKDIR /app

# Install minimal build deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies with fallback for cache mount
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Verify critical imports work
RUN python -c "import numpy; print(f'NumPy {numpy.__version__} OK')" && \
    python -c "import torch; print(f'PyTorch {torch.__version__} OK')" && \
    python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers OK')"

# Copy only necessary project files
COPY app/ .

# Set Python path to app so imports work correctly
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Update uvicorn path to match new structure
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
