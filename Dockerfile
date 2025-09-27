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

# Use BuildKit cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy only necessary project files
COPY app/ ./app/

# Set Python path to app/api so imports work correctly
ENV PYTHONPATH=/app/app/api
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Update uvicorn path to match new structure
CMD ["sh", "-c", "uvicorn app.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]