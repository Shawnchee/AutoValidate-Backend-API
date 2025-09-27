FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching (expects requirements.txt at project root)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project into image
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Use $PORT provided by App Platform
CMD ["sh", "-c", "uvicorn app.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]