# 🚗 Developer-First Smart Vehicle Data Validation SDK & API (AutoValidate)

## Other Related Repository Links
- [Demo Frontend](https://github.com/Seann004/AutoValidate-Frontend)
- [SDK Repo](https://github.com/Seann004/Auto-Input-Validator-SDK)
- [Weekly Schedule Training Airflow](https://github.com/Shawnchee/AutoValidate-Training-Airflow)
- [Developer Documentation - Work in Progress v0](https://sss-45.gitbook.io/sss-docs/)

## 👥 Team Members  
- 👨‍💻Shawn Chee (Final-Year @ UM)
- 👨‍💻Sean Sean (3rd-Year @ UM)

---
## Track & Problem Statement  

**Track:** Industry Collaboration  
**Problem Statement:** Smart Vehicle Data Validation & Error Detection by BJAK  

**Description:**  
When buying or renewing car insurance online, users often mistype or enter incorrect vehicle details (e.g., plate number, car model, year of manufacture).  

**The Problem:**  
These mistakes can:  
- Delay policy approval  
- Cause pricing errors  
- Lead to invalid insurance coverage  

**What the Solution Should Solve:**  
Build a smart system that **detects and corrects typos or inaccurate vehicle input specifications in real time** – ensuring smoother, faster, and more reliable insurance applications.  

---
## 🔧 Project Overview  
Smart Vehicle Data Validation is a two-layer system designed to **detect and correct typos or inaccurate vehicle details in real time** during online insurance applications.  

### 🔹 The Problem  
When users mistype or enter incorrect details (plate number, car model, year):  
- 🚨 Policy approvals get delayed  
- 💸 Pricing errors occur  
- ❌ Invalid insurance coverage risks arise  

### 🔹 The Solution  
Our system combines **instant SDK field validation** (frontend) with a **multi-stage backend API** for correction, ensuring smoother, faster, and more reliable insurance applications.  

### Developer-First Design 🧑‍💻
AutoValidate is built for plug-and-play integration, allowing developers to leverage a complex validation pipeline without deep expertise.
- Lightweight SDK: A standalone JS/TS module for instant client-side validation.
- API Integration: Standard HTTP requests return clean JSON, making it framework-agnostic.

---
## FastAPI Implementation for Semantic Search, OCR, and Auto-Correction (This is the API REPO)

## Introduction

This directory contains the FastAPI backend for the AutoValidate vehicle-insurance system. It implements high-performance input validation, VLM OCR-based document extraction, and typo-correction pipelines. The service uses SentenceTransformers embeddings, a Qdrant vector store, and Redis Cache for persistence and Supabase for saving training_dataset for finetuning purposes. Models are versioned on Hugging Face Hub and can be retrained and redeployed automatically.

## Summary of Capabilities
```bash
1. Frequent-typo cache lookup for instant corrections  
2. Fast fuzzy matching as a first-pass filter  
3. Semantic search based on pretrained and fine-tuned embeddings  
4. Weekly scheduled fine-tuning of the embedding model 
5. Optional Vehicle Ownership Certificate (VOC) OCR upload for deterministic extraction
```
This implementation serves as the core validation layer for insurance applications, providing endpoints for brand/model validation, VLM OCR document processing, and typo correction.

---

## **Table of Contents**

- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [API Components](#api-components)
- [Endpoints Reference](#endpoints-reference)
- [Vector Search Pipeline](#vector-search-pipeline)
- [Deployment Configuration](#deployment-configuration)
- [Environment Variables](#environment-variables)
- [Performance Characteristics](#performance-characteristics)

---

## **Architecture**

This FastAPI backend implements a layered architecture:

- **API Layer**: FastAPI endpoints with request/response validation schemas
- **Core Service Layer**: Embedding generation, search functionality, typo detection
- **Data Access Layer**: Integration with Qdrant, Supabase and Redis Cache
- **OCR Module**: Document processing with vision model integration (Gemini 2.5 Flash)

---

## **Technical Stack**

### Core Technologies
- **FastAPI**: Asynchronous Python web framework with automatic OpenAPI doc generation
- **SentenceTransformers**: Deep learning framework for embedding generation (in this case is intfloat/multilingual-e5-small, performs best among other lightweight embedding model)
- **Qdrant**: Vector database with hybrid filtering for efficient semantic search
- **Supabase**: PostgreSQL backend with REST API for session management, used to store training_dataset for weekly scheduled finetuning with Airflow
- **HuggingFace Hub**: Model hosting and versioning for embedding models
- **Google Gemini Vision API**: Document OCR processing for VOC extraction
- **Redis**: In-memory caching for high-throughput validation responses (Frequent Typo Lookup)
- **Docker**: Container-based deployment with multi-stage build optimization (For deployment purposes)
- **Airflow**: Scheduling finetune / retrain embedding model based on frequent typos for better performance

### Dependencies
- Python 3.12+
- PyTorch 2.1+
- SentenceTransformers 2.2.2+
- FastAPI 0.103.1+
- Uvicorn (ASGI server)
- Qdrant Client 1.6.0+
- Supabase Client 1.0.3+

---

## **API Components**

### Core Modules
- **app/api/app.py**: Main FastAPI application with endpoint definitions
- **app/api/core/**: Core business logic implementation
  - **embedding.py**: Embedding model loading and vector generation
  - **search.py**: Hybrid search implementation (vector + fuzzy)
  - **ingestion.py**: Data ingestion pipeline for vector DB
  - **db_lookup.py**: Database operations for typo correction
- **app/api/services/**: Service integrations
  - **config.py**: Configuration and environment variables
  - **qdrant.py**: Qdrant client initialization and search methods
  - **redis.py**: Redis cache implementation
- **app/api/ocr/**: OCR processing module (VLM)
  - **main.py**: VOC extraction implementation
- **app/models/**: Data models and schemas
  - **schemas.py**: Pydantic models for request/response validation

### Concurrency Model
- **Thread Safety**: Load-time model initialization with locks to prevent race conditions
- **Asynchronous Endpoints**: Non-blocking IO for database and vector operations
- **Background Tasks**: Asynchronous processing for non-critical operations
- **Connection Pooling**: Optimized database connections

---

## **Endpoints Reference**

### Input Validation

#### `POST /search`
Main semantic search endpoint for input validation.

**Request Body Schema**:
```python
class SearchRequest(BaseModel):
    query: str
    domain: DomainType  # Enum: brand, model
    fuzzy_threshold: Optional[float] = 0.75
    max_results: Optional[int] = 3
    session_id: Optional[str] = None
```

**Response Schema**:
```python
class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    domain: DomainType
    voc_result: Optional[Dict[str, Any]] = None
```

### Document Processing

#### `POST /upload-voc`
OCR processing endpoint for Vehicle Ownership Certificates.

**Request Body**: Multipart form with file upload and optional session ID.

**Response Schema**:
```python
class UploadVOCResponse(BaseModel):
    status: str
    message: str
    session_id: str
    car_brand: Optional[str] = None
    car_model: Optional[str] = None
    manufactured_year: Optional[str] = None
    voc_valid: Optional[bool] = False
```

### Additional Endpoints

- **GET /health**: Health check endpoint with 200 OK response
- **GET /ready**: Readiness check for model and dependencies
- **POST /ingest**: Asynchronous data ingestion into vector DB
- **GET /detect/brand/{query}**: Direct brand validation endpoint
- **GET /detect/model/{query}**: Direct model validation endpoint
- **POST /save-correction**: Feedback endpoint for typo corrections

---

## **Vector Search Pipeline**

### 1. Preprocessing
```python
def normalize_case(text: str) -> str:
    """Normalize text to title case (first letter uppercase, rest lowercase)"""
    if not text:
        return ""
    return text.strip().title()
```

### 2. Embedding Generation
```python
def load_embedding_model_hf() -> SentenceTransformer:
    """Load embedding model from HuggingFace Hub with caching"""
    model = SentenceTransformer(
        os.path.join(MODEL_PATH, "finetuned-embedding-model"),
        use_auth_token=HF_TOKEN
    )
    return model
```

### 3. Vector Search
```python
def hybrid_search(
    query: str,
    choices: List[str],
    vector_type: str,
    fuzzy_threshold: float = 0.8,
    top_k: int = 3,
    model: Optional[SentenceTransformer] = None
) -> List[Dict[str, Any]]:
    """Hybrid search combining semantic similarity and fuzzy string matching"""
    # Implementation details in core/search.py
```

### 4. Typo Detection
```python
async def typo_lookup(query: str, domain: str) -> Tuple[bool, Optional[str]]:
    """Look up a known typo in the database for instant correction"""
    # Implementation details in core/db_lookup.py
```

---

## **Deployment Configuration**

### Docker Configuration

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
```


---

## **Environment Variables**

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `HF_TOKEN` | HuggingFace API token | Yes | - |
| `HF_REPO` | HuggingFace repo ID | Yes | - |
| `SUPABASE_URL` | Supabase project URL | Yes | - |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | Yes | - |
| `SUPABASE_SERVICE_KEY` | Supabase service key | Yes | - |
| `QDRANT_URL` | Qdrant vector DB URL | Yes | - |
| `QDRANT_API_KEY` | Qdrant API key | Yes | - |
| `COLLECTION_NAME` | Qdrant collection name | Yes | - |
| `REDIS_HOST` | Redis host address | Yes | - |
| `REDIS_PORT` | Redis port number | Yes | 6379 |
| `REDIS_PASSWORD` | Redis password | Yes | - |
| `MODEL_NAME` | Embedding model name | No | intfloat/multilingual-e5-small |
| `GEMINI_API_KEY` | Google Gemini API key | Yes | - |


---

## **Performance Characteristics**

| Metric | Value |
|--------|-------|
|Frequent lookup in Redis| ~0.02s
|Fuzzy Matching (including saving typo to Redis cache and training_dataset table)| ~0.1s
|Embeddings semantic search| ~1–2s
|VLM Processing for VOC| ~5–8s
