# Car Input Validation API & SDK

A FastAPI-based API and TypeScript SDK for real-time car insurance form validation.  
Features include typo detection, top-3 suggestion generation, optional VOC biasing, and dynamic input validation on frontend fields.

---

## **Table of Contents**

- [Overview](#overview)  
- [Tech Stack](#tech-stack)  
- [Frontend SDK](#frontend-sdk)  
- [API Endpoints](#api-endpoints)  
  - [`POST /validate/brand`](#post-validatebrand)  
  - [`POST /validate/model`](#post-validatemodel)  
- [Request/Response Examples](#requestresponse-examples)  
- [Docker Deployment](#docker-deployment)  
- [Usage](#usage) 

---

## **Overview**

This API validates and corrects user inputs for car insurance forms:  

- **Brand** → Detect typos, suggest top-3 correct brands.  
- **Model** → Detect typos, suggest top-3 correct models.  
- **Optional VOC upload** → Temporarily bias suggestions during that session.  
- **Feedback loop** → Selected suggestions can be stored to fine-tune the reranker over time.  
- **Frontend SDK:** TypeScript + React hooks (dynamic input validation, instant feedback)  
- **Backend API:** FastAPI (Python), SentenceTransformers embeddings, HuggingFace reranker 

---

## **Frontend SDK**

### Features

- Dynamic field validation for:  
  - **Manufactured Year:** Blocks alphabet input, max 4 digits, range check  
  - **IC Number:** Validates Malaysian IC format (e.g., `YYMMDD-##-####`)  
  - **Car Plate Number:** Validates format rules (letters, digits, spacing)  
- Instant, zero-latency feedback  
- Integrates seamlessly with API for typo detection & suggestions  

---

## **API Endpoints**

### `POST /validate/brand`

Validate car brand input. Returns top-3 suggestions if a typo is detected.

**Request Body:**

```json
{
  "input": "Prootn",
  "session_id": "abc-123",
  "voc_data": {
    "brand": "Proton",
    "model": "Saga",
    "manufactured_year": 2018
  }
}
```
**Response:**
```json
{
  "input": "Prootn",
  "is_valid": false,
  "suggestions": ["Proton", "Perodua", "Honda"]
}
```

### `POST /validate/model`

Validate car model input. Returns top-3 suggestions if a typo is detected.

**Request Body:**

```json
{
  "input": "Sgaa",
  "brand": "Proton",
  "session_id": "abc-123",
  "voc_data": {
    "brand": "Proton",
    "model": "Saga",
    "registration_year": 2018
  }
}
```

**Response:**

```json
{
  "input": "Sgaa",
  "brand": "Proton",
  "is_valid": false,
  "suggestions": ["Saga", "Perdana", "Exora"]
}
```

---

## **Docker Deployment**

### Build the Docker image:

```bash
docker build -t car-validation-api .
```

### Run locally:

```bash
docker run -p 8000:8000 car-validation-api
```

### Access API:

Visit the API documentation at:  
[http://localhost:8000/docs](http://localhost:8000/docs)

---

## **Notes**

- **Real-time validation:** SDK blocks invalid inputs immediately; API provides typo detection + top-3 suggestions.  
- **VOC Biasing:** Optional, session-level only. No data persisted unless using Supabase.  
- **Feedback Loop:** Store user-selected suggestions for reranker fine-tuning via Airflow.  
- **Developer-first:** SDK + API designed for easy integration into any insurance form.
