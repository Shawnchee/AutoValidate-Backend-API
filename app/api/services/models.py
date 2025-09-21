# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class VectorType(str, Enum):
    BRAND = "brand"
    MODEL = "model"

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query text")
    vector_type: VectorType = Field(default=VectorType.BRAND, description="Whether to search for brands or models")
    fuzzy_threshold: int = Field(default=75, ge=50, le=100, description="Threshold for fuzzy matching (50-100)")
    max_results: int = Field(default=3, ge=1, le=10, description="Maximum number of results to return")

class SearchResult(BaseModel):
    text: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    vector_type: str

class IngestRequest(BaseModel):
    recreate_collection: bool = Field(default=False, description="Whether to recreate the collection")

class IngestResponse(BaseModel):
    status: str
    message: str
    points_count: int