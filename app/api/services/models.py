# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class DomainType(str, Enum):
    BRAND = "brand"
    MODEL = "model"

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query text")
    domain: DomainType = Field(default=DomainType.BRAND, description="Whether to search for brands or models")
    fuzzy_threshold: int = Field(default=75, ge=50, le=100, description="Threshold for fuzzy matching (50-100)")
    max_results: int = Field(default=3, ge=1, le=10, description="Maximum number of results to return")
    session_id: Optional[str] = None 

class UploadVOCResponse(BaseModel):
    status: str
    message: str
    session_id: str
    car_brand: Optional[str] = None
    car_model: Optional[str] = None
    manufactured_year: Optional[str] = None
    voc_valid: bool = False
    
class VOCResult(BaseModel):
    """VOC extraction results from OCR"""
    car_brand: Optional[str] = None
    car_model: Optional[str] = None
    manufactured_year: Optional[str] = None
    voc_valid: bool = False

class ManufacturedYearRequest(BaseModel):
    car_brand: str
    car_model: str

class ManufacturedYearResult(BaseModel):
    year_start: Optional[str] = None
    year_end: Optional[str] = None

class SearchResult(BaseModel):
    text: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    domain: str
    voc_result: Optional[VOCResult] = None 

class IngestRequest(BaseModel):
    recreate_collection: bool = Field(default=False, description="Whether to recreate the collection")

class IngestResponse(BaseModel):
    status: str
    message: str
    points_count: int