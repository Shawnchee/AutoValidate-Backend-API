from pydantic import BaseModel, Field
from typing import List, Optional

class ValidationRequest(BaseModel):
    query: str = Field(..., description="The query string to validate")

class BrandValidationRequest(ValidationRequest):
    brand: str = Field(..., description="The brand of the vehicle")

class ModelValidationRequest(ValidationRequest):
    model: str = Field(..., description="The model of the vehicle")

class ValidationSuggestion(BaseModel):
    suggestion: List[str] = Field(..., description="List of suggested valid entries")
