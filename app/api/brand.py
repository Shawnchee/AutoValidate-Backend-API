from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import ValidationRequest, ValidationSuggestion, BrandValidationRequest

router = APIRouter()

@router.post("/brand", response_model=ValidationResponse)
async def validate_brand(request: BrandValidationRequest)
    query = request.query.strip()

    # Normalize input
    normalized = query.title()

    # Data fetching here
    # db = 

    