from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import brand

app = FastAPI(
    title="Vehicle Insurance Validation API"
    description="Developer API for validating vehicle insurance forms in real-time quick and accurately"
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(brand.router, prefix="/api/validate", tags=["validation"])
app.include_router(model.router, prefix="/api/validate", tags=["validation"])
app.include_router(year.router, prefix="/api/validate", tags=["validation"])
app.include_router(voc.router, prefix="/api", tags=["voc"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}