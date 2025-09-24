from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form
from contextlib import asynccontextmanager
import asyncio
import logging
import threading
import shutil
import tempfile
import os
import datetime
from uuid import uuid4
from timeit import default_timer as timer
from supabase import create_client, Client

from services.models import SearchRequest, SearchResponse, SearchResult, IngestRequest, IngestResponse, DomainType, ManufacturedYearRequest, ManufacturedYearResult
from core.search import hybrid_search, load_choices
from core.embedding import get_embedding_model
from core.ingestion import ingest_data
from core.db_lookup import typo_lookup, save_typo_correction
from services.config import API_TITLE, API_DESCRIPTION, API_VERSION,SUPABASE_URL, SUPABASE_ANON_KEY
from services.qdrant import get_qdrant_client
from ocr.main import VOCExtractor

# Set up logging
logger = logging.getLogger(__name__)

# Lock to prevent concurrent lazy loads
_load_lock = threading.Lock()

def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

async def save_correction_async(typo: str, corrected: str, domain: str):
    """Async wrapper to save typo correction without blocking"""
    try:
        # Direct await the coroutine - don't use asyncio.to_thread for an async function
        await save_typo_correction(typo, corrected, domain)
    except Exception as e:
        logger.error(f"Failed to save correction: {e}")

def ensure_model_loaded():
    """
    Ensure embedding model and choice lists are loaded into app.state.
    Safe to call from endpoints (will lazy-load on first access).
    """
    if not hasattr(app.state, "model"):
        with _load_lock:
            if not hasattr(app.state, "model"):
                try:
                    app.state.model = get_embedding_model()
                    brand_choices, model_choices = load_choices()
                    app.state.brand_choices = brand_choices
                    app.state.model_choices = model_choices
                    logger.info("Lazy-loaded embedding model and choices")
                except Exception as e:
                    logger.exception("Failed to lazy-load model/choices: %s", e)

# Define application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle: guarantee the embedding model and choices
    are loaded during process startup (not on first request).
    Uses asyncio.to_thread to avoid blocking the event loop while loading.
    """
    logger.info("Starting up, loading embedding model and choices")
    try:
        ts1 = timer()
        app.state.model = await asyncio.to_thread(get_embedding_model)
        brand_choices, model_choices = await asyncio.to_thread(load_choices)
        app.state.brand_choices = brand_choices
        app.state.model_choices = model_choices 
        ts2 = timer()
        logger.info(f"Startup completed in {ts2 - ts1:.2f} seconds")
        logger.info("Startup: embedding model and choices loaded")
    except Exception as e:
        logger.exception("Startup initialization failed: %s", e)

    yield

    # Cleanup on shutdown
    logger.info("Shutting down")

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Dependency to get model
def get_model():
    return app.state.model

# Home endpoint
@app.get("/")
def read_root():
    ensure_model_loaded()
    return {
        "message": "Vehicle Search API",
        "docs": "/docs",
        "endpoints": [
            "/detect/brand",
            "/detect/model",
            "/search",
            "/ingest"
        ]
    }
# VOC Upload endpoint
@app.post("/upload-voc", response_model=dict)
async def upload_voc(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Upload VOC image and extract car information using OCR.
    Stores results in Supabase linked to session_id for later retrieval.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Use provided session ID or generate a new one
    session_id = session_id or str(uuid4())

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
         # Process the OCR extraction in background to avoid blocking
        async def process_and_save():
            try:
                # Extract car info using VOCExtractor
                extractor = VOCExtractor()
                result = extractor.extract_from_image(temp_path)
                
                # Save to Supabase with session ID
                supabase = get_supabase_client()
                data = {
                    "session_id": session_id,
                    "car_brand": result.get("car_brand", ""),
                    "car_model": result.get("car_model", ""),
                    "manufactured_year": result.get("manufactured_year", ""),
                    "voc_valid": bool(result.get("car_brand") or result.get("car_model")),
                    "created_at": datetime.datetime.utcnow().isoformat()
                }
                
                # Upsert to voc_session table
                supabase.table("voc_session").upsert(
                    data, on_conflict=["session_id"]
                ).execute()
                
                logger.info(f"VOC data saved for session {session_id}: {data}")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                logger.error(f"Error processing VOC image: {e}")
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Run processing in background
        if background_tasks:
            background_tasks.add_task(process_and_save)
        else:
            # For testing or immediate response
            await process_and_save()
        
        return {
            "status": "success",
            "message": "VOC image uploaded and processing started",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error uploading VOC: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@app.post("/get-manufactured-year-range", response_model=ManufacturedYearResult)
async def get_manufactured_year_range(request: ManufacturedYearRequest):
    """
    Get the manufactured year range for a given car brand and model.
    """
    client = get_qdrant_client()

            # Create payload indexes for car_brand and car_model
    client.create_payload_index(
            collection_name="car_data_modelbrand",
            field_name="car_brand",
            field_schema="keyword"  
    )
        
    client.create_payload_index(
            collection_name="car_data_modelbrand",
            field_name="car_model",
            field_schema="keyword"
    )
    try:
        # Use correct parameters for search() method
        scroll_result = client.search(
            collection_name="car_data_modelbrand",
            query_vector=[0] * 384,  
            query_filter={  
                "must": [
                    {"key": "car_brand", "match": {"value": request.car_brand}},
                    {"key": "car_model", "match": {"value": request.car_model}}
                ]
            },
            with_payload=True,
            limit=1
        )
        
        # Extract points from the search result
        points = []
        if hasattr(scroll_result, 'points'):
            points = scroll_result.points
        elif isinstance(scroll_result, list):
            points = scroll_result
            
        if points and len(points) > 0:
            point = points[0]
            year_start = str(point.payload.get("year_start")) if hasattr(point, "payload") else None
            year_end = str(point.payload.get("year_end")) if hasattr(point, "payload") else None
            return ManufacturedYearResult(
                year_start=year_start, 
                year_end=year_end
            )
        else:
            logger.info(f"No data found for brand '{request.car_brand}' and model '{request.car_model}'")
            return ManufacturedYearResult()
    except Exception as e:
        logger.error(f"Error retrieving manufactured year range: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving manufactured year range: {str(e)}")

# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, model=Depends(get_model)):
    try:
        domain = request.domain.value # Get string value of the enum DomainType
        voc_result = None
        logger.info(f"Is there request session ID? {request.session_id is not None}")
        

        if request.session_id:
            try:
                supabase = get_supabase_client()
                voc_response = supabase.table("voc_session") \
                    .select("*") \
                    .eq("session_id", request.session_id) \
                    .execute()
                logger.info(f"Session ID is currently {request.session_id}")
                if voc_response.data and len(voc_response.data) > 0:
                    voc_data = voc_response.data[0]
                    voc_result = {
                        "car_brand": voc_data.get("car_brand"),
                        "car_model": voc_data.get("car_model"),
                        "manufactured_year": voc_data.get("manufactured_year"),
                        "voc_valid": voc_data.get("voc_valid", False)
                    }
                    logger.info(f"VOC data found for session {request.session_id}")
            except Exception as e:
                logger.warning(f"Failed to retrieve VOC data: {e}")
                # Continue with search even if VOC lookup fails

        # 1. Try DB lookup for known typo first
        found , corrected = await typo_lookup(request.query, domain)
        if found and corrected:
            return SearchResponse(
                results=[SearchResult(text=corrected)],
                query=request.query,
                domain=request.domain,
                voc_result=voc_result
            )

        # 2. If not found in DB, do hybrid search

        # Select choices based on vector_type
        choices = app.state.brand_choices if request.domain == DomainType.BRAND else app.state.model_choices
        
        if not choices:
            raise HTTPException(status_code=500, detail=f"No {request.domain} choices available")

        # Perform search
        ts1 = timer()
        results = hybrid_search(
            query=request.query,
            choices=choices,
            vector_type=domain,
            fuzzy_threshold=request.fuzzy_threshold,
            top_k=request.max_results,
            model=model
        )
        ts2 = timer()
        logger.info(f"Search for '{request.query}' ({domain}) returned {len(results)} results in {ts2 - ts1:.2f} seconds")

        # Format response
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query=request.query,
            domain=request.domain,  
            voc_result=voc_result
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-correction", response_model=dict)
async def save_correction(
    background_tasks: BackgroundTasks,
    typo: str,
    corrected: str,
    domain: str
):
    """
    Save an explicitly accepted typo correction
    Called when the user selects a correction from the UI
    """
    try:
        if domain not in [d.value for d in DomainType]:
            raise HTTPException(status_code=400, detail="Invalid domain")
        # Create a wrapper function that can run in background tasks
        async def bg_save_correction():
            await save_typo_correction(typo, corrected, domain)
            
        background_tasks.add_task(bg_save_correction)

        return {
                "status": "success",
                "message": f"Saved correction '{typo}' -> '{corrected}' in domain '{domain}'",
                "typo": typo,
                "corrected": corrected,
                "domain": domain
            }
        
    except Exception as e:
        logger.error(f"Save correction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# New detect endpoints for brand and model typo detection
@app.get("/detect/brand/{query}", response_model=SearchResponse)
async def detect_brand_typo(
    query: str, 
    model=Depends(get_model)
):
    """
    Detect and correct brand typos
    """
    request = SearchRequest(
        query=query,
        domain=DomainType.BRAND,
    )
    
    return await search(request, model)


@app.get("/detect/model/{query}", response_model=SearchResponse)
async def detect_model_typo(
    query: str, 
    model=Depends(get_model)
):
    """
    Detect and correct model typos
    """
    request = SearchRequest(
        query=query,
        domain=DomainType.MODEL,
    )
    
    
    return await search(request, model)


# Data ingestion endpoint
@app.post("/ingest", response_model=IngestResponse)
async def ingest_data_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    try:
        # Run ingestion in background
        def run_ingestion():
            try:
                points_count = ingest_data(recreate_collection=request.recreate_collection)
                logger.info(f"Ingestion completed: {points_count} points")
            except Exception as e:
                logger.error(f"Background ingestion failed: {e}")
        
        # Add to background tasks
        background_tasks.add_task(run_ingestion)
        
        return IngestResponse(
            status="success",
            message="Data ingestion started in the background",
            points_count=0
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Ready check endpoint
@app.get("/ready")
def ready_check():
    ensure_model_loaded()
    return {
        "ready": hasattr(app.state, "model"),
        "model_loaded": hasattr(app.state, "model"),
        "brand_choices_loaded": hasattr(app.state, "brand_choices") and len(app.state.brand_choices) > 0,
        "model_choices_loaded": hasattr(app.state, "model_choices") and len(app.state.model_choices) > 0
    }



