from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from contextlib import asynccontextmanager
import asyncio
import logging
import threading
from timeit import default_timer as timer

from services.models import SearchRequest, SearchResponse, SearchResult, IngestRequest, IngestResponse, DomainType
from core.search import hybrid_search, load_choices
from core.embedding import get_embedding_model
from core.ingestion import ingest_data
from services.config import API_TITLE, API_DESCRIPTION, API_VERSION

# Set up logging
logger = logging.getLogger(__name__)

# Lock to prevent concurrent lazy loads
_load_lock = threading.Lock()

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

# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, model=Depends(get_model)):
    try:
        # Select choices based on vector_type
        choices = app.state.brand_choices if request.domain == DomainType.BRAND else app.state.model_choices
        
        if not choices:
            raise HTTPException(status_code=500, detail=f"No {request.domain} choices available")

        # Perform search
        ts1 = timer()
        results = hybrid_search(
            query=request.query,
            choices=choices,
            domain=request.domain,
            fuzzy_threshold=request.fuzzy_threshold,
            top_k=request.max_results,
            model=model
        )
        ts2 = timer()
        logger.info(f"Search for '{request.query}' ({request.vector_type}) returned {len(results)} results in {ts2 - ts1:.2f} seconds")
        
        # Format response
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query=request.query,
            vector_type=request.vector_type
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
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

