import pandas as pd
import logging
from qdrant_client.models import PointStruct
from services.config import COLLECTION_NAME, DATA_PATH
from app.api.services.qdrant import get_qdrant_client, create_collection
from core.embedding import get_embedding_model

logger = logging.getLogger(__name__)

def create_points_from_df(df, model):
    """Create points from DataFrame with enhanced embedding context"""
    brand_points = []
    model_points = []
    
    for idx, row in df.iterrows():
        try:
            # Brand point with enhanced context
            brand_embedding = model.encode(f"car brand: {row['car_brand']} models: {row['car_model']}")
            brand_point = PointStruct(
                id=int(row['id']),
                vector=brand_embedding.tolist(),
                payload={
                    "id": int(row['id']),
                    "car_brand": row['car_brand'],
                    "car_model": row['car_model'],
                    "year_start": int(row['year_start']),
                    "year_end": int(row['year_end']),
                    "vector_type": "brand"
                }
            )
            brand_points.append(brand_point)
            
            # Model point with enhanced context
            model_embedding = model.encode(f"car model: {row['car_model']} by {row['car_brand']}")
            model_point = PointStruct(
                id=int(row['id']) + 10000,  # Offset for model IDs
                vector=model_embedding.tolist(),
                payload={
                    "id": int(row['id']),
                    "car_brand": row['car_brand'],
                    "car_model": row['car_model'],
                    "year_start": int(row['year_start']),
                    "year_end": int(row['year_end']),
                    "vector_type": "model"
                }
            )
            model_points.append(model_point)
            
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
    
    return brand_points + model_points

def ingest_data(recreate_collection=False):
    """Ingest data into Qdrant"""
    try:
        # Initialize clients
        client = get_qdrant_client()
        model = get_embedding_model()
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} records from {DATA_PATH}")
        
        # Get vector size from model
        vector_size = len(model.encode("test"))
        
        # Create collection
        create_collection(client, COLLECTION_NAME, vector_size, recreate=recreate_collection)
        
        # Create points
        points = create_points_from_df(df, model)
        logger.info(f"Created {len(points)} points for ingestion")
        
        # Ingest data
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"Successfully ingested {len(points)} points into collection '{COLLECTION_NAME}'")
        return len(points)
        
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise