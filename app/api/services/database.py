
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from services.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
import logging

logger = logging.getLogger(__name__)

def get_qdrant_client():
    """Initialize and return Qdrant client"""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        raise

def create_collection(client, collection_name=COLLECTION_NAME, vector_size=384, recreate=False):
    """Create or recreate Qdrant collection"""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name in collection_names:
            if recreate:
                # Delete existing collection
                client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection '{collection_name}'")
            else:
                logger.info(f"Collection '{collection_name}' already exists")
                return False

        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Created collection '{collection_name}'")

        # Add payload index for vector_type
        client.create_payload_index(
            collection_name=collection_name,
            field_name="vector_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logger.info(f"Added payload index on vector_type")
        return True
        
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise