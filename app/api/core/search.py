# app/search.py
import logging
import pandas as pd
import re
import os
from rapidfuzz import process as rapidfuzz_process
from rapidfuzz import fuzz
from services.config import COLLECTION_NAME, DATA_PATH
from services.qdrant import get_qdrant_client
from core.embedding import load_embedding_model_hf

logger = logging.getLogger(__name__)

def normalize_text(s):
    """Normalize text to improve matching"""
    if not isinstance(s, str):
        return s
    s = s.strip().lower()
    s = re.sub(r'[\-–—_/]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def load_choices():
    """Load brand and model choices from Qdrant payloads (fallback to CSV)."""
    try:
        client = get_qdrant_client()
        logger.info("Loading choices from Qdrant collection: %s", COLLECTION_NAME)

        brands = set()
        models = set()

        batch = 500
        offset = 0

        while True:
            # scroll returns (points, next_page_offset)
            scroll_result = client.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=batch,
                with_payload=True,
                with_vectors=False
            )
            
            # Unpack the tuple correctly: (points, next_page_offset)
            if isinstance(scroll_result, tuple) and len(scroll_result) == 2:
                points, next_offset = scroll_result
            else:
                # Handle older API versions that might return just points
                points = scroll_result
                next_offset = None
                
            if not points or len(points) == 0:
                break
                
            logger.debug(f"Processing {len(points)} points")

            for p in points:
                # Directly access the payload attribute of Record objects
                if hasattr(p, "payload"):
                    payload = p.payload  # Use direct attribute access, not getattr with default
                    b = payload.get("car_brand")
                    m = payload.get("car_model")
                    if b:
                        brands.add(b)
                    if m:
                        models.add(m)
                    logger.debug(f"Processed point ID {getattr(p, 'id', 'N/A')} - Brand: {b}, Model: {m}")
                else:
                    logger.warning(f"Point has no payload attribute: {type(p)}")

            if len(points) < batch or next_offset is None:
                break
            
            # Use the returned next offset if available
            if next_offset is not None:
                offset = next_offset
            else:
                offset += len(points)

        if brands or models:
            brand_list = sorted(brands)
            model_list = sorted(models)
            logger.info("Loaded %d brands and %d models from Qdrant", len(brand_list), len(model_list))
            return brand_list, model_list

        # Fallback to CSV if Qdrant has no payloads
        logger.warning("No brands/models found in Qdrant payloads, falling back to CSV at DATA_PATH: %s", DATA_PATH)
        df = pd.read_csv(DATA_PATH)
        brand_choices = list(df['car_brand'].dropna().unique())
        model_choices = list(df['car_model'].dropna().unique())
        logger.info("Loaded %d unique brands and %d unique models from CSV", len(brand_choices), len(model_choices))
        return brand_choices, model_choices

    except Exception as e:
        logger.exception("Error loading choices from Qdrant/CSV: %s", e)
        return [], []

def hybrid_search(query, choices, vector_type="brand", fuzzy_threshold=75, top_k=3, model=None):
    """
    Hybrid search that combines fuzzy matching and embeddings
    """
    if not model:
        model = load_embedding_model_hf()

    # Normalize query
    query_norm = normalize_text(query)

    # Select appropriate fuzzy matching algorithm
    scorer = fuzz.token_sort_ratio if ' ' in query_norm or len(query_norm) > 10 else fuzz.ratio
    
    # Step 1: Try fuzzy matching first
    fuzzy_matches = rapidfuzz_process.extract(
        query_norm,
        [normalize_text(c) for c in choices],
        scorer=scorer,
        limit=top_k * 2
    )
    
    # Map normalized choices back to original labels
    norm_to_orig = {normalize_text(c): c for c in choices}
    fuzzy_matches = [(norm_to_orig.get(match, match), score, idx) for match, score, idx in fuzzy_matches]
    
    # Filter matches that meet threshold
    good_fuzzy_matches = [(match, score) for match, score, _ in fuzzy_matches if score >= fuzzy_threshold]
    
    results = []
    
    # If we have good fuzzy matches, use those
    if good_fuzzy_matches:
        logger.info(f"Found {len(good_fuzzy_matches)} good fuzzy matches for '{query}'")
        for match, score in good_fuzzy_matches:
            results.append({
                "text": match,
                "score": score / 100.0,
                "source": "fuzzy"
            })
    
    # Step 2: If not enough good fuzzy matches, use embeddings
    if len(results) == 0:
        logger.info(f"No good fuzzy matches above threshold {fuzzy_threshold}, using embeddings")
        client = get_qdrant_client()
        
        # Use appropriate context
        query_with_context = f"car {vector_type}: {query}"
        query_vector = model.encode(query_with_context)
        
        # Search Qdrant
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k * 3,
            query_filter={"must": [{"key": "vector_type", "match": {"value": vector_type}}]}
        )
        
        # Process results
        embedding_results = []
        unique_results = {}
        
        for result in search_result:
            # Get text based on vector_type
            text = result.payload.get('car_brand' if vector_type == 'brand' else 'car_model')
            
            if text not in unique_results:
                unique_results[text] = result
                
                # Skip if already in results
                if any(r["text"] == text for r in results):
                    continue
                
                # Add to results
                results.append({
                    "text": text,
                    "score": result.score,
                    "source": "embedding"
                })
    
    # Return top_k results, sorted by score
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
