import supabase

import datetime
import logging
from timeit import default_timer as timer

from api.services.config import SUPABASE_ANON_KEY, SUPABASE_URL
from api.services.redis import redis

logging = logging.getLogger(__name__)

logging.info(f"Redis client initialized: {redis}")

def get_supabase_client():
    """Initialize and return Supabase client"""
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        return supabase
    except Exception as e:
        logging.error(f"Failed to initialize Supabase client: {e}")
        raise

async def typo_lookup(query: str, domain: str):
    """
    Lookup typo corrections with Redis caching
    Returns (found, correction) where:
    - found: boolean indicating if typo was found
    - correction: corrected text if found, None otherwise
    """
    try:
        ts1 = timer()
        normalized_query = query.lower().strip()

        redis_key=f"typo:{domain}:{normalized_query}"
        # Query typo_lookup table for exact match (redis cache)
        try:
            cached = await redis.get(redis_key)
            if cached:
                ts2 = timer()
                logging.info(f"Redis hit: '{query}' -> '{cached}' ({domain}) in {ts2-ts1:.4f}s")
                return True, cached
        except Exception as e:
            logging.warning(f"Redis error: {e}")
        return False, None
        
    except Exception as e:
        logging.error(f"Error in typo lookup: {e}")
        # On error, return not found so we fall back to hybrid search
        return False, None
        
async def save_typo_correction(typo: str, corrected: str, domain:str):
    """
    Save into redis cache and typo_training_dataset db when typo correction happens
    """
    supabase = get_supabase_client()
    try:
        normalized_typo = typo.lower().strip()
        created_at = datetime.datetime.utcnow().isoformat()

        redis_key = f"typo:{domain}:{normalized_typo}"
        # Upsert - insert or update if unique constraint violated
        # TODO: redis
        try:
            await redis.set(redis_key, corrected)
            logging.debug(f"Redis set: '{normalized_typo}' -> '{corrected}'")
        except Exception as e:
            logging.warning(f"Redis error: {e}")
        
        # 2. Also save to typo_training_dataset for future model training
        training_data = {
            "typo": typo,
            "corrected": corrected,
            "domain": domain,
            "created_at": created_at,        
        }
        response_training = supabase.table("typo_training_dataset").insert(training_data).execute()

        logging.info(f"Saved typo correction: '{typo}' -> '{corrected}' ({domain})")
        return True
    except Exception as e:
        logging.error(f"Error saving typo correction: {e}")
        return False
