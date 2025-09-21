import supabase

import datetime
import logging
from timeit import default_timer as timer

from services.config import SUPABASE_ANON_KEY, SUPABASE_URL

logging = logging.getLogger(__name__)

def get_supabase_client():
    """Initialize and return Supabase client"""
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        return supabase
    except Exception as e:
        logging.error(f"Failed to initialize Supabase client: {e}")
        raise

def typo_lookup(query: str, domain: str):
    """
    Lookup existing brand and model typos in Supabase table
    Returns (found, correction) where:
    - found: bool indicating if a typo was found
    - correction: corrected string if found, else None
    """
    supabase = get_supabase_client()
    try:
        ts1 = timer()

        # Query typo_lookup table for exact match
        response = supabase.table("typo_lookup") \
        .select("corrected") \
        .eq("typo", query.lower().strip()) \
        .eq("domain", domain) \
        .limit(1) \
        .execute()

        ts2 = timer()

        # Validate results
        lookup_data = response.data
        found = lookup_data and len(lookup_data) > 0

        if found:
            correction = lookup_data[0].get("corrected")
            logging.info(f"Typo lookup hit: '{query} -> {correction} ({domain} in {ts2-ts1:.2f}seconds)")
            return found, correction
        else:
            logging.debug(f"Typo lookup miss: '{query}' ({domain}) in {ts2-ts1:.2f}s")
            return False, None
        
    except Exception as e:
        logging.error(f"Error in typo lookup: {e}")
        # On error, return not found so we fall back to hybrid search
        return False, None
        
def save_typo_correction(typo: str, corrected: str, domain:str):
    """
    Save into typo_lookup db when typo correction happens
    """
    supabase = get_supabase_client()
    try:
        typo = typo.lower().strip()
        created_at = datetime.datetime.utcnow().isoformat()


        # Upsert - insert or update if unique constraint violated
        typo_lookup_data = {
            "typo": typo,
            "corrected": corrected,
            "domain": domain,
        }
        response_typo_lookup = supabase.table("typo_lookup").upsert(typo_lookup_data).execute()
        
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
