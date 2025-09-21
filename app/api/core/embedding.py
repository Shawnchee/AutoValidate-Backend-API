import os
import logging
from sentence_transformers import SentenceTransformer
from services.config import MODEL_NAME, MODEL_PATH
import datetime
import torch

logger = logging.getLogger(__name__)

def get_embedding_model(use_finetuned=True):
    """
    Load embedding model - either fine-tuned or base model
    """
    try:
        if use_finetuned and os.path.exists(MODEL_PATH):
            # Try to load fine-tuned model
            logger.info(f"Loading fine-tuned model from {MODEL_PATH}")
            model = SentenceTransformer(MODEL_PATH)
            logger.info("Fine-tuned model loaded successfully")
            return model
        else:
            # Load base model
            logger.info(f"Loading base model: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            logger.info("Base model loaded successfully")
            return model
        
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info(f"Falling back to base model: {MODEL_NAME}")
        return SentenceTransformer(MODEL_NAME)