import os
import logging
import zipfile
import tempfile
import shutil
from sentence_transformers import SentenceTransformer
from services.config import MODEL_NAME, MODEL_PATH
import datetime
import torch
from huggingface_hub import snapshot_download, HfApi

logger = logging.getLogger(__name__)

def load_embedding_model_hf(
    repo_id="ShawnSean/AutoValidate-Embedding-Model", 
    model_date=None,
    force_reload=False
):
    """
    Load embedding model from HuggingFace Hub with caching
    
    Args:
        repo_id: Hugging Face repo ID
        model_date: Specific date of model to load (YYYYMMDD)
        force_reload: Force redownload even if cached
        
    Returns:
        Loaded SentenceTransformer model
    """
    # Set up cache key
    cache_key = f"{repo_id}_{model_date or 'latest'}"
    
    # Check for cached model path
    cache_dir = os.path.join(os.path.dirname(MODEL_PATH), "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set up HF_TOKEN from env
    if "HF_TOKEN" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv()
    
    # Disable progress bars for cleaner logs
    os.environ["HF_HUB_DISABLE_PROGRESS"] = "1"
    
    try:
        api = HfApi()
        
        # List model files in the repo
        logger.info(f"Listing model files in {repo_id}...")
        model_files = [
            f for f in api.list_repo_files(repo_id=repo_id)
            if f.startswith("models/finetuned-embedding-model-") and f.endswith(".zip")
        ]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {repo_id}")
        
        # Select appropriate model file
        if model_date is None:
            # Find the latest model
            model_files.sort(reverse=True)
            model_file = model_files[0]
            date_from_file = model_file.split('-')[-1].split('.')[0]
            logger.info(f"Using latest model: {model_file} (date: {date_from_file})")
        else:
            # Find model with specified date
            date_str = str(model_date)
            matching_files = [f for f in model_files if date_str in f]
            if not matching_files:
                raise FileNotFoundError(f"No model found for date {model_date}")
            model_file = matching_files[0]
            date_from_file = date_str
            logger.info(f"Using model from {model_date}: {model_file}")
        
        # Create dated directory for this specific model version
        extract_dir = os.path.join(cache_dir, date_from_file)
        model_dir = os.path.join(extract_dir, "finetuned-embedding-model")
        
        # Skip download if model already exists and not forcing reload
        if os.path.exists(model_dir) and os.path.isdir(model_dir) and not force_reload:
            logger.info(f"Using cached model at {model_dir}")
        else:
            # Create temp directory for safe extraction
            temp_dir = tempfile.mkdtemp(prefix="hf_model_")
            
            try:
                # Download the specific file from the repo
                logger.info(f"Downloading model from Hugging Face Hub...")
                repo_dir = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[model_file],
                    repo_type="model"
                )
                
                # Path to the downloaded zip file
                zip_path = os.path.join(repo_dir, model_file)
                
                # Extract the model to temp directory first
                logger.info(f"Extracting model...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Ensure extract directory exists
                os.makedirs(os.path.dirname(extract_dir), exist_ok=True)
                
                # If extraction was successful, replace the old version (if any)
                if os.path.exists(extract_dir):
                    logger.info(f"Removing old model version: {extract_dir}")
                    shutil.rmtree(extract_dir)
                
                # Move from temp to final location
                shutil.move(temp_dir, extract_dir)
                
                # Clean up the download cache to save space
                if os.path.exists(repo_dir) and repo_dir != extract_dir:
                    shutil.rmtree(repo_dir)
                    
            except Exception as e:
                # Clean up temp dir on error
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise e
        
        # Load the model with SentenceTransformer
        logger.info(f"Loading model from {model_dir}...")
        model = SentenceTransformer(model_dir)
        logger.info("Model loaded successfully!")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from Hugging Face: {e}")