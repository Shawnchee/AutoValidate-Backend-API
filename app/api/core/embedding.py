import os
import logging
import zipfile
import tempfile
import shutil
from sentence_transformers import SentenceTransformer
from api.services.config import MODEL_NAME, MODEL_PATH
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
        # Check if model already exists and is valid
        model_dir = None
        possible_model_dirs = [
            os.path.join(extract_dir, "finetuned-embedding-model"),
            extract_dir,
            os.path.join(extract_dir, "model"),
        ]
        
        if not force_reload:
            for possible_dir in possible_model_dirs:
                if os.path.exists(possible_dir) and os.path.isdir(possible_dir):
                    # Validate it's a proper SentenceTransformer model
                    required_files = ["config.json"]
                    model_files_check = ["pytorch_model.bin", "model.safetensors"]
                    
                    has_config = any(os.path.exists(os.path.join(possible_dir, f)) for f in required_files)
                    has_model = any(os.path.exists(os.path.join(possible_dir, f)) for f in model_files_check)
                    
                    if has_config and has_model:
                        model_dir = possible_dir
                        logger.info(f"Using cached model at {model_dir}")
                        break
        
        if not model_dir:
            # Download and extract model
            logger.info(f"Downloading and extracting model...")
            
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
                
                if not os.path.exists(zip_path):
                    raise FileNotFoundError(f"Downloaded zip file not found: {zip_path}")
                
                # Extract the model to temp directory first
                logger.info(f"Extracting model from {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the actual model directory in the extracted content
                extracted_model_dir = None
                
                # Walk through extracted content to find SentenceTransformer model
                for root, dirs, files in os.walk(temp_dir):
                    # Check if this directory contains SentenceTransformer files
                    has_config = "config.json" in files
                    has_model = any(f in files for f in ["pytorch_model.bin", "model.safetensors"])
                    
                    if has_config and has_model:
                        extracted_model_dir = root
                        logger.info(f"Found model directory in extraction: {extracted_model_dir}")
                        break
                
                if not extracted_model_dir:
                    # Fallback: look for any directory named with "model" or "finetuned"
                    for root, dirs, files in os.walk(temp_dir):
                        if any(keyword in os.path.basename(root).lower() 
                               for keyword in ["model", "finetuned", "embedding"]):
                            if files:  # Directory has some files
                                extracted_model_dir = root
                                logger.info(f"Using fallback model directory: {extracted_model_dir}")
                                break
                
                if not extracted_model_dir:
                    # List what we actually extracted for debugging
                    logger.error("Could not find model directory. Extracted content:")
                    for root, dirs, files in os.walk(temp_dir):
                        logger.error(f"  Directory: {root}")
                        for file in files[:5]:  # Limit output
                            logger.error(f"    File: {file}")
                        if len(files) > 5:
                            logger.error(f"    ... and {len(files)-5} more files")
                    raise FileNotFoundError("No valid SentenceTransformer model found in extracted ZIP")
                
                # Ensure extract directory parent exists
                os.makedirs(os.path.dirname(extract_dir), exist_ok=True)
                
                # Remove old version if it exists
                if os.path.exists(extract_dir):
                    logger.info(f"Removing old model version: {extract_dir}")
                    shutil.rmtree(extract_dir)
                
                # Create the final directory structure
                final_model_dir = os.path.join(extract_dir, "finetuned-embedding-model")
                os.makedirs(extract_dir, exist_ok=True)
                
                # Copy the model files to the final location
                shutil.copytree(extracted_model_dir, final_model_dir)
                model_dir = final_model_dir
                
                logger.info(f"Model successfully extracted to: {model_dir}")
                
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
                # Clean up the download cache to save space
                if os.path.exists(repo_dir) and repo_dir != extract_dir:
                    shutil.rmtree(repo_dir)
                    
            except Exception as e:
                # Clean up temp dir on error
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise e
        
        # Validate the model directory before loading
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
        
        required_files = ["config.json"]
        model_files_check = ["pytorch_model.bin", "model.safetensors"]
        
        has_config = any(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
        has_model = any(os.path.exists(os.path.join(model_dir, f)) for f in model_files_check)
        
        if not has_config or not has_model:
            logger.warning(f"Model directory exists but missing required files: {model_dir}")
            logger.warning(f"Contents: {os.listdir(model_dir) if os.path.exists(model_dir) else 'Directory not found'}")
            raise FileNotFoundError(f"Invalid model directory structure: {model_dir}")
        
        # Load the model with SentenceTransformer
        logger.info(f"Loading model from {model_dir}...")
        model = SentenceTransformer(model_dir)
        logger.info("Model loaded successfully!")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from Hugging Face: {e}")
        # Try fallback to base model
        logger.warning(f"Attempting fallback to base model: {MODEL_NAME}")
        try:
            fallback_model = SentenceTransformer(MODEL_NAME)
            logger.info("Fallback model loaded successfully!")
            return fallback_model
        except Exception as fallback_error:
            logger.error(f"Fallback model also failed: {fallback_error}")
            raise Exception(f"Both HuggingFace model and fallback model failed. Original error: {e}, Fallback error: {fallback_error}")