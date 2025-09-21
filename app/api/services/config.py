
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

BASE_API_DIR = Path(__file__).resolve().parent.parent

# Supabase Config (or any other db)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')

# Qdrant configuration
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'car_data_modelbrand')

# Model configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'intfloat/multilingual-e5-small')
_env_model_path = os.getenv('MODEL_PATH')
if _env_model_path:
    _m = Path(_env_model_path)
    if not _m.is_absolute():
        _m = (Path.cwd() / _m).resolve()
    MODEL_PATH = str(_m)
else:
    # candidate locations (module-relative and repo notebook folder)
    candidates = [
        BASE_API_DIR / "model" / "finetuned-embedding-model",               # app/api/model/...
        Path.cwd() / "model" / "finetuned-embedding-model",                # ./model/...
        Path.cwd() / "notebook" / "model" / "finetuned-embedding-model",   # ./notebook/model/...
        BASE_API_DIR.parent.parent / "notebook" / "model" / "finetuned-embedding-model",  # repo-root/notebook/...
    ]
    found = None
    for c in candidates:
        try:
            if c.exists():
                found = c.resolve()
                break
        except Exception:
            continue
    if found:
        MODEL_PATH = str(found)
    else:
        # fallback to original relative default
        MODEL_PATH = os.getenv('MODEL_PATH', './model/finetuned-embedding-model')

# Data configuration
_env_data_path = os.getenv('DATA_PATH')
if _env_data_path:
    _p = Path(_env_data_path)
    # If a relative path was provided, resolve relative to the repo CWD
    if not _p.is_absolute():
        _p = (Path.cwd() / _p).resolve()
    DATA_PATH = str(_p)
else:
    DATA_PATH = str((BASE_API_DIR / "car_dataset.csv").resolve())
    
# API configuration
API_TITLE = "Vehicle Search API"
API_DESCRIPTION = "API for searching vehicle brands and models with typo tolerance"
API_VERSION = "1.1.1"