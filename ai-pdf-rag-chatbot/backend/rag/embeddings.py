import time
from typing import Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings

# Registry caching loaded embedding models in-memory
_embeddings_registry: Dict[str, HuggingFaceEmbeddings] = {}

# Standard model dimensions
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-large-en-v1.5": 1024
}

def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbeddings:
    """
    Initializes and caches the requested HuggingFace embeddings model.
    """
    global _embeddings_registry
    
    # Normalize model name input
    if model_name not in MODEL_DIMENSIONS:
        model_name = "BAAI/bge-small-en-v1.5"
        
    if model_name not in _embeddings_registry:
        print(f"Loading embedding model: {model_name}...")
        t_start = time.time()
        _embeddings_registry[model_name] = HuggingFaceEmbeddings(
            model_name=model_name
        )
        duration = time.time() - t_start
        print(f"Loaded {model_name} in {duration:.2f}s")
        
    return _embeddings_registry[model_name]

def get_model_dimension(model_name: str) -> int:
    """Returns the dimension output size of the embedding model."""
    return MODEL_DIMENSIONS.get(model_name, 384)
