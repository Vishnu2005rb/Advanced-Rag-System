import os
import requests
import time
from typing import Dict, List, Optional
from langchain_core.embeddings import Embeddings

# Registry caching loaded embedding models in-memory
_embeddings_registry: Dict[str, Embeddings] = {}

# Standard model dimensions
MODEL_DIMENSIONS = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-large-en-v1.5": 1024
}

class HFInferenceAPIEmbeddings(Embeddings):
    """
    Lightweight embeddings class that calls Hugging Face's serverless Inference API.
    Avoids downloading models locally and does not require torch or sentence-transformers.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError(
                "Hugging Face API Token is required for serverless embeddings. "
                "Please set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in your environment/variables."
            )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            self.api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        if response.status_code != 200:
            raise Exception(f"Hugging Face Inference API error: {response.text}")
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        if not self.api_key:
            raise ValueError(
                "Hugging Face API Token is required for serverless embeddings. "
                "Please set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in your environment/variables."
            )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            self.api_url,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}}
        )
        if response.status_code != 200:
            raise Exception(f"Hugging Face Inference API error: {response.text}")
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            return result[0]
        return result

def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> Embeddings:
    """
    Initializes and caches the requested serverless embeddings model.
    """
    global _embeddings_registry
    
    # Normalize model name input
    if model_name not in MODEL_DIMENSIONS:
        model_name = "BAAI/bge-small-en-v1.5"
        
    if model_name not in _embeddings_registry:
        print(f"Initializing API-based embedding model: {model_name}...")
        t_start = time.time()
        _embeddings_registry[model_name] = HFInferenceAPIEmbeddings(
            model_name=model_name
        )
        duration = time.time() - t_start
        print(f"Initialized API client for {model_name} in {duration:.2f}s")
        
    return _embeddings_registry[model_name]

def get_model_dimension(model_name: str) -> int:
    """Returns the dimension output size of the embedding model."""
    return MODEL_DIMENSIONS.get(model_name, 384)
