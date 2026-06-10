import os
import math
import time
import requests
from typing import List, Tuple, Optional
from langchain_core.documents import Document

MODEL_NAME = "BAAI/bge-reranker-base"

def sigmoid(x: float) -> float:
    """Applies sigmoid function to map raw logit score to a [0.0 - 1.0] probability range."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5
) -> List[Tuple[Document, float]]:
    """
    Reranks a candidate list of Document objects against a query using Hugging Face's serverless Inference API.
    If the API key is missing or the request fails, falls back to the original retriever ranking.
    """
    if not documents:
        return []

    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    
    if api_key:
        api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # BAAI/bge-reranker-base expects cross-encoder pairs.
        payload = {
            "inputs": [
                {"text": query, "text_pair": doc.page_content}
                for doc in documents
            ]
        }
        
        try:
            print(f"Calling Hugging Face Inference API for reranking using {MODEL_NAME}...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                scores = []
                
                # Check response format
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and 'score' in item:
                            scores.append(item['score'])
                        elif isinstance(item, (int, float)):
                            scores.append(item)
                        elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict) and 'score' in item[0]:
                            scores.append(item[0]['score'])
                
                if len(scores) == len(documents):
                    ranked_results = []
                    for doc, score in zip(documents, scores):
                        normalized_score = float(score)
                        # Apply sigmoid if the API returns raw logits (outside [0, 1])
                        if any(s > 1.0 or s < 0.0 for s in scores):
                            normalized_score = float(sigmoid(score))
                        
                        normalized_score = max(0.05, min(0.99, normalized_score))
                        ranked_results.append((doc, normalized_score))
                        
                    ranked_results.sort(key=lambda item: item[1], reverse=True)
                    print("Successfully reranked documents via Hugging Face Inference API.")
                    return ranked_results[:top_n]
                else:
                    print(f"Unexpected response length from reranker API: {len(result)} vs {len(documents)}")
            else:
                print(f"Hugging Face reranker API returned status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Failed to rerank documents via Hugging Face API: {e}")

    # Fallback: return documents with dummy scores matching their current order
    print("Falling back to original retriever scores (no reranking applied).")
    fallback_results = []
    for idx, doc in enumerate(documents):
        # Generate a dummy score that preserves original ranking order
        score = max(0.05, min(0.99, 1.0 - (idx * 0.05)))
        fallback_results.append((doc, score))
        
    return fallback_results[:top_n]
