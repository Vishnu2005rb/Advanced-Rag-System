import math
import time
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

_reranker_instance = None
MODEL_NAME = "BAAI/bge-reranker-base"

def get_reranker() -> CrossEncoder:
    """
    Initializes and caches the local CrossEncoder reranker model.
    This model computes cross-attention relevance scores between queries and contexts.
    """
    global _reranker_instance
    if _reranker_instance is None:
        print(f"Loading reranker model: {MODEL_NAME}...")
        t_start = time.time()
        # BAAI/bge-reranker-base is a highly accurate cross-encoder model (~270MB)
        _reranker_instance = CrossEncoder(MODEL_NAME)
        duration = time.time() - t_start
        print(f"Loaded {MODEL_NAME} in {duration:.2f}s")
    return _reranker_instance

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
    Reranks a candidate list of Document objects against a query using BGE Reranker.
    
    Args:
        query: The user search query.
        documents: A list of candidate Document chunks (e.g. top 20).
        top_n: The number of top-scoring chunks to select.
        
    Returns:
        A list of tuples containing the ranked Document and its normalized relevance score.
    """
    if not documents:
        return []
        
    # Get cached model
    model = get_reranker()
    
    # Construct input pairs: [[query, doc1_text], [query, doc2_text], ...]
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Compute relevance scores (logits)
    raw_scores = model.predict(pairs)
    
    # Zip and normalize scores
    ranked_results = []
    for doc, score in zip(documents, raw_scores):
        normalized_score = float(sigmoid(score))
        # Ensure it fits within bounds
        normalized_score = max(0.05, min(0.99, normalized_score))
        ranked_results.append((doc, normalized_score))
        
    # Sort descending by score
    ranked_results.sort(key=lambda item: item[1], reverse=True)
    
    # Return best N chunks
    return ranked_results[:top_n]
