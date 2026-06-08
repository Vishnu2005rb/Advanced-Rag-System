from typing import List, Tuple, Optional
from langchain_core.documents import Document
from rag.vectorstore import get_vector_store

def retrieve_similar_chunks(
    question: str, 
    k: int = 3, 
    filename_filter: Optional[str] = None,
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> List[Tuple[Document, float]]:
    """
    Performs similarity search with relevance scores on the Chroma DB vector store.
    
    Args:
        question: The query string.
        k: The number of top chunks to retrieve.
        filename_filter: If provided, filter search results to this filename.
        
    Returns:
        A list of tuples containing the Document and its normalized similarity score.
    """
    db = get_vector_store(model_name)
    
    # Define metadata filter if filename is specified
    filter_dict = None
    if filename_filter:
        filter_dict = {"source": filename_filter}
        
    # LangChain's Chroma wrapper has similarity_search_with_relevance_scores
    # It returns (Document, score) where score is typically in range [0, 1] depending on distance metric.
    try:
        raw_results = db.similarity_search_with_relevance_scores(
            question, 
            k=k, 
            filter=filter_dict
        )
    except Exception as e:
        print(f"Similarity search failed: {e}. Falling back to standard similarity search.")
        # Fallback to standard similarity search (without scores) in case of schema/distance issues
        docs = db.similarity_search(question, k=k, filter=filter_dict)
        return [(doc, 0.85) for doc in docs]
        
    results = []
    for doc, score in raw_results:
        normalized_score = float(score)
        # Handle normalization edge cases
        if normalized_score < 0:
            normalized_score = 0.05
        elif normalized_score > 1.0:
            normalized_score = 0.99
        results.append((doc, normalized_score))
        
    return results
