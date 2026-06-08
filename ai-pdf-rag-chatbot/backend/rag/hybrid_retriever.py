from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from rag.vectorstore import get_vector_store
from rag.retriever import retrieve_similar_chunks

def hybrid_retrieve(
    question: str,
    k: int = 5,
    score_threshold: float = 0.0,
    filename_filter: Optional[str] = None,
    model_name: str = "BAAI/bge-small-en-v1.5",
    search_type: str = "similarity", # 'similarity' or 'mmr'
    fetch_k: int = 20
) -> List[Tuple[Document, float]]:
    """
    Performs Hybrid Search combining BM25 Keyword Search and Vector DB Search.
    Merges results using Reciprocal Rank Fusion (RRF).
    """
    db = get_vector_store(model_name)
    filter_dict = {"source": filename_filter} if filename_filter else None
    
    # 1. Get all document chunks for BM25 initialization
    try:
        if filter_dict:
            chroma_data = db.get(where=filter_dict)
        else:
            chroma_data = db.get()
            
        all_chunks = []
        if chroma_data and "documents" in chroma_data and chroma_data["documents"]:
            for content, metadata in zip(chroma_data["documents"], chroma_data["metadatas"]):
                all_chunks.append(Document(page_content=content, metadata=metadata))
    except Exception as e:
        print(f"Error fetching chunks for BM25: {e}")
        all_chunks = []
        
    # 2. Execute BM25 Keyword Search
    bm25_docs = []
    if all_chunks:
        try:
            bm25_retriever = BM25Retriever.from_documents(all_chunks)
            bm25_retriever.k = max(k * 2, fetch_k) # retrieve extra candidate chunks
            bm25_docs = bm25_retriever.invoke(question)
        except Exception as e:
            print(f"BM25 retrieval failed: {e}")
            bm25_docs = []
            
    # 3. Execute Vector Search (Similarity or MMR)
    vector_results: List[Tuple[Document, float]] = []
    try:
        if search_type.lower() == "mmr":
            # MMR Search
            # langchain-chroma's max_marginal_relevance_search returns Documents
            raw_mmr = db.max_marginal_relevance_search(
                question,
                k=max(k * 2, fetch_k),
                fetch_k=fetch_k * 2,
                filter=filter_dict
            )
            # Tag with a mock semantic score
            vector_results = [(doc, 0.85) for doc in raw_mmr]
        else:
            # Standard similarity search with scores
            vector_results = retrieve_similar_chunks(
                question=question,
                k=max(k * 2, fetch_k),
                filename_filter=filename_filter,
                model_name=model_name
            )
    except Exception as e:
        print(f"Vector retrieval failed: {e}")
        vector_results = []
        
    # 4. Merge results using Reciprocal Rank Fusion (RRF)
    # RRF Score formula: RRF_Score = 1 / (60 + rank_bm25) + 1 / (60 + rank_vector)
    rrf_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Document] = {}
    base_scores: Dict[str, float] = {} # Keep track of original semantic score for reference
    
    # Process Vector search ranks
    for rank, (doc, score) in enumerate(vector_results):
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        doc_registry[chunk_id] = doc
        base_scores[chunk_id] = score
        
        # 1-indexed rank
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + (rank + 1)))
        
    # Process BM25 ranks
    for rank, doc in enumerate(bm25_docs):
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        doc_registry[chunk_id] = doc
        if chunk_id not in base_scores:
            base_scores[chunk_id] = 0.50 # default middle ground score for keyword-only matches
            
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + (rank + 1)))
        
    # 5. Sort by RRF score and select top K candidates
    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    top_candidates = sorted_chunks[:k]
    
    # 6. Re-map candidates back to (Document, score) format and apply threshold
    final_results = []
    for chunk_id, rrf_val in top_candidates:
        doc = doc_registry[chunk_id]
        orig_score = base_scores[chunk_id]
        
        # Apply score threshold filter
        if orig_score >= score_threshold:
            final_results.append((doc, orig_score))
            
    return final_results
