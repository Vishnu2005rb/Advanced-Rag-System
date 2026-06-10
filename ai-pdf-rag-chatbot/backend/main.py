import time
import os
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Advanced RAG components
from rag.pdf_loader import load_pdf_bytes
from rag.chunker import chunk_documents
from rag.embeddings import get_embeddings, get_model_dimension
from rag.vectorstore import add_documents_to_store, clear_vector_store, get_vector_store, delete_document_from_store
from rag.query_transform import rewrite_query
from rag.hybrid_retriever import hybrid_retrieve
from rag.reranker import rerank_documents
from rag.prompts import QA_SYSTEM_PROMPT
from rag.memory import add_message, get_history, format_history, clear_history
from rag.evaluation import evaluate_rag
from rag.llm import get_groq_llm
from rag.database import init_db, save_document, get_all_documents, delete_document as db_delete_document, clear_registry, get_document_by_filename

# Initialize SQLite database schema
init_db()

app = FastAPI(title="Advanced AI PDF RAG Chatbot Backend", version="2.0.0")

# Enable CORS for Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Document registry database is persisted in SQLite registry.db

class ChatRequest(BaseModel):
    question: str
    filename_filter: Optional[str] = None
    model_name: Optional[str] = "llama-3.3-70b-versatile"
    temperature: Optional[float] = 0.3
    api_key: Optional[str] = None
    
    # Advanced RAG Configurations
    session_id: Optional[str] = "default_session"
    embedding_model: Optional[str] = "BAAI/bge-small-en-v1.5"
    retriever_type: Optional[str] = "hybrid" # similarity, mmr, hybrid
    score_threshold: Optional[float] = 0.70
    enable_query_rewrite: Optional[bool] = True
    enable_reranking: Optional[bool] = True
    enable_memory: Optional[bool] = True
    run_evaluation: Optional[bool] = True
    top_k: Optional[int] = 5
    fetch_k: Optional[int] = 20

class SourceReference(BaseModel):
    page: int
    content: str
    score: float
    source: str
    chunk_id: Optional[str] = None

class RagasScores(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    retrieved_chunks: List[SourceReference]
    scores: Optional[RagasScores] = None
    rewritten_query: Optional[str] = None
    time_taken: float
    model_used: str

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Advanced AI PDF RAG Chatbot Backend API!",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": time.time(), "version": "2.0.0"}

@app.get("/documents")
def get_documents():
    return get_all_documents()

@app.get("/stats")
def get_global_stats(model_name: str = "BAAI/bge-small-en-v1.5"):
    embeddings = get_embeddings(model_name)
    vector_dim = get_model_dimension(model_name)
    
    # Filter documents matching embedding model
    model_docs = [doc for doc in get_all_documents() if doc.get("embedding_model") == model_name]
    total_pages = sum(doc["pages"] for doc in model_docs)
    total_chunks = sum(doc["chunks"] for doc in model_docs)
    
    return {
        "total_documents": len(model_docs),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "vector_dimension": vector_dim,
        "embedding_model": model_name,
        "llm_model": "Llama 3.3 70B"
    }

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    chunker_strategy: str = Form("recursive"), # recursive or semantic
    embedding_model: str = Form("BAAI/bge-small-en-v1.5")
):
    """
    Ingests a PDF document using the Advanced RAG pipeline:
    1. Loads PDF page-by-page using PyPDFLoader (pdf_loader)
    2. Segments text using either Recursive Character or Semantic splitters (chunker)
    3. Generates embedding vectors dynamically mapping BGE or MiniLM models (embeddings)
    4. Indexes chunks persistently inside ChromaDB (vectorstore)
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        file_bytes = await file.read()
        filename = file.filename
        
        # Load PDF
        docs = load_pdf_bytes(file_bytes, filename)
        if not docs:
            raise HTTPException(
                status_code=400, 
                detail="PDF could not be parsed or contains no extractable text."
            )
            
        pages_count = len(docs)
        
        # Get active embedding model for semantic splitting / stats calculation
        embeddings = get_embeddings(embedding_model)
        
        # Chunk text
        chunks = chunk_documents(
            documents=docs,
            strategy=chunker_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embeddings_model=embeddings
        )
        chunks_count = len(chunks)
        
        # Index chunks in persistent ChromaDB store
        add_documents_to_store(chunks, model_name=embedding_model)
        
        # Record stats
        file_size_mb = f"{len(file_bytes) / (1024 * 1024):.2f} MB"
        doc_stats = {
            "filename": filename,
            "size": file_size_mb,
            "pages": pages_count,
            "chunks": chunks_count,
            "embedding_model": embedding_model,
            "vector_count": chunks_count,
            "status": "Processed",
            "timestamp": time.time()
        }
        save_document(doc_stats)
        
        return {
            "status": "success",
            "filename": filename,
            "pages": pages_count,
            "chunks": chunks_count,
            "embedding_model": embedding_model,
            "vector_count": chunks_count
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Advanced Ingestion failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
def query_chatbot(request: ChatRequest):
    """
    Advanced RAG chat workflow:
    1. Query Transform: Rewrites input question based on history (query_transform)
    2. Retrieval: Similarity / MMR / Hybrid keyword + vector searches (hybrid_retriever)
    3. Reranking: Sorts retrieved chunks using CrossEncoder re-rankers (reranker)
    4. Prompting: Builds contexts grounding prompt schemas (prompts)
    5. Reasoning: Generates answers using Groq Llama 3.3
    6. Memory: Updates conversation logs (memory)
    7. Evaluation: Computes RAGAS audit scores (evaluation)
    """
    t_start = time.time()
    
    # 1. Conversation Memory retrieval
    session_id = request.session_id or "default_session"
    history_list = get_history(session_id) if request.enable_memory else []
    
    # 2. Query Transformation (Rewrite)
    rewritten_query = request.question
    if request.enable_query_rewrite and request.enable_memory:
        rewritten_query = rewrite_query(
            question=request.question,
            chat_history=history_list,
            model_name=request.model_name or "llama-3.3-70b-versatile",
            api_key=request.api_key
        )
        
    # 3. Retrieve chunks (Hybrid, MMR, or Similarity)
    # If Reranking is enabled, retrieve top 20 candidate chunks
    retrieval_k = request.fetch_k if request.enable_reranking else request.top_k
    
    retrieved_results = hybrid_retrieve(
        question=rewritten_query,
        k=retrieval_k or 5,
        score_threshold=request.score_threshold or 0.0,
        filename_filter=request.filename_filter,
        model_name=request.embedding_model or "BAAI/bge-small-en-v1.5",
        search_type="mmr" if request.retriever_type == "mmr" else "similarity",
        fetch_k=request.fetch_k or 20
    )
    
    # Map raw retrieved chunks
    raw_sources = []
    for doc, score in retrieved_results:
        raw_sources.append(SourceReference(
            page=doc.metadata.get("page", 1),
            content=doc.page_content,
            score=score,
            source=doc.metadata.get("source", "Unknown"),
            chunk_id=doc.metadata.get("chunk_id")
        ))
        
    # 4. Reranking Pipeline
    final_sources = []
    if request.enable_reranking and retrieved_results:
        candidate_docs = [doc for doc, _ in retrieved_results]
        reranked = rerank_documents(
            query=rewritten_query,
            documents=candidate_docs,
            top_n=request.top_k or 5
        )
        for doc, score in reranked:
            final_sources.append(SourceReference(
                page=doc.metadata.get("page", 1),
                content=doc.page_content,
                score=score,
                source=doc.metadata.get("source", "Unknown"),
                chunk_id=doc.metadata.get("chunk_id")
            ))
    else:
        # If no reranking, use the sorted retrieval results up to top_k
        final_sources = raw_sources[:request.top_k]
        
    # If no contexts are found
    if not final_sources:
        return ChatResponse(
            answer="Information not available in document",
            sources=[],
            retrieved_chunks=[],
            scores=RagasScores(faithfulness=0.0, answer_relevance=0.0, context_precision=0.0, context_recall=0.0),
            rewritten_query=rewritten_query,
            time_taken=round(time.time() - t_start, 2),
            model_used=request.model_name or "llama-3.3-70b-versatile"
        )
        
    # 5. Format prompt templates
    context_text = "\n\n".join([
        f"[Source: {src.source} | Page: {src.page} | ID: {src.chunk_id}]\n{src.content}"
        for src in final_sources
    ])
    
    history_str = format_history(session_id) if request.enable_memory else "No prior conversation history."
    
    prompt = QA_SYSTEM_PROMPT.format(
        context=context_text,
        history=history_str,
        question=request.question
    )
    
    # 6. Invoke Llama 3.3 LLM via Groq
    try:
        llm = get_groq_llm(
            model_name=request.model_name or "llama-3.3-70b-versatile",
            temperature=request.temperature,
            api_key=request.api_key
        )
        
        # Invoke prompt
        response = llm.invoke(prompt)
        answer_text = response.content.strip()
        
    except Exception as e:
        answer_text = f"Error generating answer from LLM: {str(e)}"
        
    # 7. Add responses to chat memory
    if request.enable_memory:
        add_message(session_id, "user", request.question)
        add_message(session_id, "assistant", answer_text)
        
    # 8. RAG Evaluation audit
    ragas_scores = None
    if request.run_evaluation:
        chunks_text_list = [src.content for src in final_sources]
        scores_dict = evaluate_rag(
            question=request.question,
            answer=answer_text,
            retrieved_chunks=chunks_text_list,
            model_name=request.model_name or "llama-3.3-70b-versatile",
            api_key=request.api_key
        )
        ragas_scores = RagasScores(**scores_dict)
        
    duration = time.time() - t_start
    
    # Return source metadata with text snippets for main list, but full chunks inside retrieved_chunks
    sources_snippets = []
    for src in final_sources:
        sources_snippets.append(SourceReference(
            page=src.page,
            content=src.content[:200] + "...", # Snippet for context list preview
            score=src.score,
            source=src.source,
            chunk_id=src.chunk_id
        ))
        
    return ChatResponse(
        answer=answer_text,
        sources=sources_snippets,
        retrieved_chunks=final_sources, # Full text blocks
        scores=ragas_scores,
        rewritten_query=rewritten_query,
        time_taken=round(duration, 2),
        model_used=request.model_name or "llama-3.3-70b-versatile"
    )

@app.post("/clear")
def clear_backend_data(session_id: Optional[str] = "default_session"):
    """Clears indexed databases and chat history memory."""
    try:
        clear_vector_store()
        clear_registry()
        if session_id:
            clear_history(session_id)
        return {"status": "success", "message": "All vector spaces and chat histories cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/clear_history")
def clear_session_chat_history(session_id: str = "default_session"):
    """Resets memory history for a session."""
    clear_history(session_id)
    return {"status": "success", "message": f"Chat history for session {session_id} successfully cleared."}

@app.delete("/documents/{filename}")
def delete_document(filename: str, embedding_model: str = "BAAI/bge-small-en-v1.5"):
    """Deletes a specific document index from vector db."""
    try:
        doc = get_document_by_filename(filename)
        if doc:
            # Delete from vector store
            delete_document_from_store(filename, model_name=embedding_model)
            # Delete from registry
            db_delete_document(filename)
            return {"status": "success", "message": f"Document {filename} deleted from indexing."}
        else:
            raise HTTPException(status_code=404, detail="Document not found in registry.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
