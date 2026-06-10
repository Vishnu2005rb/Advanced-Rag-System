import os
import shutil
import tempfile
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.embeddings import get_embeddings

# Use writable directory (defaulting to system temp, but customizable via DATA_DIR environment variable)
DATA_DIR = os.getenv("DATA_DIR", tempfile.gettempdir())
VECTOR_DB_ROOT = os.path.join(DATA_DIR, "vector_db")
os.makedirs(VECTOR_DB_ROOT, exist_ok=True)

def get_persist_dir_for_model(model_name: str) -> str:
    """Creates a filesystem-safe directory path based on the embedding model name."""
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    return os.path.join(VECTOR_DB_ROOT, safe_name)

def get_vector_store(model_name: str = "BAAI/bge-small-en-v1.5") -> Chroma:
    """
    Initializes and returns the persistent ChromaDB instance for the given embedding model.
    Isolates databases by embedding model to prevent dimension mismatches.
    """
    persist_dir = get_persist_dir_for_model(model_name)
    embeddings = get_embeddings(model_name)
    
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="pdf_advanced_rag"
    )

def add_documents_to_store(documents: List[Document], model_name: str = "BAAI/bge-small-en-v1.5") -> int:
    """Adds a list of LangChain Document objects to the Chroma DB store."""
    db = get_vector_store(model_name)
    
    # Extract existing document chunk IDs first to handle updates/overwrites
    # Chroma allows adding documents. If they contain IDs, it will update or insert.
    ids = [doc.metadata.get("chunk_id") for doc in documents]
    db.add_documents(documents, ids=ids)
    return len(documents)

def delete_document_from_store(filename: str, model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Deletes all chunks associated with a specific filename from the vector store.
    """
    db = get_vector_store(model_name)
    
    # Retrieve documents matching metadata filter source
    try:
        # Chroma client allows deleting by metadata where source matches
        collection = db._collection
        collection.delete(where={"source": filename})
        print(f"Deleted index chunks for {filename} under {model_name}.")
    except Exception as e:
        print(f"Error deleting documents for {filename}: {e}")

def clear_vector_store(model_name: Optional[str] = None):
    """
    Deletes all indexed vector files. 
    If model_name is provided, clears only that model's database folder.
    Otherwise, deletes the entire vector_db root.
    """
    if model_name:
        persist_dir = get_persist_dir_for_model(model_name)
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
                print(f"Cleared database folder for {model_name}.")
            except Exception as e:
                print(f"Error clearing model directory {persist_dir}: {e}")
    else:
        # Delete root vector_db directory
        if os.path.exists(VECTOR_DB_ROOT):
            try:
                shutil.rmtree(VECTOR_DB_ROOT)
                os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
                print("Cleared entire vector database repository.")
            except Exception as e:
                print(f"Error clearing root directory {VECTOR_DB_ROOT}: {e}")
