from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

def chunk_documents(
    documents: List[Document],
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embeddings_model: Optional[any] = None
) -> List[Document]:
    """
    Applies the specified chunking strategy on the list of documents.
    
    Args:
        documents: List of parsed LangChain Document objects.
        strategy: Either 'recursive' or 'semantic'.
        chunk_size: Token size limit for Recursive splitter.
        chunk_overlap: Character overlap for Recursive splitter.
        embeddings_model: Required if strategy is 'semantic'.
        
    Returns:
        A list of split Document objects with unique chunk_ids in metadata.
    """
    if strategy.lower() == "semantic":
        if embeddings_model is None:
            # Fall back to recursive if no model is provided
            print("Warning: SemanticChunker requires an embeddings model. Falling back to recursive.")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            # Semantic chunking from langchain_experimental
            splitter = SemanticChunker(
                embeddings_model,
                breakpoint_threshold_type="percentile"
            )
    else:
        # Standard recursive text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    chunks = splitter.split_documents(documents)
    
    # Enrich metadata with a unique chunk ID: {filename}_chunk_{index}
    # Grouped by filename to allow unique incremental index
    file_chunk_counters = {}
    for chunk in chunks:
        filename = chunk.metadata.get("filename", "unknown_doc")
        
        if filename not in file_chunk_counters:
            file_chunk_counters[filename] = 0
            
        chunk_index = file_chunk_counters[filename]
        chunk.metadata["chunk_id"] = f"{filename}_chunk_{chunk_index}"
        file_chunk_counters[filename] += 1
        
    return chunks
