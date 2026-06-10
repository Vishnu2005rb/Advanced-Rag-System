import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Use writable temp directory (defaulting to system temp)
TEMP_DIR = os.path.join(tempfile.gettempdir(), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

def load_pdf_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Loads text page-by-page from an uploaded PDF using LangChain's PyPDFLoader.
    Saves bytes temporarily inside the workspace directory, parses, and cleans up.
    
    Returns:
        A list of LangChain Document objects with:
        - page_content
        - metadata = {filename, page, source}
    """
    temp_path = os.path.join(TEMP_DIR, filename)
    
    # Save the file bytes to a temporary location in the workspace
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
        
    try:
        # Load using LangChain's PyPDFLoader
        loader = PyPDFLoader(temp_path)
        raw_documents = loader.load()
        
        processed_documents = []
        for idx, doc in enumerate(raw_documents):
            # PyPDFLoader returns 0-indexed page numbers in metadata under "page"
            # Let's ensure it is 1-indexed for standard human display
            raw_page = doc.metadata.get("page", idx)
            page_num = raw_page + 1
            
            # Re-wrap to ensure metadata contains exactly filename, page, and source
            new_doc = Document(
                page_content=doc.page_content.strip(),
                metadata={
                    "filename": filename,
                    "page": page_num,
                    "source": filename,
                    "source_path": temp_path
                }
            )
            processed_documents.append(new_doc)
            
        return processed_documents
        
    finally:
        # Securely clean up the temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_path}: {e}")
