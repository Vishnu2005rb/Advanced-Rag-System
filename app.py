import streamlit as st
import os
import time
import html
from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load Environment Variables
load_dotenv()

# =================================================
# 1. CORE RAG BACKEND FUNCTIONS
# =================================================

@st.cache_resource
def get_embeddings_model():
    """Load and cache the HuggingFace sentence embedding model (MiniLM 384 dimensions)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_groq_client():
    """Initialize ChatGroq LLM client using custom or env key."""
    api_key = st.session_state.custom_groq_api_key.strip()
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")
    return ChatGroq(
        model=st.session_state.llm_model,
        api_key=api_key,
        temperature=st.session_state.temperature
    )

def index_pdf_document(file_object):
    """
    Executes the ingestion pipeline steps sequentially,
    updating the UI step indicators dynamically.
    """
    status_placeholder = st.session_state.get("stepper_placeholder")
    
    # Step 2: Extract Text
    st.session_state.current_step = 2
    if status_placeholder:
        status_placeholder.markdown(get_stepper_html(2), unsafe_allow_html=True)
    time.sleep(0.3)
    
    reader = PdfReader(file_object)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    st.session_state.pdf_num_pages = len(reader.pages)
    
    # Step 3: Create Chunks
    st.session_state.current_step = 3
    if status_placeholder:
        status_placeholder.markdown(get_stepper_html(3), unsafe_allow_html=True)
    time.sleep(0.3)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap
    )
    chunks = splitter.split_text(text)
    st.session_state.chunks = chunks
    st.session_state.total_chunks = len(chunks)
    
    # Step 4: Generate Embeddings
    st.session_state.current_step = 4
    if status_placeholder:
        status_placeholder.markdown(get_stepper_html(4), unsafe_allow_html=True)
    embeddings = get_embeddings_model()
    
    # Step 5: Create Vector Store
    st.session_state.current_step = 5
    if status_placeholder:
        status_placeholder.markdown(get_stepper_html(5), unsafe_allow_html=True)
    time.sleep(0.3)
    
    # Get vector dimensions
    sample_vector = embeddings.embed_query("Check dimensions")
    st.session_state.vector_dimension = len(sample_vector)
    
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    st.session_state.vector_store = vector_store
    
    # Finish Ingestion
    st.session_state.current_step = 6
    st.session_state.processed = True
    st.rerun()

def query_rag_pipeline(question):
    """
    Performs Similarity Search (Step 6) and invokes Groq Llama LLM (Step 7)
    to generate the final response.
    """
    db = st.session_state.vector_store
    k_val = st.session_state.top_k
    
    # Step 6: Similarity Search
    raw_results = db.similarity_search_with_relevance_scores(question, k=k_val)
    
    # Normalize cosine distances to similarity scores [0.0 - 1.0]
    results = []
    for doc, score in raw_results:
        score = float(score)
        if score < 0: score = 0.05
        elif score > 1: score = 0.99
        results.append((doc, score))
        
    st.session_state.retrieved_docs = results
    
    # Step 7: LLM Answer Generation
    t_start = time.time()
    context_text = "\n".join([doc.page_content for doc, _ in results])
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI PDF assistant. Answer the question using ONLY the given PDF context.
        If you do not know the answer based on the context, state that you cannot find the answer in the document.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Give a clear and comprehensive answer:
        """
    )
    
    try:
        llm = get_groq_client()
        chain = prompt | llm
        response = chain.invoke({
            "context": context_text,
            "question": question
        })
        t_end = time.time()
        
        st.session_state.latest_answer = response.content
        st.session_state.latest_time = t_end - t_start
        
        # Append to Chat page conversation history
        st.session_state.chat_history.append({
            "question": question,
            "answer": response.content,
            "time": f"{st.session_state.latest_time:.2f}",
            "retrieved": results
        })
    except Exception as e:
        st.session_state.latest_answer = f"Error generating answer: {str(e)}"
        st.session_state.latest_time = 0.0


# =================================================
# 2. REUSABLE UI COMPONENT RENDERERS
# =================================================

def render_custom_css():
    """Inject premium futuristic dark mode styles & transitions."""
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
        /* Base Palette & Backgrounds */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif !important;
            background-color: #050814 !important;
            color: #f8fafc !important;
        }
        
        /* Remove default Streamlit Headers & Footers */
        header { visibility: hidden !important; }
        footer { visibility: hidden !important; }
        [data-testid="stHeader"] { display: none !important; }
        
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            max-width: 100% !important;
        }
        
        /* Sidebar layout styling */
        [data-testid="stSidebar"] {
            background-color: #070B18 !important;
            border-right: 1px solid rgba(148, 163, 184, 0.15) !important;
            padding: 1.5rem 1rem !important;
        }
        [data-testid="stSidebarCollapseButton"] {
            color: #cbd5e1 !important;
        }
        [data-testid="stSidebar"] hr {
            border-top-color: rgba(148, 163, 184, 0.15) !important;
        }
        
        /* Sidebar Headers */
        .nav-header {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.1em !important;
            color: #94a3b8 !important;
            margin: 24px 0 10px 10px !important;
            text-transform: uppercase;
        }
        
        /* Sidebar Link Buttons */
        div[data-testid="stSidebar"] div.stButton button {
            background-color: transparent !important;
            color: #94a3b8 !important;
            border: none !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 10px 14px !important;
            border-radius: 8px !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            width: 100% !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
        }
        div[data-testid="stSidebar"] div.stButton button:hover {
            background-color: rgba(255, 255, 255, 0.03) !important;
            color: #f8fafc !important;
        }
        div[data-testid="stSidebar"] div.stButton button[data-testid="baseButton-primary"] {
            background: rgba(99, 102, 241, 0.12) !important;
            color: #a5b4fc !important;
            border: 1px solid rgba(99, 102, 241, 0.25) !important;
            font-weight: 600 !important;
        }
        
        /* Gradient File Uploader button */
        div[data-testid="stFileUploader"] section {
            padding: 0 !important;
            border: none !important;
            background-color: transparent !important;
        }
        div[data-testid="stFileUploader"] label {
            display: none !important;
        }
        div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
            padding: 0 !important;
            border: none !important;
            background-color: transparent !important;
        }
        div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] > div {
            display: none !important;
        }
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 11px 20px !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            width: 100% !important;
            box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
            transition: all 0.2s ease !important;
            cursor: pointer;
        }
        div[data-testid="stFileUploader"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
            border: none !important;
        }
        
        /* Uploaded file metadata box */
        .file-details-box {
            background-color: rgba(15, 23, 42, 0.85) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 12px !important;
            padding: 10px 12px !important;
            margin-top: 14px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
        }
        .file-details-left {
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
        }
        .file-icon-wrapper-sidebar {
            background-color: rgba(124, 90, 237, 0.1) !important;
            color: #a78bfa !important;
            width: 32px !important;
            height: 32px !important;
            border-radius: 6px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.1rem !important;
        }
        .file-info {
            display: flex !important;
            flex-direction: column !important;
        }
        .file-name {
            color: #f8fafc !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            max-width: 140px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .file-meta {
            color: #94a3b8 !important;
            font-size: 0.7rem !important;
        }
        .file-success-badge-circle {
            background-color: rgba(34, 197, 94, 0.15) !important;
            color: #22c55e !important;
            border: 1px solid rgba(34, 197, 94, 0.3) !important;
            width: 18px !important;
            height: 18px !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 0.72rem !important;
            font-weight: bold !important;
        }
        
        /* Stats dashboard items in sidebar */
        .sidebar-stat-card {
            background-color: rgba(15, 23, 42, 0.85) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 12px !important;
            padding: 10px 12px !important;
            margin-bottom: 10px !important;
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            transition: all 0.2s ease !important;
        }
        .sidebar-stat-card:hover {
            border-color: rgba(255, 255, 255, 0.05) !important;
        }
        .stat-icon-wrapper {
            border-radius: 8px !important;
            width: 32px !important;
            height: 32px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .stat-content {
            display: flex !important;
            flex-direction: column !important;
        }
        .stat-value {
            font-weight: 700 !important;
            font-size: 1rem !important;
            line-height: 1.2 !important;
        }
        .stat-label {
            color: #94a3b8 !important;
            font-size: 0.7rem !important;
            font-weight: 500 !important;
        }
        
        /* Neon/glow stats metrics mapping */
        .blue-stat .stat-value { color: #60a5fa !important; }
        .blue-stat .stat-icon-wrapper {
            background-color: rgba(96, 165, 250, 0.1) !important;
            color: #60a5fa !important;
        }
        
        .gold-stat .stat-value { color: #fbbf24 !important; }
        .gold-stat .stat-icon-wrapper {
            background-color: rgba(251, 191, 36, 0.1) !important;
            color: #fbbf24 !important;
        }
        
        .green-stat .stat-value { color: #34d399 !important; }
        .green-stat .stat-icon-wrapper {
            background-color: rgba(52, 211, 153, 0.1) !important;
            color: #34d399 !important;
        }
        
        /* Top bar elements styling */
        .mode-badge {
            background-color: rgba(255, 255, 255, 0.01) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 20px !important;
            padding: 4px 12px !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            display: flex !important;
            align-items: center !important;
            gap: 6px !important;
        }
        .online-badge {
            background-color: rgba(34, 197, 94, 0.1) !important;
            color: #22c55e !important;
            border: 1px solid rgba(34, 197, 94, 0.2) !important;
            border-radius: 20px !important;
            padding: 4px 12px !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            display: flex !important;
            align-items: center;
            position: relative;
        }
        
        /* Pulse Animation for Online Badge Indicator */
        @keyframes pulse {
            0% { opacity: 0.4; }
            50% { opacity: 1; }
            100% { opacity: 0.4; }
        }
        .online-badge span.dot {
            animation: pulse 1.6s infinite ease-in-out;
            margin-right: 6px;
            font-size: 0.95rem;
        }
        
        /* Visual Pipeline container */
        .pipeline-container {
            background-color: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
        }
        .pipeline-header h3 {
            margin: 0 !important;
            font-size: 1.05rem !important;
            letter-spacing: 0.05em !important;
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        .pipeline-header p {
            margin: 2px 0 20px 0 !important;
            font-size: 0.82rem !important;
            color: #94a3b8 !important;
        }
        
        .steps-wrapper {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 4px !important;
            width: 100% !important;
        }
        .step-card {
            background-color: rgba(255, 255, 255, 0.01) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            border-radius: 12px !important;
            padding: 14px 10px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
            flex: 1 1 0% !important;
            min-width: 90px !important;
            transition: all 0.3s ease !important;
        }
        
        /* Hover lift animation and neon border */
        .step-card:hover {
            transform: translateY(-4px) !important;
            border-color: rgba(124, 58, 237, 0.45) !important;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.15) !important;
        }
        .step-card.active {
            background-color: rgba(99, 102, 241, 0.02) !important;
            border-color: #6366f1 !important;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.15) !important;
        }
        .step-card.completed {
            background-color: rgba(34, 197, 94, 0.02) !important;
            border-color: #22c55e !important;
        }
        
        .step-icon-wrapper {
            width: 38px !important;
            height: 38px !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin-bottom: 8px !important;
        }
        .step-title {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            margin-bottom: 4px !important;
            white-space: nowrap !important;
        }
        .step-desc {
            font-size: 0.6rem !important;
            color: #94a3b8 !important;
            line-height: 1.2 !important;
        }
        .step-arrow {
            font-size: 0.95rem !important;
            font-weight: bold !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            flex-shrink: 0 !important;
        }
        
        .timeline-wrapper {
            display: flex;
            align-items: center;
            position: relative;
            padding: 0 45px;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        .timeline-track {
            position: absolute;
            height: 4px;
            left: 70px;
            right: 70px;
            z-index: 1;
        }
        .timeline-dots {
            display: flex;
            justify-content: space-between;
            width: 100%;
            z-index: 2;
        }
        .timeline-dot {
            width: 20px;
            height: 20px;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 0.7rem !important;
            color: white !important;
            font-weight: bold !important;
            z-index: 2 !important;
        }
        .timeline-dot.green {
            background-color: #22c55e !important;
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.8) !important;
        }
        .timeline-dot.purple {
            background-color: #7c3aed !important;
            box-shadow: 0 0 10px rgba(124, 58, 237, 0.8) !important;
        }
        .pipeline-status {
            text-align: center;
            font-size: 0.9rem;
            font-weight: 700;
            margin-top: 12px;
        }
        
        /* Cards styled by targeting stVerticalBlockBorderWrapper */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: rgba(15, 23, 42, 0.85) !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
            margin-bottom: 20px !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(124, 58, 237, 0.3) !important;
            box-shadow: 0 4px 24px rgba(124, 58, 237, 0.06) !important;
        }
        
        .card-title {
            font-size: 1rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            margin-bottom: 14px !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
        }
        
        /* Modern search box input columns matching reference image exactly */
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"]:has(input) {
            background-color: #0b0d19 !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 10px !important;
            padding: 4px 8px !important;
            display: flex !important;
            align-items: center !important;
            transition: border-color 0.2s ease !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"]:has(input):focus-within {
            border-color: rgba(124, 58, 237, 0.45) !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
            padding: 0 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"]:has(input) input {
            background-color: transparent !important;
            border: none !important;
            padding: 12px 6px !important;
            font-size: 0.95rem !important;
            color: white !important;
            box-shadow: none !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"]:has(input) label {
            display: none !important;
        }
        
        /* Circular purple/indigo gradient send button inside search bar */
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"]:has(input) button {
            background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
            color: white !important;
            border: none !important;
            border-radius: 50% !important;
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 10px rgba(124, 58, 237, 0.3) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"]:has(input) button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 6px 14px rgba(124, 58, 237, 0.45) !important;
        }
        
        /* Copy action button inside AI Answer block */
        .copy-btn {
            background-color: transparent !important;
            color: #94a3b8 !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 6px !important;
            padding: 4px 10px !important;
            font-size: 0.78rem !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
        }
        .copy-btn:hover {
            color: #ffffff !important;
            background-color: rgba(255, 255, 255, 0.03) !important;
            border-color: #94a3b8 !important;
        }
        
        /* Similarity result card layout styling */
        .context-list {
            display: flex !important;
            flex-direction: column !important;
            gap: 12px !important;
        }
        .context-item {
            background-color: rgba(15, 23, 42, 0.4) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            border-radius: 10px !important;
            padding: 12px !important;
            display: flex !important;
            flex-direction: column !important;
            gap: 8px !important;
            transition: all 0.2s ease !important;
        }
        .context-item:hover {
            transform: translateX(4px) !important;
            border-color: rgba(148, 163, 184, 0.25) !important;
        }
        .context-header {
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
        }
        .context-rank {
            width: 20px !important;
            height: 20px !important;
            border-radius: 50% !important;
            color: white !important;
            font-size: 0.75rem !important;
            font-weight: bold !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .context-text-preview {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: #f8fafc !important;
        }
        .context-score {
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            padding: 2px 8px !important;
            border-radius: 12px !important;
        }
        .context-body {
            display: flex !important;
            flex-direction: column !important;
            gap: 4px !important;
        }
        .context-snippet {
            margin: 0 !important;
            font-size: 0.8rem !important;
            color: #94a3b8 !important;
            line-height: 1.45 !important;
        }
        .context-page {
            font-size: 0.72rem !important;
            color: #475569 !important;
            font-weight: 500 !important;
        }
        
        /* Source info cards */
        .source-info-card {
            background-color: rgba(15, 23, 42, 0.4) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            border-radius: 12px !important;
            padding: 12px 14px !important;
            display: flex !important;
            align-items: center !important;
            gap: 14px !important;
            position: relative !important;
        }
        .source-icon-wrapper {
            background-color: rgba(236, 72, 153, 0.1) !important;
            color: #ec4899 !important;
            font-size: 1.5rem !important;
            width: 42px !important;
            height: 42px !important;
            border-radius: 8px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .source-details {
            display: flex !important;
            flex-direction: column !important;
            flex-grow: 1 !important;
        }
        .source-filename {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            margin-bottom: 2px !important;
            max-width: 160px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .source-meta {
            color: #f8fafc !important;
            font-size: 0.75rem !important;
            margin-bottom: 2px !important;
        }
        .source-time {
            color: #94a3b8 !important;
            font-size: 0.7rem !important;
        }
        .source-status-badge {
            background-color: rgba(34, 197, 94, 0.1) !important;
            color: #22c55e !important;
            border: 1px solid rgba(34, 197, 94, 0.2) !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            padding: 3px 8px !important;
            border-radius: 20px !important;
        }
        
        /* Conversational chat bubble cards */
        .chat-bubble {
            padding: 12px 16px !important;
            border-radius: 12px !important;
            font-size: 0.9rem !important;
            color: #e2e8f0 !important;
            line-height: 1.5 !important;
        }
        .chat-avatar {
            width: 32px !important;
            height: 32px !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 0.95rem !important;
            flex-shrink: 0 !important;
        }
        
        /* Selectbox and slider customization */
        div[data-testid="stSlider"] [data-baseweb="slider"] {
            color: #7c3aed !important;
        }
        div[data-testid="stSelectbox"] select, div[data-testid="stSelectbox"] div[role="button"] {
            background-color: #0b0d19 !important;
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            color: #f8fafc !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_sidebar_stat_box(value, label, icon_svg, color_class):
    """HTML stat card template inside sidebar."""
    return f"""
    <div class="sidebar-stat-card {color_class}">
        <div class="stat-icon-wrapper">
            {icon_svg}
        </div>
        <div class="stat-content">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
    </div>
    """

def get_timeline_html(current_step):
    """Generates the progress track and neon-glowing checkbox checkpoints."""
    dots = []
    for i in range(1, 8):
        if i < 7:
            if i < current_step:
                dots.append('<div class="timeline-dot green">✓</div>')
            elif i == current_step:
                dots.append(f'<div class="timeline-dot green">{i}</div>')
            else:
                dots.append(f'<div class="timeline-dot" style="background-color: #161a26; color: #475569; box-shadow: none;">{i}</div>')
        else:
            if current_step >= 7:
                dots.append('<div class="timeline-dot purple">✓</div>')
            elif current_step == 7:
                dots.append(f'<div class="timeline-dot purple">{i}</div>')
            else:
                dots.append(f'<div class="timeline-dot" style="background-color: #161a26; color: #475569; box-shadow: none;">{i}</div>')
                
    # Ingestion lines filling up
    if current_step >= 7:
        track_bg = "linear-gradient(to right, #22c55e 83%, #7c3aed 17%)"
    elif current_step >= 6:
        track_bg = "linear-gradient(to right, #22c55e 66%, #161a26 34%)"
    elif current_step == 5:
        track_bg = "linear-gradient(to right, #22c55e 50%, #161a26 50%)"
    elif current_step == 4:
        track_bg = "linear-gradient(to right, #22c55e 33%, #161a26 67%)"
    elif current_step == 3:
        track_bg = "linear-gradient(to right, #22c55e 16%, #161a26 84%)"
    else:
        track_bg = "#161a26"
        
    return f"""
    <div class="timeline-wrapper">
        <div class="timeline-track" style="background: {track_bg};"></div>
        <div class="timeline-dots">
            {"".join(dots)}
        </div>
    </div>
    """

def get_stepper_html(current_step):
    """Renders the horizontal RAG pipeline workflow chart matching the spec exact text."""
    steps = [
        {"num": 1, "name": "1. PDF Upload", "desc": "Upload your PDF document", "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.2 15c.7-1.2 1-2.5.7-3.9-.3-2-1.5-3.8-3.2-4.7C18 3 14.8.8 11.2 1.4 8.5 1.9 6.2 3.8 5.3 6.4c-2 .3-3.6 1.8-4 3.8-.4 2.2.8 4.3 2.8 5.1"></path><path d="M12 12v9"></path><path d="m15 15-3-3-3 3"></path></svg>', "color": "#a78bfa"},
        {"num": 2, "name": "2. Extract Text", "desc": "Text extracted using PyPDF", "icon": doc_lines_svg, "color": "#60a5fa"},
        {"num": 3, "name": "3. Create Chunks", "desc": "Semantic chunks", "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"></circle><circle cx="6" cy="18" r="3"></circle><line x1="20" y1="4" x2="8.12" y2="15.88"></line><line x1="14.47" y1="14.48" x2="20" y2="20"></line><line x1="8.12" y1="8.12" x2="12" y2="12"></line></svg>', "color": "#06b6d4"},
        {"num": 4, "name": "4. Embeddings", "desc": "Generate 384D vectors", "icon": cube_svg, "color": "#34d399"},
        {"num": 5, "name": "5. Vector Store", "desc": "Store in ChromaDB", "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"></path></svg>', "color": "#fbbf24"},
        {"num": 6, "name": "6. Similarity Search", "desc": "Find relevant chunks", "icon": search_svg, "color": "#fb923c"},
        {"num": 7, "name": "7. LLM Groq", "desc": "Generate answer", "icon": robot_svg, "color": "#f472b6"}
    ]
    
    steps_html = ""
    for idx, step in enumerate(steps):
        is_completed = step["num"] < current_step
        is_active = step["num"] == current_step
        
        status_class = "completed" if is_completed else ("active" if is_active else "pending")
        border_color = "#22c55e" if is_completed else ("#7c3aed" if is_active else "rgba(148, 163, 184, 0.2)")
        text_color = "#f8fafc" if (is_completed or is_active) else "#64748b"
        icon_color = step["color"] if (is_completed or is_active) else "#475569"
        bg_circle = step["color"] + "15" if (is_completed or is_active) else "transparent"
        border_circle = step["color"] + "44" if (is_completed or is_active) else "rgba(148, 163, 184, 0.2)"
        
        steps_html += f"""
        <div class="step-card {status_class}" style="border-color: {border_color};">
            <div class="step-icon-wrapper" style="background-color: {bg_circle}; border: 1px solid {border_circle}; color: {icon_color};">
                {step["icon"]}
            </div>
            <div class="step-title" style="color: {text_color};">{step["name"]}</div>
            <div class="step-desc">{step["desc"]}</div>
        </div>
        """
        
        if idx < 6:
            if step["num"] < current_step:
                arrow_color = "#22c55e"
            elif step["num"] == current_step:
                arrow_color = "#7c3aed"
            else:
                arrow_color = "rgba(148, 163, 184, 0.2)"
            steps_html += f"""
            <div class="step-arrow" style="color: {arrow_color};">➔</div>
            """
            
    timeline_html = get_timeline_html(current_step)
    
    # Progress Badge rendering
    status_html = ""
    if current_step >= 6:
        status_html = """
        <div style="display: flex; align-items: center; justify-content: center; gap: 6px; font-weight: 700; color: #22c55e; font-size: 0.95rem; margin-top: 15px;">
            Pipeline Ready 
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="background: rgba(34,197,94,0.1); border-radius: 50%; padding: 2px;"><polyline points="20 6 9 17 4 12"></polyline></svg>
        </div>
        """
    else:
        status_msg = f"Processing Pipeline... Step {current_step}/5" if st.session_state.pdf_file_name else "Pipeline Idle"
        status_color = "#7c3aed" if st.session_state.pdf_file_name else "#475569"
        status_html = f"""
        <div style="display: flex; align-items: center; justify-content: center; gap: 6px; font-weight: 700; color: {status_color}; font-size: 0.95rem; margin-top: 15px;">
            ● {status_msg}
        </div>
        """
        
    return f"""
    <div class="pipeline-container">
        <div class="pipeline-header">
            <h3>RAG PIPELINE</h3>
            <p>From PDF to Intelligent Answers</p>
        </div>
        <div class="steps-wrapper">
            {steps_html}
        </div>
        {timeline_html}
        {status_html}
    </div>
    """


# =================================================
# 3. INITIAL STATE SETUP
# =================================================

# State mappings
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "processed" not in st.session_state:
    st.session_state.processed = False

# Hyper-parameter defaults
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "llama-3.3-70b-versatile"
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
if "custom_groq_api_key" not in st.session_state:
    st.session_state.custom_groq_api_key = ""

# File metadata
if "pdf_file_name" not in st.session_state:
    st.session_state.pdf_file_name = None
if "pdf_file_size" not in st.session_state:
    st.session_state.pdf_file_size = None
if "pdf_num_pages" not in st.session_state:
    st.session_state.pdf_num_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "vector_dimension" not in st.session_state:
    st.session_state.vector_dimension = 0
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Query values
if "latest_question" not in st.session_state:
    st.session_state.latest_question = ""
if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""
if "latest_time" not in st.session_state:
    st.session_state.latest_time = 0.0
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =================================================
# 4. SIDEBAR RENDERING (Left fixed bar)
# =================================================
render_custom_css()

# Sidebar Header Branding
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding-left: 5px;">
        <span style="font-size: 2.1rem; filter: drop-shadow(0 0 10px rgba(124, 58, 237, 0.45));">🤖</span>
        <div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">RAG PDF Chatbot</div>
            <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 500;">Powered by Groq + Llama 3.3</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# File Uploader Section
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type="pdf",
    key="pdf_uploader_widget"
)

# File upload trigger action
if uploaded_file:
    if st.session_state.pdf_file_name != uploaded_file.name:
        st.session_state.processed = False
        st.session_state.current_step = 1
        st.session_state.pdf_file_name = uploaded_file.name
        
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.session_state.pdf_file_size = f"{file_size_mb:.1f} MB"
        st.rerun()

# Document Info Status Card
if st.session_state.pdf_file_name:
    st.sidebar.markdown(
        f"""
        <div class="file-details-box">
            <div class="file-details-left">
                <div class="file-icon-wrapper-sidebar">📄</div>
                <div class="file-info">
                    <span class="file-name" title="{st.session_state.pdf_file_name}">{st.session_state.pdf_file_name}</span>
                    <span class="file-meta">{st.session_state.pdf_file_size} • {st.session_state.pdf_num_pages} pages</span>
                </div>
            </div>
            <div class="file-success-badge-circle">✓</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div style="border: 1px dashed rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 14px; text-align: center; font-size: 0.78rem; color: #94a3b8; margin-top: 10px;">
            No PDF document uploaded
        </div>
        """,
        unsafe_allow_html=True
    )

# Sidebar Page Routing Menu
st.sidebar.markdown('<div class="nav-header">NAVIGATION</div>', unsafe_allow_html=True)

nav_options = [
    {"page": "Dashboard", "icon": "🏠"},
    {"page": "Chat", "icon": "💬"},
    {"page": "Documents", "icon": "📄"},
    {"page": "Settings", "icon": "⚙️"}
]

for opt in nav_options:
    is_active = st.session_state.active_page == opt["page"]
    if st.sidebar.button(
        f"{opt['icon']}  {opt['page']}",
        key=f"nav_{opt['page']}",
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        st.session_state.active_page = opt["page"]
        st.rerun()

# Stats Dashboard Items
st.sidebar.markdown('<div class="nav-header">STATS</div>', unsafe_allow_html=True)

stats_html = ""

# SVG Icons
cube_svg = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>"""
grid_svg = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>"""
search_svg = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>"""
robot_svg = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="M12 7v4"></path><line x1="8" y1="16" x2="8.01" y2="16"></line><line x1="16" y1="16" x2="16.01" y2="16"></line></svg>"""
doc_lines_svg = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><line x1="10" y1="9" x2="8" y2="9"></line></svg>"""

# 1. Total Chunks (blue)
stats_html += render_sidebar_stat_box(
    st.session_state.total_chunks,
    "Total Chunks",
    cube_svg,
    "blue-stat"
)

# 2. Vector Dimension (blue)
stats_html += render_sidebar_stat_box(
    st.session_state.vector_dimension,
    "Vector Dimension",
    grid_svg,
    "blue-stat"
)

# 3. Top K Results (gold)
stats_html += render_sidebar_stat_box(
    st.session_state.top_k,
    "Top K Results",
    search_svg,
    "gold-stat"
)

# 4. LLM Model (green)
model_short_name = "Llama 3.3 70B"
if "llama3-8b" in st.session_state.llm_model:
    model_short_name = "Llama 3 8B"
elif "mixtral" in st.session_state.llm_model:
    model_short_name = "Mixtral 8x7B"

stats_html += render_sidebar_stat_box(
    model_short_name,
    "LLM Model",
    robot_svg,
    "green-stat"
)

st.sidebar.markdown(stats_html, unsafe_allow_html=True)

# Footer Built-With
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style="font-size: 0.65rem; color: #94a3b8; text-align: center; font-weight: 500;">
        Built using: <br/> Streamlit • LangChain • ChromaDB<br/>HuggingFace • Groq
    </div>
    """,
    unsafe_allow_html=True
)


# =================================================
# 5. HEADER BAR VIEW
# =================================================
def render_top_bar_badges():
    api_key_check = os.getenv("GROQ_API_KEY", "") or st.session_state.custom_groq_api_key.strip()
    status_text = "Online" if api_key_check else "Key Missing"
    status_color = "rgba(34, 197, 94, 0.1)" if api_key_check else "rgba(239, 68, 68, 0.1)"
    status_text_color = "#22c55e" if api_key_check else "#ef4444"
    status_border = "rgba(34, 197, 94, 0.2)" if api_key_check else "rgba(239, 68, 68, 0.2)"
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; align-items: center; gap: 12px; margin-bottom: 16px; width: 100%;">
            <div class="mode-badge">🌙 Dark</div>
            <div class="online-badge" style="background-color: {status_color} !important; color: {status_text_color} !important; border-color: {status_border} !important; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                <span class="dot">●</span>{status_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =================================================
# 6. ROUTER VIEW RENDERING
# =================================================

# TAB 1: DASHBOARD
if st.session_state.active_page == "Dashboard":
    render_top_bar_badges()
    
    # Stepper component instantiation
    stepper_placeholder = st.empty()
    st.session_state["stepper_placeholder"] = stepper_placeholder
    stepper_placeholder.markdown(get_stepper_html(st.session_state.current_step), unsafe_allow_html=True)
    
    # Run pipeline automatically on PDF upload
    if uploaded_file and not st.session_state.processed:
        with st.spinner("Executing RAG Pipeline ingestion steps..."):
            index_pdf_document(uploaded_file)
            
    # Page Main Layout Grid
    col_left, col_right = st.columns([7, 5])
    
    with col_left:
        # Card 1: Ask Anything console
        st.markdown('<div class="card-title">💬 Ask Anything from Your PDF</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if not st.session_state.processed:
                st.info("⚠️ Pipeline is idle. Upload a document in the sidebar to begin searching.")
            else:
                col_inp, col_sbt = st.columns([12, 1])
                with col_inp:
                    question = st.text_input(
                        "Ask a question...",
                        placeholder="Ask a question...",
                        label_visibility="collapsed",
                        key="ask_question_input"
                    )
                with col_sbt:
                    submit_query = st.button("➔", key="ask_question_btn")
                    
                if submit_query and question.strip():
                    st.session_state.latest_question = question
                    
                    # Update Visual Stepper to Search (6)
                    st.session_state.current_step = 6
                    stepper_placeholder.markdown(get_stepper_html(6), unsafe_allow_html=True)
                    
                    with st.spinner("Querying vector databases..."):
                        # Execute Similarity Query
                        query_rag_pipeline(question)
                        
                    # Update Visual Stepper to Answer (7)
                    st.session_state.current_step = 7
                    stepper_placeholder.markdown(get_stepper_html(7), unsafe_allow_html=True)
                    st.rerun()
                    
        # Card 2: AI Answer Block
        st.markdown('<div class="card-title">🤖 AI Answer</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if not st.session_state.latest_answer:
                st.write(
                    '<div style="color: #94a3b8; font-style: italic; text-align: center; padding: 20px 0;">'
                    'Submit a question above to generate an AI response.'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                # Flow raw text directly as markdown
                st.markdown(st.session_state.latest_answer)
                st.markdown("<hr style='border-top: 1px solid rgba(148, 163, 184, 0.15); margin: 12px 0 8px 0;'>", unsafe_allow_html=True)
                
                # Metadata Footer
                col_meta, col_copy = st.columns([8, 2])
                with col_meta:
                    st.markdown(
                        f"""
                        <div style="color: #fbbf24; font-size: 0.75rem; font-weight: 500; display: flex; align-items: center; gap: 4px;">
                            ⚡ Generated by {model_short_name} (Groq) &nbsp;•&nbsp; ⏱️ Response time: {st.session_state.latest_time:.2f}s
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_copy:
                    escaped_answer = html.escape(st.session_state.latest_answer).replace('`','\\`').replace('\n','\\n')
                    copy_html = f"""
                    <div style="text-align: right;">
                        <button class="copy-btn" onclick="navigator.clipboard.writeText(`{escaped_answer}`); this.innerText='✓ Copied'; setTimeout(() => this.innerText='📋 Copy', 2000);">
                            📋 Copy
                        </button>
                    </div>
                    """
                    st.markdown(copy_html, unsafe_allow_html=True)
                    
    with col_right:
        # Card 3: Similarity Match results
        st.markdown('<div class="card-title">🔍 Retrieved Context (Top 3)</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if not st.session_state.retrieved_docs:
                st.write(
                    '<div style="color: #94a3b8; font-style: italic; text-align: center; padding: 20px 0;">'
                    'No context segments retrieved.'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                context_html = '<div class="context-list">'
                for idx, (doc, score) in enumerate(st.session_state.retrieved_docs):
                    page_num = doc.metadata.get("page", 0) + 1
                    score_val = f"{score:.2f}"
                    snippet = doc.page_content
                    if len(snippet) > 130:
                        snippet = snippet[:130] + "..."
                        
                    colors = [
                        {"badge": "#22c55e", "bg": "rgba(34, 197, 94, 0.1)"},  # Result 1: green
                        {"badge": "#60a5fa", "bg": "rgba(96, 165, 250, 0.1)"}, # Result 2: blue
                        {"badge": "#7c3aed", "bg": "rgba(124, 58, 237, 0.1)"}  # Result 3: purple
                    ]
                    color = colors[idx % len(colors)]
                    
                    context_html += f"""
                    <div class="context-item" style="border-left: 3px solid {color["badge"]};">
                        <div class="context-header">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="context-rank" style="background-color: {color["badge"]};">{idx+1}</span>
                                <span class="context-text-preview">Snippet #{idx+1}</span>
                            </div>
                            <span class="context-score" style="color: {color["badge"]}; background-color: {color["bg"]};">{score_val}</span>
                        </div>
                        <div class="context-body">
                            <p class="context-snippet">"{html.escape(snippet)}"</p>
                            <span class="context-page">Page {page_num}</span>
                        </div>
                    </div>
                    """
                context_html += '</div>'
                st.markdown(context_html, unsafe_allow_html=True)
                
        # Card 4: Source Document metadata card
        st.markdown('<div class="card-title">📄 Source Document Info</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if not st.session_state.pdf_file_name:
                st.write(
                    '<div style="color: #94a3b8; font-style: italic; text-align: center; padding: 20px 0;">'
                    'Upload a PDF document to view metrics.'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="source-info-card">
                        <div class="source-icon-wrapper">📄</div>
                        <div class="source-details">
                            <div class="source-filename" title="{st.session_state.pdf_file_name}">{st.session_state.pdf_file_name}</div>
                            <div class="source-meta">{st.session_state.pdf_file_size} • {st.session_state.pdf_num_pages} pages • {st.session_state.total_chunks} chunks</div>
                            <div class="source-time">Uploaded: Just now</div>
                        </div>
                        <div class="source-status-badge">
                            Processed
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
    # Centered Page Footer
    st.markdown("<hr style='border-top: 1px solid rgba(148, 163, 184, 0.15); margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; font-weight: 500;">
            RAG PDF Chatbot • Built with Streamlit 🚀
        </div>
        """,
        unsafe_allow_html=True
    )

# TAB 2: CHAT INTERFACE
elif st.session_state.active_page == "Chat":
    render_top_bar_badges()
    
    if not st.session_state.processed:
        st.info("⚠️ Please upload a PDF in the sidebar to initialize chat capabilities.")
    else:
        chat_container = st.container(height=500, border=True)
        
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    """
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #94a3b8; padding: 40px 0;">
                        <span style="font-size: 3rem;">💬</span>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin-top: 10px;">Document Chat Session</div>
                        <div style="font-size: 0.85rem; max-width: 350px; text-align: center; margin-top: 5px;">Ask a question regarding the uploaded PDF document below.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                for chat in st.session_state.chat_history:
                    # User dialogue bubble
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-end; margin-bottom: 16px; gap: 10px; width: 100%;">
                            <div class="chat-bubble" style="background-color: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); max-width: 70%; border-bottom-right-radius: 2px;">
                                {html.escape(chat["question"])}
                            </div>
                            <div class="chat-avatar" style="background-color: #3b82f6;">👤</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Assistant dialogue bubble
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 16px; gap: 10px; width: 100%;">
                            <div class="chat-avatar" style="background-color: #7c3aed;">🤖</div>
                            <div class="chat-bubble" style="background-color: rgba(15, 23, 42, 0.85); border: 1px solid rgba(148, 163, 184, 0.2); max-width: 70%; border-bottom-left-radius: 2px;">
                                {chat["answer"]}
                                <div style="font-size: 0.72rem; color: #fbbf24; margin-top: 8px; font-weight: 500;">
                                    ⚡ {model_short_name} • ⏱️ Response time: {chat["time"]}s
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
        chat_query = st.chat_input("Ask a question about the PDF document...")
        
        if chat_query:
            with st.spinner("Finding document contexts..."):
                query_rag_pipeline(chat_query)
            st.rerun()

# TAB 3: DOCUMENTS DETAILED EXPLORER
elif st.session_state.active_page == "Documents":
    render_top_bar_badges()
    
    if not st.session_state.processed:
        st.info("⚠️ Please upload a PDF in the sidebar to browse text chunks.")
    else:
        st.markdown('<div class="card-title">📂 Document Chunk Explorer</div>', unsafe_allow_html=True)
        with st.container(border=True):
            col_search, col_pages = st.columns([7, 3])
            with col_search:
                search_term = st.text_input("Search term", placeholder="Search within chunks...", key="chunk_search")
            with col_pages:
                st.markdown(
                    f"""
                    <div style="text-align: right; font-size: 0.88rem; margin-top: 10px; color: #94a3b8;">
                        Pages: <strong>{st.session_state.pdf_num_pages}</strong> &nbsp;|&nbsp; Chunks: <strong>{st.session_state.total_chunks}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            filtered_chunks = []
            for i, chunk in enumerate(st.session_state.chunks):
                if not search_term or search_term.lower() in chunk.lower():
                    filtered_chunks.append((i+1, chunk))
                    
            if not filtered_chunks:
                st.warning("No chunks found matching search query.")
            else:
                # Dynamic page navigation
                chunks_per_page = 5
                total_filtered = len(filtered_chunks)
                num_chunk_pages = max(1, (total_filtered + chunks_per_page - 1) // chunks_per_page)
                
                if num_chunk_pages > 1:
                    chunk_page = st.select_slider(
                        "Page Selector",
                        options=list(range(1, num_chunk_pages + 1)),
                        value=1,
                        key="chunk_page_slider"
                    )
                else:
                    chunk_page = 1
                    
                start_idx = (chunk_page - 1) * chunks_per_page
                end_idx = min(start_idx + chunks_per_page, total_filtered)
                
                st.markdown(
                    f'<div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">Showing chunks {start_idx+1} to {end_idx} of {total_filtered}</div>',
                    unsafe_allow_html=True
                )
                
                for idx, chunk_text in filtered_chunks[start_idx:end_idx]:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(15, 23, 42, 0.4); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(148, 163, 184, 0.15); padding-bottom: 8px; margin-bottom: 8px;">
                                <span style="font-weight: 700; color: #a78bfa; font-size: 0.9rem;">Chunk #{idx}</span>
                                <span style="background-color: rgba(124, 58, 237, 0.1); color: #a78bfa; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">Size: {len(chunk_text)} chars</span>
                            </div>
                            <div style="color: #e2e8f0; font-size: 0.85rem; line-height: 1.5; font-style: italic;">
                                "{html.escape(chunk_text)}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# TAB 4: APPLICATION SETTINGS
elif st.session_state.active_page == "Settings":
    render_top_bar_badges()
    
    st.markdown('<div class="card-title">⚙️ Hyper-parameters & Model Setup</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Dropdowns & key configurations
            model_options = {
                "llama-3.3-70b-versatile": "Llama 3.3 70B (Groq, Versatile)",
                "llama3-8b-8192": "Llama 3 8B (Groq, Fast)",
                "mixtral-8x7b-32768": "Mixtral 8x7B (Groq, Mixture of Experts)"
            }
            try:
                model_idx = list(model_options.keys()).index(st.session_state.llm_model)
            except ValueError:
                model_idx = 0
                
            selected_model_val = st.selectbox(
                "LLM Model",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=model_idx
            )
            st.session_state.llm_model = selected_model_val
            
            st.session_state.custom_groq_api_key = st.text_input(
                "Custom Groq API Key (Optional)",
                value=st.session_state.custom_groq_api_key,
                type="password",
                placeholder="Leave blank to use default key"
            )
            
            st.session_state.temperature = st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.05
            )
            
        with col2:
            st.session_state.chunk_size = st.slider(
                "Text Chunk Size",
                min_value=500,
                max_value=2000,
                value=st.session_state.chunk_size,
                step=100
            )
            st.session_state.chunk_overlap = st.slider(
                "Text Chunk Overlap",
                min_value=50,
                max_value=500,
                value=st.session_state.chunk_overlap,
                step=25
            )
            st.session_state.top_k = st.slider(
                "Retrieved Matches (K)",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                step=1
            )
            
        st.markdown("<hr style='border-top: 1px solid rgba(148, 163, 184, 0.15);'>", unsafe_allow_html=True)
        
        # State Reset trigger
        col_reset_left, col_reset_right = st.columns([8, 2])
        with col_reset_right:
            clear_db = st.button("Reset Database", type="secondary")
            if clear_db:
                st.session_state.processed = False
                st.session_state.current_step = 1
                st.session_state.pdf_file_name = None
                st.session_state.pdf_file_size = None
                st.session_state.pdf_num_pages = 0
                st.session_state.total_chunks = 0
                st.session_state.vector_dimension = 0
                st.session_state.chunks = []
                st.session_state.vector_store = None
                st.session_state.latest_question = ""
                st.session_state.latest_answer = ""
                st.session_state.latest_time = 0.0
                st.session_state.retrieved_docs = []
                st.session_state.chat_history = []
                
                st.success("State reset successfully!")
                st.rerun()
        with col_reset_left:
            st.markdown(
                """
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">
                    Changing chunk parameters will apply to the next PDF document upload.
                </div>
                """,
                unsafe_allow_html=True
            )
