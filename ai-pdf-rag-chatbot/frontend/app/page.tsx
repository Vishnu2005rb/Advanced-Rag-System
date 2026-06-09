"use client";

import React, { useState, useEffect } from "react";
import { Bot, Sun, Moon, Sparkles, Layers, Binary, Cpu, Trash2, Sliders, RefreshCw, FileText, CheckCircle2, Search, BarChart3, HelpCircle, Menu } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

import Sidebar from "../components/Sidebar";
import Pipeline from "../components/Pipeline";
import ChatBox, { ChatMessage } from "../components/ChatBox";
import RetrievedContext from "../components/RetrievedContext";
import RAGComparison from "../components/RAGComparison";
import { api, DocumentInfo, RagasScores, SourceReference } from "../lib/api";

export default function DashboardPage() {
  // Navigation & UI States
  const [activePage, setActivePage] = useState("Dashboard");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [onlineStatus, setOnlineStatus] = useState(true);
  const [mounted, setMounted] = useState(false);

  // Ingestion & RAG States
  const [uploadedDoc, setUploadedDoc] = useState<DocumentInfo | null>(null);
  const [documentList, setDocumentList] = useState<DocumentInfo[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [currentStep, setCurrentStep] = useState(1); // 1 to 9
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Stats
  const [totalChunks, setTotalChunks] = useState(0);
  const [vectorDimension, setVectorDimension] = useState(0);
  
  // Advanced RAG Configurations (States)
  const [embeddingModel, setEmbeddingModel] = useState("BAAI/bge-small-en-v1.5");
  const [chunkerStrategy, setChunkerStrategy] = useState("recursive"); // recursive, semantic
  const [retrieverType, setRetrieverType] = useState("hybrid"); // similarity, mmr, hybrid
  const [scoreThreshold, setScoreThreshold] = useState(0.70);
  const [enableQueryRewrite, setEnableQueryRewrite] = useState(true);
  const [enableReranking, setEnableReranking] = useState(true);
  const [enableMemory, setEnableMemory] = useState(true);
  const [runEvaluation, setRunEvaluation] = useState(true);
  const [topK, setTopK] = useState(5);
  const [fetchK, setFetchK] = useState(20);
  
  // Advanced Chunk Parameters
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [temperature, setTemperature] = useState(0.3);
  const [customGroqKey, setCustomGroqKey] = useState("");
  const [llmModelName, setLlmModelName] = useState("llama-3.3-70b-versatile");

  // Chat Feed History
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [retrievedSources, setRetrievedSources] = useState<SourceReference[]>([]);
  const [fullRetrievedChunks, setFullRetrievedChunks] = useState<SourceReference[]>([]);
  const [ragasScores, setRagasScores] = useState<RagasScores | null>(null);
  const [rewrittenQuery, setRewrittenQuery] = useState("");

  // Force dark mode permanently
  useEffect(() => {
    document.documentElement.classList.add("dark");
    document.documentElement.style.colorScheme = "dark";
  }, []);

  // Close mobile sidebar on page navigation
  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activePage]);

  // Load stats on mount
  useEffect(() => {
    setMounted(true);
    fetchInitialStats();
  }, [embeddingModel]); // Reload stats if the embedding model is toggled to match dimension isolations

  const fetchInitialStats = async () => {
    try {
      const stats = await api.getStats(embeddingModel);
      const docs = await api.getDocuments();
      
      setDocumentList(docs);
      setTotalChunks(stats.total_chunks);
      setVectorDimension(stats.vector_dimension);
      
      // Filter documents belonging to the active embedding model
      const modelDocs = docs.filter((d) => d.embedding_model === embeddingModel);
      if (modelDocs.length > 0) {
        const active = modelDocs[modelDocs.length - 1];
        setUploadedDoc(active);
        setCurrentStep(8); // Ready on reasoning state
      } else {
        setUploadedDoc(null);
        setCurrentStep(1);
      }
      setOnlineStatus(true);
    } catch (err) {
      console.warn("Backend server not started or unreachable yet.");
      setOnlineStatus(false);
    }
  };

  // Upload trigger helpers
  const handleUploadStart = () => {
    setIsUploading(true);
    setErrorMsg(null);
    setCurrentStep(1);
  };

  const handleUploadSuccess = async (
    filename: string,
    pages: number,
    chunks: number,
    vectorDim: number
  ) => {
    const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
    
    setCurrentStep(2); // Extract Text
    await delay(300);
    setCurrentStep(3); // Chunking
    await delay(300);
    setCurrentStep(4); // Embeddings
    await delay(300);
    setCurrentStep(5); // Vector storing (ChromaDB)
    await delay(300);
    
    const sizeMb = "2.4 MB"; // Mock sizing
    const newDoc: DocumentInfo = {
      filename,
      size: sizeMb,
      pages,
      chunks,
      embedding_model: embeddingModel,
      vector_count: chunks,
      status: "Processed",
      timestamp: Date.now() / 1000,
    };
    
    setUploadedDoc(newDoc);
    setTotalChunks((prev) => prev + chunks);
    setVectorDimension(vectorDim);
    setIsUploading(false);
    setCurrentStep(8); // Ready on Reasoning state (since query transform & retrievers run at search time)

    try {
      const docs = await api.getDocuments();
      setDocumentList(docs);
    } catch (e) {
      setDocumentList((prev) => [...prev, newDoc]);
    }
  };

  const handleUploadError = (error: string) => {
    setIsUploading(false);
    setErrorMsg(error);
    setCurrentStep(1);
  };

  // Chat message query runner
  const handleSendMessage = async (text: string) => {
    if (isUploading || isQuerying) return;
    
    const userMsg: ChatMessage = {
      id: Math.random().toString(36).substring(7),
      sender: "user",
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setChatMessages((prev) => [...prev, userMsg]);
    setIsQuerying(true);
    
    // Animate pipeline stages
    if (enableQueryRewrite) {
      setCurrentStep(5); // Query Rewrite
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
    
    setCurrentStep(6); // Retrieving (Hybrid search BM25 + Vector)
    await new Promise((resolve) => setTimeout(resolve, 300));

    if (enableReranking) {
      setCurrentStep(7); // Reranking (BGE cross-encoder)
      await new Promise((resolve) => setTimeout(resolve, 300));
    }

    setCurrentStep(8); // Reasoning (LLM Llama 3.3 execution)

    try {
      const filterFilename = uploadedDoc?.filename || undefined;
      
      const config = {
        sessionId: "default_session",
        embeddingModel,
        retrieverType,
        scoreThreshold,
        enableQueryRewrite,
        enableReranking,
        enableMemory,
        runEvaluation,
        topK,
        fetchK,
      };

      const response = await api.askQuestion(
        text,
        filterFilename,
        llmModelName,
        temperature,
        customGroqKey || undefined,
        config
      );

      // Auditing Evaluation Scores
      if (runEvaluation) {
        setCurrentStep(9); // RAGAS Evaluation
      }

      setRewrittenQuery(response.rewritten_query || "");
      setRetrievedSources(response.sources);
      setFullRetrievedChunks(response.retrieved_chunks);
      setRagasScores(response.scores || null);
      
      const aiMsg: ChatMessage = {
        id: Math.random().toString(36).substring(7),
        sender: "ai",
        text: response.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        timeTaken: response.time_taken,
        modelUsed: response.model_used,
        sources: response.sources,
      };

      setChatMessages((prev) => [...prev, aiMsg]);

      // Reset pipeline state back to active grounding after brief delay
      setTimeout(() => setCurrentStep(8), 1500);

    } catch (err: any) {
      const aiErrorMsg: ChatMessage = {
        id: Math.random().toString(36).substring(7),
        sender: "ai",
        text: `Error contacting RAG backend: ${err.message}. Ensure GROQ_API_KEY is configured correctly in the backend.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setChatMessages((prev) => [...prev, aiErrorMsg]);
    } finally {
      setIsQuerying(false);
    }
  };

  // Reset database helper
  const handleClearAll = async () => {
    const confirm = window.confirm("Are you sure you want to delete all indexed document chunks from ChromaDB and clear chat memory?");
    if (!confirm) return;

    try {
      await api.clearAll();
      setUploadedDoc(null);
      setDocumentList([]);
      setTotalChunks(0);
      setVectorDimension(0);
      setChatMessages([]);
      setRetrievedSources([]);
      setFullRetrievedChunks([]);
      setRagasScores(null);
      setRewrittenQuery("");
      setCurrentStep(1);
    } catch (err) {
      alert("Failed to reset storage");
    }
  };

  const handleDeleteDoc = async (filename: string) => {
    const confirm = window.confirm(`Remove document index metadata for ${filename}?`);
    if (!confirm) return;

    try {
      await api.deleteDocument(filename, embeddingModel);
      fetchInitialStats();
    } catch (err) {
      alert("Failed to delete document from index");
    }
  };

  if (!mounted) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[#050814] text-slate-100 font-sans">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 rounded-full border-4 border-purple-500/10 border-t-purple-500 animate-spin" />
          <p className="text-[10px] font-bold text-purple-400 uppercase tracking-widest animate-pulse">
            Initializing AI Sandbox...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#050814] text-slate-100 font-sans">
      {/* LEFT SIDEBAR PANEL */}
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        uploadedDoc={uploadedDoc}
        onUploadStart={handleUploadStart}
        onUploadSuccess={handleUploadSuccess}
        onUploadError={handleUploadError}
        isUploading={isUploading}
        totalChunks={totalChunks}
        vectorDimension={vectorDimension}
        topK={topK}
        llmModelName={llmModelName}
        onClearAll={handleClearAll}
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
      />

      {/* Sidebar Backdrop Overlay on Mobile */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-45 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* MAIN CONTENT WORKSPACE */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* TOP STATUS NAVIGATION BAR */}
        <header className="flex items-center justify-between border-b border-slate-800/80 px-6 md:px-8 py-3.5 bg-[#070B18]/60 backdrop-blur-md z-10 shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="lg:hidden p-1.5 rounded-lg border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-900 cursor-pointer"
            >
              <Menu className="w-4 h-4" />
            </button>
            <h2 className="text-sm font-extrabold tracking-tight text-white uppercase">
              {activePage}
            </h2>
          </div>

          <div className="flex items-center gap-3">

            {/* Online Status Indicator */}
            <div
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-[10px] font-bold transition-all ${
                onlineStatus
                  ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                  : "bg-red-500/10 border-red-500/20 text-red-400"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${onlineStatus ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
              <span>{onlineStatus ? "Online" : "Offline"}</span>
            </div>
          </div>
        </header>

        {/* WORKSPACE VIEW CONTROLLER */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          {errorMsg && (
            <motion.div
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 bg-red-950/20 border border-red-500/30 text-red-400 rounded-xl text-xs font-semibold"
            >
              {errorMsg}
            </motion.div>
          )}

          <AnimatePresence mode="wait">
            {activePage === "Dashboard" && (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="space-y-6"
              >
                {/* RAG PIPELINE VISUALIZER ON TOP */}
                <Pipeline
                  currentStep={currentStep}
                  isProcessing={isUploading || isQuerying}
                  hasFile={!!uploadedDoc}
                />

                {/* Intermediate processing steps panel (glowing query rewrites) */}
                {rewrittenQuery && enableQueryRewrite && (
                  <motion.div
                    initial={{ opacity: 0, y: 3 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3.5 bg-gradient-to-r from-purple-950/10 to-indigo-950/10 border border-purple-500/25 rounded-xl text-[11px] font-semibold text-slate-300 flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-purple-400 animate-pulse" />
                      <span>
                        LLM Query Rewriter expanded search parameters:
                        <span className="text-slate-100 italic font-bold ml-1.5">"{rewrittenQuery}"</span>
                      </span>
                    </div>
                    <span className="text-[9px] uppercase tracking-widest text-purple-400 font-extrabold border border-purple-400/20 px-2 py-0.5 rounded-full">
                      Query Rewritten
                    </span>
                  </motion.div>
                )}

                {/* TWO COLUMN GRID: CHAT VS CONTEXT */}
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                  {/* Chat input and output display */}
                  <div className="lg:col-span-3 flex flex-col gap-6">
                    <ChatBox
                      messages={chatMessages}
                      onSendMessage={handleSendMessage}
                      isLoading={isQuerying}
                      viewMode="dashboard"
                    />
                  </div>

                  {/* Context and document info */}
                  <div className="lg:col-span-2">
                    <RetrievedContext
                      sources={retrievedSources}
                      retrievedChunks={fullRetrievedChunks}
                      activeDocument={uploadedDoc}
                      ragasScores={ragasScores}
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {activePage === "Chat" && (
              <motion.div
                key="chat"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full"
              >
                {/* ChatGPT scrollable bubble screen */}
                <div className="lg:col-span-3">
                  {rewrittenQuery && enableQueryRewrite && (
                    <div className="mb-2.5 p-2 px-3 bg-purple-950/5 border border-purple-500/15 rounded-xl text-[10.5px] font-semibold text-purple-300">
                      📝 Optimizing search query: <span className="text-slate-200 italic font-bold">"{rewrittenQuery}"</span>
                    </div>
                  )}
                  <ChatBox
                    messages={chatMessages}
                    onSendMessage={handleSendMessage}
                    isLoading={isQuerying}
                    viewMode="chat"
                  />
                </div>

                {/* Similarity preview panel on the side */}
                <div className="lg:col-span-1">
                  <RetrievedContext
                    sources={retrievedSources}
                    retrievedChunks={fullRetrievedChunks}
                    activeDocument={uploadedDoc}
                    ragasScores={ragasScores}
                  />
                </div>
              </motion.div>
            )}

            {activePage === "Documents" && (
              <motion.div
                key="documents"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-6 shadow-xl backdrop-blur-md"
              >
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-sm font-extrabold text-white tracking-wide uppercase">
                      Indexed Documents (Advanced Database)
                    </h3>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Verify indexed vector coordinates and files partitioned by model type
                    </p>
                  </div>

                  {documentList.length > 0 && (
                    <button
                      onClick={handleClearAll}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-red-950/20 border border-red-500/30 text-red-400 text-xs font-semibold rounded-lg hover:bg-red-500/10 transition-all cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>Clear All Collections</span>
                    </button>
                  )}
                </div>

                {documentList.length === 0 ? (
                  <div className="border border-dashed border-slate-800/60 rounded-xl p-12 text-center text-slate-500 text-xs">
                    No documents indexed in the vector store database.
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400 font-bold uppercase tracking-wider">
                          <th className="py-3 px-4">Filename</th>
                          <th className="py-3 px-4">Size</th>
                          <th className="py-3 px-4">Pages</th>
                          <th className="py-3 px-4">Chunks</th>
                          <th className="py-3 px-4">Embedding Model</th>
                          <th className="py-3 px-4">Vectors Indexed</th>
                          <th className="py-3 px-4">Status</th>
                          <th className="py-3 px-4 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-medium">
                        {documentList.map((doc, index) => (
                          <tr key={index} className="hover:bg-slate-900/20 text-slate-300">
                            <td className="py-3 px-4 text-white font-bold max-w-[150px] truncate" title={doc.filename}>
                              {doc.filename}
                            </td>
                            <td className="py-3 px-4">{doc.size}</td>
                            <td className="py-3 px-4">{doc.pages}</td>
                            <td className="py-3 px-4">{doc.chunks}</td>
                            <td className="py-3 px-4 text-purple-400 font-mono">{doc.embedding_model}</td>
                            <td className="py-3 px-4">{doc.vector_count}</td>
                            <td className="py-3 px-4">
                              <span className="flex items-center gap-1 text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20 w-fit text-[10px]">
                                <CheckCircle2 className="w-3 h-3" />
                                <span>{doc.status}</span>
                              </span>
                            </td>
                            <td className="py-3 px-4 text-right">
                              <button
                                onClick={() => handleDeleteDoc(doc.filename)}
                                className="text-red-400 hover:text-red-300 font-bold text-[10px] border border-red-500/20 bg-red-500/5 px-2 py-1 rounded transition-colors cursor-pointer"
                              >
                                Delete
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </motion.div>
            )}

            {activePage === "Settings" && (
              <motion.div
                key="settings"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-6"
              >
                {/* 1. ADVANCED RAG INGESTION CONFIGS */}
                <div className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-6 shadow-xl backdrop-blur-md space-y-4">
                  <div>
                    <h3 className="text-sm font-extrabold text-white tracking-wide uppercase">
                      Advanced Ingestion settings
                    </h3>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Configure text loaders, chunk partitions, and embedding dimensions
                    </p>
                  </div>

                  {/* Embedding Model Selection */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-bold text-slate-300">
                      Embedding Model
                    </label>
                    <select
                      value={embeddingModel}
                      onChange={(e) => setEmbeddingModel(e.target.value)}
                      className="w-full bg-[#0b0d19] border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500/40"
                    >
                      <option value="BAAI/bge-small-en-v1.5">BAAI/bge-small-en-v1.5 (BGE-Small, Default)</option>
                      <option value="sentence-transformers/all-MiniLM-L6-v2">sentence-transformers/all-MiniLM-L6-v2 (MiniLM)</option>
                      <option value="BAAI/bge-large-en-v1.5">BAAI/bge-large-en-v1.5 (BGE-Large, 1024D)</option>
                    </select>
                    <span className="text-[9px] text-slate-500 font-semibold">
                      Note: Switching models changes active database vector dimensions. Index files are isolated.
                    </span>
                  </div>

                  {/* Chunker strategy selection */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-bold text-slate-300">
                      Chunker Split Strategy
                    </label>
                    <select
                      value={chunkerStrategy}
                      onChange={(e) => setChunkerStrategy(e.target.value)}
                      className="w-full bg-[#0b0d19] border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500/40"
                    >
                      <option value="recursive">Recursive Character Splitter</option>
                      <option value="semantic">Semantic Chunking (Sentence embeddings similarity)</option>
                    </select>
                  </div>

                  {/* Chunk Size */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs font-semibold text-slate-300">
                      <span>Chunk Size (Recursive characters)</span>
                      <span className="text-purple-400 font-bold">{chunkSize} chars</span>
                    </div>
                    <input
                      type="range"
                      min="100"
                      max="2000"
                      step="50"
                      value={chunkSize}
                      onChange={(e) => setChunkSize(Number(e.target.value))}
                      disabled={chunkerStrategy === "semantic"}
                      className="w-full accent-purple-500 bg-slate-950 rounded-lg cursor-pointer disabled:opacity-30"
                    />
                  </div>

                  {/* Chunk Overlap */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs font-semibold text-slate-300">
                      <span>Chunk Overlap (Recursive characters)</span>
                      <span className="text-purple-400 font-bold">{chunkOverlap} chars</span>
                    </div>
                    <input
                      type="range"
                      min="10"
                      max="500"
                      step="10"
                      value={chunkOverlap}
                      onChange={(e) => setChunkOverlap(Number(e.target.value))}
                      disabled={chunkerStrategy === "semantic"}
                      className="w-full accent-purple-500 bg-slate-950 rounded-lg cursor-pointer disabled:opacity-30"
                    />
                  </div>
                </div>

                {/* 2. ADVANCED RAG RETRIEVAL & QUERY OPTIONS */}
                <div className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-6 shadow-xl backdrop-blur-md space-y-4">
                  <div>
                    <h3 className="text-sm font-extrabold text-white tracking-wide uppercase">
                      Advanced Retrieval settings
                    </h3>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Configure search retrievers, reranking thresholds, and memory states
                    </p>
                  </div>

                  {/* Retriever type selector */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-bold text-slate-300">
                      Retriever Query Type
                    </label>
                    <select
                      value={retrieverType}
                      onChange={(e) => setRetrieverType(e.target.value)}
                      className="w-full bg-[#0b0d19] border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500/40"
                    >
                      <option value="similarity">Vector Similarity Search</option>
                      <option value="mmr">MMR Search (Maximum Marginal Relevance diversity)</option>
                      <option value="hybrid">BM25 + Vector Hybrid Search (RRF merged)</option>
                    </select>
                  </div>

                  {/* Toggle controls */}
                  <div className="grid grid-cols-2 gap-3.5 pt-1.5">
                    {/* Query Rewrite */}
                    <label className="flex items-center gap-2 cursor-pointer text-xs font-semibold text-slate-300 hover:text-slate-100 transition-colors">
                      <input
                        type="checkbox"
                        checked={enableQueryRewrite}
                        onChange={(e) => setEnableQueryRewrite(e.target.checked)}
                        className="rounded accent-purple-500 bg-slate-950 border-slate-800 w-4 h-4"
                      />
                      <span>Query Rewrite</span>
                    </label>

                    {/* BGE Reranker */}
                    <label className="flex items-center gap-2 cursor-pointer text-xs font-semibold text-slate-300 hover:text-slate-100 transition-colors">
                      <input
                        type="checkbox"
                        checked={enableReranking}
                        onChange={(e) => setEnableReranking(e.target.checked)}
                        className="rounded accent-purple-500 bg-slate-950 border-slate-800 w-4 h-4"
                      />
                      <span>BGE Reranking</span>
                    </label>

                    {/* Chat Memory */}
                    <label className="flex items-center gap-2 cursor-pointer text-xs font-semibold text-slate-300 hover:text-slate-100 transition-colors">
                      <input
                        type="checkbox"
                        checked={enableMemory}
                        onChange={(e) => setEnableMemory(e.target.checked)}
                        className="rounded accent-purple-500 bg-slate-950 border-slate-800 w-4 h-4"
                      />
                      <span>Chat Memory</span>
                    </label>

                    {/* RAGAS Eval */}
                    <label className="flex items-center gap-2 cursor-pointer text-xs font-semibold text-slate-300 hover:text-slate-100 transition-colors">
                      <input
                        type="checkbox"
                        checked={runEvaluation}
                        onChange={(e) => setRunEvaluation(e.target.checked)}
                        className="rounded accent-purple-500 bg-slate-950 border-slate-800 w-4 h-4"
                      />
                      <span>RAGAS Evaluation</span>
                    </label>
                  </div>

                  {/* Similarity Score Threshold */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs font-semibold text-slate-300">
                      <span>Similarity Threshold</span>
                      <span className="text-purple-400 font-bold">{(scoreThreshold * 100).toFixed(0)}% match</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.05"
                      value={scoreThreshold}
                      onChange={(e) => setScoreThreshold(Number(e.target.value))}
                      className="w-full accent-purple-500 bg-slate-950 rounded-lg cursor-pointer"
                    />
                  </div>

                  {/* Top K */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs font-semibold text-slate-300">
                      <span>Retrieve Top K Chunks</span>
                      <span className="text-purple-400 font-bold">{topK} chunks</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="1"
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      className="w-full accent-purple-500 bg-slate-950 rounded-lg cursor-pointer"
                    />
                  </div>

                  {/* MMR fetch K */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs font-semibold text-slate-300">
                      <span>MMR Candidate Fetch K</span>
                      <span className="text-purple-400 font-bold">{fetchK} chunks</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="40"
                      step="1"
                      value={fetchK}
                      onChange={(e) => setFetchK(Number(e.target.value))}
                      disabled={retrieverType !== "mmr" && !enableReranking}
                      className="w-full accent-purple-500 bg-slate-950 rounded-lg cursor-pointer disabled:opacity-30"
                    />
                  </div>
                </div>

                {/* 3. LLM CONFIGS */}
                <div className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-6 shadow-xl backdrop-blur-md space-y-4 md:col-span-2">
                  <div>
                    <h3 className="text-sm font-extrabold text-white tracking-wide uppercase">
                      Groq LLM Configuration
                    </h3>
                    <p className="text-[10px] text-slate-500 mt-0.5">
                      Select model architectures and adjust temperature limits
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Model Select */}
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-semibold text-slate-300">
                        Active LLM Model
                      </label>
                      <select
                        value={llmModelName}
                        onChange={(e) => setLlmModelName(e.target.value)}
                        className="w-full bg-[#0b0d19] border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500/40"
                      >
                        <option value="llama-3.3-70b-versatile">Llama 3.3 70B (llama-3.3-70b-versatile)</option>
                        <option value="mixtral-8x7b-32768">Mixtral 8x7B (mixtral-8x7b-32768)</option>
                        <option value="gemma2-9b-it">Gemma 2 9B (gemma2-9b-it)</option>
                      </select>
                    </div>

                    {/* Temperature */}
                    <div className="flex flex-col gap-1.5">
                      <div className="flex justify-between text-xs font-semibold text-slate-300">
                        <span>LLM Temperature</span>
                        <span className="text-purple-400 font-bold">{temperature}</span>
                      </div>
                      <input
                        type="range"
                        min="0.0"
                        max="1.0"
                        step="0.05"
                        value={temperature}
                        onChange={(e) => setTemperature(Number(e.target.value))}
                        className="w-full accent-purple-500 bg-slate-950 rounded-lg cursor-pointer"
                      />
                    </div>

                    {/* Custom API Key */}
                    <div className="flex flex-col gap-1.5 md:col-span-2">
                      <label className="text-xs font-semibold text-slate-300">
                        Custom Groq API Key (Overrides env key)
                      </label>
                      <input
                        type="password"
                        value={customGroqKey}
                        onChange={(e) => setCustomGroqKey(e.target.value)}
                        placeholder="gsk_..."
                        className="w-full bg-[#0b0d19] border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500/40 placeholder-slate-700"
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activePage === "Comparison" && (
              <motion.div
                key="comparison"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
              >
                <RAGComparison />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
