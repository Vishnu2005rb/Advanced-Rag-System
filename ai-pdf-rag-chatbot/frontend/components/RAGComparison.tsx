import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud,
  Scissors,
  Database,
  Search,
  Cpu,
  Bot,
  Zap,
  RefreshCw,
  Sliders,
  BarChart4,
  AlertTriangle,
  CheckCircle2,
  Info,
  X,
  Layers,
  Sparkles,
} from "lucide-react";

interface RAGStep {
  id: string;
  num: number;
  name: string;
  desc: string;
  icon: React.ReactNode;
  color: string;
  glowColor: string;
  detailedTitle: string;
  normalBehavior: string;
  engineeringBehavior: string;
  impact: string;
  specs: string[];
}

export default function RAGComparison() {
  const [selectedStepId, setSelectedStepId] = useState<string | null>("loader");

  const steps: RAGStep[] = [
    {
      id: "loader",
      num: 1,
      name: "Document Ingestion",
      desc: "Parsing & loading PDFs",
      icon: <UploadCloud className="w-5 h-5" />,
      color: "border-purple-500 text-purple-400 bg-purple-500/10",
      glowColor: "rgba(124, 58, 237, 0.4)",
      detailedTitle: "Document Ingestion & Metadata Processing",
      normalBehavior: "Extracts all PDF text into a single, raw string using basic text-extraction scripts. All page numbers, chapter titles, and source document metadata are discarded.",
      engineeringBehavior: "Loads PDFs page-by-page using PyPDFLoader. Each page's text is wrapped as an independent LangChain Document object and tagged with metadata including source filename, source path, and page number (normalized to 1-indexed).",
      impact: "Critical for citations. Without it, the LLM cannot state exactly which page or document it retrieved facts from, leading to ungrounded claims and a lack of auditability.",
      specs: [
        "Normal: Concatenated raw text dump",
        "Engineering: PyPDFLoader + LangChain Document wrapper",
        "Metadata preserved: filename, page number, source path",
        "Page-level tracking prevents loss of structure"
      ]
    },
    {
      id: "chunker",
      num: 2,
      name: "Text Chunking",
      desc: "Splitting document text",
      icon: <Scissors className="w-5 h-5" />,
      color: "border-blue-500 text-blue-400 bg-blue-500/10",
      glowColor: "rgba(37, 99, 235, 0.4)",
      detailedTitle: "Text Chunking Strategy",
      normalBehavior: "Uses rigid, character-count based splitting (e.g., split every 500 characters). This frequently chops sentences in half, splits numbers (like dates or financial figures) across chunks, and destroys context.",
      engineeringBehavior: "Implements RecursiveCharacterTextSplitter which splits text based on a hierarchy of characters (paragraphs, then sentences, then words) to preserve grammar. Additionally, provides Semantic Chunking which determines break-points using cosine distance between sentence embeddings.",
      impact: "Semantic and Recursive chunking ensures that paragraphs representing a single cohesive topic remain in the same chunk, improving retrieval accuracy and preventing split facts.",
      specs: [
        "Normal: Character-count splitting (breaking words/sentences)",
        "Engineering: Recursive splitting & Semantic boundary splitters",
        "Configurability: User-defined size and overlap overrides",
        "Context tracking: Unique chunk_id ({filename}_chunk_{index}) appended"
      ]
    },
    {
      id: "vector_db",
      num: 3,
      name: "Vector Database",
      desc: "Storing embedding vectors",
      icon: <Database className="w-5 h-5" />,
      color: "border-emerald-500 text-emerald-400 bg-emerald-500/10",
      glowColor: "rgba(34, 197, 94, 0.4)",
      detailedTitle: "Vector Database & Isolation",
      normalBehavior: "Stores generated vector coordinates in an ephemeral in-memory database. Vectors are deleted when the application shuts down or restarts. No partitioning exists; different models overwrite database slots, causing dimension conflicts.",
      engineeringBehavior: "Uses a persistent disk-based ChromaDB. Database folders are isolated under model-safe subdirectories (e.g., `vector_db/BAAI_bge-small-en-v1.5`) to prevent dimension mismatches. Indexes documents using the custom `chunk_id` to allow overwrites/updates without creating duplicate vectors.",
      impact: "Disk persistence avoids recreating embeddings from scratch on server restart. Model isolation guarantees that switching embedding models in Settings won't trigger dimension error crashes.",
      specs: [
        "Normal: Ephemeral memory vectors (lost on reboot)",
        "Engineering: Local disk ChromaDB storage",
        "Directory Isolation: Separate folders for BGE vs MiniLM models",
        "Deduplication: Custom chunk IDs protect index from duplicates"
      ]
    },
    {
      id: "query_rewrite",
      num: 4,
      name: "Query Optimization",
      desc: "Transforming user input",
      icon: <RefreshCw className="w-5 h-5" />,
      color: "border-amber-500 text-amber-400 bg-amber-500/10",
      glowColor: "rgba(245, 158, 11, 0.4)",
      detailedTitle: "LLM Query Transformation",
      normalBehavior: "Sends the raw user question directly to the database query. Conversational pronouns ('What did they do next?', 'How much did it increase?') are searched literally, yielding irrelevant search results.",
      engineeringBehavior: "Uses an LLM (Llama 3.3 via Groq) at query-time to analyze the user's question and the last 4 chat history turns. It rewrites the query into a single, keyword-rich search phrase that resolves pronouns and removes conversational filler.",
      impact: "Improves context matching for follow-up questions. Conversational context is preserved in the vector search query, ensuring relevant chunks are retrieved.",
      specs: [
        "Normal: Raw question searched directly",
        "Engineering: LLM-powered Query Rewriting based on history",
        "Memory Window: Evaluates last 4 message logs",
        "Focus: Coreference resolution (pronoun substitution)"
      ]
    },
    {
      id: "search",
      num: 5,
      name: "Hybrid & MMR Retrieval",
      desc: "Searching for documents",
      icon: <Search className="w-5 h-5" />,
      color: "border-orange-500 text-orange-400 bg-orange-500/10",
      glowColor: "rgba(249, 115, 22, 0.4)",
      detailedTitle: "Hybrid Keyword + Vector Search with MMR",
      normalBehavior: "Runs simple vector similarity search (k-NN). The top K closest vectors are retrieved, regardless of whether they contain exact keyword matches. This frequently misses specific terms (like ID numbers or exact product names) and suffers from redundant chunks.",
      engineeringBehavior: "Implements Hybrid Search combining BM25 keyword matching and dense vector similarity. Merges results using Reciprocal Rank Fusion (RRF). Supports MMR (Maximum Marginal Relevance) to balance relevance and chunk diversity, deduplicating based on chunk_id.",
      impact: "Keyword matching (BM25) ensures exact terms are found, while dense retrieval handles synonyms. MMR prevents sending duplicate information to the LLM.",
      specs: [
        "Normal: Vector similarity only",
        "Engineering: Dense Vector + BM25 Keyword Hybrid search",
        "Deduplication: Reciprocal Rank Fusion (RRF) on chunk_id keys",
        "Diversity: Maximum Marginal Relevance (MMR) selection"
      ]
    },
    {
      id: "reranker",
      num: 6,
      name: "Cross-Encoder Reranking",
      desc: "Re-evaluating chunk relevance",
      icon: <Sliders className="w-5 h-5" />,
      color: "border-pink-500 text-pink-400 bg-pink-500/10",
      glowColor: "rgba(236, 72, 153, 0.4)",
      detailedTitle: "Cross-Encoder Reranking Layer",
      normalBehavior: "No reranking occurs. The raw top K chunks from the vector database are sent straight to the LLM. Vector similarity (bi-encoder cosine distance) is often coarse, ranking partially off-topic chunks near the top.",
      engineeringBehavior: "Performs high-recall candidate retrieval (retrieves top 20 candidates), then passes them through a local CrossEncoder model (`BAAI/bge-reranker-base`). The cross-encoder computes full attention between the query and each chunk to output a precise relevance score.",
      impact: "Guarantees that the highest-quality context is sent to the LLM, squeezing out noise and placing crucial facts at the beginning of the prompt context.",
      specs: [
        "Normal: Coarse vector ranking",
        "Engineering: local CrossEncoder Reranking (BGE Reranker Base)",
        "Compression: Filters top 20 candidate chunks down to top 5",
        "Normalization: Maps raw logits via sigmoid to [0.05 - 0.99]"
      ]
    },
    {
      id: "reasoning",
      num: 7,
      name: "Contextual Reasoning",
      desc: "LLM generation & guardrails",
      icon: <Cpu className="w-5 h-5" />,
      color: "border-teal-500 text-teal-400 bg-teal-500/10",
      glowColor: "rgba(20, 184, 166, 0.4)",
      detailedTitle: "Contextual Reasoning & LLM Guardrails",
      normalBehavior: "Retrieved chunks are pasted into a basic prompt template. The LLM is allowed to generate answers using its pre-trained external knowledge. When the context is missing information, the LLM hallucinates a plausible answer.",
      engineeringBehavior: "Uses `QA_SYSTEM_PROMPT` containing critical grounding guardrails. The prompt strictly instructs the LLM to use only the provided context. If the answer cannot be found in the context, it must answer exactly: 'Information not available in document'. It also forces source citation formatting.",
      impact: "Eliminates hallucination in production. Users are guaranteed that answers are grounded strictly in their uploaded documents, with clear source file and page citations.",
      specs: [
        "Normal: Free-form generation (hallucination risk)",
        "Engineering: Grounded prompt templates with strict rule sets",
        "Fallback: Enforces 'Information not available in document' on context gaps",
        "Citations: Structured source blocks with page indicators"
      ]
    },
    {
      id: "evaluation",
      num: 8,
      name: "RAGAS Audit Audit",
      desc: "Real-time quality checking",
      icon: <BarChart4 className="w-5 h-5" />,
      color: "border-indigo-500 text-indigo-400 bg-indigo-500/10",
      glowColor: "rgba(99, 102, 241, 0.4)",
      detailedTitle: "Real-Time RAGAS Quality Evaluation Audit",
      normalBehavior: "No auditing or quality tracking exists. System administrators and users have no automated way of knowing if the LLM hallucinated, if the retriever returned irrelevant noise, or if the answer missed the user's question.",
      engineeringBehavior: "Automatically audits the QA turn in real-time. Binds the LLM in JSON mode and acts as an independent Judge to calculate scores [0.0 - 1.0] for Faithfulness (groundedness), Answer Relevance, Context Precision (retrieval signal-to-noise), and Context Recall.",
      impact: "Creates a continuous feedback loop. Provides an immediate quality check for every query, flagging potential failures or retrieval issues before they impact the user experience.",
      specs: [
        "Normal: Unaudited LLM outputs",
        "Engineering: Real-time LLM-as-a-Judge RAGAS audit",
        "Metrics: Faithfulness, Relevance, Precision, Recall",
        "JSON Mode: Structured JSON metrics mapped directly in the chat sidebar"
      ]
    }
  ];

  const selectedStep = steps.find((s) => s.id === selectedStepId);

  return (
    <div className="space-y-6">
      {/* Intro Banner */}
      <div className="bg-gradient-to-br from-[#0c0f24] to-[#050814] border-2 border-purple-500/35 rounded-2xl p-6 shadow-[0_0_20px_rgba(124,58,237,0.15)] relative overflow-hidden">
        <div className="absolute -top-10 -right-10 w-40 h-40 bg-purple-600/10 rounded-full blur-3xl pointer-events-none" />
        <div className="flex items-center gap-3 mb-2">
          <Sparkles className="w-6 h-6 text-purple-400 animate-pulse" />
          <h3 className="text-sm font-extrabold text-white tracking-widest uppercase bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-purple-400">
            Normal RAG vs. Production Engineering RAG
          </h3>
        </div>
        <p className="text-xs text-slate-400 font-medium leading-relaxed max-w-4xl">
          Compare the difference between a basic, naive RAG pipeline (often used for quick demos) and a production-grade 
          **Engineering RAG** architecture. Click on any step in either flow to compare the mechanisms, technical specs, and impacts.
        </p>
      </div>

      {/* Side-by-Side Comparison Pipelines */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Left Column: Naive/Normal RAG */}
        <div className="bg-[#0f172a]/40 border border-red-500/20 rounded-2xl p-5 shadow-lg relative overflow-hidden flex flex-col justify-between">
          <div className="absolute -top-12 -left-12 w-28 h-28 bg-red-500/5 rounded-full blur-2xl pointer-events-none" />
          
          <div className="flex items-center justify-between mb-4 border-b border-slate-900 pb-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4.5 h-4.5 text-red-400" />
              <h4 className="text-xs font-black text-red-300 uppercase tracking-widest">
                Naive RAG Pipeline (Demo-Grade)
              </h4>
            </div>
            <span className="text-[9px] font-extrabold text-red-400 bg-red-500/10 px-2 py-0.5 rounded-full border border-red-500/20 uppercase">
              High Risk
            </span>
          </div>

          <div className="flex flex-col gap-3">
            {/* Step 1: Ingestion */}
            <div 
              onClick={() => setSelectedStepId("loader")}
              className={`p-3.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-3.5 ${
                selectedStepId === "loader"
                  ? "bg-red-500/10 border-red-500/50 shadow-[0_0_12px_rgba(239,68,68,0.15)]"
                  : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
              }`}
            >
              <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center shrink-0 border border-slate-800 text-slate-500">
                1
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-bold text-slate-200">Raw Text Extraction</span>
                <span className="text-[10px] text-slate-500 font-medium">Pages concatenated into single text string</span>
              </div>
            </div>

            {/* Step 2: Chunking */}
            <div 
              onClick={() => setSelectedStepId("chunker")}
              className={`p-3.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-3.5 ${
                selectedStepId === "chunker"
                  ? "bg-red-500/10 border-red-500/50 shadow-[0_0_12px_rgba(239,68,68,0.15)]"
                  : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
              }`}
            >
              <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center shrink-0 border border-slate-800 text-slate-500">
                2
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-bold text-slate-200">Rigid Character Splitting</span>
                <span className="text-[10px] text-slate-500 font-medium">Splits exactly at character count limits</span>
              </div>
            </div>

            {/* Step 3: DB */}
            <div 
              onClick={() => setSelectedStepId("vector_db")}
              className={`p-3.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-3.5 ${
                selectedStepId === "vector_db"
                  ? "bg-red-500/10 border-red-500/50 shadow-[0_0_12px_rgba(239,68,68,0.15)]"
                  : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
              }`}
            >
              <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center shrink-0 border border-slate-800 text-slate-500">
                3
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-bold text-slate-200">In-Memory Store</span>
                <span className="text-[10px] text-slate-500 font-medium">Temporary array wiped on server shutdown</span>
              </div>
            </div>

            {/* Step 4: Retrieval */}
            <div 
              onClick={() => setSelectedStepId("search")}
              className={`p-3.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-3.5 ${
                selectedStepId === "search"
                  ? "bg-red-500/10 border-red-500/50 shadow-[0_0_12px_rgba(239,68,68,0.15)]"
                  : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
              }`}
            >
              <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center shrink-0 border border-slate-800 text-slate-500">
                4
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-bold text-slate-200">Basic Vector Similarity</span>
                <span className="text-[10px] text-slate-500 font-medium">Naive k-NN search only (no BM25 or Reranking)</span>
              </div>
            </div>

            {/* Step 5: Generation */}
            <div 
              onClick={() => setSelectedStepId("reasoning")}
              className={`p-3.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-3.5 ${
                selectedStepId === "reasoning"
                  ? "bg-red-500/10 border-red-500/50 shadow-[0_0_12px_rgba(239,68,68,0.15)]"
                  : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
              }`}
            >
              <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center shrink-0 border border-slate-800 text-slate-500">
                5
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-bold text-slate-200">Direct LLM Prompting</span>
                <span className="text-[10px] text-slate-500 font-medium">Raw chunks pushed directly to prompt templates</span>
              </div>
            </div>
          </div>
          
          <div className="mt-5 text-[10px] text-red-400 font-bold bg-red-500/5 p-2 rounded-lg border border-red-500/10 flex items-start gap-1.5 leading-relaxed">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span>Limitations: Highly prone to hallucination, context gaps, pronoun resolution failure, vector dimension crashes, and zero metadata tracing for page sources.</span>
          </div>
        </div>

        {/* Right Column: Engineering RAG */}
        <div className="bg-[#0f172a]/40 border border-purple-500/30 rounded-2xl p-5 shadow-lg relative overflow-hidden flex flex-col justify-between">
          <div className="absolute -top-12 -right-12 w-28 h-28 bg-purple-500/5 rounded-full blur-2xl pointer-events-none" />
          
          <div className="flex items-center justify-between mb-4 border-b border-slate-900 pb-3">
            <div className="flex items-center gap-2">
              <Zap className="w-4.5 h-4.5 text-purple-400 animate-pulse" />
              <h4 className="text-xs font-black text-purple-300 uppercase tracking-widest bg-clip-text text-transparent bg-gradient-to-r from-white via-purple-300 to-purple-400">
                Engineering RAG Pipeline (Production-Grade)
              </h4>
            </div>
            <span className="text-[9px] font-extrabold text-purple-300 bg-purple-500/15 px-2 py-0.5 rounded-full border border-purple-500/30 uppercase tracking-widest shadow-[0_0_8px_rgba(168,85,247,0.2)]">
              Grounded
            </span>
          </div>

          <div className="flex flex-col gap-2.5">
            {/* Step 1 & 2: loader & Chunker */}
            <div className="grid grid-cols-2 gap-2.5">
              <div 
                onClick={() => setSelectedStepId("loader")}
                className={`p-2.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-2.5 ${
                  selectedStepId === "loader"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                    : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
                }`}
              >
                <div className="w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
                  1
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-bold text-slate-200 truncate">Metadata Tagging</span>
                  <span className="text-[9px] text-slate-500 truncate">Page-by-page preservation</span>
                </div>
              </div>

              <div 
                onClick={() => setSelectedStepId("chunker")}
                className={`p-2.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-2.5 ${
                  selectedStepId === "chunker"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                    : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
                }`}
              >
                <div className="w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
                  2
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-bold text-slate-200 truncate">Advanced Chunker</span>
                  <span className="text-[9px] text-slate-500 truncate">Recursive & Semantic splitting</span>
                </div>
              </div>
            </div>

            {/* Step 3 & 4: DB */}
            <div className="grid grid-cols-2 gap-2.5">
              <div 
                onClick={() => setSelectedStepId("vector_db")}
                className={`p-2.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-2.5 ${
                  selectedStepId === "vector_db"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                    : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
                }`}
              >
                <div className="w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
                  3
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-bold text-slate-200 truncate">Model Isolation</span>
                  <span className="text-[9px] text-slate-500 truncate">Persistent divided DB folders</span>
                </div>
              </div>

              <div 
                onClick={() => setSelectedStepId("query_rewrite")}
                className={`p-2.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-2.5 ${
                  selectedStepId === "query_rewrite"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                    : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
                }`}
              >
                <div className="w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
                  4
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-bold text-slate-200 truncate">Query Rewriter</span>
                  <span className="text-[9px] text-slate-500 truncate">Coreference pronoun resolving</span>
                </div>
              </div>
            </div>

            {/* Step 5 & 6: Search & Reranking */}
            <div className="grid grid-cols-2 gap-2.5">
              <div 
                onClick={() => setSelectedStepId("search")}
                className={`p-2.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-2.5 ${
                  selectedStepId === "search"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                    : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
                }`}
              >
                <div className="w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
                  5
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-bold text-slate-200 truncate">Hybrid & MMR Search</span>
                  <span className="text-[9px] text-slate-500 truncate">BM25 + Vector dense search</span>
                </div>
              </div>

              <div 
                onClick={() => setSelectedStepId("reranker")}
                className={`p-2.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-2.5 ${
                  selectedStepId === "reranker"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                    : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
                }`}
              >
                <div className="w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
                  6
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-bold text-slate-200 truncate">BGE Reranking</span>
                  <span className="text-[9px] text-slate-500 truncate">Cross-attention sort relevance</span>
                </div>
              </div>
            </div>

            {/* Step 7 & 8: Reasoning & Eval */}
            <div className="grid grid-cols-2 gap-2.5">
              <div 
                onClick={() => setSelectedStepId("reasoning")}
                className={`p-2.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-2.5 ${
                  selectedStepId === "reasoning"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                    : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
                }`}
              >
                <div className="w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
                  7
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-bold text-slate-200 truncate">Context Grounding</span>
                  <span className="text-[9px] text-slate-500 truncate">Prompt safety guardrails</span>
                </div>
              </div>

              <div 
                onClick={() => setSelectedStepId("evaluation")}
                className={`p-2.5 rounded-xl border transition-all duration-200 cursor-pointer flex items-center gap-2.5 ${
                  selectedStepId === "evaluation"
                    ? "bg-purple-500/10 border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]"
                    : "bg-slate-950/40 border-slate-900 text-slate-400 hover:border-slate-800"
                }`}
              >
                <div className="w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 flex items-center justify-center shrink-0 text-[10px] font-extrabold">
                  8
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="text-[11px] font-bold text-slate-200 truncate">Real-Time RAGAS Audit</span>
                  <span className="text-[9px] text-slate-500 truncate">Quality metrics evaluation</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-5 text-[10px] text-purple-300 font-semibold bg-purple-500/5 p-2 rounded-lg border border-purple-500/10 flex items-start gap-1.5 leading-relaxed">
            <CheckCircle2 className="w-3.5 h-3.5 text-purple-400 shrink-0 mt-0.5" />
            <span>Advantages: Strictly halts hallucinations, provides page and document references, handles follow-up context, combines keyword-vector matches, and calculates live quality audits.</span>
          </div>
        </div>
      </div>

      {/* Comparison Details Panel */}
      <AnimatePresence mode="wait">
        {selectedStep && (
          <motion.div
            key={selectedStep.id}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            className={`border-l-4 border-purple-500 bg-[#0f172a]/80 rounded-r-2xl p-5 shadow-2xl backdrop-blur-md`}
          >
            <div className="flex justify-between items-start mb-3 border-b border-slate-900 pb-2.5">
              <div className="flex items-center gap-2">
                <Info className="w-4.5 h-4.5 text-purple-400 animate-pulse" />
                <h4 className="text-xs font-black text-white uppercase tracking-wider">
                  {selectedStep.detailedTitle}
                </h4>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-4 mt-2">
              <div className="bg-red-500/5 border border-red-500/15 rounded-xl p-3 flex flex-col gap-1.5">
                <span className="text-[9px] font-black text-red-400 uppercase tracking-widest">Naive RAG Mechanism</span>
                <p className="text-xs text-slate-400 font-medium leading-relaxed">
                  {selectedStep.normalBehavior}
                </p>
              </div>

              <div className="bg-purple-500/5 border border-purple-500/15 rounded-xl p-3 flex flex-col gap-1.5">
                <span className="text-[9px] font-black text-purple-300 uppercase tracking-widest">Engineering RAG Mechanism</span>
                <p className="text-xs text-slate-300 font-medium leading-relaxed">
                  {selectedStep.engineeringBehavior}
                </p>
              </div>
            </div>

            <div className="p-3 bg-slate-950/40 rounded-xl border border-slate-900 mb-4 flex items-start gap-2.5">
              <div className="flex items-center justify-center w-6 h-6 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 shrink-0 text-[10px] font-black uppercase mt-0.5">
                Why
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] font-black text-purple-400 uppercase tracking-wider">Architectural Relevance & Impact</span>
                <p className="text-xs text-slate-400 font-medium leading-relaxed mt-0.5">
                  {selectedStep.impact}
                </p>
              </div>
            </div>

            <div className="flex flex-col gap-1.5 pt-1.5 border-t border-slate-900/60">
              <span className="text-[9px] font-extrabold text-slate-500 uppercase tracking-widest">
                Technical Specifications & Comparison
              </span>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[10.5px] text-slate-400 font-semibold list-none pl-0 mt-1">
                {selectedStep.specs.map((spec, sIdx) => (
                  <li key={sIdx} className="flex items-start gap-1.5">
                    <span className="text-purple-400 shrink-0">✦</span>
                    <span>{spec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* RAGAS Tutorial/Explanation Box */}
      <div className="bg-gradient-to-br from-[#0c0f24] to-[#050814] border border-slate-800/80 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="absolute -bottom-10 -right-10 w-36 h-36 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />
        
        <div className="flex items-center gap-2 mb-4 border-b border-slate-900 pb-3">
          <BarChart4 className="w-5 h-5 text-indigo-400 animate-pulse" />
          <h4 className="text-xs font-black text-white uppercase tracking-widest">
            RAGAS Audit Metrics Explanation
          </h4>
        </div>

        <p className="text-xs text-slate-400 leading-relaxed font-medium mb-5">
          **RAGAS (Retrieval Augmented Generation Assessment)** is the industry-standard evaluation framework for auditing 
          the quality of RAG pipelines in real-time. Instead of simple semantic evaluation, RAGAS splits RAG quality 
          assessment into **four core dimensions** to isolate and resolve hallucination, retrieval gaps, and response focus:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Faithfulness */}
          <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-200">1. Faithfulness (Groundedness)</span>
              <span className="text-[9px] font-extrabold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full uppercase">
                Checks Hallucination
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed font-medium">
              Checks whether the generated response is grounded **strictly in the retrieved chunks**. The judge prompts the LLM to count the facts presented in the response, and verifies if every fact can be explicitly deduced from the context chunks.
            </p>
            <div className="text-[10px] text-slate-400 font-semibold italic bg-slate-950 p-2 rounded-lg border border-slate-900/60 mt-1">
              Formula: (Facts in answer supported by context) / (Total facts in answer)
            </div>
          </div>

          {/* Answer Relevance */}
          <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-200">2. Answer Relevance</span>
              <span className="text-[9px] font-extrabold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full uppercase">
                Checks Response Focus
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed font-medium">
              Evaluates how directly the answer addresses the user's question. The model generates semantic questions back from the output and calculates cosine similarity vs the original query. It penalizes redundant, verbose, or off-topic filler.
            </p>
            <div className="text-[10px] text-slate-400 font-semibold italic bg-slate-950 p-2 rounded-lg border border-slate-900/60 mt-1">
              Benefit: Ensures answers are clear, direct, and concise, improving user satisfaction.
            </div>
          </div>

          {/* Context Precision */}
          <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-200">3. Context Precision (Signal-to-Noise)</span>
              <span className="text-[9px] font-extrabold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded-full uppercase">
                Checks Search Accuracy
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed font-medium">
              Measures whether the retrieved chunks are highly relevant to answering the query. It checks if the database placed the most important, fact-rich chunks at the very top of the retrieved set, penalizing noise or irrelevant passages.
            </p>
            <div className="text-[10px] text-slate-400 font-semibold italic bg-slate-950 p-2 rounded-lg border border-slate-900/60 mt-1">
              Formula: (Sum of precision at rank K for all relevant chunks) / (Total relevant chunks)
            </div>
          </div>

          {/* Context Recall */}
          <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-200">4. Context Recall</span>
              <span className="text-[9px] font-extrabold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded-full uppercase">
                Checks Document Coverage
              </span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed font-medium">
              Verifies if the database retriever successfully fetched **all** the necessary facts required to answer the query completely. The judge aligns the response facts with the retrieved context to see what details were missed.
            </p>
            <div className="text-[10px] text-slate-400 font-semibold italic bg-slate-950 p-2 rounded-lg border border-slate-900/60 mt-1">
              Formula: (Ground-truth facts found in retrieved context) / (Total ground-truth facts)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
