import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud,
  Scissors,
  Network,
  Database,
  RefreshCw,
  Search,
  Sliders,
  Cpu,
  BarChart4,
  CheckCircle2,
  Sparkles,
  Info,
  X,
} from "lucide-react";

interface PipelineProps {
  currentStep: number; // 1 to 9
  isProcessing: boolean;
  hasFile: boolean;
}

export default function Pipeline({ currentStep, isProcessing, hasFile }: PipelineProps) {
  const [selectedDetailStep, setSelectedDetailStep] = useState<number | null>(null);

  const steps = [
    {
      num: 1,
      name: "1. PDF Loader",
      desc: "PyPDFLoader + Meta",
      icon: <UploadCloud className="w-5 h-5" />,
      color: "border-purple-500 text-purple-400 bg-purple-500/10",
      glowColor: "rgba(124, 58, 237, 0.45)",
      accentColor: "border-purple-500",
      detailsTitle: "Step 1: Document Loader & Metadata Tagging",
      detailedDesc: "Loads the PDF document using PyPDFLoader. Each page's text is parsed and enriched with metadata keys (filename, page number, source path) to support exact source citation in answers.",
      technicalSpecs: [
        "Library: langchain_community.document_loaders (PyPDFLoader)",
        "Metadata tags: filename, page (1-indexed), source, source_path",
        "Stream handling: Buffered memory bytes saved to temporary workspace directory",
        "Verification: Logs page counts and validation hashes"
      ]
    },
    {
      num: 2,
      name: "2. Chunker",
      desc: "Semantic / Recursive",
      icon: <Scissors className="w-5 h-5" />,
      color: "border-blue-500 text-blue-400 bg-blue-500/10",
      glowColor: "rgba(37, 99, 235, 0.45)",
      accentColor: "border-blue-500",
      detailsTitle: "Step 2: Advanced Chunker strategies",
      detailedDesc: "Segments text pages into manageable chunks. Supports standard Recursive Splitting (character limits) and Semantic Chunking (breaking paragraphs dynamically based on embedding distances).",
      technicalSpecs: [
        "Recursive: RecursiveCharacterTextSplitter (chunk_size, chunk_overlap)",
        "Semantic: langchain_experimental SemanticChunker (embedding boundaries)",
        "Indexing: Appends incremental unique chunk_id to each chunk",
        "Integrity: Prevents split text from breaking mid-topic boundaries"
      ]
    },
    {
      num: 3,
      name: "3. Embeddings",
      desc: "BGE / MiniLM Models",
      icon: <Network className="w-5 h-5" />,
      color: "border-cyan-500 text-cyan-400 bg-cyan-500/10",
      glowColor: "rgba(6, 182, 212, 0.45)",
      accentColor: "border-cyan-500",
      detailsTitle: "Step 3: Multi-Model Embeddings Generator",
      detailedDesc: "Translates split text chunks into vector coordinates. Supports multiple models (BGE-small, MiniLM-L6, and BGE-large) dynamically cached to avoid reload delays.",
      technicalSpecs: [
        "Models: BAAI/bge-small-en-v1.5, sentence-transformers, BAAI/bge-large-en-v1.5",
        "Dimensions: 384 dimensions (small/MiniLM) or 1024 dimensions (large)",
        "Caching: Singleton dictionary caching in python memory",
        "Auditing: Measures and returns embedding generation duration"
      ]
    },
    {
      num: 4,
      name: "4. Vector Store",
      desc: "Persistent Chroma",
      icon: <Database className="w-5 h-5" />,
      color: "border-emerald-500 text-emerald-400 bg-emerald-500/10",
      glowColor: "rgba(34, 197, 94, 0.45)",
      accentColor: "border-emerald-500",
      detailsTitle: "Step 4: Persistent Vector Database",
      detailedDesc: "Persists the generated embedding vectors and chunk metadata to a folder. Databases are isolated by embedding model names to prevent vector dimension mismatches.",
      technicalSpecs: [
        "Engine: ChromaDB (langchain_chroma)",
        "Persistence: Saved locally inside the backend directory under vector_db/",
        "ID mapping: Uses chunk_id keys to allow document overwrites/updates",
        "Cleanup: Supports single-document deletes and global collection resets"
      ]
    },
    {
      num: 5,
      name: "5. Query Rewrite",
      desc: "LLM Transform",
      icon: <RefreshCw className="w-5 h-5" />,
      color: "border-amber-500 text-amber-400 bg-amber-500/10",
      glowColor: "rgba(245, 158, 11, 0.45)",
      accentColor: "border-amber-500",
      detailsTitle: "Step 5: LLM Query Transformation",
      detailedDesc: "Rewrites imprecise user questions into search-optimized keywords. Uses Llama 3.3 via Groq to resolve coreferences (like 'it' or 'who') based on past history turns.",
      technicalSpecs: [
        "Inference: ChatGroq (llama-3.3-70b-versatile)",
        "History limit: Evaluates the last 4 turns of memory",
        "Clean query: Outputs only the search-optimized query without comments",
        "Bypass: Can be enabled or disabled in Settings"
      ]
    },
    {
      num: 6,
      name: "6. Retriever",
      desc: "BM25 + Vector Hybrid",
      icon: <Search className="w-5 h-5" />,
      color: "border-orange-500 text-orange-400 bg-orange-500/10",
      glowColor: "rgba(249, 115, 22, 0.45)",
      accentColor: "border-orange-500",
      detailsTitle: "Step 6: Hybrid Search & MMR Retrieval",
      detailedDesc: "Combines BM25 token keyword matches with Chroma vector search. Merges both retrieval streams using Reciprocal Rank Fusion (RRF) and supports MMR for selection diversity.",
      technicalSpecs: [
        "Keyword: BM25Retriever (initialized on-the-fly from active chunks)",
        "Vector search: Cosine similarity search or MMR diversity search",
        "Fusion: Reciprocal Rank Fusion (RRF) sorting",
        "Filtering: Rejects chunks below configurable similarity thresholds (e.g. 0.70)"
      ]
    },
    {
      num: 7,
      name: "7. Reranker",
      desc: "BGE Cross-Encoder",
      icon: <Sliders className="w-5 h-5" />,
      color: "border-pink-500 text-pink-400 bg-pink-500/10",
      glowColor: "rgba(236, 72, 153, 0.45)",
      accentColor: "border-pink-500",
      detailsTitle: "Step 7: BGE Cross-Encoder Reranking",
      detailedDesc: "Scores the top 20 candidate chunks against the optimized search query. Unlike bi-encoders, the cross-encoder runs full cross-attention over query-context pairs for high relevance sorting.",
      technicalSpecs: [
        "Model: sentence-transformers CrossEncoder (BAAI/bge-reranker-base)",
        "Compression: Reranks top 20 retrieved candidates down to the best 5",
        "Normalization: Maps logits via sigmoid into [0.05 - 0.99] scores",
        "Resource: Loaded locally and cached in python memory"
      ]
    },
    {
      num: 8,
      name: "8. Reasoning",
      desc: "Llama 3.3 70B Groq",
      icon: <Cpu className="w-5 h-5" />,
      color: "border-teal-500 text-teal-400 bg-teal-500/10",
      glowColor: "rgba(20, 184, 166, 0.45)",
      accentColor: "border-teal-500",
      detailsTitle: "Step 8: Contextual Reasoning & Response Generation",
      detailedDesc: "Combines the reranked chunks, query, and history into a strict context prompt. Llama 3.3 parses the context and answers using only the provided facts, creating structural sources lists.",
      technicalSpecs: [
        "Model: ChatGroq (llama-3.3-70b-versatile)",
        "Prompt: QA_SYSTEM_PROMPT with strict context limits",
        "Safety: Fallback to 'Information not available' on context gaps",
        "Formatting: Bullet points, bold headers, and page citations"
      ]
    },
    {
      num: 9,
      name: "9. Evaluation",
      desc: "RAGAS Scores Audit",
      icon: <BarChart4 className="w-5 h-5" />,
      color: "border-indigo-500 text-indigo-400 bg-indigo-500/10",
      glowColor: "rgba(99, 102, 241, 0.45)",
      accentColor: "border-indigo-500",
      detailsTitle: "Step 9: RAGAS Quality Evaluation Audit",
      detailedDesc: "Audits the quality of the generated response in real-time using LLM-as-a-Judge patterns. Computes performance metrics on a 0-100% scale to detect hallucinations, query focus, and context completeness before results are delivered.",
      technicalSpecs: [
        "Faithfulness (Checks Hallucination): Verifies if the answer contains only facts explicitly mentioned in context chunks.",
        "Answer Relevance (Checks Focus): Measures how directly the response addresses the prompt's question, penalizing verbose filler.",
        "Context Precision (Checks Search Accuracy): Evaluates if retrieved context chunks are highly relevant, placing crucial facts at the top.",
        "Context Recall (Checks Document Coverage): Verifies if retrieved context contains all necessary facts to answer the question fully."
      ]
    },
  ];

  const handleStepClick = (stepNum: number) => {
    setSelectedDetailStep((prev) => (prev === stepNum ? null : stepNum));
  };

  const getProgressPercentage = () => {
    if (!hasFile) return 0;
    if (currentStep >= 9) return 100;
    return Math.round(((currentStep - 1) / 8) * 100);
  };

  const trackWidth = getProgressPercentage();
  const selectedStepData = steps.find((s) => s.num === selectedDetailStep);

  return (
    <div className="w-full bg-gradient-to-br from-[#0c0f24] to-[#050814] border-2 border-purple-500/35 rounded-2xl p-5 shadow-[0_0_30px_rgba(124,58,237,0.2)] backdrop-blur-md relative overflow-hidden flex flex-col gap-4">
      {/* Decorative Neon Accents */}
      <div className="absolute -top-10 -right-10 w-40 h-40 bg-purple-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-10 -left-10 w-40 h-40 bg-cyan-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header Banner */}
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400 animate-spin [animation-duration:10s]" />
            <h3 className="text-sm font-black text-white tracking-widest uppercase bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-purple-400">
              ADVANCED RAG PIPELINE
            </h3>
          </div>
          <p className="text-[11px] text-slate-400 font-medium mt-1">
            Click any pipeline stage below to view its technical specifications and algorithms
          </p>
        </div>

        {/* Action Status Check */}
        <div className="hidden sm:block">
          {currentStep >= 6 && hasFile ? (
            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full shadow-[0_0_15px_rgba(34,197,94,0.1)]">
              Pipeline: Active & Grounded
            </span>
          ) : isProcessing ? (
            <span className="text-[10px] font-bold text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-3 py-1 rounded-full animate-pulse">
              Pipeline: Computing
            </span>
          ) : (
            <span className="text-[10px] font-bold text-slate-500 bg-slate-950 border border-slate-900 px-3 py-1 rounded-full">
              Pipeline: Awaiting File
            </span>
          )}
        </div>
      </div>

      {/* Steps Grid */}
      <div className="flex items-stretch justify-between gap-1.5 w-full overflow-x-auto pb-4 pt-1 scrollbar-thin">
        {steps.map((step, idx) => {
          const isCompleted = hasFile && step.num < currentStep;
          const isActive = hasFile && step.num === currentStep;
          const isSelected = selectedDetailStep === step.num;

          let cardClass = "border-slate-800/80 bg-slate-950/20 text-slate-500 cursor-pointer";
          let iconClass = "bg-slate-900 border-slate-800 text-slate-600";
          let nameColor = "text-slate-500";
          let descColor = "text-slate-600";
          let glowShadow = {};

          if (isCompleted) {
            cardClass = "border-emerald-500/50 bg-emerald-950/5 text-emerald-400 cursor-pointer";
            iconClass = "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
            nameColor = "text-slate-200 font-extrabold";
            descColor = "text-slate-400";
          } else if (isActive) {
            cardClass = "border-purple-500 bg-purple-950/15 text-purple-400 cursor-pointer";
            iconClass = "bg-purple-500/20 text-purple-300 border-purple-500/40";
            nameColor = "text-white font-black";
            descColor = "text-slate-200";
            glowShadow = {
              boxShadow: `0 0 20px ${step.glowColor}`,
            };
          }

          if (isSelected) {
            cardClass += " ring-2 ring-purple-400 border-transparent bg-purple-950/30";
          }

          return (
            <React.Fragment key={step.num}>
              {/* Step Card Wrapper */}
              <motion.div
                onClick={() => handleStepClick(step.num)}
                whileHover={{ y: -3, scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                style={glowShadow}
                className={`flex flex-col items-center text-center p-3 rounded-2xl border-2 flex-1 min-w-[110px] max-w-[130px] transition-all duration-300 relative ${cardClass}`}
              >
                {/* Active Indicator Glow */}
                {isActive && (
                  <span className="absolute -top-1 -right-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-purple-500"></span>
                  </span>
                )}

                {/* Step Icon */}
                <div
                  className={`flex items-center justify-center w-10 h-10 rounded-xl border-2 mb-2.5 transition-all ${iconClass}`}
                >
                  {isCompleted ? (
                    <motion.span
                      initial={{ scale: 0, rotate: -45 }}
                      animate={{ scale: 1, rotate: 0 }}
                      className="text-emerald-400 font-bold"
                    >
                      ✓
                    </motion.span>
                  ) : (
                    step.icon
                  )}
                </div>

                <span className={`text-[10px] tracking-tight leading-none ${nameColor}`}>
                  {step.name}
                </span>
                <span className={`text-[8.5px] leading-snug mt-1.5 font-medium ${descColor}`}>
                  {step.desc}
                </span>
              </motion.div>

              {/* Linking arrow */}
              {idx < steps.length - 1 && (
                <div
                  className={`text-xs font-black px-1 flex items-center justify-center shrink-0 transition-colors ${
                    hasFile && currentStep > step.num
                      ? "text-emerald-400 drop-shadow-[0_0_5px_rgba(52,211,153,0.5)]"
                      : hasFile && currentStep === step.num
                      ? "text-purple-400 drop-shadow-[0_0_5px_rgba(124,58,237,0.5)] animate-pulse"
                      : "text-slate-800"
                  }`}
                >
                  ➔
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Connecting Timeline Progress Bar */}
      <div className="relative px-6 mb-2">
        <div className="absolute top-1/2 left-6 right-6 h-[4px] bg-slate-900 -translate-y-1/2 rounded-full" />
        
        <motion.div
          initial={{ width: "0%" }}
          animate={{ width: `${trackWidth}%` }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="absolute top-1/2 left-6 h-[4px] bg-gradient-to-r from-emerald-400 via-cyan-400 to-purple-500 -translate-y-1/2 rounded-full shadow-[0_0_12px_rgba(34,197,94,0.65)]"
        />

        <div className="relative flex justify-between w-full">
          {steps.map((step) => {
            const isCompleted = hasFile && step.num < currentStep;
            const isActive = hasFile && step.num === currentStep;

            let dotClass = "bg-slate-950 border-slate-800 text-slate-500";
            if (isCompleted) {
              dotClass = "bg-emerald-500 border-emerald-400 text-white shadow-[0_0_12px_rgba(34,197,94,0.8)]";
            } else if (isActive) {
              dotClass = "bg-purple-500 border-purple-400 text-white shadow-[0_0_12px_rgba(124,58,237,0.8)] scale-110";
            }

            return (
              <div
                key={step.num}
                className={`relative z-10 flex items-center justify-center w-6 h-6 rounded-full border-2 text-[10px] font-black transition-all duration-300 ${dotClass}`}
              >
                {isCompleted ? "✓" : step.num}
              </div>
            );
          })}
        </div>
      </div>

      {/* Details Box Section */}
      <AnimatePresence mode="wait">
        {selectedStepData && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className={`border-l-4 ${selectedStepData.accentColor} bg-[#0b0d19]/80 rounded-r-2xl p-4.5 shadow-inner mt-2`}
          >
            <div className="flex justify-between items-start mb-2.5">
              <div className="flex items-center gap-2">
                <Info className="w-4 h-4 text-purple-400" />
                <h4 className="text-xs font-black text-white uppercase tracking-wider">
                  {selectedStepData.detailsTitle}
                </h4>
              </div>
              <button
                onClick={() => setSelectedDetailStep(null)}
                className="text-slate-500 hover:text-white transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed font-medium mb-3">
              {selectedStepData.detailedDesc}
            </p>

            <div className="flex flex-col gap-1.5 border-t border-slate-900/60 pt-3">
              <span className="text-[10px] font-extrabold text-slate-400 uppercase tracking-widest">
                Technical Specifications
              </span>
              <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[10.5px] text-slate-400 font-semibold list-none pl-0 mt-1">
                {selectedStepData.technicalSpecs.map((spec, sIdx) => (
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

      {/* Highlighted Banner Status */}
      <div className="flex justify-center mt-1">
        {currentStep >= 8 && hasFile ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-2 font-black text-xs text-emerald-400 bg-emerald-500/10 border-2 border-emerald-500/30 px-5 py-2 rounded-full shadow-[0_0_20px_rgba(34,197,94,0.18)]"
          >
            <CheckCircle2 className="w-4.5 h-4.5 animate-bounce" />
            <span className="tracking-wide uppercase">Pipeline Active & Grounded ✓</span>
          </motion.div>
        ) : isProcessing ? (
          <motion.div
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 1.2, repeat: Infinity }}
            className="text-xs font-black text-cyan-400 bg-cyan-500/10 border-2 border-cyan-500/30 px-5 py-2 rounded-full flex items-center gap-2 shadow-[0_0_20px_rgba(6,182,212,0.15)]"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping shrink-0" />
            <span className="tracking-wide uppercase">Processing: RAG Execution (Step {currentStep}/9)</span>
          </motion.div>
        ) : (
          <div className="text-xs font-black text-slate-500 bg-slate-950 border border-slate-900 px-5 py-2 rounded-full flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-slate-800" />
            <span className="tracking-wide uppercase">Pipeline Idle - Awaiting Document</span>
          </div>
        )}
      </div>
    </div>
  );
}
